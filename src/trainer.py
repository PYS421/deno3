
import json
import math
import random
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.metrics import calculate_entity_metrics


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CustomTrainer:

    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        eval_samples,
        data_collator,
        output_dir,
        learning_rate,
        weight_decay,
        epochs,
        batch_size,
        gradient_accumulation_steps,
        warmup_steps,
        max_grad_norm,
        max_new_tokens,
        num_workers,
        bf16,
        seed,
        save_total_limit,
        early_stopping_patience,
        swanlab_run,
        lr_scheduler_type,
        memory_monitor=None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("没有检测到 CUDA GPU。")

        if gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps 必须大于 0。"
            )

        if epochs < 1:
            raise ValueError("epochs 必须大于 0。")

        if lr_scheduler_type not in {
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        }:
            raise ValueError(
                f"不支持的学习率调度器：{lr_scheduler_type}"
            )

        set_seed(seed)

        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_samples = eval_samples
        self.data_collator = data_collator

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.max_new_tokens = max_new_tokens
        self.num_workers = num_workers
        self.save_total_limit = save_total_limit
        self.early_stopping_patience = early_stopping_patience
        self.swanlab_run = swanlab_run
        self.lr_scheduler_type = lr_scheduler_type
        self.memory_monitor = memory_monitor

        self.use_bf16 = (
            bf16 and torch.cuda.is_bf16_supported()
        )

        self.device = next(
            model.parameters()
        ).device

        self.global_step = 0
        self.best_f1 = -1.0
        self.best_epoch = 0

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=data_collator,
            num_workers=num_workers,
            pin_memory=True,
        )

        if len(self.train_loader) == 0:
            raise ValueError("训练数据集为空。")

        self.updates_per_epoch = math.ceil(
            len(self.train_loader)
            / self.gradient_accumulation_steps
        )

        self.total_steps = (
            self.updates_per_epoch * self.epochs
        )

        trainable_params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        if not trainable_params:
            raise ValueError(
                "没有可训练参数，请检查 LoRA 配置。"
            )

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.scheduler = get_scheduler(
            name=self.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        print("CustomTrainer 初始化完成")
        print(f"训练样本数：{len(self.train_dataset)}")
        print(f"验证样本数：{len(self.eval_samples)}")
        print(f"每轮 batch 数：{len(self.train_loader)}")
        print(f"每轮优化器更新次数：{self.updates_per_epoch}")
        print(f"计划总更新次数：{self.total_steps}")
        print(f"学习率调度器：{self.lr_scheduler_type}")
        print(f"BF16：{self.use_bf16}")
        print(f"输出目录：{self.output_dir}")

    def _log(self, metrics):
        print(
            json.dumps(
                metrics,
                ensure_ascii=False,
                indent=2,
            )
        )

        if self.swanlab_run is not None:
            self.swanlab_run.log(metrics)

    def train(self, resume_from_checkpoint=None):
        start_epoch = 0
        stale_epochs = 0

        if resume_from_checkpoint is not None:
            start_epoch = self.load_checkpoint(
                resume_from_checkpoint
            )

        if self.memory_monitor is not None:
            self.memory_monitor.start()

        for epoch in range(start_epoch, self.epochs):
            if self.memory_monitor is not None:
                self.memory_monitor.mark(
                    "train",
                    epoch=epoch + 1,
                    global_step=self.global_step,
                )

            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            epoch_loss = 0.0
            epoch_batches = 0

            num_batches = len(self.train_loader)

            for batch_index, batch in enumerate(
                self.train_loader
            ):
                batch = {
                    key: value.to(self.device)
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in batch.items()
                }

                group_start = (
                    batch_index
                    // self.gradient_accumulation_steps
                ) * self.gradient_accumulation_steps

                group_size = min(
                    self.gradient_accumulation_steps,
                    num_batches - group_start,
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16,
                ):
                    outputs = self.model(**batch)
                    loss = outputs.loss

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"epoch={epoch + 1}, "
                        f"batch={batch_index + 1}, "
                        f"loss={loss.detach().item()}"
                    )

                epoch_loss += loss.detach().float().item()
                epoch_batches += 1

                (loss / group_size).backward()

                should_update = (
                    (batch_index + 1)
                    % self.gradient_accumulation_steps == 0
                    or batch_index + 1 == num_batches
                )

                if should_update:
                    torch.nn.utils.clip_grad_norm_(
                        [
                            parameter
                            for parameter in self.model.parameters()
                            if parameter.requires_grad
                        ],
                        self.max_grad_norm,
                    )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.global_step += 1

            train_loss = (
                epoch_loss / max(epoch_batches, 1)
            )

            learning_rate = (
                self.optimizer.param_groups[0]["lr"]
            )

            print(
                f"\nEpoch {epoch + 1}/{self.epochs} "
                f"| train_loss={train_loss:.6f} "
                f"| lr={learning_rate:.8g}"
            )

            if self.memory_monitor is not None:
                self.memory_monitor.mark(
                    "evaluation",
                    epoch=epoch + 1,
                    global_step=self.global_step,
                )

            eval_metrics = self.evaluate()

            metrics = {
                **eval_metrics,
                "epoch": epoch + 1,
                "global_step": self.global_step,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
            }

            self._log(metrics)

            current_f1 = eval_metrics["eval_f1"]

            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.best_epoch = epoch + 1
                stale_epochs = 0

                self.save_checkpoint(
                    self.output_dir / "best",
                    epoch=epoch + 1,
                    is_best=True,
                )

                print(
                    f"最佳验证 F1 更新为 "
                    f"{self.best_f1:.6f}"
                )
            else:
                stale_epochs += 1

            self.save_checkpoint(
                self.output_dir
                / f"checkpoint-epoch-{epoch + 1}",
                epoch=epoch + 1,
                is_best=False,
            )

            self._limit_checkpoints()

            if (
                self.early_stopping_patience is not None
                and stale_epochs
                >= self.early_stopping_patience
            ):
                print("验证集 F1 未提升，提前停止训练。")
                break

        print(
            f"\n训练结束：best_epoch={self.best_epoch}, "
            f"best_f1={self.best_f1:.6f}"
        )

        return {
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
            "global_step": self.global_step,
        }

    @torch.inference_mode()
    def evaluate(self):
        was_training = self.model.training
        self.model.eval()

        predictions = []
        references = []

        try:
            for sample in self.eval_samples:
                messages = sample.get("messages")

                if not isinstance(messages, list) or not messages:
                    raise ValueError(
                        "验证样本缺少有效的 messages。"
                    )

                if messages[-1].get("role") != "assistant":
                    raise ValueError(
                        "验证样本最后一条消息必须是 assistant。"
                    )

                if not any(
                    item.get("role") == "user"
                    for item in messages
                ):
                    raise ValueError(
                        "验证样本缺少 user 消息。"
                    )

                prompt_messages = messages[:-1]

                prompt_text = (
                    self.tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )

                inputs = self.tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    add_special_tokens=False,
                )

                inputs = {
                    key: value.to(self.device)
                    for key, value in inputs.items()
                }

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.use_bf16,
                ):
                    generated = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                prompt_length = inputs["input_ids"].shape[1]

                generated_tokens = generated[
                    0, prompt_length:
                ]

                prediction_text = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                )

                predictions.append(prediction_text)
                references.append(messages[-1]["content"])

                del inputs, generated, generated_tokens

        finally:
            if was_training:
                self.model.train()

        return calculate_entity_metrics(
            predictions,
            references,
        )

    def save_checkpoint(self, path, epoch, is_best=False):
        path = Path(path)
        path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        state = {
            "epoch": epoch,
            "global_step": self.global_step,
            "best_f1": self.best_f1,
            "best_epoch": self.best_epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }

        torch.save(
            state,
            path / "trainer_state.pt",
        )

        with open(
            path / "trainer_state.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "best_f1": self.best_f1,
                    "best_epoch": self.best_epoch,
                    "is_best": is_best,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    def load_checkpoint(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        state_file = checkpoint_path / "trainer_state.pt"

        if not state_file.is_file():
            raise FileNotFoundError(
                f"训练状态文件不存在：{state_file}"
            )

        state = torch.load(
            state_file,
            map_location="cpu",
            weights_only=False,
        )

        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])

        self.global_step = state["global_step"]
        self.best_f1 = state["best_f1"]
        self.best_epoch = state["best_epoch"]

        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])

        if (
            state["cuda_rng"] is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(state["cuda_rng"])

        print(
            f"训练状态已读取：{checkpoint_path}\n"
            f"上次完成 epoch：{state['epoch']}"
        )

        return state["epoch"]

    def _limit_checkpoints(self):
        if (
            self.save_total_limit is None
            or self.save_total_limit < 1
        ):
            return

        checkpoints = sorted(
            self.output_dir.glob("checkpoint-epoch-*"),
            key=lambda path: int(
                path.name.rsplit("-", 1)[-1]
            ),
        )

        while len(checkpoints) > self.save_total_limit:
            old_checkpoint = checkpoints.pop(0)

            if old_checkpoint.exists():
                shutil.rmtree(old_checkpoint)
                print(
                    f"已删除旧 checkpoint：{old_checkpoint}"
                )
