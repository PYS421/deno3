import json
import sys
import random
from pathlib import Path

import torch
import swanlab

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    CONFIG,
    TRAIN_FILE,
    DEV_FILE,
    OUTPUT_DIR,
    LOG_DIR,
    RESULT_DIR,
    MAX_LENGTH,
    MAX_NEW_TOKENS,
    SEED,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BF16,
    DATALOADER_NUM_WORKERS,
    SAVE_TOTAL_LIMIT,
    WARMUP_STEPS,
    MAX_GRAD_NORM,
    SWANLAB_PROJECT,
    SWANLAB_EXPERIMENT_NAME,
)

from src.model import load_tokenizer, build_model
from src.dataset import Mydataset
from src.collator import Mycollator
from src.trainer import CustomTrainer


def load_eval_samples(file_path):

    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(
            f"验证集文件不存在：{file_path}"
        )

    try:
        with file_path.open("r", encoding="utf-8") as file:
            samples = json.load(file)

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"验证集不是有效的 JSON 文件：{file_path}，"
            f"第 {exc.lineno} 行，第 {exc.colno} 列。"
        ) from exc

    if not isinstance(samples, list):
        raise ValueError(
            f"验证集顶层必须是 JSON 数组：{file_path}"
        )

    if not samples:
        raise ValueError(
            f"验证集为空：{file_path}"
        )

    for index, item in enumerate(samples, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"验证集第 {index} 条数据必须是 JSON 对象。"
            )

        messages = item.get("messages")

        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError(
                f"验证集第 {index} 条数据缺少有效的 messages 字段。"
            )

        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(
                    f"验证集第 {index} 条数据的消息格式错误。"
                )

            if "role" not in message or "content" not in message:
                raise ValueError(
                    f"验证集第 {index} 条消息缺少 role 或 content 字段。"
                )

        if messages[-1].get("role") != "assistant":
            raise ValueError(
                f"验证集第 {index} 条数据的最后一条消息"
            )

    print(
        f"验证集 JSON 读取成功：{file_path.name}，"
        f"样本数={len(samples)}"
    )

    return samples


def main():
    method = str(CONFIG.get("method", "lora")).lower()

    if method not in {"lora", "qlora"}:
        raise ValueError(
            f"不支持的微调方式：{method}，只支持 lora 或 qlora。"
        )

    method_name = method.upper()
    model_name = f"Qwen2.5-7B-Instruct-{method_name}"

    print("=" * 60)
    print(f"Qwen2.5-7B-Instruct {method_name} NER 微调")
    print("=" * 60)

    print(f"微调方式：{method_name}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"训练数据：{TRAIN_FILE}")
    print(f"验证数据：{DEV_FILE}")

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 CUDA GPU，停止训练。")

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


    for directory in (OUTPUT_DIR, LOG_DIR, RESULT_DIR):
        Path(directory).mkdir(
            parents=True,
            exist_ok=True,
        )

    swanlab_run = None

    try:
        print("\n[1/6] 加载 Tokenizer")

        tokenizer = load_tokenizer()

        print("词表大小：", len(tokenizer))
        print("Pad token ID：", tokenizer.pad_token_id)

        print("\n[2/6] 加载训练数据")

        train_dataset = Mydataset(
            data_path=TRAIN_FILE,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
            use_cache=True,
        )

        if len(train_dataset) == 0:
            raise ValueError("训练数据集为空，停止训练。")

        print("训练样本数：", len(train_dataset))

        print("\n[3/6] 加载验证数据")

        eval_samples = load_eval_samples(DEV_FILE)

        print("验证样本数：", len(eval_samples))

        print("\n[4/6] 加载模型和 Collator")

        data_collator = Mycollator(tokenizer)
        model = build_model()

        trainable_params = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        total_params = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        print(f"模型总参数量：{total_params:,}")
        print(f"可训练参数量：{trainable_params:,}")
        print("GPU：", torch.cuda.get_device_name(0))

        if trainable_params == 0:
            raise RuntimeError(
                "没有可训练参数，请检查 LoRA/QLoRA 配置。"
            )

        base_model = model.get_base_model()
        is_loaded_in_4bit = getattr(
            base_model,
            "is_loaded_in_4bit",
            False,
        )

        print(f"实际 4-bit 量化状态：{is_loaded_in_4bit}")

        if method == "qlora" and not is_loaded_in_4bit:
            raise RuntimeError(
                "当前配置指定 QLoRA，但基座模型未检测到 4-bit 量化。"
            )

        if method == "lora" and is_loaded_in_4bit:
            raise RuntimeError(
                "当前配置指定 LoRA，但基座模型被加载为 4-bit。"
            )

        print("\n[5/6] 初始化 SwanLab")

        try:
            swanlab_run = swanlab.init(
                project=SWANLAB_PROJECT,
                experiment_name=SWANLAB_EXPERIMENT_NAME,
                config={
                    "model": model_name,
                    "method": method_name,
                    "train_samples": len(train_dataset),
                    "eval_samples": len(eval_samples),
                    "max_length": MAX_LENGTH,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "batch_size": BATCH_SIZE,
                    "gradient_accumulation_steps":
                        GRADIENT_ACCUMULATION_STEPS,
                    "epochs": EPOCHS,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "seed": SEED,
                    "bf16": BF16,
                    "gpu": torch.cuda.get_device_name(0),
                    "output_dir": str(OUTPUT_DIR),
                },
            )

        except Exception as exc:
            print(
                "SwanLab 初始化失败，将继续训练但不记录指标："
                f"{exc}"
            )
            swanlab_run = None

        print("\n[6/6] 初始化自定义 Trainer")

        trainer = CustomTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_samples=eval_samples,
            data_collator=data_collator,
            output_dir=OUTPUT_DIR,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            gradient_accumulation_steps=(
                GRADIENT_ACCUMULATION_STEPS
            ),
            warmup_steps=WARMUP_STEPS,
            max_grad_norm=MAX_GRAD_NORM,
            max_new_tokens=MAX_NEW_TOKENS,
            num_workers=DATALOADER_NUM_WORKERS,
            bf16=BF16,
            seed=SEED,
            save_total_limit=SAVE_TOTAL_LIMIT,
            early_stopping_patience=int(CONFIG["early_stopping_patience"]),
            swanlab_run=swanlab_run,
            lr_scheduler_type=CONFIG["lr_scheduler_type"],
        )

        print("\n" + "=" * 60)
        print(f"开始正式训练：{method_name}")
        print(f"Epochs：{EPOCHS}")
        print(f"Batch size：{BATCH_SIZE}")
        print(
            "Gradient accumulation："
            f"{GRADIENT_ACCUMULATION_STEPS}"
        )
        print(f"输出目录：{OUTPUT_DIR}")
        print(f"日志目录：{LOG_DIR}")
        print(f"结果目录：{RESULT_DIR}")
        print("=" * 60, flush=True)

        result = trainer.train()

        result_path = (
            Path(RESULT_DIR) / "training_summary.json"
        )

        with result_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print("\n训练结束。")
        print("结果文件：", result_path)
        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        if swanlab_run is not None:
            try:
                swanlab.finish()
            except Exception as exc:
                print(f"SwanLab 关闭提示：{exc}")


if __name__ == "__main__":
    main()

