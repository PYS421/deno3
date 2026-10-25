import hashlib
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from config.config import MAX_LENGTH, SYSTEM_PROMPT


CACHE_VERSION = 2


class Mydataset(Dataset):

    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=MAX_LENGTH,
        use_cache=True,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_cache = use_cache

        if not self.data_path.is_file():
            raise FileNotFoundError(
                f"数据文件不存在：{self.data_path}"
            )

        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "Tokenizer 的 pad_token_id 为空，请先设置 pad_token。"
            )

        if self.use_cache and self.load_cache():
            return

        # 读取标准 JSON 数组
        self.items = self._load_json()

        self.samples = [
            self.encode_item(item)
            for item in self.items
        ]

        if self.use_cache:
            self.save_cache()

        print(
            f"数据集加载完成：{self.data_path.name}，"
            f"样本数={len(self.samples)}"
        )

    def _load_json(self):
        """读取 JSON 数组格式的指令数据集。"""

        try:
            with self.data_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                items = json.load(file)

        except json.JSONDecodeError as error:
            raise ValueError(
                f"{self.data_path} 不是有效的 JSON 文件："
                f"第 {error.lineno} 行，第 {error.colno} 列。"
            ) from error

        if not isinstance(items, list):
            raise ValueError(
                f"{self.data_path} 的顶层必须是 JSON 数组。"
            )

        if not items:
            raise ValueError(
                f"{self.data_path} 数据集为空。"
            )

        for index, item in enumerate(items, start=1):

            if not isinstance(item, dict):
                raise ValueError(
                    f"{self.data_path} 第 {index} 条数据必须是 JSON 对象。"
                )

            messages = item.get("messages")

            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(
                    f"{self.data_path} 第 {index} 条数据的 "
                    "messages 字段无效。"
                )

            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError(
                        f"{self.data_path} 第 {index} 条数据中，"
                        "messages 的每个元素都必须是对象。"
                    )

                if "role" not in message or "content" not in message:
                    raise ValueError(
                        f"{self.data_path} 第 {index} 条数据中，"
                        "消息必须包含 role 和 content 字段。"
                    )

            if messages[-1].get("role") != "assistant":
                raise ValueError(
                    f"{self.data_path} 第 {index} 条数据的"
                    "最后一条消息必须是 assistant。"
                )

        print(
            f"JSON 数据读取成功：{self.data_path.name}，"
            f"原始样本数={len(items)}"
        )

        return items

    def cache_path(self):

        data_hash = hashlib.sha256(
            self.data_path.read_bytes()
        ).hexdigest()

        fingerprint = {
            "version": CACHE_VERSION,
            "data_sha256": data_hash,
            "max_length": self.max_length,
            "system_prompt": SYSTEM_PROMPT,
            "tokenizer_name": getattr(
                self.tokenizer,
                "name_or_path",
                "",
            ),
            "tokenizer_size": len(self.tokenizer),
            "chat_template": str(
                getattr(
                    self.tokenizer,
                    "chat_template",
                    "",
                )
            ),
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        digest = hashlib.sha256(
            json.dumps(
                fingerprint,
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:20]

        cache_dir = self.data_path.parent / ".cache"

        return cache_dir / (
            f"{self.data_path.stem}-{digest}.pt"
        )

    def load_cache(self):

        cache_path = self.cache_path()

        if not cache_path.is_file():
            return False

        try:
            samples = torch.load(
                cache_path,
                map_location="cpu",
                weights_only=True,
            )

            if not isinstance(samples, list):
                return False

            for sample in samples:
                if not isinstance(sample, dict):
                    return False

                if not all(
                    key in sample
                    for key in (
                        "input_ids",
                        "attention_mask",
                        "labels",
                    )
                ):
                    return False

            self.samples = samples

            print(f"读取缓存：{cache_path}")

            return True

        except Exception as error:
            print(
                f"缓存读取失败，将重新编码：{error}"
            )
            return False

    def save_cache(self):


        cache_path = self.cache_path()

        cache_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = cache_path.with_name(
            f"{cache_path.name}.{os.getpid()}.tmp"
        )

        torch.save(
            self.samples,
            temporary_path,
        )

        os.replace(
            temporary_path,
            cache_path,
        )

        print(f"已保存缓存：{cache_path}")

    def encode_item(self, item):
        """将一条 messages 数据编码为训练所需的张量。"""

        messages = item["messages"]

        prompt_messages = messages[:-1]
        full_messages = messages

        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        full_ids = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        # 确定提示词与完整对话实际一致的前缀长度
        prefix_length = 0
        limit = min(
            len(prompt_ids),
            len(full_ids),
        )

        while (
            prefix_length < limit
            and prompt_ids[prefix_length]
            == full_ids[prefix_length]
        ):
            prefix_length += 1

        if prefix_length >= len(full_ids):
            raise ValueError(
                "完整对话中没有可用于训练的后续 token，"
                "请检查 messages 和 Tokenizer 的 chat_template。"
            )

        # 保留答案监督信号
        target_ids = full_ids[prefix_length:]

        if len(target_ids) >= self.max_length:
            target_ids = target_ids[:self.max_length]
            prompt_ids = []
        else:
            prompt_budget = (
                self.max_length - len(target_ids)
            )

            prompt_ids = prompt_ids[:prefix_length]

            prompt_ids = (
                prompt_ids[-prompt_budget:]
                if prompt_budget
                else []
            )

        input_ids = prompt_ids + target_ids

        # 仅对提示词部分屏蔽 loss
        labels = (
            [-100] * len(prompt_ids)
            + target_ids
        )

        if not any(
            label != -100
            for label in labels
        ):
            raise ValueError(
                "样本没有有效的监督 token。"
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.ones(
                len(input_ids),
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
