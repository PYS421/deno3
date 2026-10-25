import torch


class Mycollator:

    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

        if self.pad_token_id is None:
            raise ValueError("Tokenizer 没有 pad_token_id。")

    def __call__(self, features):
        if not features:
            raise ValueError("收到空 batch。")

        max_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for feature in features:
            input_ids = feature["input_ids"].tolist()
            attention_mask = feature["attention_mask"].tolist()
            labels = feature["labels"].tolist()

            pad_len = max_length - len(input_ids)

            input_ids += [self.pad_token_id] * pad_len
            attention_mask += [0] * pad_len
            labels += [-100] * pad_len

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)

        return {
            "input_ids": torch.tensor(
                batch_input_ids, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                batch_attention_mask, dtype=torch.long
            ),
            "labels": torch.tensor(
                batch_labels, dtype=torch.long
            ),
        }
