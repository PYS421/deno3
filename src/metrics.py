import json


def parse_entities(text):
    if not isinstance(text, str):
        return set(), False

    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")

    if start < 0 or end < start:
        return set(), False

    try:
        data = json.loads(text[start:end + 1])

        if not isinstance(data, list):
            return set(), False

        entities = set()

        for item in data:
            if not isinstance(item, dict):
                continue

            name = item.get("name")
            entity_type = item.get("type")

            if (
                isinstance(name, str)
                and name.strip()
                and isinstance(entity_type, str)
                and entity_type.strip()
            ):
                entities.add(
                    (name.strip(), entity_type.strip())
                )

        return entities, True

    except (json.JSONDecodeError, TypeError):
        return set(), False


def calculate_entity_metrics(predictions, references):
    if len(predictions) != len(references):
        raise ValueError(
            "预测列表和标准答案列表的长度不一致。"
        )

    tp = 0
    fp = 0
    fn = 0
    parse_errors = 0

    for pred_text, gold_text in zip(predictions, references):
        pred_entities, pred_valid = parse_entities(pred_text)
        gold_entities, gold_valid = parse_entities(gold_text)

        if not pred_valid:
            parse_errors += 1

        if not gold_valid:
            raise ValueError(
                "标准答案中存在无法解析的 JSON 数组。"
            )

        tp += len(pred_entities & gold_entities)
        fp += len(pred_entities - gold_entities)
        fn += len(gold_entities - pred_entities)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_f1": f1,
        "eval_tp": tp,
        "eval_fp": fp,
        "eval_fn": fn,
        "eval_parse_errors": parse_errors,
        "eval_samples": len(predictions),
    }
