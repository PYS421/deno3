import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "outputs/qwen2.5-7b-lora"
TEST_FILE = "data/instruction/test.jsonl"
OUTPUT_DIR = Path("outputs/qwen2.5-7b-lora")

SYSTEM_PROMPT = (
    "You are a biomedical named entity recognition system. "
    "Extract gene entities from the input sentence. "
    'Return only a valid JSON array. Each entity must contain '
    'the keys name and type. The type must be "GENE". '
    "If no gene entity exists, return []."
)


def parse_entities(text):
    text = text.strip()
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end < start:
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
            entity_type = item.get("type", "GENE")

            if isinstance(name, str) and name.strip():
                entities.add((name.strip(), entity_type))

        return entities, True

    except (json.JSONDecodeError, TypeError):
        return set(), False


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("1/4 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("2/4 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto"
    )

    print("3/4 加载 LoRA Adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("4/4 读取完整测试集...")
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    print(f"测试样本数量：{len(samples)}")

    tp = fp = fn = 0
    parse_errors = 0
    prediction_file = OUTPUT_DIR / "test_predictions.jsonl"

    with open(prediction_file, "w", encoding="utf-8") as out:
        for idx, sample in enumerate(
            tqdm(samples, desc="测试集推理", unit="条"), start=1
        ):
            messages = sample["messages"]

            user_message = next(
                x["content"] for x in messages
                if x["role"] == "user"
            )
            gold_text = next(
                x["content"] for x in messages
                if x["role"] == "assistant"
            )

            gold, gold_ok = parse_entities(gold_text)
            if not gold_ok:
                raise ValueError(f"第 {idx} 条真实标签无法解析，请检查数据。")

            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]

            prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(model.device)

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )

            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            raw_prediction = tokenizer.decode(
                new_tokens, skip_special_tokens=True
            )

            pred, pred_ok = parse_entities(raw_prediction)
            if not pred_ok:
                parse_errors += 1

            tp += len(pred & gold)
            fp += len(pred - gold)
            fn += len(gold - pred)

            record = {
                "index": idx,
                "input": user_message,
                "gold": [
                    {"name": name, "type": typ}
                    for name, typ in sorted(gold)
                ],
                "prediction": [
                    {"name": name, "type": typ}
                    for name, typ in sorted(pred)
                ],
                "raw_prediction": raw_prediction,
                "prediction_parse_ok": pred_ok
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    metrics = {
        "test_samples": len(samples),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "parse_errors": parse_errors,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

    metrics_file = OUTPUT_DIR / "test_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n========== 完整测试集评估结果 ==========")
    print(f"测试样本数：{len(samples)}")
    print(f"TP：{tp}")
    print(f"FP：{fp}")
    print(f"FN：{fn}")
    print(f"无法解析的预测数：{parse_errors}")
    print(f"Precision：{precision:.4f}")
    print(f"Recall：   {recall:.4f}")
    print(f"F1：       {f1:.4f}")
    print(f"\n预测结果已保存：{prediction_file}")
    print(f"评估指标已保存：{metrics_file}")


if __name__ == "__main__":
    main()
