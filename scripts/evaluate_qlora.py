import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==================== 配置 ====================
BASE_MODEL = (
    "/root/autodl-tmp/hf_cache/hub/"
    "models--Qwen--Qwen2.5-7B-Instruct/"
    "snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
)
ADAPTER_PATH = "outputs/qwen2.5-7b-qlora/best"
TEST_FILE = "data/instruction/test.json"
OUTPUT_DIR = Path("results/eval-test-qlora-best")

MAX_LENGTH = 512
MAX_NEW_TOKENS = 128

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


def load_samples(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    if content.startswith("["):
        samples = json.loads(content)
    else:
        samples = [
            json.loads(line)
            for line in content.splitlines()
            if line.strip()
        ]

    if not isinstance(samples, list):
        raise ValueError("测试集应为 JSON 数组或 JSONL 格式。")

    return samples


def main():
    if not Path(ADAPTER_PATH, "adapter_config.json").exists():
        raise FileNotFoundError(f"未找到适配器配置：{ADAPTER_PATH}")

    if not Path(BASE_MODEL, "config.json").exists():
        raise FileNotFoundError(f"未找到基础模型配置：{BASE_MODEL}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== 1. 加载 Tokenizer =====")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    print("===== 2. 加载基础模型 =====")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print("===== 3. 加载 QLoRA 最佳适配器 =====")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("===== 4. 读取测试集 =====")
    samples = load_samples(TEST_FILE)
    print(f"测试样本数量：{len(samples)}")

    if len(samples) != 5000:
        raise ValueError(
            f"预期测试集为 5000 条，实际读取 {len(samples)} 条；"
            "请先检查数据文件。"
        )

    tp = fp = fn = 0
    parse_errors = 0
    prediction_file = OUTPUT_DIR / "test_predictions.jsonl"

    print("===== 5. 开始完整测试集推理 =====")
    with open(prediction_file, "w", encoding="utf-8") as out:
        for idx, sample in enumerate(
            tqdm(samples, desc="QLoRA 测试集推理", unit="条"),
            start=1
        ):
            messages = sample["messages"]

            user_message = next(
                m["content"] for m in messages
                if m["role"] == "user"
            )
            gold_text = next(
                m["content"] for m in messages
                if m["role"] == "assistant"
            )

            gold, gold_ok = parse_entities(gold_text)
            if not gold_ok:
                raise ValueError(f"第 {idx} 条真实标签无法解析。")

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
                max_length=MAX_LENGTH
            ).to(model.device)

            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            raw_prediction = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
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

            if idx % 100 == 0:
                out.flush()

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    metrics = {
        "model": BASE_MODEL,
        "adapter": ADAPTER_PATH,
        "test_file": TEST_FILE,
        "test_samples": len(samples),
        "max_length": MAX_LENGTH,
        "max_new_tokens": MAX_NEW_TOKENS,
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

    print("\n========== QLoRA 完整测试集结果 ==========")
    print(f"测试样本数：{len(samples)}")
    print(f"TP：{tp}")
    print(f"FP：{fp}")
    print(f"FN：{fn}")
    print(f"解析错误：{parse_errors}")
    print(f"Precision：{precision:.6f} ({precision * 100:.2f}%)")
    print(f"Recall：   {recall:.6f} ({recall * 100:.2f}%)")
    print(f"F1：       {f1:.6f} ({f1 * 100:.2f}%)")
    print(f"\n预测结果：{prediction_file}")
    print(f"指标文件：{metrics_file}")


if __name__ == "__main__":
    main()
