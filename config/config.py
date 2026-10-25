import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--config",
    default="config/lora_config.json",
    help="config/qlora_config.json",
)
_args, _unknown_args = _parser.parse_known_args()

CONFIG_PATH = Path(_args.config)
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = PROJECT_ROOT / CONFIG_PATH

if not CONFIG_PATH.is_file():
    raise FileNotFoundError(
        f"配置文件不存在：{CONFIG_PATH}"
    )

with CONFIG_PATH.open("r", encoding="utf-8") as _file:
    CONFIG = json.load(_file)


def _project_path(key):
    path = Path(CONFIG[key])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


MODEL_NAME = CONFIG["model_name"]
TOKENIZER_PATH = _project_path("tokenizer_path")

TRAIN_FILE = _project_path("train_file")
DEV_FILE = _project_path("dev_file")
TEST_FILE = _project_path("test_file")

OUTPUT_DIR = _project_path("output_dir")
LOG_DIR = _project_path("log_dir")
RESULT_DIR = _project_path("result_dir")

MAX_LENGTH = int(CONFIG["max_length"])
MAX_NEW_TOKENS = int(CONFIG["max_new_tokens"])
METRIC_EVAL_LIMIT = CONFIG.get("metric_eval_limit")

SEED = int(CONFIG["seed"])
BATCH_SIZE = int(CONFIG["batch_size"])
GRADIENT_ACCUMULATION_STEPS = int(
    CONFIG["gradient_accumulation_steps"]
)
EPOCHS = int(CONFIG["epochs"])

LEARNING_RATE = float(CONFIG["learning_rate"])
WEIGHT_DECAY = float(CONFIG["weight_decay"])
LR_SCHEDULER_TYPE = CONFIG["lr_scheduler_type"]

BF16 = bool(CONFIG["bf16"])
GRADIENT_CHECKPOINTING = bool(
    CONFIG["gradient_checkpointing"]
)
DATALOADER_NUM_WORKERS = int(
    CONFIG["dataloader_num_workers"]
)
SAVE_TOTAL_LIMIT = int(CONFIG["save_total_limit"])
WARMUP_STEPS = int(CONFIG["warmup_steps"])
MAX_GRAD_NORM = float(CONFIG["max_grad_norm"])

USE_QLORA = bool(CONFIG["use_qlora"])
LOAD_IN_4BIT = bool(CONFIG["load_in_4bit"])
BNB_4BIT_QUANT_TYPE = CONFIG["bnb_4bit_quant_type"]
BNB_4BIT_USE_DOUBLE_QUANT = bool(
    CONFIG["bnb_4bit_use_double_quant"]
)
BNB_4BIT_COMPUTE_DTYPE = CONFIG[
    "bnb_4bit_compute_dtype"
]

LORA_R = int(CONFIG["lora_r"])
LORA_ALPHA = int(CONFIG["lora_alpha"])
LORA_DROPOUT = float(CONFIG["lora_dropout"])
LORA_TARGET_MODULES = CONFIG["lora_target_modules"]

SWANLAB_PROJECT = CONFIG["swanlab_project"]
SWANLAB_EXPERIMENT_NAME = CONFIG[
    "swanlab_experiment_name"
]

SYSTEM_PROMPT = (
    "You are a biomedical named entity recognition system. "
    "Extract gene entities from the input sentence. "
    "Return only a valid JSON array. "
    "Each entity must contain the keys name and type. "
    'The type must be "GENE". '
    "If no gene entity exists, return []."
)

print(f"已加载配置：{CONFIG_PATH}")
print(f"微调方式：{CONFIG['method']}")