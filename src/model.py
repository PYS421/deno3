from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config.config import (
    MODEL_NAME,
    TOKENIZER_PATH,
    BF16,
    GRADIENT_CHECKPOINTING,
    USE_QLORA,
    LOAD_IN_4BIT,
    BNB_4BIT_QUANT_TYPE,
    BNB_4BIT_USE_DOUBLE_QUANT,
    BNB_4BIT_COMPUTE_DTYPE,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
)


def load_tokenizer():
    if not TOKENIZER_PATH.is_dir():
        raise FileNotFoundError(
            f"本地 Tokenizer 目录不存在：{TOKENIZER_PATH}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(TOKENIZER_PATH),
        local_files_only=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def build_model():
    if not torch.cuda.is_available():
        raise RuntimeError("没有检测到可用的 CUDA GPU。")

    if USE_QLORA and not LOAD_IN_4BIT:
        raise ValueError(
            "QLoRA 配置错误：必须启用 4-bit 量化。"
        )

    if USE_QLORA and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "当前 GPU 不支持 BF16，无法使用当前 QLoRA 配置。"
        )

    dtype = torch.bfloat16 if BF16 else torch.float32

    model_kwargs = {
        "local_files_only": True,
        "device_map": {"": torch.cuda.current_device()},
    }

    if USE_QLORA:
        # QLoRA：4-bit NF4 量化。
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }

        if BNB_4BIT_COMPUTE_DTYPE not in dtype_map:
            raise ValueError(
                "不支持的量化计算精度："
                f"{BNB_4BIT_COMPUTE_DTYPE}"
            )

        compute_dtype = dtype_map[BNB_4BIT_COMPUTE_DTYPE]

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=BNB_4BIT_QUANT_TYPE,
            bnb_4bit_use_double_quant=(
                BNB_4BIT_USE_DOUBLE_QUANT
            ),
            bnb_4bit_compute_dtype=compute_dtype,
        )

        model_kwargs["quantization_config"] = quantization_config
        model_kwargs["torch_dtype"] = compute_dtype

        print("加载方式：QLoRA")
        print("4-bit 量化：开启")
        print("量化类型：", BNB_4BIT_QUANT_TYPE)
        print("双重量化：", BNB_4BIT_USE_DOUBLE_QUANT)
        print("计算精度：", BNB_4BIT_COMPUTE_DTYPE)

    else:
        # 普通 LoRA：基座模型不进行 4-bit 量化。
        model_kwargs["torch_dtype"] = dtype

        print("加载方式：LoRA")
        print("4-bit 量化：关闭")
        print("模型精度：", dtype)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        **model_kwargs,
    )

    model.config.use_cache = False

    if USE_QLORA:
        # 量化模型训练前的准备。
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=GRADIENT_CHECKPOINTING,
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            },
        )

    elif GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            }
        )

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    if USE_QLORA:
        base_model = model.get_base_model()
        is_quantized = getattr(
            base_model,
            "is_loaded_in_4bit",
            False,
        )

        if not is_quantized:
            raise RuntimeError(
                "QLoRA 检查失败：基座模型未报告 4-bit 加载状态。"
            )

        print("检查通过：基座模型已加载为 4-bit。")

    return model
