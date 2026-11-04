# Demo 3：基于 Qwen2.5-7B-Instruct 的生物医学命名实体识别

本项目基于 **Qwen2.5-7B-Instruct**，使用 BC2GM（BioCreative II Gene Mention）生物医学数据集开展命名实体识别（Named Entity Recognition，NER）任务。通过指令微调，使大语言模型能够从生物医学文本中识别基因实体（GENE），并采用实体级 Precision、Recall 和 F1-score 评估模型性能。

项目实现了 LoRA 和 QLoRA 两种参数高效微调方案，并对数据集处理、模型加载、批次数据整理、训练流程和评估指标进行模块化设计。同时，结合自定义 Trainer 和 SwanLab 实验跟踪，对训练过程、模型性能和显存使用情况进行监控，并通过漏检、误检分析进一步研究模型的识别错误。

## 1. 项目介绍

### 1.1 项目目标

本项目的主要目标是将 Qwen2.5-7B-Instruct 应用于生物医学基因实体识别任务。

给定一段生物医学文本，模型需要识别其中出现的基因名称，并按照指定的 JSON 格式输出实体及其类别。

**输入文本：**

```text
BRCA1 mutations are associated with breast cancer.
```

**期望输出：**

```json
[
  {
    "name": "BRCA1",
    "type": "GENE"
  }
]
```

当文本中不存在目标实体时，模型应输出空数组：

```json
[]
```

### 1.2 项目特点

- **基座模型：** Qwen2.5-7B-Instruct
- **任务类型：** 生物医学命名实体识别（NER）
- **数据集：** BC2GM
- **识别类别：** GENE
- **微调方法：** LoRA、QLoRA
- **深度学习框架：** PyTorch
- **模型训练框架：** Hugging Face Transformers、PEFT
- **实验跟踪：** SwanLab
- **项目实现：** 自定义 Dataset、Collator、Trainer 和实体级评估指标
- **显存优化：** Gradient Checkpointing、QLoRA 4-bit 量化
- **显存监控：** 自定义 MemoryMonitor 模块
- **错误分析：** 分析漏检、误检及输出格式解析错误

## 2. 项目目录

```text
demo3_qwen_ner/
├── config/
│   ├── lora_config.json
│   └── qlora_config.json
├── data/
│   ├── dev.json
│   ├── test.json
│   ├── labels.json
│   └── instruction/
│       ├── train.json
│       ├── dev.json
│       ├── test.json
│       ├── train.jsonl
│       ├── dev.jsonl
│       └── test.jsonl
├── picture/
│   └── 训练过程与实验结果截图
├── monitoring/
│   ├── memory/
│   │   └── 显存监控记录
│   ├── notes/
│   │   └── 实验监控说明
│   └── screenshots/
│       └── 监控截图
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── collator.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── memory_monitor.py
│   └── train_lora_round2.py
├── scripts/
│   ├── evaluate_lora.py
│   └── evaluate_qlora.py
├── results/
│   ├── lora-config/
│   └── qlora/
├── .gitignore
└── README.md
```

主要模块说明：

| 文件或目录 | 功能 |
|---|---|
| `config/` | 保存 LoRA、QLoRA 训练配置 |
| `src/model.py` | 模型加载、量化配置及 LoRA 适配器构建 |
| `src/dataset.py` | 数据集读取、样本处理及编码 |
| `src/collator.py` | 将多个样本整理为批次张量 |
| `src/metrics.py` | 计算实体级 Precision、Recall、F1-score 等指标 |
| `src/trainer.py` | 自定义训练器及训练流程 |
| `src/memory_monitor.py` | 训练和验证阶段的显存监控 |
| `src/train_lora_round2.py` | LoRA、QLoRA 训练入口 |
| `scripts/` | 模型评估及辅助脚本 |
| `data/` | 原始处理数据和指令微调数据 |
| `results/` | 保存实验摘要与分析结果 |
| `monitoring/` | 保存显存监控记录、说明和截图 |
| `picture/` | 保存训练曲线、指标和实验截图 |

注：以上为项目主要结构示意，具体文件以仓库中实际提交的内容为准。模型权重、训练检查点、缓存和大型临时日志不应作为普通源码提交。

## 3. 数据集

本项目使用 BC2GM 生物医学基因实体识别数据集，并将任务限定为 `GENE` 单类别实体识别。

### 3.1 数据集划分

| 数据集 | 样本数量 |
|---|---:|
| Train | 12,500 |
| Dev | 2,500 |
| Test | 5,000 |
| Total | 20,000 |

### 3.2 指令数据格式

为适配指令微调，将数据整理为包含任务指令、输入文本和目标输出的格式。

示例：

```json
{
  "instruction": "Extract all GENE entities from the text.",
  "input": "BRCA1 mutations are associated with breast cancer.",
  "output": "[{\"name\": \"BRCA1\", \"type\": \"GENE\"}]"
}
```

模型根据输入文本生成实体列表。目标输出采用 JSON 数组形式，每个实体包含实体名称 `name` 和实体类型 `type`。

无目标实体的样本使用 `[]` 表示。

### 3.3 数据注意事项

请在公开仓库前确认 BC2GM 数据集的许可协议是否允许重新分发。如果不允许公开原始数据，应从仓库中排除数据文件，并在 README 中说明数据获取方式及预处理流程。

## 4. 环境配置

项目在 CUDA 环境下进行训练。实际运行时，需要根据 GPU、CUDA 版本和依赖兼容性选择合适的软件版本。

### 4.1 创建 Conda 环境

```bash
conda create -n demo3_ner python=3.11 -y
conda activate demo3_ner
```

### 4.2 主要依赖

项目主要使用以下依赖：

| 依赖 | 用途 |
|---|---|
| PyTorch | 深度学习计算 |
| Transformers | 加载 Qwen 模型及 Tokenizer |
| PEFT | LoRA、QLoRA 参数高效微调 |
| BitsAndBytes | 4-bit 量化 |
| SwanLab | 实验跟踪与可视化 |
| Pandas | 实验监控数据处理 |

具体依赖版本应根据实际运行环境配置。

## 5. 模型与微调方法

本项目采用 Qwen2.5-7B-Instruct 作为基座模型，并使用参数高效微调方法完成 GENE 实体识别任务。

### 5.1 LoRA

LoRA（Low-Rank Adaptation）通过在指定线性层中引入低秩适配器，冻结基座模型的大部分参数，仅训练新增的适配器参数，从而降低训练成本。

本项目使用 PEFT 构建 LoRA 适配器，并针对注意力模块和 MLP 模块中的指定线性层进行微调。

历史第二轮 LoRA 实验的主要配置如下：

| 参数 | 配置 |
|---|---|
| Base model | Qwen2.5-7B-Instruct |
| Batch size | 4 |
| Gradient accumulation steps | 2 |
| Epochs | 3 |
| Learning rate | 5e-5 |
| Scheduler | Cosine |
| Weight decay | 0.01 |
| Precision | BF16 |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| LoRA dropout | 0.05 |
| Seed | 42 |

以上参数对应历史第二轮 LoRA 实验。本项目后续还进行了不同 LoRA 参数配置的实验，具体参数以对应配置文件为准。

### 5.2 QLoRA

QLoRA（Quantized Low-Rank Adaptation）结合低比特量化和 LoRA，在降低基座模型显存占用的同时训练适配器参数。

本项目使用 BitsAndBytes 实现基座模型的 4-bit 量化，采用 NF4 量化类型和双重量化，并使用 BF16 作为计算精度。

主要配置：

| 参数 | 配置 |
|---|---|
| Quantization | 4-bit |
| Quantization type | NF4 |
| Double quantization | True |
| Compute dtype | BF16 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |

QLoRA 使用独立的配置文件 `config/qlora_config.json`，其他训练参数以实际配置文件为准。

### 5.3 Gradient Checkpointing

为了进一步降低训练过程中的显存占用，本项目在 LoRA 和 QLoRA 实验中启用了 Gradient Checkpointing（梯度检查点）。

其基本思路是在前向传播过程中不保存所有中间激活值，而是在反向传播时重新计算部分中间结果，从而降低激活值占用的显存。

本项目在两种微调方式下均支持梯度检查点：

- LoRA：通过模型的 `gradient_checkpointing_enable()` 方法启用。
- QLoRA：通过 `prepare_model_for_kbit_training()` 中的 `use_gradient_checkpointing` 参数启用。

同时，将模型的 `use_cache` 设置为 `False`，以适配梯度检查点训练。

Gradient Checkpointing 能够降低显存使用量，但会增加部分计算开销，因此可能影响训练速度。

### 5.4 LoRA 与 QLoRA 对比

| 对比维度 | LoRA | QLoRA |
|---|---|---|
| 基座模型 | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| 微调方式 | 低秩适配器 | 量化基座模型 + 低秩适配器 |
| 基座模型量化 | 无 | 4-bit NF4 |
| 计算精度 | BF16 | BF16 |
| 显存需求 | 通常较高 | 通常较低 |
| 训练目标 | 训练适配器参数 | 训练适配器参数 |
| 主要优势 | 实现简单，训练效率较高 | 更节省显存 |

## 6. 模型训练与显存监控

### 6.1 进入项目目录

```bash
cd ~/autodl-tmp/demo3_qwen_ner
conda activate demo3_ner
```

训练前请检查：

1. 基座模型路径或本地模型缓存是否正确。
2. 配置文件中的训练集、验证集路径是否存在。
3. 输出目录是否设置正确。
4. GPU 显存是否满足当前配置。
5. SwanLab 实验记录配置是否正确。

### 6.2 LoRA 训练

使用 LoRA 配置文件启动训练：

```bash
python -m src.train_lora_round2 --config config/lora_config.json
```

训练过程包括：

1. 加载 Tokenizer。
2. 加载训练集和验证集。
3. 加载 Qwen2.5-7B-Instruct 基座模型。
4. 构建 LoRA 适配器。
5. 初始化 SwanLab。
6. 初始化自定义 Trainer。
7. 执行训练、验证和模型检查点保存。

### 6.3 QLoRA 训练

使用 QLoRA 配置文件启动训练：

```bash
python -m src.train_lora_round2 --config config/qlora_config.json
```

训练入口与 LoRA 相同，但通过独立配置文件控制 4-bit 量化及其他 QLoRA 参数。

启动后应检查日志中的以下信息：

- 加载方式是否为 QLoRA。
- 4-bit 量化是否开启。
- 量化类型是否为 NF4。
- 基座模型是否通过 4-bit 状态检查。
- Gradient Checkpointing 是否启用。

### 6.4 自定义 Trainer

本项目实现了自定义 Trainer，用于控制模型训练、梯度累积、优化器更新、学习率调度、验证集评估和模型检查点保存等过程。

通过自定义训练流程，可以更灵活地记录训练状态、评估指标和显存变化，并为后续实验分析提供统一的数据来源。

### 6.5 SwanLab 实验跟踪

训练过程中使用 SwanLab 跟踪实验数据，包括：

- 训练损失（Train Loss）。
- 验证集 Precision。
- 验证集 Recall。
- 验证集 F1-score。
- 学习率变化。
- 实验配置和训练轮次。

SwanLab 项目地址：

[Demo3 Qwen NER - SwanLab](https://swanlab.cn/@2004421/demo3-qwen-ner)

不同训练实验使用独立的实验名称，以便进行实验结果比较。

### 6.6 显存监控

为了分析不同参数高效微调方法的显存开销，本项目实现了独立的显存监控模块：

```text
src/memory_monitor.py
```

主要监控内容包括：

- PyTorch 当前显存分配量（Allocated Memory）。
- PyTorch 显存保留量（Reserved Memory）。
- PyTorch 峰值分配显存。
- PyTorch 峰值保留显存。
- GPU 整体显存占用。
- GPU 利用率。

监控模块支持按照训练阶段记录显存变化，包括：

- Train：模型训练阶段。
- Evaluation：验证集评估阶段。

监控结果以 CSV 和 JSON 格式保存：

```text
monitoring/
└── memory/
    ├── LORA_实验时间/
    │   ├── memory_log.csv
    │   └── memory_summary.json
    └── QLORA_实验时间/
        ├── memory_log.csv
        └── memory_summary.json
```

通过监控数据，可以进一步比较 LoRA 和 QLoRA 在模型训练及验证阶段的显存使用情况。

## 7. 模型评估

本项目使用实体级 Precision、Recall 和 F1-score 评估模型表现，并记录 TP、FP、FN 及输出解析错误数量。

### 7.1 评价指标

**Precision（精确率）**

模型预测出的实体中，正确实体所占的比例：

\[
P=\frac{TP}{TP+FP}
\]

**Recall（召回率）**

真实实体中被模型正确识别的比例：

\[
R=\frac{TP}{TP+FN}
\]

**F1-score**

Precision 与 Recall 的调和平均数：

\[
F_1=\frac{2PR}{P+R}
\]

其中：

- TP：正确识别的实体数量。
- FP：误检实体数量。
- FN：漏检实体数量。

实体级评估需要根据项目定义的实体匹配规则判断预测结果与真实标注是否一致。输出格式解析失败的样本也应单独记录，便于进一步分析。

### 7.2 评估脚本

LoRA 评估脚本：

```bash
python scripts/evaluate_lora.py
```

QLoRA 评估脚本：

```bash
python scripts/evaluate_qlora.py
```

运行前，请检查脚本中的适配器路径、测试数据路径和生成参数，确保加载的是目标实验的模型检查点。

## 8. 实验结果

本节汇总项目中已有的实验记录。不同训练轮次、不同参数配置和不同模型检查点的结果分别记录，避免将不同实验的数据混合比较。

### 8.1 历史 LoRA 实验结果

历史第二轮自定义 Trainer LoRA 实验在完整验证集上的最佳结果：

| 指标 | 数值 |
|---|---:|
| Best Epoch | 3 |
| Precision | 84.60% |
| Recall | 80.48% |
| F1-score | 82.49% |
| TP | 2,334 |
| FP | 425 |
| FN | 566 |
| Parse Errors | 15 |

该实验采用历史第二轮的 LoRA 配置，LoRA rank 为 32，alpha 为 64。

### 8.2 历史 QLoRA 实验结果

此前 QLoRA 实验记录中的一组验证集结果：

| 指标 | 数值 |
|---|---:|
| Best Epoch | 3 |
| Precision | 84.55% |
| Recall | 80.93% |
| F1-score | 82.70% |
| TP | 2,347 |
| FP | 429 |
| FN | 553 |
| Parse Errors | 11 |

该结果属于历史实验记录，与本节后续最新 LoRA、QLoRA 对比实验分开统计。

### 8.3 历史测试集结果

根据此前保存的评估记录，已有测试集结果如下：

| 实验 | Precision | Recall | F1-score |
|---|---:|---:|---:|
| LoRA（已有检查点） | 约 83.47% | 约 79.05% | 81.20% |
| QLoRA（已有实验记录） | 83.75% | 77.97% | 80.76% |

**结果说明：**

1. 以上测试集结果来自已有评估记录。
2. 不能自动视为最新一组 LoRA 和 QLoRA 检查点的测试集表现。
3. 不同实验可能使用不同的训练配置、模型检查点或生成参数，因此比较时应同时核对实验配置。
4. 正式报告中应保留对应的评估日志、模型检查点标识和数据集划分信息。

### 8.4 最新 LoRA 与 QLoRA 验证集对比实验

本次实验于 2026 年 11 月 4 日完成，在 NVIDIA RTX 5090 GPU 上进行。

两种方法均采用 BF16，并开启 Gradient Checkpointing。除基座模型量化方式外，主要训练参数保持一致。

#### （1）训练配置

| 参数 | LoRA | QLoRA |
|---|---:|---:|
| Batch Size | 4 | 4 |
| Gradient Accumulation | 2 | 2 |
| Epochs | 3 | 3 |
| Learning Rate | 5e-5 | 5e-5 |
| Weight Decay | 0.01 | 0.01 |
| Max Length | 512 | 512 |
| LoRA Rank | 16 | 16 |
| LoRA Alpha | 32 | 32 |
| LoRA Dropout | 0.05 | 0.05 |
| Precision | BF16 | BF16 |
| Quantization | None | 4-bit NF4 |
| Double Quantization | — | True |
| Gradient Checkpointing | True | True |

#### （2）验证集结果

| 指标 | LoRA | QLoRA |
|---|---:|---:|
| Best Epoch | 3 | 3 |
| Precision | 84.43% | 84.66% |
| Recall | 80.79% | 80.69% |
| **F1-score** | **82.57%** | **82.63%** |
| TP | 2,343 | 2,340 |
| FP | 432 | 424 |
| FN | 557 | 560 |
| Parse Errors | 16 | 16 |

两种方法均在第三轮取得最佳验证集结果。

QLoRA 的 F1-score 为 82.63%，LoRA 的 F1-score 为 82.57%，两者相差约 0.05 个百分点，整体表现非常接近。

#### （3）显存使用情况

| 指标 | LoRA | QLoRA |
|---|---:|---:|
| GPU | RTX 5090 | RTX 5090 |
| PyTorch 峰值分配显存 | 19.32 GB | 13.74 GB |
| GPU 整体峰值占用 | 26.17 GB | 18.85 GB |
| Gradient Checkpointing | 开启 | 开启 |

显存对比结果：

- QLoRA 的 PyTorch 峰值分配显存比 LoRA 减少约 28.9%。
- QLoRA 的 GPU 整体峰值占用比 LoRA 减少约 28.0%。
- 两种方法均成功完成三个 Epoch 的训练，没有发生显存溢出。

#### （4）实验分析

从验证集结果来看，LoRA 和 QLoRA 的实体识别效果较为接近。QLoRA 在本次实验中取得了略高的 F1-score，同时具有更低的显存占用。

这表明，在本次实验配置下，QLoRA 能够以较低的显存开销取得与普通 LoRA 接近的识别效果。

需要注意的是，本次结果来自单次实验，F1-score 的微小差异不足以证明某种微调方法具有显著的泛化性能优势。后续仍需要结合独立测试集评估进一步分析。

## 9. 错误分析

除整体指标外，本项目还对漏检（FN）、误检（FP）和输出解析错误进行分析，以便确定下一步改进方向。

重点分析的问题包括：

- **实体边界错误：** 预测的基因实体边界与标注不一致。
- **基因名称变体：** 基因缩写、别名或特殊字符影响识别。
- **上下文理解错误：** 模型未能正确判断文本中的词语是否为基因实体。
- **生成格式错误：** 输出内容不符合预期 JSON 格式，导致解析失败。
- **数据与标注差异：** 原文、标注与预测结果之间存在差异。

对错误样本进行分类，有助于判断改进工作应优先集中在数据处理、提示词设计、模型训练参数还是生成格式约束。

## 10. 项目总结

本项目基于 Qwen2.5-7B-Instruct 实现了面向 BC2GM 数据集的生物医学 GENE 命名实体识别任务，完成了从数据预处理、指令构建、模型加载、参数高效微调到实体级评估的完整流程。

项目主要工作包括：

1. 将 BC2GM 数据集转换为适用于大语言模型指令微调的数据格式。
2. 使用自定义 Dataset 和 Collator 完成训练数据处理。
3. 基于 PEFT 实现 LoRA 和 QLoRA 两种参数高效微调方案。
4. 实现自定义 Trainer，控制训练、验证和模型检查点保存流程。
5. 引入 Gradient Checkpointing，降低训练过程中的显存开销。
6. 实现独立的 MemoryMonitor 模块，记录训练和验证阶段的显存变化。
7. 使用 SwanLab 跟踪实验过程与模型性能。
8. 通过 Precision、Recall、F1-score 及 TP、FP、FN 分析模型识别效果。
9. 对漏检、误检和输出格式错误进行进一步分析。

最新一组实验中，LoRA 和 QLoRA 分别取得 82.57% 和 82.63% 的验证集 F1-score。QLoRA 在本次实验中将 PyTorch 峰值分配显存降低约 28.9%，表现出较好的显存效率。

后续可以进一步开展独立测试集评估、不同随机种子实验和错误样本分析，以更全面地评价模型的实体识别能力及不同微调方法的实际表现。

