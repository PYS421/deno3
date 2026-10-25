# Demo 3：基于 Qwen2.5-7B-Instruct 的生物医学命名实体识别

本项目基于 **Qwen2.5-7B-Instruct**，使用 BC2GM（BioCreative II Gene Mention）生物医学数据集开展命名实体识别（Named Entity Recognition，NER）任务。通过指令微调，使大语言模型能够从生物医学文本中识别基因实体（GENE），并采用实体级 Precision、Recall 和 F1-score 评估模型性能。

项目实现了 LoRA 和 QLoRA 两种参数高效微调方案，并对数据集处理、模型加载、批次数据整理、训练流程和评估指标进行模块化设计。同时，结合 SwanLab 对训练过程和实验结果进行跟踪，并通过漏检、误检分析进一步研究模型的识别错误。

## 1. 项目介绍

### 1.1 项目目标

本项目的主要目标是将 Qwen2.5-7B-Instruct 应用于生物医学基因实体识别任务。给定一段生物医学文本，模型需要识别其中出现的基因名称，并按照指定的 JSON 格式输出实体及其类别。

例如：

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
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── collator.py
│   ├── metrics.py
│   ├── trainer.py
│   └── train_lora_round2.py
├── scripts/
│   ├── evaluate_lora.py
│   ├── evaluate_qlora.py
├── .gitignore
└── README.md
```

主要模块说明：

| 文件或目录 | 功能 |
|---|---|
| `config/` | 保存 LoRA、QLoRA 训练配置 |
| `src/model.py` | 模型加载及微调相关逻辑 |
| `src/dataset.py` | 数据集读取、样本处理及编码 |
| `src/collator.py` | 将多个样本整理为批次张量 |
| `src/metrics.py` | 计算实体级 Precision、Recall、F1-score 等指标 |
| `src/trainer.py` | 自定义训练器及训练流程相关逻辑 |
| `src/train_lora_round2.py` | LoRA 第二轮训练入口 |
| `scripts/` | 模型评估、错误分析及辅助脚本 |
| `data/` | 原始处理数据和指令微调数据 |
| `results/` | 保存实验结果与分析文件 |
| `picture/` | 保存训练曲线、指标和实验截图 |
| `monitoring/` | 保存实验监控说明与记录 |

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
conda create -n demo3_ner python=3.10 -y
conda activate demo3_ner
```

## 5. 模型与微调方法

本项目采用 Qwen2.5-7B-Instruct 作为基座模型，并使用参数高效微调方法完成 GENE 实体识别任务。

### 5.1 LoRA

LoRA（Low-Rank Adaptation）通过在指定线性层中引入低秩适配器，冻结基座模型的大部分参数，仅训练新增的适配器参数，从而降低训练成本。

第二轮 LoRA 实验的主要配置如下：

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

以上参数对应第二轮 LoRA 实验。项目其他轮次的配置可能不同，应以相应配置文件和实验记录为准。

### 5.2 QLoRA

QLoRA（Quantized Low-Rank Adaptation）结合低比特量化和 LoRA，在降低基座模型显存占用的同时训练适配器参数。

本项目设置了独立的 QLoRA 配置文件 `config/qlora_config.json`。量化位数、量化类型、计算精度、LoRA 参数、梯度检查点和训练超参数均应以该文件及实际训练代码为准。

### 5.3 LoRA 与 QLoRA

| 对比维度 | LoRA | QLoRA |
|---|---|---|
| 基座模型 | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| 微调方式 | 低秩适配器 | 量化基座模型 + 低秩适配器 |
| 显存需求 | 通常较高 | 通常较低 |
| 训练目标 | 训练适配器参数 | 训练适配器参数 |
| 主要优势 | 实现简单，训练效率较高 | 更节省显存 |
| 配置文件 | `config/lora_config.json` | `config/qlora_config.json` |

## 6. 模型训练

进入项目目录并激活环境：

```bash
cd ~/autodl-tmp/demo3_qwen_ner
conda activate demo3_ner
```

训练前请检查：

1. 基座模型路径或本地模型缓存是否正确。
2. 配置文件中的训练集、验证集路径是否存在。
3. 输出目录是否设置正确。
4. GPU 显存是否满足当前配置。
5. SwanLab 的实验记录配置是否正确。

### 6.1 LoRA 训练

项目中的 LoRA 训练入口：

```bash
python src/train_lora_round2.py
```

该命令适用于脚本默认配置与当前环境一致的情况。如果训练脚本需要额外的参数，请根据实际代码传入相应配置。

### 6.2 QLoRA 训练

QLoRA 使用独立的配置和训练流程。请使用仓库中实际提供的 QLoRA 训练入口，并检查 `config/qlora_config.json` 中的数据路径、模型路径、量化参数和输出目录。

### 6.3 训练过程监控

训练过程中可借助 SwanLab 跟踪训练损失、验证指标、实验参数和不同实验之间的表现差异。

实验记录应与具体训练配置对应，避免将不同实验的曲线和指标混用。

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

本节汇总已有实验记录。验证集和测试集分别列出；不同训练轮次或不同模型检查点的结果不直接混为同一组对比实验。

### 8.1 LoRA 验证集结果

第二轮 LoRA 实验的最佳检查点在完整验证集上的结果如下：

| 指标 | 数值 |
|---|---:|
| Precision | 84.60% |
| Recall | 80.48% |
| F1-score | 82.49% |
| TP | 2,334 |
| FP | 425 |
| FN | 566 |
| Parse errors | 15 |

### 8.2 QLoRA 验证集结果

根据已有 QLoRA 实验记录，第 3 轮训练取得最佳验证集结果：

| 指标 | 数值 |
|---|---:|
| Epoch | 3 |
| Precision | 84.55% |
| Recall | 80.93% |
| F1-score | 82.70% |
| TP | 2,347 |
| FP | 429 |
| FN | 553 |
| Parse errors | 11 |

### 8.3 测试集结果

根据此前保存的评估记录，以下结果来自已有测试集评估：

| 实验 | Precision | Recall | F1-score |
|---|---:|---:|---:|
| LoRA（已有检查点） | 约 83.47% | 约 79.05% | 81.20% |
| QLoRA（已有实验记录） | 83.75% | 77.97% | 80.76% |

**结果说明：**

1. LoRA 第二轮的验证集结果来自该轮最佳检查点。
2. QLoRA 验证集结果来自已有记录的第 3 轮训练。
3. 测试集结果来自已有评估记录，不应自动视为上述验证集最佳检查点的测试结果。
4. 不同实验可能使用不同的训练配置、模型检查点或生成参数，因此比较时应同时核对实验配置。
5. 正式报告中应保留对应的评估日志、模型检查点标识和数据集划分信息。

## 9. 错误分析

除整体指标外，本项目还对漏检（FN）、误检（FP）和输出解析错误进行分析，以便确定下一步改进方向。

重点分析的问题包括：

- **实体边界错误：** 预测的基因实体边界与标注不一致。
- **基因名称变体：** 基因缩写、别名或特殊字符影响识别。
- **上下文理解错误：** 模型未能正确判断文本中的词语是否为基因实体。
- **生成格式错误：** 输出内容不符合预期 JSON 格式，导致解析失败。
- **数据与标注差异：** 原文、标注与预测结果之间存在差异。

对错误样本进行分类，有助于判断改进工作应优先集中在数据处理、提示词设计、模型训练参数还是生成格式约束。
