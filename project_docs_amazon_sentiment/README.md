# Amazon 商品评论情感与评分预测项目执行方案

## 1. 项目基本信息

项目英文名称：**Amazon Review Sentiment and Rating Classification Using Deep Learning**

项目中文名称：**基于深度学习的 Amazon 商品评论情感分类与评分预测**

项目方向：自然语言处理、文本分析、深度学习。

项目形式：使用 Python 对商品评论文本进行建模，完成模型训练、结果评估、错误分析和 demo 展示。

## 2. 项目主要内容

本项目使用英文 Amazon 商品评论数据，根据评论文本完成两个分类任务。

任务 A：三分类情感分析。

- 输入：一条英文商品评论。
- 输出：`negative`、`neutral` 或 `positive`。
- 目的：判断评论整体情感倾向。

任务 B：五星评分预测。

- 输入：同一条英文商品评论。
- 输出：`1`、`2`、`3`、`4` 或 `5` stars。
- 目的：预测更细粒度的用户评分。

两个任务使用同一数据集，但难度不同。三分类任务更关注整体情感，五星预测任务需要区分更细的评价强度，因此更适合用于结果讨论和错误分析。

## 3. 是否符合课程要求

课程项目要求包含：自选主题、Python 编程、深度学习或 NLP 方向、结果展示、报告、展示和个人贡献说明。本项目对应关系如下。

| 课程要求 | 本项目对应内容 |
| --- | --- |
| Free topic project | 主题为商品评论情感与评分预测 |
| Text analysis / NLP | 对评论文本进行分类分析 |
| Deep learning techniques | 使用 DistilBERT 进行文本分类 |
| Python programming | 使用 Python 完成数据处理、训练、评估和 demo |
| Specific purpose | 根据商品评论自动判断情感和评分 |
| Findings and results | 展示模型指标、对比结果、混淆矩阵和错误案例 |
| Report | 包含 objectives、methods、results、discussion、contribution |
| Presentation | 每位成员负责对应 slides 并发言 |

项目工作量来源：

- 两个任务：三分类情感分析和五星评分预测。
- 两类方法：传统机器学习 baseline 和深度学习模型。
- 多种评估：Accuracy、Precision、Recall、Macro F1、classification report、confusion matrix。
- 结果分析：成功案例、失败案例、类别混淆、任务难度对比。
- Demo 展示：输入新评论，输出情感类别和预测星级。

因此，本项目不是单一情感分类任务，而是一个包含数据分析、模型对比、深度学习训练、细粒度预测和错误分析的完整文本分析项目。

## 4. 数据集说明

本项目明确区分“原始数据集”和“后处理数据集”。

原始数据集：

- 名称：**Amazon Reviews Multi / en**
- 来源：Hugging Face `mteb/amazon_reviews_multi`
- 链接：https://huggingface.co/datasets/mteb/amazon_reviews_multi
- 内容：英文 Amazon 评论文本与原始星级信息

后处理数据集（项目实际训练使用）：

- 名称：**Amazon Review Rating Processed 3-Class**
- 来源：Kaggle `yxggla/amazon-review-rating-processed-3class`
- 链接：https://www.kaggle.com/datasets/yxggla/amazon-review-rating-processed-3class
- 处理说明：在保留 `stars`（1-5 星）基础上，新增 `label_3class` 三分类情感标签

本地文件：

- `data/amazon_reviews_multi_en/summary.json`
- `data/amazon_reviews_multi_en/processed_3class/train_3class.csv`
- `data/amazon_reviews_multi_en/processed_3class/validation_3class.csv`
- `data/amazon_reviews_multi_en/processed_3class/test_3class.csv`

数据规模：

| 数据划分 | 样本数 | 三分类标签分布 |
| --- | ---: | --- |
| Train | 200,000 | negative 80,000；neutral 40,000；positive 80,000 |
| Validation | 5,000 | negative 2,000；neutral 1,000；positive 2,000 |
| Test | 5,000 | negative 2,000；neutral 1,000；positive 2,000 |

主要字段：

| 字段 | 含义 |
| --- | --- |
| `id` | 评论编号 |
| `raw_label_5way` | 原始 5 类标签，取值为 0-4 |
| `stars` | 星级评分，取值为 1-5 |
| `label_3class` | 三分类情感标签 |
| `text` | 商品评论文本 |

## 5. 标签定义

五星预测任务直接使用 `stars` 字段，取值为 1 到 5。

三分类情感标签由星级映射得到：

| 星级 | 三分类标签 | 说明 |
| --- | --- | --- |
| 1-2 stars | `negative` | 明显负面评价 |
| 3 stars | `neutral` | 中立、一般或混合评价 |
| 4-5 stars | `positive` | 明显正面评价 |

该映射可以保证项目不需要额外大规模人工标注，同时让三分类任务和五星预测任务建立清晰联系。

## 6. 技术路线

### 6.1 数据处理

使用 pandas 读取 train、validation、test 三个 CSV 文件。检查字段、样本数量、空文本、标签分布和文本长度。

### 6.2 Baseline 模型

使用 `TF-IDF + Logistic Regression` 作为 baseline。

- 对 `text` 字段提取 TF-IDF 特征。
- 训练三分类 baseline。
- 训练五星预测 baseline。
- 在 test set 上输出最终指标。

### 6.3 深度学习模型

使用 `DistilBERT` 作为主模型。

- 使用 `distilbert-base-uncased` tokenizer 编码文本。
- 分别 fine-tune 三分类模型和五星预测模型。
- 使用 validation set 观察训练效果。
- 使用 test set 报告最终结果。

### 6.4 评估与分析

统一使用以下指标：

- Accuracy
- Precision
- Recall
- Macro F1
- Classification report
- Confusion matrix

分析重点：

- Baseline 与 DistilBERT 的效果差异。
- 三分类任务与五星预测任务的难度差异。
- `neutral` 类的识别难点。
- 五星预测中相邻星级的混淆。
- 成功案例和失败案例。

## 7. 项目整体流程

1. 项目确认：确定题目、任务、数据集和分工。
2. 数据检查：读取数据，确认字段、样本数和标签分布。
3. 数据分析：生成类别分布、星级分布、文本长度统计和样例。
4. Baseline 实验：训练三分类和五星预测的 Logistic Regression 模型。
5. 深度学习实验：训练三分类和五星预测的 DistilBERT 模型。
6. 模型评估：统一输出指标、classification report 和 confusion matrix。
7. 错误分析：整理成功案例、失败案例和类别混淆原因。
8. Demo 制作：输入新评论，输出情感类别和预测星级。
9. 报告撰写：完成 objectives、methods、results、discussion 和 contribution。
10. 展示准备：完成 slides、讲稿、Q&A 和最终提交材料。

## 8. 成员分工原则

分工按项目流程划分，每位成员有独立交付物，并与其他成员形成上下游衔接。

- 成员 1 负责项目技术主线、DistilBERT 训练、demo 和最终整合。
- 成员 2 负责数据检查、数据统计和 baseline 实验。
- 成员 3 负责统一评估、指标表、confusion matrix 和预测结果整理。
- 成员 4 负责数据集来源、字段说明、标签映射和任务定义。
- 成员 5 负责实验记录、结果解释、错误案例和 discussion 草稿。
- 成员 6 负责 report、PPT、讲稿、contribution 和提交材料整理。

| 成员 | 角色 | 是否写代码 | 独立负责内容 | 详细文档 |
| --- | --- | --- | --- | --- |
| 成员 1 | 组长 / 深度学习与整合负责人 | 是 | DistilBERT 训练主流程、demo、最终整合 | [成员 1 文档](member_1_leader_integration.md) |
| 成员 2 | 数据处理与 baseline 负责人 | 是 | 数据检查、统计图、TF-IDF + Logistic Regression | [成员 2 文档](member_2_data_baseline.md) |
| 成员 3 | 评估与可视化负责人 | 是 | 统一评估脚本、结果表、confusion matrix、预测结果文件 | [成员 3 文档](member_3_model_evaluation.md) |
| 成员 4 | 数据集与任务说明负责人 | 否 | 数据来源、字段解释、标签映射、任务定义 | [成员 4 文档](member_4_dataset_task_design.md) |
| 成员 5 | 结果分析与讨论负责人 | 否 | 实验记录、成功失败案例、错误分析、discussion 草稿 | [成员 5 文档](member_5_results_error_analysis.md) |
| 成员 6 | 报告与展示负责人 | 否 | report、PPT、讲稿、contribution、提交清单 | [成员 6 文档](member_6_report_presentation.md) |

## 9. 阶段安排

| 阶段 | 工作内容 | 主要负责人 | 阶段产出 |
| --- | --- | --- | --- |
| 第 1 阶段 | 项目确认、数据集说明、标签定义 | 成员 1、成员 4 | 题目、任务定义、数据说明 |
| 第 2 阶段 | 数据检查、数据统计、baseline | 成员 2 | 数据统计图、baseline 指标 |
| 第 3 阶段 | DistilBERT 训练和预测 | 成员 1、成员 3 | 深度学习模型预测结果 |
| 第 4 阶段 | 指标评估、图表、错误分析 | 成员 3、成员 5 | 结果表、混淆矩阵、案例分析 |
| 第 5 阶段 | 报告、PPT、demo 和彩排 | 成员 1、成员 6 | 最终提交材料 |

## 10. Demo 示例

Demo 输入：

```text
The headphones are comfortable and the sound quality is great, but the battery life is shorter than expected.
```

Demo 输出：

```text
Sentiment prediction: positive
Rating prediction: 4 stars
```

展示说明：评论整体偏正面，但包含 battery life 的轻微负面反馈，因此星级可能低于 5 stars。

## 11. 最终提交清单

- Report PDF，约 4 页，最多 5 页。
- Presentation slides，PPT 或 PDF 格式。
- Python code files 或 Jupyter Notebook。
- Demo 截图或运行说明。
- 实验结果表。
- Confusion matrix。
- 成功与失败案例分析。
- 每位成员 contribution 说明。
- 每位成员负责发言的 slides。

## 12. 建议 PPT 结构

1. 项目标题与团队成员
2. 研究背景与项目目标
3. 数据集与标签定义
4. 方法一：TF-IDF + Logistic Regression
5. 方法二：DistilBERT
6. 三分类情感分析结果
7. 五星评分预测结果
8. 错误分析与讨论
9. Demo 展示
10. 总结与成员贡献
