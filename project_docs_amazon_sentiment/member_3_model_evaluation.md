# 成员 3：评估与可视化负责人

## 1. 角色定位

成员 3 是代码成员，负责统一评估、结果表和可视化图表。

成员 3 负责把成员 1 和成员 2 的模型输出整理成可比较的实验结果。

## 2. 负责范围

成员 3 主要负责：

- 统一评估函数。
- Baseline 与 DistilBERT 的指标计算。
- Classification report。
- Confusion matrix。
- 预测结果格式整理。
- 结果对比表。

## 3. 输入材料

- 成员 1 输出的 DistilBERT 预测结果。
- 成员 2 输出的 baseline 预测结果。
- `test_3class.csv`。

## 4. 具体工作步骤

1. 统一预测结果格式。
   - 每个结果文件至少包含 `id`、`text`、`true_label`、`pred_label`。
   - 确保 baseline 和 DistilBERT 文件格式一致。
   - 确保三分类和五星预测文件可以区分。

2. 编写评估函数。
   - 输入真实标签和预测标签。
   - 输出 Accuracy。
   - 输出 Precision。
   - 输出 Recall。
   - 输出 Macro F1。
   - 输出 classification report。

3. 评估四组实验。
   - 三分类 baseline。
   - 五星预测 baseline。
   - 三分类 DistilBERT。
   - 五星预测 DistilBERT。

4. 生成 confusion matrix。
   - 为三分类任务生成 confusion matrix。
   - 为五星预测任务生成 confusion matrix。
   - 图中标签顺序必须清楚。
   - 图标题需要包含任务名称和模型名称。

5. 整理结果总表。
   - 表格字段包括 Task、Model、Accuracy、Precision、Recall、Macro F1。
   - 将四组实验放在同一个结果表中。
   - 标明哪个是 baseline，哪个是 deep learning model。

6. 输出给其他成员。
   - 将结果表交给成员 5 做分析。
   - 将 confusion matrix 图交给成员 5 和成员 6。
   - 将评估代码交给成员 1 归入最终代码。

## 5. 交付物

- 统一评估脚本。
- 四组实验的指标表。
- 四组实验的 classification report。
- 三分类 confusion matrix。
- 五星预测 confusion matrix。
- 格式统一的预测结果文件。

## 6. 验收标准

- 四组实验都有 Accuracy 和 Macro F1。
- 四组实验都能生成 classification report。
- 三分类和五星预测都有 confusion matrix。
- 所有图表标题和标签清楚。
- 成员 5 能直接根据结果表写分析。
- 成员 6 能直接将图表放入 report/PPT。
- 结果表中的模型名称、任务名称和指标名称统一。
- Confusion matrix 的标签顺序与任务定义一致。

## 7. 可写入报告的贡献说明

`Member 3 was responsible for implementing the evaluation pipeline, calculating performance metrics, generating classification reports and confusion matrices, and organizing model comparison results.`
