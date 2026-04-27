# 成员 2：数据处理与 Baseline 负责人

## 1. 角色定位

成员 2 是代码成员，负责项目前期数据处理和传统机器学习 baseline。

成员 2 的任务是为后续 DistilBERT 实验提供可靠的数据基础和可对比的 baseline 结果。

## 2. 负责范围

成员 2 主要负责：

- 数据文件读取与字段检查。
- 标签分布统计。
- 文本长度统计。
- 数据统计图。
- 三分类 baseline。
- 五星预测 baseline。

## 3. 输入材料

- `summary.json`
- `train_3class.csv`
- `validation_3class.csv`
- `test_3class.csv`

## 4. 具体工作步骤

1. 读取数据。
   - 使用 pandas 读取 train、validation、test。
   - 检查字段是否包含 `id`、`raw_label_5way`、`stars`、`label_3class`、`text`。
   - 检查空文本数量。
   - 检查异常标签。

2. 核对样本数量。
   - 统计 train 样本数。
   - 统计 validation 样本数。
   - 统计 test 样本数。
   - 与 `summary.json` 中的数字对照。

3. 统计标签分布。
   - 统计三分类标签数量。
   - 统计五星评分标签数量。
   - 生成表格。
   - 生成柱状图。

4. 统计文本长度。
   - 计算每条评论的词数或字符数。
   - 输出平均值、中位数、最大值。
   - 可选：生成文本长度分布图。

5. 训练三分类 baseline。
   - 使用 `TfidfVectorizer` 提取文本特征。
   - 使用 `LogisticRegression` 训练模型。
   - 标签使用 `label_3class`。
   - 在 test set 上输出预测结果。

6. 训练五星预测 baseline。
   - 使用同样的 TF-IDF 特征。
   - 使用 `LogisticRegression` 训练模型。
   - 标签使用 `stars`。
   - 在 test set 上输出预测结果。

7. 保存 baseline 输出。
   - 保存两个任务的预测结果。
   - 保存两个任务的指标。
   - 将结果交给成员 3 和成员 5。

## 5. 交付物

- 数据检查结果。
- 标签分布表。
- 标签分布图。
- 文本长度统计。
- 三分类 baseline 代码。
- 五星预测 baseline 代码。
- 两个 baseline 的预测结果。
- 两个 baseline 的初步指标。

## 6. 验收标准

- 能证明三个 CSV 文件可正常读取。
- 样本数与 `summary.json` 一致。
- 三分类和五星标签分布都有表格或图。
- 三分类 baseline 有 test set 预测结果。
- 五星预测 baseline 有 test set 预测结果。
- 输出结果格式能被成员 3 继续评估。
- 图表能被成员 6 放入 PPT 或 report。
- Baseline 代码能重复运行，并输出相同格式的结果文件。
- 能说明 baseline 的作用是为 DistilBERT 提供对比参照。

## 7. 可写入报告的贡献说明

`Member 2 was responsible for data loading, label verification, exploratory data analysis, and the implementation of TF-IDF plus Logistic Regression baseline models for both classification tasks.`
