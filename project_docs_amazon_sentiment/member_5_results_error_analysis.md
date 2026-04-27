# 成员 5：结果分析与讨论负责人

## 1. 角色定位

成员 5 是非代码成员，负责实验结果整理、错误案例分析和 discussion 草稿。

成员 5 的重点是解释实验结果，形成 findings and discussion 的主要内容。

## 2. 负责范围

成员 5 主要负责：

- 实验记录表。
- 结果对比表。
- 成功案例整理。
- 失败案例整理。
- Confusion matrix 文字解读。
- Discussion 草稿。

## 3. 输入材料

- 成员 2 提供的 baseline 指标。
- 成员 3 提供的最终结果表。
- 成员 3 提供的 confusion matrix。
- 成员 1 和成员 2 提供的预测结果文件。
- 测试集 `test_3class.csv`。

## 4. 具体工作步骤

1. 建立实验记录表。
   - 记录四组实验：三分类 baseline、五星 baseline、三分类 DistilBERT、五星 DistilBERT。
   - 每组记录模型名称、任务名称、测试集、主要参数、Accuracy、Macro F1。

2. 整理模型结果对比。
   - 对比 baseline 和 DistilBERT。
   - 对比三分类和五星预测。
   - 找出最重要的 3 条结果发现。

3. 解读三分类结果。
   - 观察 `negative`、`neutral`、`positive` 哪一类最容易预测错误。
   - 重点分析 `neutral` 是否容易被预测成正面或负面。
   - 写出三分类任务的主要发现。

4. 解读五星预测结果。
   - 观察 1-5 stars 中哪些类别容易混淆。
   - 重点分析相邻星级混淆，例如 2 vs 3、3 vs 4。
   - 写出五星预测比三分类更难的原因。

5. 整理成功案例。
   - 三分类任务至少 5 个成功案例。
   - 五星预测任务至少 5 个成功案例。
   - 每个案例包含评论片段、真实标签、预测标签和简短解释。

6. 整理失败案例。
   - 三分类任务至少 5 个失败案例。
   - 五星预测任务至少 5 个失败案例。
   - 每个案例包含评论片段、真实标签、预测标签和可能失败原因。

7. 撰写 discussion 草稿。
   - 说明 DistilBERT 相对 baseline 的优势。
   - 说明五星预测难于三分类的原因。
   - 说明模型局限，例如文本截断、讽刺表达、星级与文本不完全一致。
   - 提出未来改进方向，例如使用更大模型、加入更多上下文、进行类别平衡处理。

## 5. 交付物

- 实验记录表。
- 结果对比表。
- 三分类结果分析文字。
- 五星预测结果分析文字。
- 成功案例表。
- 失败案例表。
- Discussion 草稿。

## 6. 验收标准

- 四组实验都有记录。
- 至少总结 3 条主要实验发现。
- 三分类任务至少有 5 个成功案例和 5 个失败案例。
- 五星预测任务至少有 5 个成功案例和 5 个失败案例。
- 能解释 `neutral` 类为什么较难。
- 能解释五星预测为什么较难。
- 成员 6 可以直接使用该内容写 Findings and Results 与 Discussion。
- 每个案例包含评论片段、真实标签、预测标签和简短原因。
- Discussion 至少包含模型优势、错误原因和未来改进三个部分。

## 7. 可写入报告的贡献说明

`Member 5 was responsible for experiment documentation, result comparison, confusion matrix interpretation, qualitative case analysis, and preparation of the findings and discussion sections.`
