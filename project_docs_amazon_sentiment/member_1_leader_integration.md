# 成员 1：组长 / 深度学习与整合负责人

## 1. 角色定位

成员 1 是项目组长，也是主要代码负责人之一，承担项目中难度最高的技术部分。

核心职责是负责项目技术主线、DistilBERT 训练主流程、demo 和最终整合。

## 2. 负责范围

成员 1 主要负责：

- 项目整体技术路线确认。
- DistilBERT 三分类模型训练。
- DistilBERT 五星预测模型训练。
- Demo 主流程。
- 最终代码整合。
- 最终结果核对。
- 答辩主线和 Q&A 准备。

## 3. 输入材料

- 训练数据：`train_3class.csv`
- 验证数据：`validation_3class.csv`
- 测试数据：`test_3class.csv`
- 成员 2 输出的 baseline 结果。
- 成员 3 输出的评估脚本和结果表格式。
- 成员 4 输出的数据说明和标签映射说明。
- 成员 5 输出的错误分析结论。
- 成员 6 输出的 report/PPT 结构。

## 4. 具体工作步骤

1. 确认项目主线。
   - 项目主题为商品评论情感分类与评分预测。
   - 保留两个任务：三分类情感分析和五星预测。
   - 主模型使用 `DistilBERT`。

2. 搭建深度学习训练代码结构。
   - 建立数据读取模块。
   - 建立 tokenizer 编码流程。
   - 建立 Dataset 或 DataLoader。
   - 建立训练函数。
   - 建立预测函数。
   - 建立模型保存和加载流程。

3. 训练三分类 DistilBERT。
   - 输入字段使用 `text`。
   - 标签字段使用 `label_3class`。
   - 输出标签为 `negative`、`neutral`、`positive`。
   - 记录训练参数，包括 max length、batch size、learning rate、epoch。

4. 训练五星预测 DistilBERT。
   - 输入字段使用 `text`。
   - 标签字段使用 `stars`。
   - 输出标签为 `1`、`2`、`3`、`4`、`5`。
   - 记录训练参数，保持与三分类实验尽量一致。

5. 导出模型预测结果。
   - 对 test set 生成预测结果。
   - 每个任务输出一个预测结果文件。
   - 文件至少包含 `id`、`text`、`true_label`、`pred_label`。
   - 将预测结果交给成员 3 和成员 5。

6. 制作 demo。
   - 输入任意英文商品评论。
   - 输出三分类情感标签。
   - 输出五星预测结果。
   - 准备至少 3 条展示样例。

7. 整合最终代码。
   - 确保所有代码路径清楚。
   - 确保 notebook 或主程序能按顺序运行。
   - 确保 report/PPT 中使用的结果与代码输出一致。

8. 准备答辩问答。
   - 为什么选择 DistilBERT？
   - 为什么做两个任务？
   - 为什么五星预测更难？
   - 为什么使用英文 Amazon 数据？
   - 如果模型效果不高，原因是什么？

## 5. 交付物

- DistilBERT 三分类训练代码。
- DistilBERT 五星预测训练代码。
- 两个任务的 test set 预测结果。
- Demo 代码或 notebook cell。
- 最终代码入口说明。
- 答辩 Q&A 草稿。

## 6. 验收标准

- 三分类 DistilBERT 能完成训练和预测。
- 五星预测 DistilBERT 能完成训练和预测。
- 两个任务都有 test set 预测结果文件。
- Demo 能输入新评论并输出两个预测结果。
- 所有模型参数和路径记录清楚。
- 成员 3 能使用预测结果完成评估。
- 成员 6 能根据项目主线完成 report/PPT。
- 最终代码有清晰入口，组员能按说明运行主要流程。
- 能解释模型输入、输出、标签含义和训练流程。

## 7. 可写入报告的贡献说明

`Member 1 served as the project leader and was responsible for the overall technical pipeline, DistilBERT training for both sentiment classification and rating prediction, demo implementation, final code integration, and presentation coordination.`
