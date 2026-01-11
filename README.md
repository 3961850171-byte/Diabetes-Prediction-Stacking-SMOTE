# Diabetes-Prediction-Stacking-SMOTE
# 基于 Stacking 集成学习与 SMOTE 算法的糖尿病风险预测系统

## 1. 项目简介
本项目旨在利用机器学习技术，通过 2015 年 BRFSS 健康指标数据预测个体的糖尿病患病风险。针对医疗数据中普遍存在的**类别不平衡**（样本分布不均）问题，本项目通过技术改进显著提升了模型的实用性。

## 2. 核心技术
- **数据预处理**：缺失值清理、重复值剔除及特征标准化（`StandardScaler`）。
- **样本平衡 (Innovation)**：引入 **SMOTE** (*Synthetic Minority Over-sampling Technique*) 对患病样本进行过采样，有效解决了模型对少数类识别率低的问题。
- **集成学习 (Stacking)**：
    - **第一层 (Base Models)**：RandomForest（随机森林）与 XGBoost。
    - **第二层 (Meta Model)**：Logistic Regression（逻辑回归）。

## 3. 主要成果
- 模型准确率稳定在较高水平。
- **关键优化**：通过引入 SMOTE 技术，模型对糖尿病阳性样本的捕捉能力（Recall/召回率）得到显著提升，降低了漏诊率。

## 4. 文件说明
- `stacking.py`: 基础集成模型实现（记录了从基础模型到集成的演进过程）。
- `改进.py`: 最终方案，包含 SMOTE 数据增强的优化模型实现。
