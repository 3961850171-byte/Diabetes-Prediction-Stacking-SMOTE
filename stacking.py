import pandas as pd
import numpy as np
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns

print("正在从 Kaggle 下载数据集...")

path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
target_file = 'diabetes_binary_health_indicators_BRFSS2015.csv'
file_path = os.path.join(path, target_file)

df = pd.read_csv(file_path)
print(f"数据加载成功！样本总数: {len(df)}")

null_info = df.isnull().sum()
if null_info.sum() == 0:
    print("Step 1: 扫描完成，数据集中无缺失值。")
else:
    df.dropna(inplace=True)
    print("Step 1: 已清理缺失值。")

duplicates = df.duplicated().sum()
print(f"Step 2: 发现并移除重复记录数: {duplicates}")
df.drop_duplicates(inplace=True)

scaler = StandardScaler()
numerical_features = ['BMI', 'GenHlth', 'MentHlth', 'PhysHlth', 'Age', 'Education', 'Income']
print(f"Step 3: 正在执行特征标准化...")

df[numerical_features] = scaler.fit_transform(df[numerical_features])

X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("\n" + "="*40)
print("数据预处理最终结果汇总:")
print(f"- 训练集样本 (X_train): {X_train.shape[0]} 条")
print(f"- 测试集样本 (X_test): {X_test.shape[0]} 条")
print(f"- 特征维度: {X_train.shape[1]} 个特征")
print("="*40)
print("\n标准化后的训练集数据预览 (前5行):")
print(X_train.head())

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
base_models = [
('rf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)),
('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, use_label_encoder=False, eval_metric='logloss'))
]

meta_model = LogisticRegression()
stacking_clf = StackingClassifier(estimators=base_models,final_estimator=meta_model,cv=5,passthrough=False)

print("正在训练 Stacking 集成模型，这可能需要一点时间...")
stacking_clf.fit(X_train, y_train)

y_pred_stack = stacking_clf.predict(X_test)
stacking_acc = accuracy_score(y_test, y_pred_stack)

print("\n" + "="*30)
print(f"Stacking 集成模型准确率: {stacking_acc:.4f}")
print("="*30)
print("\n分类报告:\n", classification_report(y_test, y_pred_stack))
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred_stack), annot=True, fmt='d', cmap='Purples')
plt.title('Stacking Ensemble Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

accuracy = accuracy_score(y_test, y_pred_stack)
precision = precision_score(y_test, y_pred_stack)
recall = recall_score(y_test, y_pred_stack)
f1 = f1_score(y_test, y_pred_stack)

y_pred_proba = stacking_clf.predict_proba(X_test)[:, 1]
auc_score = roc_auc_score(y_test, y_pred_proba)

print(f"准确率 (Accuracy): {accuracy:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall): {recall:.4f}")
print(f"F1 分数: {f1:.4f}")
print(f"AUC 值: {auc_score:.4f}")