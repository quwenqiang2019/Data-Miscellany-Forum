#!/usr/bin/env python3
"""
Eli5 特征重要性分析 - 基于心脏疾病预测数据集
使用 Permutation Importance 方法评估特征对模型性能的重要性
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import export_text
import eli5
from eli5.sklearn import PermutationImportance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data_path = "dataset.csv"
df = pd.read_csv(data_path)

print("=" * 60)
print("数据集基本信息")
print("=" * 60)
print(f"数据维度: {df.shape}")
print(f"特征列: {list(df.columns)}")
print(f"\n目标变量分布:\n{df['target'].value_counts()}")
print(f"\n特征统计描述:\n{df.describe()}")

# 特征与目标
feature_names = [col for col in df.columns if col != 'target']
X = df[feature_names]
y = df['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")

# 网格搜索优化随机森林
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

print("\n正在执行网格搜索优化...")
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1
)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print(f"最优参数: {grid_search.best_params_}")
print(f"最优交叉验证准确率: {grid_search.best_score_:.4f}")
print(f"测试集准确率: {best_model.score(X_test, y_test):.4f}")

# 1. Feature Importances (训练过程中的分裂贡献)
print("\n" + "=" * 60)
print("特征重要性 (feature_importances_)")
print("=" * 60)
fi_importances = best_model.feature_importances_
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': fi_importances
}).sort_values('Importance', ascending=False)
print(fi_df.to_string(index=False))

# 2. Permutation Importance (通过打乱特征值评估对模型性能的影响)
print("\n" + "=" * 60)
print("Permutation Importance (基于准确率)")
print("=" * 60)
perm = PermutationImportance(best_model, scoring='accuracy', random_state=42)
perm.fit(X_train, y_train)

pi_results = eli5.format_as_dataframe(
    eli5.explain_weights(perm, feature_names=feature_names, top=15)
)
print(pi_results.to_string(index=False))

# 3. 决策树可视化
print("\n" + "=" * 60)
print("单棵决策树结构 (前5层)")
print("=" * 60)
tree_text = export_text(
    best_model.estimators_[0],
    feature_names=feature_names,
    max_depth=5
)
print(tree_text[:2000])

# 4. 生成可视化图表
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Permutation Importance 图
pi_sorted = pi_results.sort_values('weight', ascending=True)
colors = ['#3b6ee0' if w > 0 else '#ccc' for w in pi_sorted['weight']]
axes[0].barh(pi_sorted['feature'], pi_sorted['weight'], color=colors, edgecolor='white')
axes[0].set_xlabel('Permutation Importance (Weight)', fontsize=12)
axes[0].set_title('Eli5 Permutation Importance', fontsize=14, fontweight='bold')
axes[0].axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
for i, (w, s) in enumerate(zip(pi_sorted['weight'], pi_sorted['std'])):
    axes[0].text(w + 0.002, i, f'{w:.3f}±{s:.3f}', va='center', fontsize=9)
axes[0].grid(axis='x', alpha=0.3)

# Feature Importances 对比图
fi_sorted = fi_df.sort_values('Importance', ascending=True)
axes[1].barh(fi_sorted['Feature'], fi_sorted['Importance'], color='#e8833a', edgecolor='white')
axes[1].set_xlabel('Feature Importance', fontsize=12)
axes[1].set_title('RandomForest Feature Importances', fontsize=14, fontweight='bold')
for i, v in enumerate(fi_sorted['Importance']):
    axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('el5_importance_chart.png', dpi=150, bbox_inches='tight', facecolor='white')
print("\n图表已保存: el5_importance_chart.png")

# 保存结果到CSV
pi_results.to_csv('permutation_importance_results.csv', index=False)
fi_df.to_csv('feature_importance_results.csv', index=False)
print("结果已保存: permutation_importance_results.csv, feature_importance_results.csv")

# 汇总
print("\n" + "=" * 60)
print("分析总结")
print("=" * 60)
top3 = pi_results.head(3)
print(f"Top 3 重要特征 (Permutation Importance):")
for _, row in top3.iterrows():
    print(f"  - {row['feature']}: weight={row['weight']:.4f}, std={row['std']:.4f}")
print(f"\n模型测试集准确率: {best_model.score(X_test, y_test):.4f}")
print(f"最优参数: {grid_search.best_params_}")
