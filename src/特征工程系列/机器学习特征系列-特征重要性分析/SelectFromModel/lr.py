import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# 1. 准备数据
data = pd.read_csv(r'dataset.csv')
df = pd.DataFrame(data)
print(df)
# 2. 提取目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
X = df[features] 
y = df[target]


# 3. 数据预处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. 标准化特征（对于基于树的模型不是必须的，但对于线性模型很重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "=" * 60)
print("第一步：使用随机森林作为基学习器进行特征选择")
print("=" * 60)

# 5. 首先训练一个逻辑回归模型来获取特征重要性，L1正则化会自动进行特征选择（系数为0的特征被剔除）
lr_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
lr_l1.fit(X_train_scaled, y_train)


# 5.1 查看特征重要性(系数)
lr_coef = pd.DataFrame({
    'feature': features,
    'coefficient': np.abs(lr_l1.coef_[0])
}).sort_values('coefficient', ascending=False)

print("\n逻辑回归L1正则化特征系数（绝对值）:")
print(lr_coef)

# 5.2. 使用SelectFromModel进行特征选择
# 方法1：使用默认阈值（均值）
selector_mean = SelectFromModel(lr_l1, threshold='mean', prefit=True)
X_train_selected_mean = selector_mean.transform(X_train)
X_test_selected_mean = selector_mean.transform(X_test)

selected_features_mean = X.columns[selector_mean.get_support()].tolist()
print(f"\n使用默认阈值（mean）选择的特征 ({len(selected_features_mean)}个):")
print(selected_features_mean)

# 方法2：使用中位数阈值
selector_median = SelectFromModel(lr_l1, threshold='median', prefit=True)
selected_features_median = X.columns[selector_median.get_support()].tolist()
print(f"\n使用中位数阈值（median）选择的特征 ({len(selected_features_median)}个):")
print(selected_features_median)

# 方法3：使用自定义阈值（例如：重要性 > 0.05）
selector_custom = SelectFromModel(lr_l1, threshold=0.05, prefit=True)
selected_features_custom = X.columns[selector_custom.get_support()].tolist()
print(f"\n使用自定义阈值（0.05）选择的特征 ({len(selected_features_custom)}个):")
print(selected_features_custom)


# 6. 对比不同特征子集的模型性能
print("\n" + "=" * 60)
print("第二步：对比特征选择前后的模型性能")
print("=" * 60)

def evaluate_model(X_tr, X_te, y_tr, y_te, feature_names_subset):
    """评估模型性能"""
    lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    cv_scores = cross_val_score(lr, X_tr, y_tr, cv=5)
    return {
        'accuracy': acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_features': X_tr.shape[1],
        'features': feature_names_subset
    }

# 6.1. 使用全部特征
results_all = evaluate_model(X_train, X_test, y_train, y_test, features)
print(f"\n1. 全部特征 ({results_all['n_features']}个):")
print(f"   测试集准确率: {results_all['accuracy']:.4f}")
print(f"   交叉验证准确率: {results_all['cv_mean']:.4f} (+/- {results_all['cv_std']*2:.4f})")

# 6.2. 使用SelectFromModel选择的特征（均值阈值）
results_selected = evaluate_model(
    X_train_selected_mean, X_test_selected_mean, 
    y_train, y_test, selected_features_mean
)
print(f"\n2. SelectFromModel选择特征 ({results_selected['n_features']}个):")
print(f"   测试集准确率: {results_selected['accuracy']:.4f}")
print(f"   交叉验证准确率: {results_selected['cv_mean']:.4f} (+/- {results_selected['cv_std']*2:.4f})")
print(f"   选择的特征: {selected_features_mean}")



# 7. 创建可视化
# 图1：特征重要性条形图
fig1, ax1 = plt.subplots(figsize=(10, 8))

colors = ['#e74c3c' if imp > lr_coef['coefficient'].mean() else '#3498db' 
          for imp in lr_coef['coefficient']]

bars = ax1.barh(lr_coef['feature'], lr_coef['coefficient'], color=colors)
ax1.axvline(lr_coef['coefficient'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean Threshold ({lr_coef["coefficient"].mean():.3f})')

ax1.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
ax1.set_ylabel('Features', fontsize=12, fontweight='bold')
ax1.set_title('Random Forest Feature Importance\n(Red = Selected by Mean Threshold)', 
              fontsize=14, fontweight='bold', pad=20)
ax1.legend(loc='lower right', fontsize=10)
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3)

for i, (idx, row) in enumerate(lr_coef.iterrows()):
    ax1.text(row['coefficient'] + 0.002, i, f'{row["coefficient"]:.3f}', 
             va='center', fontsize=9)

plt.tight_layout()
plt.savefig('./fig1_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("图1保存完成：特征重要性条形图")


# 图2：不同选择方法的特征数量对比
fig2, ax2 = plt.subplots(figsize=(10, 6))

methods = ['None\n(All)', 'Mean', 'Median', 'Custom\n0.05']
n_features = [len(features), len(selected_features_mean), len(selected_features_median), len(selected_features_custom)]
colors_bar = ['#95a5a6', '#e74c3c', '#e67e22', '#f39c12']

bars = ax2.bar(methods, n_features, color=colors_bar, edgecolor='black', linewidth=1.5, width=0.6)

ax2.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
ax2.set_xlabel('Selection Method', fontsize=12, fontweight='bold')
ax2.set_title('Feature Count by Selection Method', fontsize=14, fontweight='bold', pad=20)
ax2.set_ylim(0, 15)
ax2.grid(axis='y', alpha=0.3)

for bar, n in zip(bars, n_features):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
             str(n), ha='center', va='bottom', fontsize=12, fontweight='bold')

reductions = [
            '0%', 
            "{:.2%}".format((len(features)-len(selected_features_mean))/len(features)), 
            "{:.2%}".format((len(features)-len(selected_features_median))/len(features)), 
            "{:.2%}".format((len(features)-len(selected_features_custom))/len(features))
            ]
for i, (bar, red) in enumerate(zip(bars, reductions)):
    if i > 0:  # 跳过第一个
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                 f'↓{red}', ha='center', va='center', fontsize=10, 
                 color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('./fig2_feature_count.png', dpi=150, bbox_inches='tight')
plt.show()
print("图2保存完成：特征数量对比图")


# 图3：模型性能对比
fig3, ax3 = plt.subplots(figsize=(10, 7))

methods = ['All Features\n(13)', 'LR Select\n(7)']
test_acc = [results_all['accuracy'], results_selected['accuracy']]
cv_acc = [results_all['cv_mean'], results_selected['cv_mean']]

x = np.arange(len(methods))
width = 0.35

bars1 = ax3.bar(x - width/2, test_acc, width, label='Test Accuracy', 
                color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = ax3.bar(x + width/2, cv_acc, width, label='CV Accuracy', 
                color='#2ecc71', edgecolor='black', linewidth=1.5)

ax3.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax3.set_xlabel('Feature Selection Method', fontsize=12, fontweight='bold')
ax3.set_title('Model Performance Comparison\nBefore vs After Feature Selection', 
              fontsize=14, fontweight='bold', pad=20)
ax3.set_xticks(x)
ax3.set_xticklabels(methods)
ax3.legend(loc='upper left', fontsize=11)
ax3.set_ylim(0.75, 0.90)
ax3.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('./fig3_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("图3保存完成：模型性能对比图")