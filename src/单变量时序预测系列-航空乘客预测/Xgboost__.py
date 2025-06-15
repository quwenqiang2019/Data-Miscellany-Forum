import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# 1. 读取时间序列数据
data = pd.read_csv('data.csv')
data['Month'] = pd.to_datetime(data['Month'])
df = data

sns.set(font_scale=1.2)
plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)

plt.figure()
plt.plot(df['Month'], df['Passengers'], color='b', alpha=0.6, label='Original Time Series')
plt.title('Original Time Series', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

df['Month'] = df['Month'].apply(lambda x: x.timestamp())
print(df)


# 2. 数据转换为监督学习格式
def create_lag_features(data, lags=10):
    df = data.copy()
    for i in range(1, lags + 1):
        df[f'lag_{i}'] = df['Passengers'].shift(i)
    df.dropna(inplace=True)
    return df

df_lagged = create_lag_features(df, lags=10)
X = df_lagged.drop(columns=['Passengers'])
y = df_lagged['Passengers']

# 3. 数据集拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# 4. 训练 XGBoost 模型
model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# 5. 预测
y_pred = model.predict(X_test)

# 6. 评估模型
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
plt.figure()
plt.plot(y_test.values, label='Actual', color='g')
plt.plot(y_pred, label='Predicted', color='r', linestyle='dashed')
plt.title('Actual vs Predicted', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

# 误差分布
plt.figure()
sns.histplot(y_test - y_pred, bins=30, kde=True, color='purple')
plt.title('Error Distribution', fontsize=12)
plt.tight_layout()
plt.show()

# 7、特征重要性
plt.figure()
feature_importance = model.feature_importances_
sns.barplot(x=X.columns, y=feature_importance, palette='viridis')
plt.title('Feature Importance', fontsize=12)
plt.xticks(X.columns, rotation=45)
plt.tight_layout()
plt.show()

# 8. 调参优化
grid_params = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search = GridSearchCV(XGBRegressor(objective='reg:squarederror'), grid_params, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
print(f'Best Parameters: {grid_search.best_params_}')