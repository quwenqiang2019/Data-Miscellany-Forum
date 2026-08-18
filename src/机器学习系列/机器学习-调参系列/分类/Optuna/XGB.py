import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import log_loss
import optuna
import warnings
warnings.filterwarnings('ignore')


# 准备数据
data = pd.read_csv(r'Dataset.csv')
df = pd.DataFrame(data)
print(df.head())

# 提取目标变量和特征变量
target = 'target'
features = df.columns.drop(target)
print(data["target"].value_counts()) # 顺便查看一下样本是否平衡

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

# optuna优化并获取最优超参数
# 定义目标函数
def objective(trial):
    # 定义超参数搜索范围
    params = {
        'objective': 'multi:softprob',
        'num_class': 2,
        'booster': 'gbtree',
        'eval_metric': 'mlogloss',
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
        'subsample': trial.suggest_uniform('subsample', 0.1, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
    }

    # 训练和评估模型
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    loss = log_loss(y_test, y_pred)
    return loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=True)
best_params = study.best_params
print(f"Best Params: {best_params}")
