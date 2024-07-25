import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import shap


def shap_feature_analysis(X_train, y_train):
    # 模型的构建与训练
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)

    # 创建Explainer
    explainer = shap.TreeExplainer(model, X_test)
    # 输出SHAP值
    shap_obj = explainer(X_test)
    # 特征分析
    # shap.plots.bar(shap_obj[:, :, 0], show=False)  # 全局条形图
    # plt.savefig(os.path.join(base_dir, 'result', 'shap_bar.png'), bbox_inches='tight', dpi=600)
    shap.plots.beeswarm(shap_obj[:, :, 0], show=False)  # 全局蜂群图
    plt.savefig(os.path.join(base_dir, 'result', 'shap_beeswarm.jpg'), bbox_inches='tight', dpi=600)

    return



def rf_feature_analysis(X_train, y_train):
    # 训练随机森林模型
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    # 提取特征重要性
    feature_importance = model.feature_importances_
    feature_names = features

    # 创建特征重要性的DataFrame
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})

    # 对特征重要性进行排序
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.savefig(os.path.join(base_dir, 'result', 'rf_feature_analysis.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    return

def rf_model_pred(X_train, X_test, y_train, y_test):
    # 模型的构建与训练
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # 模型推理与评价
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)  # 准确率acc
    cm = confusion_matrix(y_test, y_pred)  # 混淆矩阵
    cr = classification_report(y_test, y_pred)  # 分类报告
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1], pos_label=1)  # 计算ROC曲线和AUC值,绘制ROC曲线
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(base_dir, 'result', 'rf_roc_analysis.jpg'), bbox_inches='tight', dpi=600)
    plt.show()

    return acc, cm, cr

if __name__ == '__main__':
    # 准备数据
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(base_dir)
    data = pd.read_csv(os.path.join(base_dir, 'data', 'dataset.csv'))
    df = pd.DataFrame(data)

    # 目标变量和特征变量
    target = 'target'
    features = df.columns.drop(target)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)

    #  特征工程---shap特征重要性分析
    shap_feature_analysis(X_train, y_train)

    # 分类预测
    acc, cm, cr = model = rf_model_pred(X_train, X_test, y_train, y_test)

    #  特征工程---RF特征重要性分析
    rf_feature_analysis(X_train, y_train)