import os
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import warnings
import shap
warnings.filterwarnings('ignore')


def corr_analysis():
    # =====================4、 相关性分析特征重要性分析=========================
    # 重构训练集dataframe
    train_X = pd.DataFrame(X_train, columns=features)
    train_y = pd.DataFrame(y_train, columns=[target])
    train = pd.concat([train_X, train_y], axis=1)

    # 对训练集进行相关性分析
    sns.set(font_scale=1.2)
    plt.rc('font', family=['Times New Roman', 'SimSun'], size=12)
    plt.figure(figsize=(20, 16))
    plt.subplots_adjust()
    ax = sns.heatmap(train.corr(), annot=True, xticklabels=False, fmt=".2f")
    ax.set_title('相关性热力图')  # 图标题
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'result', 'corr.jpg'), bbox_inches='tight', dpi=600)
    plt.show()


def lr_classifier():
    # 模型的构建与训练
    model = LogisticRegression()
    model.fit(X_train_m, y_train_m)

    # 10折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    scores_model1 = cross_val_score(model, X_train_m, y_train_m, cv=kf)
    print("LR准确率为：", scores_model1)
    print("LR平均准确率为：", scores_model1.mean())

    # 模型推理与评价
    X_test_m = mm1.transform(X_test)

    # 创建Explainer
    explainer = shap.KernelExplainer(model.predict, X_test_m[:100])
    # 以SHAP的Explanation对象形式输出SHAP值
    shap_obj = explainer(X_test_m[:100])
    # 特征分析
    # shap.summary_plot(shap_obj, X_test_m[:100], plot_type="bar", feature_names=features, show=False)
    # plt.savefig(os.path.join(base_dir, 'result', 'lr_bar.png'), bbox_inches='tight', dpi=600)
    shap.summary_plot(shap_obj, X_test_m[:100], feature_names=features, show=False)
    plt.savefig(os.path.join(base_dir, 'result', 'lr_beeswarm.png'), bbox_inches='tight', dpi=600)

    y_pred_m = model.predict(X_test_m)
    y_scores = model.predict_proba(X_test_m)
    y_pred = mm2.inverse_transform(np.reshape(y_pred_m, (-1, 1)))

    acc = accuracy_score(y_test, y_pred) # 准确率acc
    print("准确率：\n", acc)
    cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
    sns.heatmap(cm, annot=True,fmt='g')
    cr = classification_report(y_test, y_pred) # 分类报告
    print("分类报告：\n", cr)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1], pos_label=1) # 计算ROC曲线和AUC值,绘制ROC曲线
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
    plt.show()


def svc_classifier():
    # 模型的构建与训练
    model = SVC(probability=True)
    model.fit(X_train_m, y_train_m)

    # 10折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    scores_model1 = cross_val_score(model, X_train_m, y_train_m, cv=kf)
    print("SVC准确率为：", scores_model1)
    print("SVC平均准确率为：", scores_model1.mean())

    # 模型推理与评价
    X_test_m = mm1.transform(X_test)

    # 创建Explainer
    explainer = shap.KernelExplainer(model.predict, X_test_m[:100])
    # 以SHAP的Explanation对象形式输出SHAP值
    shap_obj = explainer(X_test_m[:100])
    # 特征分析
    # shap.summary_plot(shap_obj, X_test_m[:100], plot_type="bar", feature_names=features, show=False)
    # plt.savefig(os.path.join(base_dir, 'result', 'svc_bar.png'), bbox_inches='tight', dpi=600)
    shap.summary_plot(shap_obj, X_test_m[:100], feature_names=features, show=False)
    plt.savefig(os.path.join(base_dir, 'result', 'svc_beeswarm.png'), bbox_inches='tight', dpi=600)

    y_pred_m = model.predict(X_test_m)
    y_scores = model.predict_proba(X_test_m)
    y_pred = mm2.inverse_transform(np.reshape(y_pred_m, (-1, 1)))

    acc = accuracy_score(y_test, y_pred) # 准确率acc
    print("准确率：\n", acc)
    cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
    sns.heatmap(cm, annot=True,fmt='g')
    cr = classification_report(y_test, y_pred) # 分类报告
    print("分类报告：\n", cr)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:, 1], pos_label=1) # 计算ROC曲线和AUC值,绘制ROC曲线
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
    plt.show()

def xgb_classifier():
    # 模型的构建与训练
    model = XGBClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # 10折交叉验证
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    scores_model1 = cross_val_score(model, X_train, y_train, cv=kf)
    print("XGB准确率为：", scores_model1)
    print("XGB平均准确率为：", scores_model1.mean())

    # 模型推理与评价
    # 创建Explainer
    explainer = shap.TreeExplainer(model, X_test[:100])
    # # 以SHAP的Explanation对象形式输出SHAP值
    shap_obj = explainer(X_test[:100])
    # 特征分析
    # shap.summary_plot(shap_obj, X_test[:100], plot_type="bar", feature_names=features, show=False)
    # plt.savefig(os.path.join(base_dir, 'result', 'xgb_bar.png'), bbox_inches='tight', dpi=600)
    shap.summary_plot(shap_obj, X_test[:100],  feature_names=features, show=False)
    plt.savefig(os.path.join(base_dir, 'result', 'xgb_beeswarm.png'), bbox_inches='tight', dpi=600)

    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)  # 准确率acc
    print("准确率：\n", acc)
    cm = confusion_matrix(y_test, y_pred)  # 混淆矩阵
    sns.heatmap(cm, annot=True,fmt='g')

    cr = classification_report(y_test, y_pred)  # 分类报告
    print("分类报告：\n", cr)
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
    plt.show()


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
    data = pd.read_csv(os.path.join(base_dir, 'data', 'heartbalance.csv'))
    df = pd.DataFrame(data)
    target = 'target'
    features = df.columns.drop(target)
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[[target]], test_size=0.2, random_state=0)


    mm1 = MinMaxScaler()   # 特征进行归一化
    X_train_m = mm1.fit_transform(X_train)
    mm2 = MinMaxScaler()     # 标签进行归一化
    y_train_m = mm2.fit_transform(y_train)

    # corr_analysis()
    # lr_classifier()
    # svc_classifier()
    xgb_classifier()

