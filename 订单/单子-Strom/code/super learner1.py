import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.optimize import minimize
from scipy.optimize import nnls
from sklearn.preprocessing import PolynomialFeatures#
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, BayesianRidge
from sklearn.svm import SVR, SVC
import random
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import os
np.random.seed(0)
random.seed(0)


''' An example dictionary of estimators can be specified as follows:
Gest_dict = {'LR': LogisticRegression(), 'SVC': SVC(probability=True),
                                 'RF': RandomForestClassifier(), 'KNN': KNeighborsClassifier(),
                                 'AB': AdaBoostClassifier(), 'poly': 'poly'}

Note that 'poly' is specified as a string because there are no default polynomial feature regressors in sklearn.
This one defaults to 2nd order features (e.g. x1*x2, x1*x3 etc...)'''


def fn(x, A, b):
    return np.linalg.norm(A.dot(x) - b)


def combiner_solve(x, y):
    # adapted from https://stackoverflow.com/questions/33385898/how-to-include-constraint-to-scipy-nnls-function-solution-so-that-it-sums-to-1/33388181
    beta_0, rnorm = nnls(x, y)
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = [[0.0, None]] * x.shape[1]
    minout = minimize(fn, beta_0, args=(x, y), method='SLSQP', bounds=bounds, constraints=cons)
    beta = minout.x
    return beta


class SuperLearner(object):
    def __init__(self, output, est_dict, k, standardized_outcome=False):

        self.k = k  # number of cross validation folds
        self.beta = None
        self.output = output  # 'reg' for regression, 'proba' or 'cls' classification
        self.trained_superlearner = None
        self.est_dict = est_dict  # dictionary of learners/algos
        self.standardized_outcome = standardized_outcome

        self.x_std = None
        self.x_mean = None
        self.y_std = None
        self.y_mean = None
        self.num_classes = None

    def fit(self, x, y):
        x = x.values if isinstance(x, pd.DataFrame) else x
        y = y.values[:, 0] if isinstance(y, pd.DataFrame) else y

        # mean and std for full dataset (can be reused wth new data at prediction time)
        self.x_std = x.std(0)
        self.x_mean = x.mean(0)

        if self.standardized_outcome:
            self.y_std = y.std(0)
            self.y_mean = y.mean(0)

        if (self.output == 'cls') or (self.output == 'proba'):
            self.num_classes = np.unique(y)

            if len(self.num_classes) == 2:
                self.num_classes = 1
            elif len(self.num_classes) > 2:
                self.num_classes = len(self.num_classes)
                self.output = 'cat'
        else:
            self.num_classes = 1

        kf = KFold(n_splits=self.k, shuffle=True, random_state=0)

        all_preds = np.zeros((len(y), len(self.est_dict)))  # for test preds

        i = 0
        for key in self.est_dict.keys():
            print('Training estimator:', key)

            est = self.est_dict[key]

            preds = []
            gts = []

            for train_index, test_index in kf.split(x):
                x_train = x[train_index]
                x_test = x[test_index]
                y_train = y[train_index]
                y_test = y[test_index]

                # per train/test fold means and standard deviations
                x_std = x_train.std(0)
                x_mean = x_train.mean(0)
                x_train = (x_train - x_mean) / x_std
                x_test = (x_test - x_mean) / x_std

                if self.standardized_outcome:
                    y_std = y_train.std(0)
                    y_mean = y_train.mean(0)
                    y_train = (y_train - y_mean) / y_std
                    y_test = (y_test - y_mean) / y_std

                if key == 'poly':
                    est = LogisticRegression(C=1e2, max_iter=350) if ((self.output == 'cls') or (
                            self.output == 'proba') or (self.output == 'cat')) else LinearRegression()
                    poly = PolynomialFeatures(2)
                    x_train_poly = poly.fit_transform(x_train)
                    x_test_poly = poly.fit_transform(x_test)

                    est.fit(x_train_poly, y_train)

                else:
                    est.fit(x_train, y_train)

                p = est.predict(x_test_poly) if key == 'poly' else est.predict(x_test)
                preds.append(p)
                gts.append(y_test)

            preds = np.concatenate(preds)
            gts = np.concatenate(gts)

            all_preds[:, i] = preds

            i += 1

        # estimate betas on test predictions
        self.beta = combiner_solve(all_preds, gts)  # all_preds is of shape [batch, categories, predictors]

        # now train each estimator on full dataset

        x = (x - self.x_mean) / self.x_std
        if self.standardized_outcome:
            y = (y - self.y_mean) / self.y_std

        for key in self.est_dict.keys():
            print('Training estimator on full data:', key)

            est = self.est_dict[key]

            if key == 'poly':
                est = LogisticRegression(C=1e2, max_iter=350) if ((self.output == 'cls') or (
                        self.output == 'proba') or (self.output == 'cat')) else LinearRegression()
                poly = PolynomialFeatures(2)
                x_poly = poly.fit_transform(x)

                est.fit(x_poly, y)

            else:
                est.fit(x, y)

            self.est_dict[key] = est

    def predict(self, x):
        x = x.values if isinstance(x, pd.DataFrame) else x
        x_ = (x - self.x_mean) / self.x_std
        all_preds = np.zeros((len(x_), len(self.est_dict)))
        i = 0

        for key in self.est_dict.keys():
            est = self.est_dict[key]
            if key == 'poly':
                poly = PolynomialFeatures(2)
                x_scaled = poly.fit_transform(x_)

            preds = est.predict(x_) if key != 'poly' else est.predict(x_scaled)

            all_preds[:, i] = preds

            i += 1

        weighted_preds = np.dot(all_preds, self.beta)
        weighted_preds = weighted_preds.reshape(-1, 1)
        if self.standardized_outcome:
            weighted_preds = (weighted_preds * self.y_std) + self.y_mean
        return weighted_preds

    def predict_proba(self, x):
        x = x.values if isinstance(x, pd.DataFrame) else x

        x_ = (x - self.x_mean) / self.x_std

        all_preds = np.zeros((len(x), self.num_classes, len(self.est_dict)))
        i = 0

        for key in self.est_dict.keys():
            est = self.est_dict[key]
            if key == 'poly':
                poly = PolynomialFeatures(2)
                x_poly = poly.fit_transform(x_)

            preds = est.predict_proba(x_)[:, 1] if key != 'poly' else est.predict_proba(x_poly)[:, 1]
            preds = preds.reshape(-1, 1)
            all_preds[:, :, i] = preds

            i += 1

        weighted_preds = []
        for cl in range(self.num_classes):
            preds = np.dot(all_preds[:, cl, :], self.beta)

            if self.standardized_outcome:
                preds = (preds * self.y_std) + self.y_mean

            weighted_preds.append(preds)

        weighted_preds = np.asarray(weighted_preds).T

        return weighted_preds


if __name__ == '__main__':
    N = 1000
    output = 'cls'  # 'cls' for classification 'proba' for probabilities, 'reg' for regression
    k = 10

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 准备数据
    df = pd.DataFrame(pd.read_excel(os.path.join(base_dir, 'data', '副本data-0416-49变量.xlsx')))
    ## 数据基本信息
    print(df.head())
    print(df.columns)
    cat_cols = [col for col in df.columns if df[col].dtype == "object"]  # 类别型变量名
    num_cols = [col for col in df.columns if df[col].dtype != "object"]  # 数值型变量名

    # 提取目标变量和特征变量
    target = 'Outcome'
    features = df.columns.drop(target)
    class_count_0, class_count_1 = df["Outcome"].value_counts()  # 顺便查看一下样本是否平衡
    class_0_under = df[df[target] == 0].sample(class_count_1)
    df = pd.concat([class_0_under, df[df[target] == 1]], axis=0)
    print(df["Outcome"].value_counts())

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=0)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 模型的构建与训练
    est_dict = {'LR': LogisticRegression(), 'SVC': SVC(probability=True), 'RF': RandomForestClassifier()}
    SL = SuperLearner(output='cls', est_dict=est_dict, k=k, standardized_outcome=False)
    SL.fit(X_train, y_train)
    preds = SL.predict(X_test)
    y_pred = np.round(preds)
    # print(y_pred)
    y_scores = SL.predict_proba(X_test)
    # print(y_scores)
    print(SL.beta, balanced_accuracy_score(y_test, y_pred))

    acc = accuracy_score(y_test, y_pred) # 准确率acc
    cm = confusion_matrix(y_test, y_pred) # 混淆矩阵
    cr = classification_report(y_test, y_pred) # 分类报告
    print(acc, cm, cr, sep='\n')
    fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1) # 计算ROC曲线和AUC值,绘制ROC曲线
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
    plt.savefig(os.path.join(base_dir, 'result', f'super learner_roc1.png'), bbox_inches='tight', dpi=600)
    plt.show()