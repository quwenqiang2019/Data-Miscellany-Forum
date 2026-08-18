import os
import numpy as np
from sklearn import tree
from sklearn import svm
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn import neighbors
from sklearn import ensemble
from sklearn import neural_network
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Create a new classifier which is based on the sckit-learn BaseEstimator and ClassifierMixin classes
class SuperLearnerClassifier(BaseEstimator, ClassifierMixin):
    """An ensemble classifier that uses heterogeneous models at the base layer and a aggregatnio model at the aggregation layer. A k-fold cross validation is used to gnerate training data for the stack layer model.

    Parameters
    ----------

    Attributes
    ----------
    Notes
    -----

    See also
    --------
    ----------
    .. [1]  van der Laan, M., Polley, E. & Hubbard, A. (2007).
            Super Learner. Statistical Applications in Genetics
            and Molecular Biology, 6(1)
            doi:10.2202/1544-6115.1309
    Examples
    --------
    """

    # Define base estimators
    def base_learners():
        svc = svm.SVC(C=100, probability=True)
        knn = neighbors.KNeighborsClassifier(n_neighbors=5)
        nb = naive_bayes.GaussianNB()
        dt = tree.DecisionTreeClassifier()
        lr = linear_model.LogisticRegression()
        rf = ensemble.RandomForestClassifier()
        gb = ensemble.GradientBoostingClassifier(n_estimators=100)

        all_base = {'SVC': svc,
                    'DT': dt,
                    'KNN': knn,
                    'LR': lr,
                    'NB': nb,
                    'RF': rf,
                    'GB': gb
                    }

        return all_base

    # Define stack layer estimators
    def meta_learners():
        meta_dt = tree.DecisionTreeClassifier()
        meta_lr = linear_model.LogisticRegression()
        meta_nb = naive_bayes.GaussianNB()
        meta_svc = svm.SVC(C=100, probability=True)
        meta_knn = neighbors.KNeighborsClassifier(n_neighbors=5)
        meta_rf = ensemble.RandomForestClassifier()
        meta_gb = ensemble.GradientBoostingClassifier(n_estimators=100)
        meta_nn = neural_network.MLPClassifier()

        all_meta = {'DecisionTree': meta_dt,
                    'LogisticRegression': meta_lr,
                    'NaiveBayes': meta_nb,
                    'SVC': meta_svc,
                    'KNN': meta_knn,
                    'RandomForest': meta_rf,
                    'GradientBoosting': meta_gb,
                    'MLPClassifier': meta_nn
                    }

        return all_meta

    # Constructor for the classifier object
    def __init__(self, probability='False', meta_clf='DecisionTree', K=5, num_class=2, \
                 base_models=base_learners(), meta_models=meta_learners(), \
                 dataset='only_meta'):

        """Setup a SuperLearner classifier .
        Parameters
        ----------

        Returns
        -------

        """
        self.base_models = base_models
        self.meta_models = meta_models
        self.K = K
        self.num_class = num_class
        self.probability = probability
        self.meta_clf = meta_clf
        self.dataset = dataset

    # The fit function to train a classifier
    def fit(self, X, y, diversity='False'):
        """Build a SuperLearner classifier from the training set (X, y).
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values (class labels) as integers or strings.
        Returns
        -------
        self : object
        """

        count = 1
        kf = KFold(self.K)
        self.diversity = diversity

        meta_train = np.zeros((len(y), len(self.base_models)), dtype=int)  # Define layout of training set for meta
        meta_train_proba = np.zeros(
            (len(y), len(self.base_models) * self.num_class))  # Define layout of training set on probability

        if self.diversity == 'False':
            print("Number of KFold:", self.K)

        # Split X, y for K-fold
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            if self.diversity == 'False':
                print("Fold:", count)

            # Fit base leaners on labels
            if self.probability == 'False':
                if self.diversity == 'False':
                    print('Fitting of base estimators(Label based): \x1b[33mIn Progress\x1b[0m')

                for i, mod in enumerate(self.base_models):
                    clf_train = self.base_models[mod]
                    clf_train.fit(X_train, y_train)  # Fit base learners
                    train_pred = clf_train.predict(X_test)  # Do prediction on fitted base learners
                    meta_train[test_index, i] = train_pred  # Put the predicted data to training dataset for Superlearner

                count = count + 1

            # Fit base learners on probability
            else:
                print('Fitting of base estimators(probability based): \x1b[33mIn Progress\x1b[0m')
                for i, mod in enumerate(self.base_models):
                    clf_train_proba = self.base_models[mod]  # Fit base learners
                    clf_train_proba.fit(X_train, y_train)  # Do probability prediction on fitted base learners
                    train_pred_proba = clf_train_proba.predict_proba(X_test)

                    # Put the predicted data to training dataset for SuperLearner
                    for l in range(0, 10):
                        meta_train_proba[test_index, i * l + 1] = train_pred_proba[:, l]

                count = count + 1

            if self.diversity == 'False':
                print('Fitting of Base estimators: \x1b[32mSUCCESS\x1b[0m')

        if self.diversity == 'True':
            return meta_train

        # Check if stack layer needs probabiltiy based outputs training
        if self.probability == 'False':
            print('Stack Layer Model:', self.meta_clf)
            print('Fitting Stack Layer on label outputs: \x1b[33mIn Progress\x1b[0m')
            # Check if original data needs to add at meta training set
            if self.dataset == 'Full':
                meta_train_full = np.column_stack([meta_train, X])  # Adding original input to the meta training set
                self.meta_models[self.meta_clf].fit(meta_train_full,
                                                    y)  # Fitting meta learner with full data(original+meta train set)
            else:
                self.meta_models[self.meta_clf].fit(meta_train, y)  # Fitting meta learner

            print('Fitting of Stack Layer: \x1b[32mSUCCESS\x1b[0m')


        else:
            print('Stack Layer Model:', self.meta_clf)
            print('Fitting Stack Layer on probability outputs: \x1b[33mIn Progress\x1b[0m')
            if self.dataset == 'Full':
                meta_train_proba_full = np.column_stack([meta_train_proba, X])
                self.meta_models[self.meta_clf].fit(meta_train_proba_full, y)

            else:
                self.meta_models[self.meta_clf].fit(meta_train_proba, y)  # Fitting meta learner

            print('Fitting of Stack Layer: \x1b[32mSUCCESS\x1b[0m')

        # Return the classifier
        return self

    # The predict function to make a set of predictions for a set of query instances
    def predict_base(self, X, diversity='False'):
        """Predict class labels of the input samples X.
        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        p : array of shape = [n_samples, ].
            The predicted class labels of the input samples.
        """

        X_test = X
        self.diversity = diversity

        # Create test data for meta learner
        meta_test = np.zeros((len(X_test[0:]), len(self.base_models)), dtype=int)

        # Do the prediction on each base learners
        for i, k in enumerate(self.base_models):
            clf_test = self.base_models[k]
            test_pred = clf_test.predict(X_test)
            meta_test[:, i] = test_pred  # Append the predictions to the meta test

        if self.diversity == 'True':
            return meta_test

        # Check if prediction should be done with adding original test set
        if self.dataset == 'Full':
            meta_test_full = np.column_stack([meta_test, X_test])
            meta_pred = self.meta_models[self.meta_clf].predict(meta_test_full)
        else:
            meta_pred = self.meta_models[self.meta_clf].predict(meta_test)

        return meta_pred

    def predict(self, X):
        """Predict class labels of the input samples X.
        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        p : array of shape = [n_samples, ].
            The predicted class labels of the input samples.
        """
        # NOTE: GridSearchCV usually doesn't call predict_proba. Below condition will help GridSearchCV to know when
        # to run with predict_proba method.

        # Predict on the basis of probability outputs or label outputs
        if self.probability == 'False':
            meta_pred = self.predict_base(X)

        else:
            meta_pred = self.predict_proba(X)

        return meta_pred

    # The predict function to make a set of predictions for a set of query instances
    def predict_proba(self, X):
        """Predict class probabilities of the input samples X.
        Parameters
        ----------
        X : array-like matrix of shape = [n_samples, n_features]
            The input samples.
        Returns
        -------
        p : array of shape = [n_samples, n_labels].
            The predicted class label probabilities of the input samples.
        """

        X_test = X

        # Create test data for SuperLearner
        meta_test_proba = np.zeros((len(X_test[0:]), len(self.base_models) * 1))
        print(meta_test_proba.shape)

        # Do the prediction on each base learners
        for i, k in enumerate(self.base_models):
            clf_test_proba = self.base_models[k]
            test_pred_proba = clf_test_proba.predict_proba(X_test)[:, 1]
            # test_pred_proba = test_pred_proba.reshape(-1, 1)
            # print(test_pred_proba.shape)

            meta_test_proba[:, i] = test_pred_proba  # Append the probability predictions to the meta test

        # Check if predict_proba to be done including orignal test data
        if self.dataset == 'Full':
            meta_test_proba_full = np.column_stack([meta_test_proba, X_test])
            meta_pred = self.meta_models[self.meta_clf].predict(meta_test_proba_full)
        else:
            meta_pred = self.meta_models[self.meta_clf].predict(meta_test_proba)

        return meta_pred

    # Method to find diversities among base classifiers
    # The idea of find_diversity method is to get the prediction of each base classifiers
    # and use those predictions to find correlation between them
    def diversity(self, X, y, x):

        meta_data = self.fit(X, y, diversity='True')
        meta_data_predict = self.predict_base(x, diversity='True')
        label = []

        # Get the base models used for create training data and append them to empty list
        for k in self.base_models.keys():
            label.append(k)

        df = pd.DataFrame(meta_data_predict)  # Create a dataframe using training set which was used for stack layer
        df.columns = label  # Add columns(base models) to the dataframe
        div_matrix = df.corr(method='pearson', min_periods=1)  # Find correlation matrix using Pearson Correlation
        div_matrix = round(div_matrix, 2)
        print("\nDiversity Between Base Classifiers:")
        print("-----------------------------------\n")
        print(div_matrix)

        print("\nDiversity Plot:")
        print("-----------------\n")
        plt.imshow(div_matrix)  # Plot the matrix with legends
        plt.colorbar()
        plt.show()

        return self



if __name__ == "__main__":
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
    model = SuperLearnerClassifier()
    model.fit(np.array(X_train), np.array(y_train))

    # 模型推理与评价
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
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
    plt.savefig(os.path.join(base_dir, 'result', f'super learner_roc.png'), bbox_inches='tight', dpi=600)
    plt.show()




