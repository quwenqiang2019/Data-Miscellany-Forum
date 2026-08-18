import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss

# 读取数据
df = pd.read_csv('E:\data\PubMed Multi Label Text Classification Dataset Processed.csv')
df = df.drop(['Title', 'meshMajor', 'pmid', 'meshid', 'meshroot'], axis =1)
print(df)

X = df["abstractText"]
y = np.asarray(df[df.columns[1:]])

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# 分词embedding模型
vectorizer = TfidfVectorizer(max_features=2500, max_df=0.9).fit(X)
# 分词embedding
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 模型训练
clf = MultiOutputClassifier(LogisticRegression()).fit(X_train_tfidf, y_train)

# 模型预测
prediction = clf.predict(X_test_tfidf)
print(prediction)

# 模型评估
print('Accuracy Score: ', accuracy_score(y_test, prediction))
print('Hamming Loss: ', round(hamming_loss(y_test, prediction),2))