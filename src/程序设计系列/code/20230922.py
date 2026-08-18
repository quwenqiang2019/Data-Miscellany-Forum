from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
df = pd.DataFrame(X, columns=range(30))
df['y'] = y

rf = RandomForestClassifier()

rfe = RFE(rf, n_features_to_select=10)
rfe.fit(X, y)

print(rfe.ranking_)