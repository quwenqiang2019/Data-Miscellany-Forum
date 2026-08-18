from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)

rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(X, y)

importances = rf.feature_importances_

# Plot importances
plt.style.use('ggplot')
plt.figure(figsize=(10, 8))
plt.bar(range(X.shape[1]), importances)
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.show()