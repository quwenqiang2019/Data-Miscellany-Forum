from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Load sample data
X, y = load_breast_cancer(return_X_y=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train a random forest model
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(X_train, y_train)

# Get baseline accuracy on test data
base_acc = accuracy_score(y_test, rf.predict(X_test))

# Initialize empty list to store importances
importances = []

# Iterate over all columns and remove one at a time
for i in range(X_train.shape[1]):
    X_temp = np.delete(X_train, i, axis=1)
    rf.fit(X_temp, y_train)
    acc = accuracy_score(y_test, rf.predict(np.delete(X_test, i, axis=1)))
    importances.append(base_acc - acc)

# Plot importance scores
plt.style.use('ggplot')
plt.figure(figsize=(10, 8))
plt.bar(range(len(importances)), importances)
plt.xlabel('Feature Index')
plt.ylabel('Feature Importance')
plt.show()