import xgboost
from sklearn.tree import DecisionTreeClassifier
import shap

# train XGBoost model
X, y = shap.datasets.adult()
model = DecisionTreeClassifier().fit(X, y)

# compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

shap.plots.beeswarm(shap_values[:,:,0])