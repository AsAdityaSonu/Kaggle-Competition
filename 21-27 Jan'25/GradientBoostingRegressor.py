#R Square: 0.48870815007168156

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor  # Changed from ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from tqdm import tqdm  # Importing tqdm

# Load training data
trainData = pd.read_csv("/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/train.csv")

# Imputation
numeric_features = trainData.drop(columns=["target"]).columns
imputer = SimpleImputer(strategy='mean')
trainData_imputed = pd.DataFrame(imputer.fit_transform(trainData[numeric_features]), columns=numeric_features)
trainData_imputed["target"] = trainData["target"]

# Correlation heatmap
corr = trainData_imputed.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Feature-target split
x_train = trainData_imputed.drop(columns=["target"])
y_train = trainData_imputed["target"]

# Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
x_train_poly = poly.fit_transform(x_train_scaled)

# Gradient Boosting Regressor Model (changed from ExtraTreesRegressor)
gbr_model = GradientBoostingRegressor(random_state=42)

# Cross-validation for Gradient Boosting Regressor with tqdm progress bar
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Wrapping the cross-validation scores with tqdm to track progress
cv_scores = []
for train_index, val_index in tqdm(kf.split(x_train_poly), total=kf.get_n_splits(), desc="Cross-validation"):
    X_train_fold, X_val_fold = x_train_poly[train_index], x_train_poly[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    gbr_model.fit(X_train_fold, y_train_fold)
    fold_score = gbr_model.score(X_val_fold, y_val_fold)
    cv_scores.append(fold_score)

cv_scores = np.array(cv_scores)
print(f"R² scores for each fold: {cv_scores}")
print(f"Mean R² score from cross-validation: {cv_scores.mean()}")

# Fit model on the entire training set
gbr_model.fit(x_train_poly, y_train)

# Evaluate on training data
train_r2 = r2_score(y_train, gbr_model.predict(x_train_poly))
print(f"R² score on training data: {train_r2}")

# Load test data
testData = pd.read_csv("/kaggle/input/ucs-654-kaggle-hack-lab-exam-ii/test.csv")

# Preprocess test data
test_features = testData.drop(columns=["id"]).columns
testData_imputed = pd.DataFrame(imputer.transform(testData[test_features]), columns=test_features)

x_test_scaled = scaler.transform(testData_imputed)
x_test_poly = poly.transform(x_test_scaled)

# Predict on test data
test_predictions = gbr_model.predict(x_test_poly)

# Submission
submission = pd.DataFrame({
    "id": testData["id"],
    "target": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("ho gaya")