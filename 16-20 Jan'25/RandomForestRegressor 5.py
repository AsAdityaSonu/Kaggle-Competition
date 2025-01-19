# Public R Square Score: 0.84399
# Model used: RandomForestRegressor

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score

# Load training data
trainData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /16-20 Jan'25/Dataset/train_data.csv")

# Impute missing values
numeric_features = trainData.drop(columns=["target"]).columns
imputer = KNNImputer(n_neighbors=5)
trainData_imputed = pd.DataFrame(imputer.fit_transform(trainData[numeric_features]), columns=numeric_features)
trainData_imputed["target"] = trainData["target"]

# Handle invalid values in f3
trainData_imputed["f3"] = trainData_imputed["f3"].clip(lower=0)
trainData_imputed["f3_log"] = trainData_imputed["f3"].apply(lambda x: np.log(x + 1))

# Prepare training data
x_train = trainData_imputed.drop(columns=["target"])
y_train = trainData_imputed["target"]

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
x_train_poly = poly.fit_transform(x_train_scaled)

rf_model = RandomForestRegressor(random_state=42)
selector = RFE(rf_model, n_features_to_select=6)
x_train_selected = selector.fit_transform(x_train_poly, y_train)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['log2', 'sqrt'],
    'bootstrap': [True, False]
}
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(x_train_selected, y_train)

best_rf_model = grid_search.best_estimator_

# Calculate cross-validation R² scores
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_rf_model, x_train_selected, y_train, cv=kf, scoring='r2')

# Fit final model
best_rf_model.fit(x_train_selected, y_train)

# Training R² score
train_r2_score = r2_score(y_train, best_rf_model.predict(x_train_selected))

# Print R² scores
print("Cross-Validation R² Scores:", cv_scores)
print("Mean Cross-Validation R² Score:", np.mean(cv_scores))
print("Training R² Score:", train_r2_score)

# Load test data
testData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /16-20 Jan'25/Dataset/test_data.csv")

# Impute missing values for test data
test_features = testData.drop(columns=["id"]).columns
testData_imputed = pd.DataFrame(imputer.transform(testData[test_features]), columns=test_features)

# Handle invalid values in f3
testData_imputed["f3"] = testData_imputed["f3"].clip(lower=0)
testData_imputed["f3_log"] = testData_imputed["f3"].apply(lambda x: np.log(x + 1))

x_test_scaled = scaler.transform(testData_imputed)
x_test_poly = poly.transform(x_test_scaled)
x_test_selected = selector.transform(x_test_poly)

# Make predictions on test data
test_predictions = best_rf_model.predict(x_test_selected)

# Prepare submission
submission = pd.DataFrame({
    "id": testData["id"],
    "target": test_predictions
})

submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' has been saved.")