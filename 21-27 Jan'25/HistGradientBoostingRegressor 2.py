import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.experimental import enable_hist_gradient_boosting  # Enabling HistGradientBoosting
from sklearn.ensemble import HistGradientBoostingRegressor  # Changed from GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from tqdm import tqdm  # Importing tqdm

# Load training data
trainData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /21-27 Jan'25/Dataset/train.csv")

# Imputation
numeric_features = trainData.drop(columns=["target"]).columns
imputer = SimpleImputer(strategy='mean')
trainData_imputed = pd.DataFrame(imputer.fit_transform(trainData[numeric_features]), columns=numeric_features)
trainData_imputed["target"] = trainData["target"]

# Correlation heatmap to identify highly correlated features
corr = trainData_imputed.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# Feature-target split
x_train = trainData_imputed.drop(columns=["target"])
y_train = trainData_imputed["target"]

# Check if log transformation is necessary
# If target variable isn't skewed, avoid log transformation
# y_train = np.log1p(y_train)  # Uncomment only if target is skewed

# Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# Polynomial Features (Consider lower degree or no polynomial features)
poly = PolynomialFeatures(degree=1, interaction_only=False, include_bias=False)  # Changed degree to 1
x_train_poly = poly.fit_transform(x_train_scaled)

# HistGradientBoosting Regressor Model (changed from GradientBoostingRegressor)
hgb_model = HistGradientBoostingRegressor(random_state=42, early_stopping=False)  # Removed early stopping for now

# Hyperparameter tuning with GridSearchCV (narrowed search space)
param_grid = {
    'learning_rate': [0.05, 0.1],
    'max_iter': [100, 150],
    'max_depth': [3, 5],
    'min_samples_leaf': [20, 50]
}
grid_search = GridSearchCV(hgb_model, param_grid, cv=3, n_jobs=-1, scoring='r2')
grid_search.fit(x_train_poly, y_train)

print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best R² score from GridSearchCV: {grid_search.best_score_}")

# Train the model with the best parameters found
hgb_model = grid_search.best_estimator_

# Cross-validation for HistGradientBoosting Regressor with tqdm progress bar
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []
for train_index, val_index in tqdm(kf.split(x_train_poly), total=kf.get_n_splits(), desc="Cross-validation"):
    X_train_fold, X_val_fold = x_train_poly[train_index], x_train_poly[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    hgb_model.fit(X_train_fold, y_train_fold)
    fold_score = hgb_model.score(X_val_fold, y_val_fold)
    cv_scores.append(fold_score)

cv_scores = np.array(cv_scores)
print(f"R² scores for each fold: {cv_scores}")
print(f"Mean R² score from cross-validation: {cv_scores.mean()}")

# Fit model on the entire training set
hgb_model.fit(x_train_poly, y_train)

# Evaluate on training data
train_r2 = r2_score(y_train, hgb_model.predict(x_train_poly))
print(f"R² score on training data: {train_r2}")

# Load test data
testData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /21-27 Jan'25/Dataset/test.csv")

# Preprocess test data
test_features = testData.drop(columns=["id"]).columns
testData_imputed = pd.DataFrame(imputer.transform(testData[test_features]), columns=test_features)

x_test_scaled = scaler.transform(testData_imputed)
x_test_poly = poly.transform(x_test_scaled)

# Predict on test data
test_predictions = hgb_model.predict(x_test_poly)

# Inverse log transformation of predictions (if you applied log transformation on y_train)
test_predictions = np.expm1(test_predictions)  # Uncomment if log transformation was applied

# Submission
submission = pd.DataFrame({
    "id": testData["id"],
    "target": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("ho gaya")