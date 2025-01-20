import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score

# Load training data
trainData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /20-27 Jan'25/Dataset/train.csv")

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
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)  # Changed to interaction-only
x_train_poly = poly.fit_transform(x_train_scaled)

# Feature Selection
et_model = ExtraTreesRegressor(random_state=42)
selector = RFE(estimator=et_model, n_features_to_select=10)  # Increased features
x_train_selected = selector.fit_transform(x_train_poly, y_train)

# Parameter grid for ExtraTreesRegressor
param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 3, 5],
    'max_features': ['auto', 'log2', 'sqrt'],
    'bootstrap': [True, False]
}

# GridSearchCV
grid_search = GridSearchCV(estimator=et_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(x_train_selected, y_train)

best_et_model = grid_search.best_estimator_

# Cross-validation scores
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_et_model, x_train_selected, y_train, cv=kf, scoring='r2')
print(f"R² scores for each fold: {cv_scores}")
print(f"Mean R² score from cross-validation: {cv_scores.mean()}")

# Fit best model
best_et_model.fit(x_train_selected, y_train)
train_r2 = r2_score(y_train, best_et_model.predict(x_train_selected))
print(f"R² score on training data: {train_r2}")

# Test data
testData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /20-27 Jan'25/Dataset/test.csv")

test_features = testData.drop(columns=["id"]).columns
testData_imputed = pd.DataFrame(imputer.transform(testData[test_features]), columns=test_features)

x_test_scaled = scaler.transform(testData_imputed)
x_test_poly = poly.transform(x_test_scaled)
x_test_selected = selector.transform(x_test_poly)

test_predictions = best_et_model.predict(x_test_selected)

# Submission
submission = pd.DataFrame({
    "id": testData["id"],
    "target": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("ho gaya")