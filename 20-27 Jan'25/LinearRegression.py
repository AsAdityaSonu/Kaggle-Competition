# Public R Square Score: 1
# Model used: Linear Regression

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
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
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
x_train_poly = poly.fit_transform(x_train_scaled)

# Linear Regression Model
lr_model = LinearRegression()

# Cross-validation for Linear Regression
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(lr_model, x_train_poly, y_train, cv=kf, scoring='r2')
print(f"R² scores for each fold: {cv_scores}")
print(f"Mean R² score from cross-validation: {cv_scores.mean()}")

# Fit model on the entire training set
lr_model.fit(x_train_poly, y_train)

# Evaluate on training data
train_r2 = r2_score(y_train, lr_model.predict(x_train_poly))
print(f"R² score on training data: {train_r2}")

# Load test data
testData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /20-27 Jan'25/Dataset/test.csv")

# Preprocess test data
test_features = testData.drop(columns=["id"]).columns
testData_imputed = pd.DataFrame(imputer.transform(testData[test_features]), columns=test_features)

x_test_scaled = scaler.transform(testData_imputed)
x_test_poly = poly.transform(x_test_scaled)

# Predict on test data
test_predictions = lr_model.predict(x_test_poly)

# Submission
submission = pd.DataFrame({
    "id": testData["id"],
    "target": test_predictions
})
submission.to_csv("submission.csv", index=False)
print("ho gaya")