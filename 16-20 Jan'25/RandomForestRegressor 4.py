# Public Score: 0.85963

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer

trainData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /16-20 Jan'25/Dataset/train_data.csv")

print(trainData.head(5))
print(trainData.columns)
print(trainData.info())

imputer = SimpleImputer(strategy='mean')
trainData_imputed = pd.DataFrame(imputer.fit_transform(trainData), columns=trainData.columns)

corr = trainData_imputed.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

x_train = trainData_imputed.drop(columns=["target"])
y_train = trainData_imputed["target"]

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
x_train_poly = poly.fit_transform(x_train_scaled)

rf_model = RandomForestRegressor(random_state=42)
selector = RFE(rf_model, n_features_to_select=6)
x_train_selected = selector.fit_transform(x_train_poly, y_train)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(x_train_selected, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

best_rf_model = grid_search.best_estimator_

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_rf_model, x_train_selected, y_train, cv=kf, scoring='r2')

print(f"R² scores for each fold: {cv_scores}")
print(f"Mean R² score from k-fold cross-validation: {cv_scores.mean()}")

best_rf_model.fit(x_train_selected, y_train)

testData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /16-20 Jan'25/Dataset/test_data.csv")
x_test = testData[["f1", "f2", "f3", "f4", "f5", "f6"]]

x_test_scaled = scaler.transform(x_test)

x_test_poly = poly.transform(x_test_scaled)

x_test_selected = selector.transform(x_test_poly)

test_predictions = best_rf_model.predict(x_test_selected)

submission = pd.DataFrame({
    "id": testData["id"],
    "target": test_predictions
})

submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' has been saved.")