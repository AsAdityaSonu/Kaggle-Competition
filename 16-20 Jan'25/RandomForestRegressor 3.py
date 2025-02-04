# Public R Square Score: 0.80855
# Model used: RandomForestRegressor

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

trainData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /16-20 Jan'25/Dataset/train_data.csv")

print(trainData.head(5))
print(trainData.columns)
print(trainData.info())

corr = trainData.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

x_train = trainData.drop(columns=["target"])
y_train = trainData["target"]

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_model = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='r2')
grid_search.fit(x_train_scaled, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

best_rf_model = grid_search.best_estimator_

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_rf_model, x_train_scaled, y_train, cv=kf, scoring='r2')

print(f"R² scores for each fold: {cv_scores}")
print(f"Mean R² score from k-fold cross-validation: {cv_scores.mean()}")

best_rf_model.fit(x_train_scaled, y_train)

testData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /16-20 Jan'25/Dataset/test_data.csv")
x_test = testData[["f1", "f2", "f3", "f4", "f5", "f6"]]
x_test_scaled = scaler.transform(x_test)

test_predictions = best_rf_model.predict(x_test_scaled)

submission = pd.DataFrame({
    "id": testData["id"],
    "target": test_predictions
})

submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' has been saved.")
