import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import Draft3 as xgb

trainData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /16-20 Jan'25/Dataset/train_data.csv")

print(trainData.head(5))
print(trainData.columns)
print(trainData.info())

# ----------------------------------------------------------
corr = trainData.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# ----------------------------------------------------------
x_train = trainData.drop(columns=["target"])
y_train = trainData["target"]

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

# ----------------------------------------------------------
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2')
grid_search.fit(x_train_scaled, y_train)

print(f"Best parameters from GridSearchCV: {grid_search.best_params_}")

best_rf_model = grid_search.best_estimator_

# ----------------------------------------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_rf_model, x_train_scaled, y_train, cv=kf, scoring='r2')

print(f"R² scores for each fold (Random Forest): {cv_scores}")
print(f"Mean R² score from k-fold cross-validation (Random Forest): {cv_scores.mean()}")

best_rf_model.fit(x_train_scaled, y_train)

# ----------------------------------------------------------
testData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /16-20 Jan'25/Dataset/test_data.csv")

print(testData.head(5))
print(testData.columns)
print(testData.info())

# ----------------------------------------------------------
x_test = testData[["f1", "f2", "f3", "f4", "f5", "f6"]]
x_test_scaled = scaler.transform(x_test)

test_predictions_rf = best_rf_model.predict(x_test_scaled)

# ----------------------------------------------------------
xg_model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
xg_model.fit(x_train_scaled, y_train)

test_predictions_xgb = xg_model.predict(x_test_scaled)

# ----------------------------------------------------------
final_predictions = 0.5 * test_predictions_rf + 0.5 * test_predictions_xgb

# ----------------------------------------------------------
submission = pd.DataFrame({
    "id": testData["id"],
    "target": final_predictions
})

submission.to_csv("submission.csv", index=False)

print("Submission file 'submission.csv' has been saved.")