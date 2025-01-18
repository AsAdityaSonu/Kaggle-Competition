import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

testData = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /16-20 Jan'25/Dataset/train_data.csv")

print(testData.head(5))
print(testData.columns)
print(testData.info())

# ----------------------------------------------------------
corr = testData.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()

# ----------------------------------------------------------
x = testData.drop(columns=["target"])
y = testData["target"]

# ----------------------------------------------------------
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# ----------------------------------------------------------
ridge_model = Ridge()
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_grid_search = GridSearchCV(ridge_model, ridge_params, cv=5)
ridge_grid_search.fit(X_train, y_train)

print(f"Best alpha for Ridge Regression: {ridge_grid_search.best_params_}")

# Evaluate Ridge Regression with best hyperparameters
ridge_best_model = ridge_grid_search.best_estimator_
train_r2_ridge = ridge_best_model.score(X_train, y_train)
test_r2_ridge = ridge_best_model.score(X_test, y_test)

print(f"Ridge Regression Training R² score: {train_r2_ridge}")
print(f"Ridge Regression Testing R² score: {test_r2_ridge}")

# ----------------------------------------------------------
# Lasso Regression (L1 Regularization)
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

train_r2_lasso = lasso_model.score(X_train, y_train)
test_r2_lasso = lasso_model.score(X_test, y_test)

print(f"Lasso Regression Training R² score: {train_r2_lasso}")
print(f"Lasso Regression Testing R² score: {test_r2_lasso}")

# ----------------------------------------------------------
# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

train_r2_rf = rf_model.score(X_train, y_train)
test_r2_rf = rf_model.score(X_test, y_test)

print(f"Random Forest Regressor Training R² score: {train_r2_rf}")
print(f"Random Forest Regressor Testing R² score: {test_r2_rf}")