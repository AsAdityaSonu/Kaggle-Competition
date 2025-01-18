# Model Used: RandomForestRegressor
# Training R² score: 0.967916105562378
# Testing R² score: 0.7542721422687347

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

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

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

train_predictions = rf_model.predict(X_train)
test_predictions = rf_model.predict(X_test)

train_r2 = r2_score(y_train, train_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(f"Random Forest Regressor Training R² score: {train_r2}")
print(f"Random Forest Regressor Testing R² score: {test_r2}")