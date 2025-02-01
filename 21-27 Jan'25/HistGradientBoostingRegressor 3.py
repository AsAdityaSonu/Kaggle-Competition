# R square: 0.7530049162963477

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from tqdm import tqdm

# Load training data
train_df = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /21-27 Jan'25/Dataset/train.csv")
X = train_df.drop(columns=['target'])
y = train_df['target']

# Step 1: Correlation Analysis
correlation_matrix = train_df.corr()
threshold = 0.8
correlation_pairs = correlation_matrix.abs().unstack().sort_values(ascending=False)
correlated_features = correlation_pairs[(correlation_pairs > threshold) & (correlation_pairs < 1)]
correlated_pairs = correlated_features.index.tolist()

features_to_drop = set()
for feature_a, feature_b in correlated_pairs:
    if feature_a not in features_to_drop and feature_b not in features_to_drop:
        features_to_drop.add(feature_b)

X_filtered = X.drop(columns=list(features_to_drop), errors='ignore')

# Step 2: Polynomial Features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X_filtered)

# Step 3: Data Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# Step 4: Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Hyperparameter Tuning for HistGradientBoostingRegressor
param_grid = {
    'max_iter': [200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [5, 10, 20],
    'l2_regularization': [0.0, 0.1, 1.0]
}

best_hist_model = HistGradientBoostingRegressor(random_state=42)
best_params = None
best_score = float('-inf')

param_combinations = list(ParameterGrid(param_grid))
total_combinations = len(param_combinations)

with tqdm(total=total_combinations, desc="GridSearch Progress") as pbar:
    for params in param_combinations:
        model = HistGradientBoostingRegressor(random_state=42, **params)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        if score > best_score:
            best_score = score
            best_params = params
            best_hist_model = model
        pbar.update(1)

print(f'Best Parameters: {best_params}')
print(f'Best R-squared Score: {best_score}')

# Step 6: Evaluate the best model
y_pred = best_hist_model.predict(X_val)
r2 = r2_score(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)

print(f'R-squared Score: {r2}')
print(f'Mean Squared Error: {mse}')

# Step 7: Predictions on Test Data
test_df = pd.read_csv("/Users/adityapandey/PycharmProjects/Kaggle-Competition /21-27 Jan'25/Dataset/test.csv")
test_ids = test_df['id']
test_features = test_df.drop(columns=['id'])
test_features_filtered = test_features[X_filtered.columns]

# Apply polynomial transformation and scaling to test features
test_features_poly = poly.transform(test_features_filtered)
test_features_scaled = scaler.transform(test_features_poly)

test_predictions = best_hist_model.predict(test_features_scaled)
output_df = pd.DataFrame({'id': test_ids, 'predicted_target': test_predictions})
output_df.to_csv('predictions_histgradientboosting_poly.csv', index=False)
print('Predictions saved to predictions_histgradientboosting_poly.csv')