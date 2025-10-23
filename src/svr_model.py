import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

df = pd.read_csv('../Datasets/Crops_data.csv')

rice_area_col = 'RICE AREA (1000 ha)'
rice_prod_col = 'RICE PRODUCTION (1000 tons)'
rice_yield_col = 'RICE YIELD (Kg per ha)'

rice_cols = ['Year', 'State Name', 'Dist Name', rice_area_col, rice_prod_col, rice_yield_col]
df_rice = df[rice_cols].copy()
df_clean = df_rice.dropna()

state = LabelEncoder()
distt = LabelEncoder()
df_clean['State_Encoded'] = state.fit_transform(df_clean['State Name'])
df_clean['Dist_Encoded'] = distt.fit_transform(df_clean['Dist Name'])

Q1, Q3 = df_clean[rice_yield_col].quantile([0.25, 0.75])
IQR = Q3 - Q1
df_final = df_clean[(df_clean[rice_yield_col] >= Q1 - 1.5*IQR) & 
                    (df_clean[rice_yield_col] <= Q3 + 1.5*IQR)]

df_final = df_final.copy()
df_final['Year_Area'] = df_final['Year'] * df_final[rice_area_col]
df_final['State_Area'] = df_final['State_Encoded'] * df_final[rice_area_col]
df_final['Prod_Area_Ratio'] = df_final[rice_prod_col] / (df_final[rice_area_col] + 1e-6)

features = ['Year', rice_area_col, rice_prod_col, 'State_Encoded', 'Dist_Encoded',
            'Year_Area', 'State_Area', 'Prod_Area_Ratio']
X = df_final[features]
y = df_final[rice_yield_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

linear = SVR(kernel='linear')
linear.fit(X_train, y_train)
linear_pred = linear.predict(X_test)

param_grid = {'C': [10, 100, 1000], 'gamma': [0.001, 0.01, 0.1], 'epsilon': [0.01, 0.1, 0.5]}
rbf = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, scoring='r2', n_jobs=-1)
rbf.fit(X_train, y_train)
rbf_pred = rbf.predict(X_test)

linear_mse = mean_squared_error(y_test, linear_pred)
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_r2 = r2_score(y_test, linear_pred)

rbf_mse = mean_squared_error(y_test, rbf_pred)
rbf_mae = mean_absolute_error(y_test, rbf_pred)
rbf_r2 = r2_score(y_test, rbf_pred)

print(f"Linear: R²={linear_r2:.4f}, MSE={linear_mse:.2f}, MAE={linear_mae:.2f}")
print(f"RBF: R²={rbf_r2:.4f}, MSE={rbf_mse:.2f}, MAE={rbf_mae:.2f}")
print(f"Best Params: {rbf.best_params_}")
print(f"\nBest Model: {'Linear' if linear_r2 > rbf_r2 else 'RBF'} (R²={max(linear_r2, rbf_r2):.4f})")

os.makedirs('../Models', exist_ok=True)
with open('../Models/svr_linear_model.pkl', 'wb') as f:
    pickle.dump(linear, f)
with open('../Models/svr_rbf_model.pkl', 'wb') as f:
    pickle.dump(rbf.best_estimator_, f)
print("Models saved successfully!")
