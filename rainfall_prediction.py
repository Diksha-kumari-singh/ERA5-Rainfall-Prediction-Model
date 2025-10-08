"""
import pandas as pd
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. Raw Data (Loading and Combining the Two Files) ---

# Define the expected file names for the instantaneous (temperature) and accumulated (rainfall) data.
# NOTE: These names have been simplified. You must RENAME and MOVE the files to this folder.
INSTANT_FILE = 'instant.nc'
ACCUM_FILE = 'accum.nc'

try:
    if not os.path.exists(INSTANT_FILE) or not os.path.exists(ACCUM_FILE):
        # This error message guides the user on the required file structure
        print("CRITICAL ERROR: Data files not found.")
        print(f"Please ensure you have extracted and renamed the files to: {INSTANT_FILE} and {ACCUM_FILE}")
        print("Both files must be in the same folder as the script.")
        exit()

    # Load Instantaneous data (t2m - Temperature)
    ds_instant = xr.open_dataset(INSTANT_FILE, engine='netcdf4')
    df_instant = ds_instant[['t2m']].to_dataframe().reset_index()

    # Load Accumulated data (tp - Total Precipitation)
    ds_accum = xr.open_dataset(ACCUM_FILE, engine='netcdf4')
    df_accum = ds_accum[['tp']].to_dataframe().reset_index()

    # Merge the two DataFrames on the shared coordinates
    df = pd.merge(df_instant, df_accum, on=['valid_time', 'latitude', 'longitude', 'expver', 'number'])

    # Drop rows where data is NaN (this handles any missing records after merging)
    df = df.dropna()
    
    print("Data loaded and combined successfully from both files.\n")
    print("Combined DataFrame head:")
    print(df.head())

except Exception as e:
    print(f"An error occurred during data loading: {e}")
    print("If the error is 'NetCDF: Unknown file format', please try reinstalling your netcdf4 library.")
    exit()

# --- 2. Exploratory Data Analysis (EDA) ---

# Rename t2m (2 metre temperature) to 2m_temperature for better readability
df = df.rename(columns={'t2m': '2m_temperature'}) 

# Drop irrelevant columns for this simple regression model
df = df.drop(columns=['expver', 'number', 'valid_time', 'latitude', 'longitude']) 

print("\n--- EDA: Missing values per column ---")
# Null check after merging and initial cleanup
print(df.isnull().sum()) 

# Visualization of total precipitation (target variable) 
plt.figure(figsize=(10, 6))
sns.histplot(df['tp'], kde=True, bins=50)
plt.title('Distribution of Total Precipitation (tp)')
plt.xlabel('Total Precipitation (m)')
plt.ylabel('Frequency')
plt.show()

# --- 3. Preprocessing, Scaling & Feature Engineering ---

# Handle high-skew in target variable (tp) by applying a log transformation (np.log1p = log(1+x))
df['tp_log'] = np.log1p(df['tp'])

# Create new feature: Temperature in Celsius (original t2m is in Kelvin)
df['t2m_celsius'] = df['2m_temperature'] - 273.15

# Drop the original precipitation and temperature columns (tp and 2m_temperature)
df = df.drop(columns=['tp', '2m_temperature'])

# Select features and target 
features = ['t2m_celsius'] 
target = 'tp_log'
X = df[features]
y = df[target]

print("\n--- Feature Engineering & Preprocessing ---")
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# --- 4. Train-Test Split & Scaling ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n--- Train-Test Split ---")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nData scaled successfully.")

# --- 5. ML Algorithms (Regression) ---
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42, n_jobs=-1),
    'Lasso': Lasso(random_state=42),
    'Ridge': Ridge(random_state=42),
    'SVR': SVR(),
    'KNN Regressor': KNeighborsRegressor(),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
}

trained_models = {}
print("\n--- Training ML Algorithms ---")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

# --- 6. Hyperparameter Tuning (Example with RandomForestRegressor) ---
print("\n--- Hyperparameter Tuning: RandomForestRegressor ---")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
}
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='r2',
    n_jobs=-1
)
grid_search.fit(X_train_scaled, y_train)
best_rf_model = grid_search.best_estimator_
print(f"Best parameters for RandomForest: {grid_search.best_params_}")
trained_models['Tuned Random Forest'] = best_rf_model

# --- 7. Model Evaluation and Comparison ---
results = {}
print("\n--- Model Evaluation and Comparison ---")
for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    
    # Inverse transform the predictions for evaluation
    y_pred_original = np.expm1(y_pred)
    y_test_original = np.expm1(y_test)
    
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)

    # Store results
    results[name] = {'R-squared': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    print(f"\n--- {name} ---")
    print(f"R-squared: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
# --- 8. Final Comparative Analysis & Visualization ---
results_df = pd.DataFrame(results).T
print("\n--- Final Comparative Analysis ---")
print(results_df.sort_values(by='R-squared', ascending=False))

# Visualization of results 
results_df['R-squared'].sort_values().plot(kind='barh', figsize=(10, 7))
plt.title('Model R-squared Comparison (Test Data)')
plt.xlabel('R-squared Score')
plt.ylabel('Model')
plt.grid(axis='x')
plt.tight_layout()
plt.show()

# Visualize the best model's predictions vs actual values 
best_model_name = results_df['R-squared'].idxmax()
best_model = trained_models[best_model_name]
y_pred_best = np.expm1(best_model.predict(X_test_scaled))
y_test_original = np.expm1(y_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_best, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.title(f'Actual vs. Predicted Rainfall for {best_model_name}')
plt.xlabel('Actual Rainfall (m)')
plt.ylabel('Predicted Rainfall (m)')
plt.grid(True)
plt.show()
"""
import pandas as pd
import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. Raw Data (Loading and Combining the Two Files) ---

# Define the expected file names for the instantaneous (temperature) and accumulated (rainfall) data.
# NOTE: These names have been simplified. 
INSTANT_FILE = 'instant.nc'
ACCUM_FILE = 'accum.nc'

try:
    if not os.path.exists(INSTANT_FILE) or not os.path.exists(ACCUM_FILE):
        print("CRITICAL ERROR: Data files not found.")
        print(f"Please ensure you have extracted and renamed the files to: {INSTANT_FILE} and {ACCUM_FILE}")
        print("Both files must be in the same folder as the script.")
        exit()

    # Load Instantaneous data (t2m - Temperature)
    ds_instant = xr.open_dataset(INSTANT_FILE, engine='netcdf4')
    df_instant = ds_instant[['t2m']].to_dataframe().reset_index()

    # Load Accumulated data (tp - Total Precipitation)
    ds_accum = xr.open_dataset(ACCUM_FILE, engine='netcdf4')
    df_accum = ds_accum[['tp']].to_dataframe().reset_index()

    # Merge the two DataFrames on the shared coordinates
    df = pd.merge(df_instant, df_accum, on=['valid_time', 'latitude', 'longitude', 'expver', 'number'])

    # Drop rows where data is NaN (this handles any missing records after merging)
    df = df.dropna()
    
    print("Data loaded and combined successfully from both files.\n")
    print("Combined DataFrame head:")
    print(df.head())

except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# --- 2. Exploratory Data Analysis (EDA) ---

# Rename t2m (2 metre temperature) to 2m_temperature for better readability
df = df.rename(columns={'t2m': '2m_temperature'}) 

# Drop irrelevant columns for this simple regression model
df = df.drop(columns=['expver', 'number', 'valid_time', 'latitude', 'longitude']) 

print("\n--- EDA: Missing values per column ---")
# Null check after merging and initial cleanup
print(df.isnull().sum()) 

# Visualization of total precipitation (target variable) 
plt.figure(figsize=(10, 6))
sns.histplot(df['tp'], kde=True, bins=50)
plt.title('Distribution of Total Precipitation (tp)')
plt.xlabel('Total Precipitation (m)')
plt.ylabel('Frequency')
plt.show()

# --- 3. Preprocessing, Scaling & Feature Engineering ---

# Handle high-skew in target variable (tp) by applying a log transformation (np.log1p = log(1+x))
df['tp_log'] = np.log1p(df['tp'])

# Create new feature: Temperature in Celsius (original t2m is in Kelvin)
df['t2m_celsius'] = df['2m_temperature'] - 273.15

# Drop the original precipitation and temperature columns (tp and 2m_temperature)
df = df.drop(columns=['tp', '2m_temperature'])

# Select features and target 
features = ['t2m_celsius'] 
target = 'tp_log'
X = df[features]
y = df[target]

print("\n--- Feature Engineering & Preprocessing ---")
print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

# --- 4. Train-Test Split & Scaling ---

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\n--- Train-Test Split ---")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nData scaled successfully.")

# --- 5. ML Algorithms (Regression) ---
# to keep the training lean and fast.
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42, n_jobs=-1),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
}

trained_models = {}
print("\n--- Training ML Algorithms (Simplified Set) ---")
for name, model in models.items():
    print(f"Training {name}...")
    # This is the line that takes the most time
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model



# --- 7. Model Evaluation and Comparison ---
results = {}
print("\n--- Model Evaluation and Comparison ---")
for name, model in trained_models.items():
    y_pred = model.predict(X_test_scaled)
    
    # Inverse transform the predictions for evaluation
    y_pred_original = np.expm1(y_pred)
    y_test_original = np.expm1(y_test)
    
    mse = mean_squared_error(y_test_original, y_pred_original)
    mae = mean_absolute_error(y_test_original, y_pred_original)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_original, y_pred_original)

    # Store results
    results[name] = {'R-squared': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}
    print(f"\n--- {name} ---")
    print(f"R-squared: {r2:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
# --- 8. Final Comparative Analysis & Visualization ---
results_df = pd.DataFrame(results).T
print("\n--- Final Comparative Analysis ---")
# The script will now quickly print this table instead of getting stuck
print(results_df.sort_values(by='R-squared', ascending=False))

# Visualization of results 
results_df['R-squared'].sort_values().plot(kind='barh', figsize=(10, 7))
plt.title('Model R-squared Comparison (Test Data)')
plt.xlabel('R-squared Score')
plt.ylabel('Model')
plt.grid(axis='x')
plt.tight_layout()
plt.show()

# Visualize the best model's predictions vs actual values 
best_model_name = results_df['R-squared'].idxmax()
best_model = trained_models[best_model_name]
y_pred_best = np.expm1(best_model.predict(X_test_scaled))
y_test_original = np.expm1(y_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test_original, y_pred_best, alpha=0.5)
plt.plot([y_test_original.min(), y_test_original.max()], [y_test_original.min(), y_test_original.max()], 'r--', lw=2)
plt.title(f'Actual vs. Predicted Rainfall for {best_model_name}')
plt.xlabel('Actual Rainfall (m)')
plt.ylabel('Predicted Rainfall (m)')
plt.grid(True)
plt.show()
