import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
from CRCEncoder import CRCTargetEncoder
import seaborn as sns

# ── Stage 1: Data Description ─────────────────────────
df_raw = pd.read_csv("Car details v3.csv")
print("Shape:", df_raw.shape)
print("\nColumn types:\n", df_raw.dtypes)
print("\nSummary statistics:\n", df_raw.describe())
print("\nMissing values:\n", df_raw.isnull().sum())

# ── Load ──────────────────────────────────────────────
df = pd.read_csv("Car details v3.csv")

# ── Feature Engineering ───────────────────────────────
df['brand'] = df['name'].str.split().str[0]
df = df.drop(columns=['name', 'torque'])

# ── Clean numeric columns ─────────────────────────────
df['mileage'] = pd.to_numeric(df['mileage'].str.replace('kmpl', '').str.replace('km/kg', '').str.strip(), errors='coerce')
df['engine'] = pd.to_numeric(df['engine'].str.replace('CC', '').str.strip(), errors='coerce')
df['max_power'] = pd.to_numeric(df['max_power'].str.replace('bhp', '').str.strip(), errors='coerce')

df = df.dropna()

# ── Feature Distribution ──────────────────────────────

numeric_cols = ['year', 'km_driven', 'selling_price', 'mileage', 'engine', 'max_power', 'seats']

print("=== SKEWNESS ===")
print(df[numeric_cols].skew().sort_values(ascending=False))

df[numeric_cols].hist(bins=30, figsize=(14, 10))
plt.tight_layout()
plt.savefig("distributions.png")
plt.show()

# ── Log Transformation ────────────────────────────────
df['km_driven'] = np.sqrt(df['km_driven'])
df['selling_price'] = np.log1p(df['selling_price'])
df['max_power'] = np.log1p(df['max_power'])

print("=== SKEWNESS AFTER TRANSFORMATION ===")
print(df[numeric_cols].skew().sort_values(ascending=False))

# ── Univariate Outlier Removal (IQR) ─────────────────
numeric_cols_to_check = ['km_driven', 'selling_price', 'max_power', 'engine', 'mileage', 'seats']

print(f"Shape before outlier removal: {df.shape}")

for col in numeric_cols_to_check:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

print(f"Shape after outlier removal: {df.shape}")


# ── Multivariate Outlier Removal (Mahalanobis) ────────
numeric_only = df.select_dtypes(include='number').dropna()

cov_matrix = np.cov(numeric_only.values.T)
inv_cov = np.linalg.pinv(cov_matrix)
mean = numeric_only.mean().values
distances = numeric_only.apply(
    lambda row: mahalanobis(row.values, mean, inv_cov), axis=1
)

threshold = chi2.ppf(0.95, df=numeric_only.shape[1])

print(f"Shape before multivariate outlier removal: {df.shape}")
df = df[distances <= threshold]
print(f"Shape after multivariate outlier removal: {df.shape}")

# ── Label Encoding ────────────────────────────────────
df_label = df.copy()
le = LabelEncoder()

df_label['fuel'] = le.fit_transform(df_label['fuel'])
df_label['seller_type'] = le.fit_transform(df_label['seller_type'])
df_label['transmission'] = le.fit_transform(df_label['transmission'])
df_label['brand'] = le.fit_transform(df_label['brand'])

owner_mapping = {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3,
                 'Fourth & Above Owner': 4, 'Test Drive Car': 0}
df_label['owner'] = df_label['owner'].str.strip().map(owner_mapping)

# ── Target Encoding ───────────────────────────────────
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']

X = df.drop(columns=['selling_price'])
y = df['selling_price']

encoder = CRCTargetEncoder(cols=categorical_cols)
X_encoded = encoder.fit_transform(X, y)

# ── Comparison ────────────────────────────────────────
print("=== ORIGINAL (before encoding) ===")
print(df[categorical_cols].head())

print("\n=== LABEL ENCODING ===")
print(df_label[categorical_cols].head())

print("\n=== CRC TARGET ENCODING ===")
print(X_encoded[categorical_cols].head())

# ── Correlation Analysis ──────────────────────────────
df_encoded = X_encoded.copy()
df_encoded['selling_price'] = y.values

pearson = df_encoded.corr(method='pearson')['selling_price'].sort_values(ascending=False)
spearman = df_encoded.corr(method='spearman')['selling_price'].sort_values(ascending=False)

print("=== PEARSON ===")
print(pearson)

print("\n=== SPEARMAN ===")
print(spearman)

mi = mutual_info_regression(df_encoded.drop(columns=['selling_price']), y)
mi_series = pd.Series(mi, index=df_encoded.drop(columns=['selling_price']).columns)
mi_series = mi_series.sort_values(ascending=False)

print("\n=== MUTUAL INFORMATION ===")
print(mi_series)

# 1. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('heatmap.png')
plt.show()

# 2. Scatter plots — numeric features vs selling_price
numeric_features = ['max_power', 'engine', 'km_driven', 'year', 'mileage']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# ── Visualizations: Feature Relationships ─────────────

# 1. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('heatmap.png')
plt.show()

# 2. Scatter plots — numeric features vs selling_price
numeric_features = ['max_power', 'engine', 'km_driven', 'year', 'mileage']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i, col in enumerate(numeric_features):
    axes[i].scatter(df_encoded[col], df_encoded['selling_price'], alpha=0.3)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('selling_price')
    axes[i].set_title(f'{col} vs selling_price')

plt.tight_layout()
plt.savefig('scatter_plots.png')
plt.show()

# 3. Box plots — categorical features vs selling_price
categorical_features = ['fuel', 'transmission', 'seller_type', 'owner']
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(categorical_features):
    df.boxplot(column='selling_price', by=col, ax=axes[i])
    axes[i].set_title(f'selling_price by {col}')
    axes[i].set_xlabel(col)

plt.tight_layout()
plt.savefig('box_plots.png')
plt.show()

from sklearn.preprocessing import StandardScaler

# ── Scaling ───────────────────────────────────────────
# Scale all numeric columns except selling_price (that's the target)
scaler = StandardScaler()

cols_to_scale = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']

X_encoded[cols_to_scale] = scaler.fit_transform(X_encoded[cols_to_scale])

print("=== AFTER SCALING ===")
print(X_encoded[cols_to_scale].describe().round(2))

from sklearn.model_selection import train_test_split

# ── Train/Test Split ──────────────────────────────────
# 80% training, 20% testing
# random_state=42 ensures reproducibility — same split every run
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# ── Stage 3: Model Training ───────────────────────────
# Random Forest chosen because:
# - handles mixed numeric/categorical features well
# - captures non-linear relationships (important for mileage/price)
# - robust to remaining outliers
rf = RandomForestRegressor(random_state=42)

# ── Grid Search ───────────────────────────────────────
# Try different combinations of hyperparameters to find the best model
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV R² score: {grid_search.best_score_:.4f}")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Stage 4: Evaluation ───────────────────────────────
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# Feature importance
feat_importance = pd.Series(
    best_model.feature_importances_,
    index=X_encoded.columns
).sort_values(ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(feat_importance)

feat_importance.plot(kind='bar', figsize=(12, 5), title='Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

import pickle

# ── Stage 5: Save Model ───────────────────────────────
with open('model.pkl', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)

print("Model, scaler and encoder saved.")