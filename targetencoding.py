import pandas as pd
from CRCEncoder import CRCTargetEncoder


df = pd.read_csv("Car details v3.csv")
print(df['name'].head(10))

df['brand'] = df['name'].str.split().str[0]
print(df['brand'].value_counts().head(10))

df = df.drop(columns=['name'])
print(df.columns.tolist())

# Drop torque
df = df.drop(columns=['torque'])

# Clean mileage, engine, max_power
df['mileage'] = pd.to_numeric(df['mileage'].str.replace('kmpl', '').str.replace('km/kg', '').str.strip(), errors='coerce')
df['engine'] = pd.to_numeric(df['engine'].str.replace('CC', '').str.strip(), errors='coerce')
df['max_power'] = pd.to_numeric(df['max_power'].str.replace('bhp', '').str.strip(), errors='coerce')

# Drop missing values
df = df.dropna()

print(df.shape)
print(df.columns.tolist())

categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner', 'brand']

encoder = CRCTargetEncoder(cols=categorical_cols)
X = df.drop(columns=['selling_price'])
y = df['selling_price']


X_before = X.copy()
X_encoded = encoder.fit_transform(X, y)

print("=== BEFORE ===")
print(X_before[categorical_cols].head())

print("\n=== AFTER ===")
print(X_encoded[categorical_cols].head())