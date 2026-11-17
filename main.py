import pandas as pd
from sklearn.preprocessing import LabelEncoder
from CRCEncoder import CRCTargetEncoder

df = pd.read_csv("Car details v3.csv")
print(df.head())

print(df.columns.tolist())
print(df.shape)
print(df.dtypes)

df = df.drop(columns=['name','torque'])
print(df.shape)

df['mileage'] = df['mileage'].str.replace('kmpl', '').str.replace('km/kg', '').str.strip()
df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce')

df['engine'] = df['engine'].str.replace('CC', '').str.strip()
df['engine'] = pd.to_numeric(df['engine'], errors='coerce')

df['max_power'] = df['max_power'].str.replace('bhp', '').str.strip()
df['max_power'] = pd.to_numeric(df['max_power'], errors='coerce')

print(df[['mileage', 'engine', 'max_power']].dtypes)
print(df[['mileage', 'engine', 'max_power']].head())

print(df.isnull().sum())

df = df.dropna()
print(df.shape)


print(df['fuel'].unique())
print(df['seller_type'].unique())
print(df['transmission'].unique())
print(df['owner'].unique())

le = LabelEncoder()

df['fuel'] = le.fit_transform(df['fuel'])
df['seller_type'] = le.fit_transform(df['seller_type'])
df['transmission'] = le.fit_transform(df['transmission'])

owner_mapping = {
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4,
    'Test Drive Car': 0
}
df['owner'] = df['owner'].map(owner_mapping)


print(df[['fuel', 'seller_type', 'transmission', 'owner']].head())

print(df.dtypes)
print(df.head())

correlation = df.corr()['selling_price'].sort_values(ascending=False)
print(correlation)

df = df.drop(columns=['seats', 'mileage'])
print(df.shape)