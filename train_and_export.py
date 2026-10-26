import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor

# Load Data
df = pd.read_csv('concrete_dataset_200.csv', encoding='latin1')
df.columns = df.columns.str.strip()

# Conversion (Ratio -> kg/m3)
materials = ['Water', 'Cement', 'Silica_Fume', 'Fly_Ash', 'Sand', 'CA_10mm', 'CA_20mm', 'Admixture']
def ratio_to_kg(row):
    SG = {"Water": 1.0, "Cement": 3.15, "Silica_Fume": 2.2, "Fly_Ash": 2.2, "Sand": 2.65, "CA_10mm": 2.7, "CA_20mm": 2.7, "Admixture": 1.1}
    total_vol = 0
    for mat in materials:
        total_vol += row[mat] / (SG[mat] * 1000)
    factor = 1 / total_vol
    for mat in materials:
        row[mat] = row[mat] * factor
    return row

df = df.apply(ratio_to_kg, axis=1)

# Preprocessing
if 'Report_Ref_No' in df.columns:
    df = df.drop('Report_Ref_No', axis=1)
if 'Comp_Str_7d' in df.columns:
    df = df.drop('Comp_Str_7d', axis=1)

df['Grade_of_Concrete'] = df['Grade_of_Concrete'].str.replace('M', '').astype(float)
df['Cement_Grade'] = df['Cement_Grade'].fillna(0).astype(int)

# Create Label Encoders / mappings for Categorical Columns
cement_company_cats = df['Cement_Company'].astype('category')
cement_type_cats = df['Cement_Type'].astype('category')

df['Cement_Company'] = cement_company_cats.cat.codes
df['Cement_Type'] = cement_type_cats.cat.codes

company_mapping = dict(enumerate(cement_company_cats.cat.categories))
type_mapping = dict(enumerate(cement_type_cats.cat.categories))

# Reverse mappings for API (String -> Code)
company_to_code = {v: k for k, v in company_mapping.items()}
type_to_code = {v: k for k, v in type_mapping.items()}

cement_features = [col for col in df.columns if 'Cement_' in col]
X = df[materials + cement_features]
y = df['Comp_Str_28d']

# Train XGBoost
print("Training XGBoost model...")
xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, random_state=42)
xgb.fit(X, y)

# Save artifacts
print("Exporting model and metadata...")
joblib.dump(xgb, 'xgboost_model.pkl')
joblib.dump(company_to_code, 'company_to_code.pkl')
joblib.dump(type_to_code, 'type_to_code.pkl')
joblib.dump(list(X.columns), 'model_features.pkl')

print("Model and encoders exported successfully to .pkl files!")
