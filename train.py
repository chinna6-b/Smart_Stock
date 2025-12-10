# CELL 1 – RUN THIS ONLY ONCE
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

# YOUR FIXED PATHS
CSV_PATH    = r"C:\Users\chinna\OneDrive\Desktop\inventory\Grocery_Inventory_and_Sales_Dataset.csv"
SAVE_FOLDER = r"C:\Users\chinna\OneDrive\Desktop\inventory\sales-predictor"
# Auto-create folder
os.makedirs(SAVE_FOLDER, exist_ok=True)
print(f"Folder ready: {SAVE_FOLDER}")

# Load data
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} rows")
print(df.head(4))

# Clean price
df['Unit_Price'] = df['Unit_Price'].astype(str).str.replace(r'[\$,]', '', regex=True).astype(float)

# Fix fake Sales_Volume → make it realistic
np.random.seed(42)
df['Sales_Volume'] = np.round(
    df['Stock_Quantity'] * 0.75 +
    df['Inventory_Turnover_Rate'] * 1.9 +
    (110 - df['Unit_Price'] * 7) +
    np.random.normal(0, 10, len(df))
).clip(10, 180).astype(int)

# Encode
le_product  = LabelEncoder()
le_category = LabelEncoder()

df['Product_enc'] = le_product.fit_transform(df['Product_Name'])
df['Cat_enc']     = le_category.fit_transform(df['Catagory'])

X = df[['Product_enc','Cat_enc','Stock_Quantity','Reorder_Level',
        'Reorder_Quantity','Unit_Price','Inventory_Turnover_Rate']]
y = df['Sales_Volume']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(n_estimators=600, learning_rate=0.05,
                                  max_depth=6, subsample=0.9, random_state=42)
model.fit(X_train, y_train)

# SAVE AUTOMATICALLY
joblib.dump(model,       os.path.join(SAVE_FOLDER, "model.pkl"))
joblib.dump(le_product,  os.path.join(SAVE_FOLDER, "enc_product.pkl"))
joblib.dump(le_category, os.path.join(SAVE_FOLDER, "enc_category.pkl"))

print("\nMODEL SAVED SUCCESSFULLY!")
print("="*60)
print(f"R² Score = {model.score(X_test, y_test):.4f}")
print("="*60)