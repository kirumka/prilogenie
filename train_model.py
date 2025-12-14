print("=== ADULT DATASET MODEL TRAINING ===")
import pandas as pd
import numpy as np

# 1. Load Adult dataset
print("1. Loading Adult dataset...")
df = pd.read_csv('data.adult.csv')
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")
print("   Columns:", df.columns.tolist())

# 2. Show sample
print("\n2. Sample data:")
print(df.head(3))

# 3. Check target variable
print("\n3. Target variable distribution:")
print(df['>50K,<=50K'].value_counts())

# 4. Basic preprocessing
print("\n4. Preprocessing...")
df_clean = df.replace('?', np.nan).dropna()
print(f"   After cleaning: {df_clean.shape}")

# 5. Prepare features (только числовые для простоты)
print("\n5. Preparing features...")
X = df_clean.select_dtypes(include=[np.number])
y = df_clean['>50K,<=50K'].apply(lambda x: 1 if '>50K' in str(x) else 0)
print(f"   X shape: {X.shape}, y shape: {y.shape}")
print("   Features:", X.columns.tolist())

# 6. Train model
print("\n6. Training GradientBoosting...")
from sklearn.ensemble import GradientBoostingClassifier
import joblib

model = GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42)
model.fit(X, y)
print(f"   Model trained. Accuracy: {model.score(X, y):.3f}")

# 7. Save
print("\n7. Saving...")
joblib.dump(model, 'model.pkl')
print("   Model saved as 'model.pkl'")
print("\n=== READY FOR STREAMLIT ===")
