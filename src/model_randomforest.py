import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data hasil preprocessing
FILE_PATH = "data/DATASET_PROCESSED.xlsx"

x_train = pd.read_excel(FILE_PATH, sheet_name='x_train')
x_test = pd.read_excel(FILE_PATH, sheet_name='x_test')
y_train = pd.read_excel(FILE_PATH, sheet_name='y_train')
y_test = pd.read_excel(FILE_PATH, sheet_name='y_test')

fitur = x_train.columns.tolist()
target = y_train.columns.tolist()

print("=== DATA LOADED ===")
print(f"Jumlah data training: {x_train.shape[0]} baris")
print(f"Jumlah data testing: {x_test.shape[0]} baris")
print(f"Fitur input: {fitur}")
print(f"Target output: {target}")

# Training model Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(x_train, y_train)

# Prediksi pada data testing
y_pred = rf.predict(x_test)

# Evaluasi per polutan
print("\n=== HASIL EVALUASI ===")
for i, col in enumerate(target):
    mse_i = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    rmse_i = mse_i ** 0.5
    r2_i = r2_score(y_test.iloc[:, i], y_pred[:, i])
    print(f"{col} -> RMSE: {rmse_i:.4f}, R2: {r2_i:.4f}")

# Visualisasi Pengaruh Kendaraan terhadap Polutan
importance = rf.feature_importances_
plt.bar(fitur, importance)
plt.title("Pengaruh Kendaraan terhadap Polutan")
plt.xlabel("Kendaraan")
plt.ylabel("Tingkat Pengaruh")
plt.show()