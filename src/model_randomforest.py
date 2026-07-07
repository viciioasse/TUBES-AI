import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

FILE_PATH = "data/DATASET_PROCESSED.xlsx"
RAW_FILE = "data/DATASET.xlsx"  # file asli sebelum normalisasi

# Data hasil preprocessing
x_train = pd.read_excel(FILE_PATH, sheet_name='x_train')
x_test = pd.read_excel(FILE_PATH, sheet_name='x_test')
y_train = pd.read_excel(FILE_PATH, sheet_name='y_train')
y_test = pd.read_excel(FILE_PATH, sheet_name='y_test')

fitur = x_train.columns.tolist()
target = y_train.columns.tolist()

print("=== DATA LOADED ===")
print(f"Jumlah data training: {x_train.shape[0]} baris")
print(f"Jumlah data testing : {x_test.shape[0]} baris\n")

# --- Ambil data raw dari file asli ---
raw_data = pd.read_excel(RAW_FILE, sheet_name='DATASHEET')  # ganti sesuai sheet asli
y_raw = raw_data[target]  # ambil kolom target polutan

# --- Fit scaler dari data raw ---
scaler_y = MinMaxScaler()
scaler_y.fit(y_raw)

def hitung_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

# --- Training model ---
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

# --- Inverse transform ke skala asli ---
y_test_raw = scaler_y.inverse_transform(y_test)
y_pred_raw = scaler_y.inverse_transform(y_pred)

print("=== HASIL EVALUASI PER POLUTAN (DATA RAW) ===")
mape_list = []

for i, col in enumerate(target):
    mse_i = mean_squared_error(y_test_raw[:, i], y_pred_raw[:, i])
    rmse_i = np.sqrt(mse_i)
    mae_i = mean_absolute_error(y_test_raw[:, i], y_pred_raw[:, i])
    r2_i = r2_score(y_test_raw[:, i], y_pred_raw[:, i])
    mape_i = hitung_mape(y_test_raw[:, i], y_pred_raw[:, i])
    mape_list.append(mape_i)
    
    print(f"Polutan {col}:")
    print(f"  - MAE  : {mae_i:.4f}")
    print(f"  - RMSE : {rmse_i:.4f}")
    print(f"  - R2   : {r2_i:.4f}")
    print(f"  - MAPE : {mape_i:.2f}%\n")

# --- Feature Importance ---
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
fitur_sorted = [fitur[idx] for idx in indices]

plt.figure(figsize=(8, 5))
plt.title("Fitur Kendaraan Paling Berpengaruh Terhadap Kualitas Udara", fontsize=11, fontweight='bold')
plt.bar(range(len(importances)), importances[indices], color='teal', edgecolor='black')
plt.xticks(range(len(importances)), fitur_sorted, rotation=45, fontweight='bold')
plt.ylabel("Bobot Kepentingan (Importance)")
plt.xlabel("Jenis Kendaraan")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("data/grafik_feature_importance.png", dpi=300)
plt.show()

# --- Perbandingan Nilai Aktual vs Prediksi per Polutan ---
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
axes = axes.flatten()

jumlah_sampel = 50  # tampilkan 50 sampel pertama

for i, col in enumerate(target):
    ax = axes[i]
    ax.plot(y_test_raw[:, i][:jumlah_sampel], label='Nilai Aktual', marker='o', color='black', linewidth=1)
    ax.plot(y_pred_raw[:, i][:jumlah_sampel], label='Prediksi RF', marker='x', linestyle='--', color='red', linewidth=1)
    ax.set_title(f"Perbandingan Aktual vs Prediksi: {col}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Indeks Data Uji", fontsize=9)
    ax.set_ylabel("Nilai Polutan (µg/m³)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("data/grafik_perbandingan_semua_polutan.png", dpi=300)
plt.show()

# --- Diagram Error MAPE per Polutan ---
plt.figure(figsize=(8, 5))
plt.title("Tingkat Error (MAPE) Per Jenis Polutan Udara", fontsize=11, fontweight='bold')
plt.bar(target, mape_list, color='salmon', edgecolor='black')
plt.axhline(y=50, color='red', linestyle='--', label='Batas Toleransi Error Layak (50%)')
plt.ylabel("Persentase Error (%)")
plt.xlabel("Jenis Polutan Udara")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("data/grafik_error_mape.png", dpi=300)
plt.show()
