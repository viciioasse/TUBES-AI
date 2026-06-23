import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

FILE_PATH = "data/DATASET_PROCESSED.xlsx"

x_train = pd.read_excel(FILE_PATH, sheet_name='x_train')
x_test = pd.read_excel(FILE_PATH, sheet_name='x_test')
y_train = pd.read_excel(FILE_PATH, sheet_name='y_train')
y_test = pd.read_excel(FILE_PATH, sheet_name='y_test')

fitur = x_train.columns.tolist()
target = y_train.columns.tolist()

print("=== DATA LOADED ===")
print(f"Jumlah data training: {x_train.shape[0]} baris")
print(f"Jumlah data testing : {x_test.shape[0]} baris\n")

def hitung_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0 
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

rf = RandomForestRegressor(
    n_estimators=200, 
    max_depth=5, 
    min_samples_split=5, 
    random_state=42, 
    n_jobs=-1
)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

print("=== HASIL EVALUASI PER POLUTAN ===")
mape_list = []

for i, col in enumerate(target):
    mse_i = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
    rmse_i = np.sqrt(mse_i)
    mae_i = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    r2_i = r2_score(y_test.iloc[:, i], y_pred[:, i])
    mape_i = hitung_mape(y_test.iloc[:, i], y_pred[:, i])
    mape_list.append(mape_i)
    
    if mape_i < 10:
        status_mape = "Sangat Akurat (Sangat Bagus)"
    elif mape_i <= 20:
        status_mape = "Baik / Akurat"
    elif mape_i <= 50:
        status_mape = "Layak / Cukup"
    else:
        status_mape = "Buruk / Kurang Akurat"
        
    print(f"Polutan {col}:")
    print(f"  - MAE  : {mae_i:.4f}")
    print(f"  - RMSE : {rmse_i:.4f}")
    print(f"  - R2   : {r2_i:.4f}")
    print(f"  - MAPE : {mape_i:.2f}% -> Kategori: {status_mape}\n")

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
fitur_sorted = [fitur[idx] for idx in indices]

plt.figure(figsize=(8, 5))
plt.title("Fitur Kendaraan Paling Berpengaruh Terhadap Kualitas Udara", fontsize=11, fontweight='bold')
plt.bar(range(len(importances)), importances[indices], color='teal', align="center", edgecolor='black')
plt.xticks(range(len(importances)), fitur_sorted, fontweight='bold')
plt.ylabel("Bobot Kepentingan (Importance)")
plt.xlabel("Jenis Kendaraan")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("data/grafik_feature_importance.png", dpi=300)
plt.show()

plt.figure(figsize=(10, 4.5))
plt.plot(y_test['PM2.5'].values[:50], label='Nilai Asli Realitas (PM2.5)', marker='o', color='black', linewidth=1.5)
plt.plot(y_pred[:, target.index('PM2.5')][:50], label='Prediksi Random Forest', marker='x', linestyle='--', color='red', linewidth=1.5)
plt.title("Perbandingan Nilai Asli vs Prediksi Polutan PM2.5 (50 Sampel Pertama)", fontsize=11, fontweight='bold')
print("[INFO] Semua grafik visualisasi berhasil disimpan di dalam folder 'data/'")
plt.xlabel("Indeks Data Uji")
plt.ylabel("Nilai Polutan (Skala Normalisasi)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("data/grafik_perbandingan_pm25.png", dpi=300)
plt.show()

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

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12))
axes = axes.flatten() 

jumlah_sampel = 50 

for i, col in enumerate(target):
    ax = axes[i]
    ax.plot(y_test[col].values[:jumlah_sampel], label='Nilai Asli', marker='o', color='black', alpha=0.7, linewidth=1)
    ax.plot(y_pred[:, i][:jumlah_sampel], label='Prediksi RF', marker='x', linestyle='--', color='crimson', alpha=0.8, linewidth=1)
    
    ax.set_title(f"Perbandingan Realitas vs Prediksi: {col}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Indeks Data Uji", fontsize=9)
    ax.set_ylabel("Nilai Skala Normalisasi", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()

plt.savefig("data/grafik_perbandingan_semua_polutan.png", dpi=300)
print("[INFO] Grafik perbandingan semua polutan berhasil disimpan di folder 'data/grafik_perbandingan_semua_polutan.png'")
plt.show()
