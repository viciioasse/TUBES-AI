import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

FILE_PATH = "data/DATASET.xlsx"

# Load data
df = pd.read_excel(FILE_PATH)
print("=== DATA AWAL ===")
print(df.shape)
print(df.head())

# Data cleaning
print("\n=== DATA CLEANING ===")
print("Missing values per kolom:")
print(df.isnull().sum())

# Drop kolom kosong & duplikat, isi nilai kosong dengan median
df.dropna(how='all', inplace=True)
df.fillna(df.median(numeric_only=True), inplace=True)
df.drop_duplicates(inplace=True)

# Function winsorize (clipping outliner)
def winsorize(dataframe, kolom):
    Q1 = dataframe[kolom].quantile(0.25)
    Q3 = dataframe[kolom].quantile(0.75)
    IQR = Q3 - Q1
    batas_bawah = Q1 - 1.5 * IQR
    batas_atas  = Q3 + 1.5 * IQR
    dataframe[kolom] = np.clip(dataframe[kolom], batas_bawah, batas_atas)
    return dataframe

# Menerapkan winsorize pada kolom kendaraan
kolom_outlier = ['SEPEDA MOTOR', 'MOBIL PRIBADI', 'BIS/TRUCK BARANG']
for kolom in kolom_outlier:
    df = winsorize(df, kolom)

print(f"\nJumlah data setelah cleaning: {df.shape[0]} baris")

# Pisahkan fitur dan target
fitur = ['SEPEDA MOTOR', 'MOBIL PRIBADI', 'BIS/TRUCK BARANG']
target = ['CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2']

x = df[fitur].copy()
y = df[target].copy()

print("\n=== FITUR INPUT (X) ===")
print(x.describe())

print("\n=== TARGET (Y) ===")
print(y.describe())

# Normalisasi data
print("\n=== NORMALISASI DATA (Min-Max Scaling) ===")
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

x_scaled = pd.DataFrame(x_scaled, columns=fitur)
y_scaled = pd.DataFrame(y_scaled, columns=target)

print("\n=== FITUR INPUT (X) Setelah Normalisasi ===")
print(x_scaled.describe())

print("\n=== TARGET (Y) Setelah Normalisasi ===")
print(y_scaled.describe())

# Split data untuk training dan testing
print("\n=== DATA SPLITTING (80:20) ===")
x_train , x_test , y_train , y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

print(f"Jumlah data training: {x_train.shape[0]} baris")
print(f"Jumlah data testing : {x_test.shape[0]} baris")

# Menyimpan data yang sudah diproses ke file baru
PROCESSED_FILE_PATH = "data/DATASET_PROCESSED.xlsx"
with pd.ExcelWriter(PROCESSED_FILE_PATH, engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Data_Cleaned', index=False)
    x_train.to_excel(writer, sheet_name='x_train', index=False)
    y_train.to_excel(writer, sheet_name='y_train', index=False)
    x_test.to_excel(writer, sheet_name='x_test', index=False)
    y_test.to_excel(writer, sheet_name='y_test', index=False)

print(f"\nData yang sudah diproses berhasil disimpan ke '{PROCESSED_FILE_PATH}'")
print("Sheet yang disimpan: Data_Cleaned, x_train, y_train, x_test, y_test")

#Kesimpulan
print("\n=== KESIMPULAN ===")
print(f"Total data bersih      : {df.shape[0]} baris")
print(f"Jumlah fitur input     : {len(fitur)} (kendaraan)")
print(f"Jumlah target output   : {len(target)} (polutan udara)")
print(f"Data training          : {x_train.shape[0]} baris (80%)")
print(f"Data testing           : {x_test.shape[0]} baris (20%)")
print(f"Metode normalisasi      : Min-Max Scaling (0-1)")
print(f"File hasil preprocessing: {PROCESSED_FILE_PATH}")