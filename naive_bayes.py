import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Membaca file CSV
data = pd.read_csv("D:\File Kuliah\Semester 7\Machine Learning\cards_data.csv")

# Menampilkan semua kolom
pd.set_option('display.max_columns', None)

# Menampilkan 100 baris pertama
data_100 = data.head(100)

print("=== 100 Data Pertama dari Dataset ===")
print(data_100)

# Menampilkan informasi dasar dataset
print("\n=== Informasi Dataset ===")
print(data.info())

# Menampilkan statistik deskriptif
print("\n=== Statistik Deskriptif ===")
print(data.describe())

# Contoh: histogram untuk kolom numerik pertama
numerik_cols = data.select_dtypes(include=['int64', 'float64']).columns

if len(numerik_cols) > 0:
    plt.figure(figsize=(8,5))
    sns.histplot(data[numerik_cols[0]], bins=20, kde=True)
    plt.title(f'Distribusi {numerik_cols[0]}')
    plt.show()
else:
    print("Tidak ada kolom numerik untuk divisualisasikan.")