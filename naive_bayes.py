import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

file_path = r"C:\Users\LENOVO\Downloads\archive (4)\users_data.csv"

df = pd.read_csv(file_path)
print("✅ Dataset berhasil dimuat! Jumlah data:", len(df))

print("\nKolom yang ditemukan di dataset:")
print(df.columns.tolist())

id_col = None
credit_col = None

for col in df.columns:
    if 'id' in col.lower():
        id_col = col
    if 'credit' in col.lower() and 'score' in col.lower():
        credit_col = col

if credit_col is None:
    raise ValueError("❌ Tidak ditemukan kolom yang mengandung 'credit_score' di dataset kamu!")

if id_col is None:
    print("⚠️ Tidak ditemukan kolom ID. Menambahkan kolom ID otomatis.")
    df['id_auto'] = range(1, len(df)+1)
    id_col = 'id_auto'

df['loyal'] = df[credit_col].apply(lambda x: 'Loyal' if x > 700 else 'Tidak Loyal')

X = df.select_dtypes(include=['number']).drop(columns=[credit_col], errors='ignore')
y = df['loyal']

for col in df.select_dtypes(include=['object']).columns:
    if col not in ['loyal']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

nb = GaussianNB()
nb.fit(X_train, y_train)

y_pred = nb.predict(X_test)
print("\n==============================")
print("  MODEL NAIVE BAYES PREDIKSI LOYALITAS ")
print("==============================")
print("Akurasi Model:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("\nLaporan Klasifikasi:\n", classification_report(y_test, y_pred))

df['Probabilitas_Loyal'] = nb.predict_proba(X_scaled)[:, list(nb.classes_).index('Loyal')]

pelanggan_paling_loyal = df.sort_values(by='Probabilitas_Loyal', ascending=False)

top_10 = pelanggan_paling_loyal[[id_col, credit_col, 'Probabilitas_Loyal', 'loyal']].head(10)
print("\n==============================")
print("  10 PELANGGAN PALING LOYAL")
print("==============================")
print(top_10.to_string(index=False))

plt.figure(figsize=(6,6))
df['loyal'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    colors=['lightgreen', 'salmon'],
    labels=['Loyal', 'Tidak Loyal'],
    startangle=90,
    explode=(0.05, 0.05)
)
plt.title('Persentase Pelanggan Loyal dan Tidak Loyal', fontsize=14)
plt.ylabel('')
plt.show()

plt.figure(figsize=(10,6))
plt.bar(top_10[id_col].astype(str), top_10['Probabilitas_Loyal'], color='skyblue')
plt.title('Top 10 Pelanggan Paling Loyal (Berdasarkan Probabilitas)', fontsize=14)
plt.xlabel('ID Pelanggan')
plt.ylabel('Probabilitas Loyalitas')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
