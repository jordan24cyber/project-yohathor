# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import category_encoders as ce 
import graphviz 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree

# %matplotlib inline
warnings.filterwarnings('ignore')

# Loop 'os.walk' ini spesifik untuk Kaggle, mungkin tidak akan menemukan apa-apa
# di lokal Anda, tapi tidak error.
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# --- 1. PERBAIKAN PATH FILE ---
# Menambahkan 'r' di depan string untuk menangani backslash '\' di Windows
data = r"D:\File Kuliah\Semester 7\Machine Learning\archive (5)\earthquake_alert_balanced_dataset.csv"

# Membaca data
# Sebaiknya gunakan try-except untuk memastikan file ada
try:
    df = pd.read_csv(data, header=None)
    print("File CSV berhasil dimuat.")
except FileNotFoundError:
    print(f"ERROR: File tidak ditemukan di path: {data}")
    print("Silakan periksa kembali path file Anda.")
    # Jika file tidak ada, kita hentikan di sini
    # raise SystemExit("Berhenti karena file tidak ditemukan.")


# Exploratory data analysis
print("\nBentuk Data (Shape):")
print(df.shape)

print("\n5 Baris Pertama (Head):")
print(df.head())

# Rename column names
col_names = ['Magnitude', 'Depth', 'CDI', 'MMI', 'SIG', 'Alert']
df.columns = col_names

print(f"\nNama kolom telah diubah menjadi: {col_names}")

print("\nInfo Dataset:")
df.info()

print("\nDistribusi Nilai (Value Counts):")
for col in col_names:
    print(f"\n--- {col} ---")
    print(df[col].value_counts().head()) # .head() agar tidak terlalu panjang

print("\nDistribusi Kelas Target 'Alert':")
print(df['Alert'].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())


# Declare feature vector and target variable
X = df.drop(['Alert'], axis=1)
y = df['Alert']

# Split data into separate training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)

print(f"\nUkuran X_train: {X_train.shape}, Ukuran X_test: {X_test.shape}")

# Feature Engineering
print(f"\nTipe data X_train sebelum encoding:\n{X_train.dtypes}")

# Encode categorical variables
encoder = ce.OrdinalEncoder(cols=['Magnitude', 'Depth', 'CDI', 'MMI', 'SIG'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

print("\n5 Baris Pertama X_train Setelah Encoding:")
print(X_train.head())


# --- Decision Tree Classifier with criterion gini index ---
print("\n--- Model Gini Index ---")
clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
clf_gini.fit(X_train, y_train)

# Predict the Test set results
y_pred_gini = clf_gini.predict(X_test)

# Check accuracy score
print('Model accuracy score (Gini): {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))

# Compare the train-set and test-set accuracy
y_pred_train_gini = clf_gini.predict(X_train)
print('Training-set accuracy score (Gini): {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_gini)))

# Check for overfitting and underfitting
print('Training set score (Gini): {:.4f}'.format(clf_gini.score(X_train, y_train)))
print('Test set score (Gini): {:.4f}'.format(clf_gini.score(X_test, y_test)))

# Visualize Decision Tree (matplotlib)
print("\nMenampilkan plot matplotlib (Gini)...")
plt.figure(figsize=(15, 10)) # Ukuran disesuaikan agar lebih mudah dibaca
tree.plot_tree(clf_gini,
               feature_names=X_train.columns,
               class_names=np.unique(y_train).astype(str), # Ambil nama kelas unik
               filled=True,
               rounded=True)

# --- 2. PERBAIKAN PLOT MATPLOTLIB ---
plt.show() # Perintah ini wajib untuk menampilkan plot di VS Code

# Visualize decision-trees with graphviz
dot_data_gini = tree.export_graphviz(clf_gini, out_file=None,
                                 feature_names=X_train.columns,
                                 class_names=np.unique(y_train).astype(str),
                                 filled=True, rounded=True,
                                 special_characters=True)
graph_gini = graphviz.Source(dot_data_gini)

# --- 3. PERBAIKAN PLOT GRAPHVIZ ---
# 'graph' saja tidak akan tampil. Gunakan .render() untuk menyimpan file
# dan view=True untuk membukanya secara otomatis.
try:
    graph_gini.render("decision_tree_gini", view=True, format="pdf")
    print("Plot 'decision_tree_gini.pdf' berhasil dibuat dan dibuka.")
except Exception as e:
    print(f"Gagal merender Graphviz (Gini): {e}")
    print("Pastikan program Graphviz (dot.exe) sudah terinstal dan ada di PATH sistem Anda.")


# --- Decision Tree Classifier with criterion entropy ---
print("\n--- Model Entropy ---")
clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_en.fit(X_train, y_train)

# Predict the Test set results
y_pred_en = clf_en.predict(X_test)

# Check accuracy score
print('Model accuracy score (Entropy): {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))

# Compare the train-set and test-set accuracy
y_pred_train_en = clf_en.predict(X_train)
print('Training-set accuracy score (Entropy): {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))

# Check for overfitting and underfitting
print('Training set score (Entropy): {:.4f}'.format(clf_en.score(X_train, y_train)))
print('Test set score (Entropy): {:.4f}'.format(clf_en.score(X_test, y_test)))

# Visualize decision-trees (matplotlib)
print("\nMenampilkan plot matplotlib (Entropy)...")
plt.figure(figsize=(15, 10))
tree.plot_tree(clf_en,
               feature_names=X_train.columns,
               class_names=np.unique(y_train).astype(str),
               filled=True,
               rounded=True)

# --- 2. PERBAIKAN PLOT MATPLOTLIB ---
plt.show()

# Visualize decision-trees with graphviz
dot_data_en = tree.export_graphviz(clf_en, out_file=None,
                                   feature_names=X_train.columns,
                                   class_names=np.unique(y_train).astype(str),
                                   filled=True, rounded=True,
                                   special_characters=True)
graph_en = graphviz.Source(dot_data_en)

# --- 3. PERBAIKAN PLOT GRAPHVIZ ---
try:
    graph_en.render("decision_tree_entropy", view=True, format="pdf")
    print("Plot 'decision_tree_entropy.pdf' berhasil dibuat dan dibuka.")
except Exception as e:
    print(f"Gagal merender Graphviz (Entropy): {e}")
    print("Pastikan program Graphviz (dot.exe) sudah terinstal dan ada di PATH sistem Anda.")


# --- Confusion matrix ---
print("\n--- Confusion Matrix (Model Entropy) ---")
cm = confusion_matrix(y_test, y_pred_en)
print('Confusion matrix\n', cm)

# --- Classification Report ---
print("\n--- Classification Report (Model Entropy) ---")
print(classification_report(y_test, y_pred_en))

print("\n--- Eksekusi Selesai ---")

import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import category_encoders as ce 

# Mengabaikan peringatan agar output lebih bersih
warnings.filterwarnings('ignore')

# --- FIX 1: Fungsi dikembalikan ke sintaks yang benar ---
def train_and_evaluate_models(csv_file):
    """
    Fungsi ini sekarang SAMA DENGAN notebook .ipynb Anda:
    1. Memuat data (tanpa header)
    2. Menambahkan OrdinalEncoder
    3. Melatih model Gini dan Entropy
    4. Menghitung 4 skor akurasi
    """
    try:
        # 1. Muat Data (Sesuai notebook, header=None)
        df = pd.read_csv(csv_file, header=None)

        # 2. Rename kolom (Sesuai notebook)
        col_names = ['Magnitude', 'Depth', 'CDI', 'MMI', 'SIG', 'Alert']
        df.columns = col_names

        # 3. Tentukan fitur (X) dan target (y)
        X = df.drop(['Alert'], axis=1)
        y = df['Alert']
        feature_names = X.columns.tolist()

        # 4. TERAPKAN ORDINAL ENCODER (Sesuai notebook)
        print("--- Menerapkan Ordinal Encoder (sesuai model asli) ---")
        encoder = ce.OrdinalEncoder(cols=feature_names)
        X_encoded = encoder.fit_transform(X)

        # 5. Lakukan Train-Test Split (Sesuai notebook)
        # test_size=0.33, random_state=42
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.33, random_state=42
        )

        print("--- Melatih Model (sesuai model asli) ---")

        # 6. Latih Model Gini (max_depth=3, random_state=0)
        clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=0)
        clf_gini.fit(X_train, y_train)

        # 7. Latih Model Entropy (max_depth=3, random_state=0)
        clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
        clf_en.fit(X_train, y_train)

        # 8. Hitung 4 Skor Akurasi
        y_pred_gini_train = clf_gini.predict(X_train)
        y_pred_gini_test = clf_gini.predict(X_test)
        y_pred_en_train = clf_en.predict(X_train)
        y_pred_en_test = clf_en.predict(X_test)

        scores = {
            "gini_train": accuracy_score(y_train, y_pred_gini_train),
            "gini_test": accuracy_score(y_test, y_pred_gini_test),
            "entropy_train": accuracy_score(y_train, y_pred_en_train),
            "entropy_test": accuracy_score(y_test, y_pred_en_test)
        }

        print("--- Model berhasil dilatih dan dievaluasi ---")

        # Kembalikan model, encoder (penting!), nama fitur, dan skor
        return clf_en, encoder, feature_names, scores

    except FileNotFoundError:
        print(f"ERROR: File dataset '{csv_file}' tidak ditemukan.")
        print("Pastikan path file sudah benar.")
        return None, None, None, None
    except Exception as e:
        print(f"Terjadi error saat melatih model: {e}")
        return None, None, None, None

def get_manual_input(features):
    """
    Meminta input manual dari user.
    """
    print("\n--- Silakan Masukkan Data Gempa Baru ---")
    print("Harap masukkan nilai persis seperti kategori (misal: 5.5, 7.2, atau 45)")

    user_data = []
    for feature in features:
        # Ambil input sebagai string, karena ini adalah kategori
        value = input(f"Masukkan nilai untuk '{feature}': ")
        user_data.append(value)
        
    return user_data

def predict_and_report(model, encoder, feature_names, new_data, scores):
    """
    Menerima model, encoder, data baru, dan skor,
    lalu mencetak prediksi DAN laporan akurasi.
    """
    
    try:
        # 1. Ubah input manual menjadi DataFrame
        input_df = pd.DataFrame([new_data], columns=feature_names)
        
        # 2. Gunakan ENCODER untuk mentransformasi data input
        input_encoded = encoder.transform(input_df)

        # 3. Lakukan prediksi pada data yang sudah di-encode
        prediction = model.predict(input_encoded)
        result = prediction[0]

        print("\n" + "="*40)
        print("---           HASIL PREDIKSI           ---")
        print("="*40)
        print("Berdasarkan data berikut:")
        for i in range(len(feature_names)):
            print(f"   - {feature_names[i]}: {new_data[i]}")

        # Menampilkan hasil prediksi
        print(f"\n==> Prediksi Level Peringatan (Alert): {result.upper()} <==")

        print("\n" + "="*40)
        print("---     PERFORMA MODEL (dari Data Asli)    ---")
        print("="*40)
        print("Ini adalah skor akurasi dari model asli Anda:")

        # Menampilkan 4 skor akurasi
        print(f"   Criterion Gini (Training Set Score) : {scores['gini_train']:.4f}")
        print(f"   Criterion Gini (Test Set Score)     : {scores['gini_test']:.4f}")
        print(f"   Criterion Entropy (Training Set Score): {scores['entropy_train']:.4f}")
        print(f"   Criterion Entropy (Test Set Score)    : {scores['entropy_test']:.4f}")
        print("="*40)

    except Exception as e:
        print(f"Error saat melakukan prediksi: {e}")

# --- FUNGSI UTAMA UNTUK MENJALANKAN SKRIP ---
if __name__ == "__main__":

    # --- FIX 2: Path file Anda ditempatkan di sini ---
    # Tambahkan 'r' di depan untuk path Windows
    dataset_file = r"D:\File Kuliah\Semester 7\Machine Learning\archive (5)\earthquake_alert_balanced_dataset.csv"

    # 1. Latih model DAN dapatkan skor akurasinya
    trained_model, data_encoder, feature_list, accuracy_scores = train_and_evaluate_models(dataset_file)

    # 2. Jika model berhasil (tidak error)
    if trained_model and data_encoder and feature_list and accuracy_scores:

        # 3. Minta input data manual
        manual_data = get_manual_input(feature_list)

        # 4. Lakukan prediksi DAN tampilkan laporannya
        predict_and_report(trained_model, data_encoder, feature_list, manual_data, accuracy_scores)

    else:
        print("\nProgram tidak dapat melanjutkan karena model gagal dilatih.")