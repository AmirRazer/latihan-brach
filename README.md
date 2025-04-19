# Laporan Proyek Machine Learning - Amir Mahmud

## Domain Proyek

Churn Prediction: Unlocking Retention Secrets merupakan upaya untuk mengidentifikasi pelanggan yang berpotensi berhenti menggunakan layanan (churn) berdasarkan perilaku dan karakteristik mereka. Dalam konteks industri keuangan dan perbankan, prediksi churn sangat penting karena kehilangan nasabah dapat berdampak besar terhadap stabilitas pendapatan dan pertumbuhan jangka panjang perusahaan.

Churn prediction membantu perusahaan dalam memahami indikator-indikator utama yang menyebabkan pelanggan berhenti, sehingga memungkinkan manajemen untuk merancang strategi retensi yang lebih tepat sasaran. Retensi pelanggan sangat krusial, karena penelitian menunjukkan bahwa mempertahankan 5% pelanggan eksisting dapat meningkatkan keuntungan sebesar 25% hingga 95%.

Dalam proyek ini, dilakukan pengembangan model prediksi churn berbasis machine learning menggunakan data pelanggan dari sektor perbankan, dengan pendekatan sebagai berikut:

- Menggunakan beberapa algoritma machine learning yaitu Logistic Regression, Support Vector Machine (SVM), Random Forest, serta Artificial Neural Network (ANN) untuk membandingkan performa masing-masing model.

- Mengatasi permasalahan class imbalance yang umum pada kasus churn prediction dengan menggunakan teknik SMOTE (Synthetic Minority Over-sampling Technique).
  
**Referensi**: Customer Churn Prediction Pada Streaming Musics Platform Menggunakan Ensemble Learning (https://openlibrarypublications.telkomuniversity.ac.id/index.php/engineering/article/view/25696) 

## Business Understanding

Dalam dunia perbankan dan layanan keuangan, mempertahankan pelanggan yang sudah ada jauh lebih hemat biaya dibandingkan mencari pelanggan baru. Namun, tidak semua pelanggan menunjukkan loyalitas tinggi; sebagian akan berhenti (churn) karena berbagai alasan, seperti ketidakpuasan layanan, kondisi keuangan pribadi, atau ketertarikan pada produk kompetitor.

Maka dari itu, memahami siapa saja pelanggan yang berisiko churn merupakan langkah penting dalam strategi bisnis yang berorientasi pada customer retention. Proyek ini bertujuan membangun model prediksi churn dengan memanfaatkan data historis pelanggan untuk membantu bank memfokuskan strategi retensi mereka secara lebih efisien.
### Problem Statements

Menjelaskan pernyataan masalah latar belakang:
- Pernyataan Masalah 1: Bagaimana mengidentifikasi pelanggan yang berpotensi melakukan churn berdasarkan data historis transaksi dan profil pelanggan?

- Pernyataan Masalah 2: Fitur-fitur apa saja yang paling berpengaruh terhadap keputusan pelanggan untuk berhenti menggunakan layanan?

- Pernyataan Masalah 3: Algoritma machine learning mana yang paling efektif untuk memprediksi churn pada data pelanggan bank yang bersifat imbalanced?

### Goals

Menjelaskan tujuan dari pernyataan masalah:
- Jawaban Masalah 1: Membangun model klasifikasi churn yang mampu mengklasifikasikan pelanggan berisiko churn dan tidak churn secara akurat.

- Jawaban Masalah 2: Melakukan eksplorasi fitur dan feature importance untuk mengidentifikasi faktor-faktor utama penyebab churn seperti skor kepuasan, saldo akun, keluhan pelanggan, dan tren aktivitas akun.

- Jawaban Masalah 3: Membandingkan performa beberapa algoritma seperti Logistic Regression, SVM, Random Forest, dan Artificial Neural Network (ANN) dalam menangani data churn yang imbalanced, serta mengukur performanya menggunakan metrik seperti precision, recall, F1-score

    ### Solution statements
    - Solusi 1: Menerapkan beberapa algoritma machine learning seperti Logistic Regression, SVM, Random Forest, dan ANN untuk membandingkan performa klasifikasi churn.

    - Solusi 2: Menggunakan teknik SMOTE (Synthetic Minority Over-sampling Technique) untuk menangani ketidakseimbangan kelas (imbalanced data) agar model lebih sensitif terhadap pelanggan yang churn.

    - Solusi 3: Melakukan evaluasi performa model dengan menggunakan metrik yang relevan seperti  F1-score, Precision, dan Recall agar hasil prediksi dapat digunakan dalam pengambilan keputusan nyata di lapangan

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah data sintetis pelanggan dari sektor perbankan, yang berisi informasi demografis, keuangan, aktivitas transaksi, dan tingkat kepuasan pelanggan. Dataset ini digunakan untuk membangun model prediksi pelanggan yang berpotensi churn (berhenti menggunakan layanan).

📌 **Sumber Data**:  
Dataset dapat diakses melalui tautan berikut:  
[https://www.kaggle.com/datasets/simronw/churn-prediction-unlocking-retention-secrets/data](https://www.kaggle.com/datasets/simronw/churn-prediction-unlocking-retention-secrets/data)

Dataset ini berisi **5.000 baris data** pelanggan dengan total **26 fitur**. Berikut adalah struktur dan deskripsi fitur-fitur yang terdapat dalam dataset:
  

### Variabel-variabel pada hurn-prediction-unlocking-retention-secrets dataset adalah sebagai berikut:
| No | Kolom                          | Deskripsi |
|----|--------------------------------|-----------|
| 1  | `Customer_ID`                  | ID unik setiap pelanggan |
| 2  | `Age`                          | Usia pelanggan |
| 3  | `Gender`                       | Jenis kelamin pelanggan |
| 4  | `Account_Type`                 | Jenis akun yang dimiliki pelanggan |
| 5  | `Account_Balance`              | Saldo rekening pelanggan |
| 6  | `Transaction_Date`             | Tanggal transaksi |
| 7  | `Transaction_Amount`           | Nominal transaksi |
| 8  | `Transaction_Type`             | Jenis transaksi (debit/kredit) |
| 9  | `Branch`                       | Cabang tempat pelanggan terdaftar |
| 10 | `Loan_Amount`                  | Jumlah pinjaman aktif |
| 11 | `Loan_Type`                    | Jenis pinjaman yang dimiliki |
| 12 | `Credit_Score`                 | Skor kredit pelanggan |
| 13 | `Is_Employed`                  | Status pekerjaan pelanggan |
| 14 | `Annual_Income`                | Pendapatan tahunan pelanggan |
| 15 | `Marital_Status`               | Status pernikahan |
| 16 | `Region`                       | Wilayah tempat tinggal pelanggan |
| 17 | `Account_Open_Date`            | Tanggal pembukaan akun |
| 18 | `Last_Transaction_Date`        | Tanggal transaksi terakhir |
| 19 | `Number_of_Transactions`       | Jumlah transaksi selama periode tertentu |
| 20 | `Account_Activity_Trend`       | Tren aktivitas akun |
| 21 | `Customer_Service_Interactions`| Jumlah interaksi dengan layanan pelanggan |
| 22 | `Recent_Complaints`            | Jumlah keluhan terbaru |
| 23 | `Change_in_Account_Balance`    | Perubahan saldo akun dalam periode tertentu |
| 24 | `Customer_Satisfaction_Score`  | Skor kepuasan pelanggan (skala 1-5) |
| 25 | `Churn_Label`                  | Label churn: 1 (churn), 0 (tidak churn) |
| 26 | `Churn_Timeframe`              | Estimasi waktu hingga churn terjadi |

---
### 📊 Profil Umum Pelanggan

- **Usia rata-rata**: 43,93 tahun  
  Rentang usia pelanggan berkisar antara 18 hingga 70 tahun.
- **Pendapatan tahunan rata-rata**: Rp 110.500.612  
  Terdapat variasi besar antar pelanggan (standar deviasi: ±52 juta).
- **Skor kredit rata-rata**: 574  
  Mengindikasikan tingkat risiko kredit sedang.

### 💰 Perilaku Keuangan Pelanggan

- **Saldo akun rata-rata**: Rp 50.378  
  Rentang saldo dari Rp 509 hingga Rp 99.994.
- **Jumlah transaksi rata-rata**: 10 transaksi  
  Maksimal mencapai 20 transaksi.
- **Jumlah pinjaman rata-rata**: Rp 7.661  
  Banyak pelanggan tidak memiliki pinjaman.
- **Perubahan saldo akun rata-rata**: Rp 46  
  Terdapat fluktuasi besar antar pelanggan (standar deviasi: ±4.999).

### 📞 Interaksi & Kepuasan Pelanggan

- **Rata-rata interaksi layanan pelanggan**: 2–3 kali  
- **Skor kepuasan rata-rata**: 3 dari 5
- **Keluhan terbaru**:  
  Rata-rata 1 keluhan, maksimum 2 keluhan.

### 🔁 Churn (Perpindahan Pelanggan)

- Sekitar **31,66% pelanggan** telah churn, ditandai dengan `Churn_Label = 1`.  
  Hal ini menunjukkan pentingnya upaya prediktif untuk mengurangi angka churn demi mempertahankan profitabilitas bisnis.
**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.
## EDA
![image](https://github.com/user-attachments/assets/de1a6893-0c48-4dff-b5ea-51492ba1d599)

- **Age**: Distribusi usia pelanggan cukup merata dari 18 hingga 70 tahun, tidak ada dominasi pada kelompok usia tertentu.
- **Annual_Income**: Penyebaran pendapatan tahunan relatif merata, banyak pelanggan berada di kisaran 50.000–150.000.
- **Account_Balance** dan **Number_of_Transactions**: Terdistribusi seimbang, menunjukkan aktivitas keuangan yang aktif dari pelanggan.


- **Loan_Amount**: Distribusinya sangat skewed ke kiri (left-skewed), mayoritas pelanggan tidak mengambil pinjaman. Produk pinjaman mungkin kurang diminati.
- **Transaction_Amount**: Terdistribusi cukup normal, menunjukkan variasi aktivitas transaksi yang aktif.
- **Change_in_Account_Balance**: Terlihat simetris di sekitar nol, menandakan adanya fluktuasi saldo baik positif maupun negatif secara seimbang.


- **Customer_Service_Interactions**: Cenderung menurun dari skor 0 hingga 5. Artinya, semakin sedikit pelanggan yang sering menghubungi layanan pelanggan.
- **Customer_Satisfaction_Score**: Didominasi oleh skor tinggi (4 dan 5). Mayoritas pelanggan merasa puas, namun ada sebagian yang tidak puas (skor 1–2).
- **Recent_Complaints**: Mayoritas pelanggan memiliki skor 0 atau 1, menandakan keluhan yang rendah dalam periode terbaru.


- **Churn_Label**: Sekitar 30% pelanggan mengalami churn, sisanya masih loyal.
- **Churn_Timeframe**: Banyak pelanggan churn dalam waktu singkat (≤ 1 bulan) setelah menunjukkan tanda churn. Ini menandakan pentingnya respons cepat dalam menjaga pelanggan.
![image](https://github.com/user-attachments/assets/346b6b84-3e7f-4f77-945b-2bd1d1521427)

- **Tidak Churn (Label 0):** 3.417 pelanggan
- **Churn (Label 1):** 1.583 pelanggan
- **Sekitar 31.7%** pelanggan mengalami churn dari total 5.000 pelanggan (`1583 / (1583 + 3417)`).
![image](https://github.com/user-attachments/assets/fc9fb43c-1516-4955-bd22-6a98ef9595a9)
![image](https://github.com/user-attachments/assets/9cca19b9-25a0-472d-824c-91ca07817d72)
![image](https://github.com/user-attachments/assets/b58ff20d-2c72-41c6-8ea2-0e4c35a5cbe7)
![image](https://github.com/user-attachments/assets/d5d1c4a6-294f-4cfa-abab-bdf295f24a2b)
![image](https://github.com/user-attachments/assets/c5c7fa22-d5d2-4897-8b17-24471e6907ef)
![image](https://github.com/user-attachments/assets/dd51d2e5-319f-4ec8-b376-263386f25015)
![image](https://github.com/user-attachments/assets/6aeca0d1-7a13-44ee-afa5-7c08cd1143dc)
![image](https://github.com/user-attachments/assets/a509c01a-d2b9-43cd-a0dd-677cf7352d96)
![image](https://github.com/user-attachments/assets/fc1b75e9-da11-4a7f-9079-7983f9a3a16f)
**Distribusi Gender terhadap Churn**

| Gender | Tidak Churn (0) | Churn (1) |
|--------|------------------|-----------|
| Female | 1.137            | 521       |
| Male   | 1.151            | 535       |
| Other  | 1.129            | 527       |

> Tidak ada perbedaan signifikan antar gender terhadap kecenderungan churn.

---

**Distribusi Account Type terhadap Churn**

| Account Type | Tidak Churn (0) | Churn (1) |
|--------------|------------------|-----------|
| Saving       | 1.141            | 517       |
| Checking     | 1.095            | 529       |
| Investment   | 1.181            | 537       |

> Nasabah dengan akun investasi sedikit lebih banyak yang churn.

---

**Distribusi Transaction Type terhadap Churn**

| Transaction Type | Tidak Churn (0) | Churn (1) |
|------------------|------------------|-----------|
| Payment          | 878              | 421       |
| Deposit          | 864              | 376       |
| Withdrawal       | 845              | 429       |
| Transfer         | 830              | 357       |

> Jenis transaksi tidak terlalu memengaruhi tingkat churn secara signifikan.

---

**Distribusi Status Verifikasi terhadap Churn**

| Verified (True/False) | Tidak Churn (0) | Churn (1) |
|------------------------|------------------|-----------|
| False                  | 1.669            | 788       |
| True                   | 1.748            | 795       |

> Tingkat churn antara yang terverifikasi dan tidak hampir seimbang.

---

**Distribusi Status Pernikahan terhadap Churn**

| Status Pernikahan | Tidak Churn (0) | Churn (1) |
|-------------------|------------------|-----------|
| Single            | 873              | 384       |
| Widowed           | 887              | 399       |
| Married           | 836              | 384       |
| Divorced          | 821              | 416       |

> Pelanggan yang bercerai (Divorced) sedikit lebih tinggi tingkat churn-nya.

---

**Distribusi Aktivitas Akun terhadap Churn**

| Account Activity | Tidak Churn (0) | Churn (1) |
|------------------|------------------|-----------|
| Increasing       | 1.157            | 572       |
| Stable           | 1.156            | 524       |
| Decreasing       | 1.104            | 487       |

> Akun dengan aktivitas menurun cenderung memiliki tingkat churn yang lebih tinggi.

---

**Distribusi Interaksi Layanan Pelanggan terhadap Churn**

| Interaksi CS | Tidak Churn (0) | Churn (1) |
|--------------|------------------|-----------|
| 0            | 580              | 245       |
| 1            | 575              | 262       |
| 2            | 527              | 258       |
| 3            | 562              | 279       |
| 4            | 590              | 290       |
| 5            | 583              | 289       |

> Frekuensi interaksi yang tinggi bisa menjadi indikasi adanya masalah atau ketidakpuasan pelanggan.

---

**Distribusi Keluhan Terbaru terhadap Churn**

| Keluhan Terakhir | Tidak Churn (0) | Churn (1) |
|------------------|------------------|-----------|
| 0                | 1.468            | 188       |
| 1                | 928              | 680       |
| 2                | 1.021            | 715       |

> Semakin banyak keluhan, semakin besar kemungkinan pelanggan akan churn.

---

**Distribusi Tipe Transaksi (Kode) terhadap Churn**

| Tipe Transaksi | Tidak Churn (0) | Churn (1) |
|----------------|------------------|-----------|
| 1              | 566              | 430       |
| 2              | 536              | 421       |
| 3              | 719              | 259       |
| 4              | 773              | 251       |
| 5              | 823              | 231       |

> Tipe transaksi 1 dan 2 menunjukkan kecenderungan churn yang lebih tinggi dibanding tipe lainnya.
![image](https://github.com/user-attachments/assets/fa415a59-7a2e-4498-9334-6659a95eec45)
- Grafik menunjukkan bahwa pelanggan yang churn masih aktif melakukan transaksi hingga mendekati akhir periode pengamatan (Maret 2025).

- Pola transaksi terlihat relatif merata, namun terdapat penurunan jumlah pelanggan churn yang melakukan transaksi di sekitar pertengahan Februari dan pertengahan Maret 2025.

- Puncak transaksi pelanggan churn terjadi di awal Januari dan pertengahan Januari, serta ada peningkatan kembali menjelang akhir Maret.
![image](https://github.com/user-attachments/assets/3e2ea8a2-21e2-4cd8-be3f-107d87a50c3c)
- Grafik menunjukkan distribusi jumlah pelanggan yang churn berdasarkan waktu berapa bulan yang lalu mereka churn (sumbu X mewakili bulan, dari 1 sampai 12 bulan yang lalu).

- Terlihat bahwa puncak churn terjadi pada bulan ke-8, di mana jumlah pelanggan yang churn mencapai lebih dari 160 orang.

- Selain bulan ke-8, bulan ke-6, 9, dan 11 juga menunjukkan frekuensi churn yang tinggi.

- Sebaliknya, jumlah pelanggan yang churn pada bulan ke-4 tercatat paling rendah dibanding bulan lainnya.
![image](https://github.com/user-attachments/assets/7fcf004b-99c6-44a1-885f-7d4be6646b0c)
- Grafik menunjukkan jumlah pelanggan yang churn di empat wilayah: East, South, North, dan West.

- Wilayah East mencatat churn pelanggan paling tinggi, dengan jumlah melebihi 410 pelanggan.

- Wilayah West memiliki jumlah churn terendah, meskipun selisihnya tidak terlalu signifikan dibandingkan wilayah lain.

- Secara umum, jumlah churn di keempat wilayah relatif merata, dengan perbedaan yang tidak terlalu mencolok.
## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.
### 1️⃣ Menghapus Kolom yang Tidak Relevan

Beberapa kolom dianggap tidak relevan karena tidak memiliki nilai prediktif, bersifat unik, atau berpotensi menyebabkan kebocoran data (*data leakage*).
```python
  columns_to_drop = [
      'Customer_ID',
      'Transaction_Date',
      'Last_Transaction_Date',
      'Account_Open_Date',
      'Churn_Timeframe',
      'Loan_Type',
      'Branch'
  ]
  
  df.drop(columns=columns_to_drop, inplace=True)
```

Alasan Penghapusan Kolom:

Customer_ID: Kolom ini hanya berisi identifikasi unik pelanggan dan tidak memberikan informasi terkait churn.
- Transaction_Date, Last_Transaction_Date, Account_Open_Date: Kolom-kolom ini tidak digunakan dalam konteks analisis dan peramalan churn berbasis fitur pelanggan.
- Churn_Timeframe: Memiliki potensi kebocoran data karena dapat mencerminkan label target itu sendiri (nilai churn).
- Loan_Type, Branch: Kolom ini terlalu spesifik atau tidak relevan dengan analisis churn.
### 2️⃣ Encoding Fitur Kategorikal
Fitur-fitur kategorikal dikonversi ke format numerik menggunakan LabelEncoder. Ini dilakukan untuk memungkinkan model machine learning bekerja dengan data numerik.
```python
from sklearn.preprocessing import LabelEncoder

categorical_cols = [
    'Gender',
    'Account_Type',
    'Transaction_Type',
    'Is_Employed',
    'Marital_Status',
    'Region',
    'Account_Activity_Trend'
]

label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Fitur '{col}' setelah Label Encoding:\n{df[col].head()}")

print("Label Encoders:", label_encoders) 
```
Alasan Encoding Kategorikal:

Model Machine Learning tidak bisa memproses data kategorikal secara langsung.
Label Encoding cocok digunakan pada fitur dengan sedikit kategori dan tidak memiliki hubungan ordinal yang kuat. Ini mengubah nilai-nilai kategori menjadi angka, sehingga model dapat memahaminya.

### 3️⃣ Memisahkan Fitur dan Target
Langkah ini memisahkan variabel independen (fitur) dan dependen (target).
```python

X = df.drop(columns='Churn_Label')
y = df['Churn_Label']

print("Shape fitur (X):", X.shape)
print("Shape target (y):", y.shape)
print("Contoh fitur (X):\n", X.head())
print("Contoh target (y):\n", y.head())
```
Alasan Pemisahan:
- X berisi fitur-fitur yang digunakan untuk prediksi churn (variabel independen).
- y adalah label target yang menunjukkan apakah pelanggan churn (1) atau tidak (0), yang akan diprediksi oleh model.

### 4️⃣ Membagi Data: Train dan Test

Dataset dibagi menjadi dua bagian: training dan testing, dengan stratifikasi agar distribusi label tetap seimbang.
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Shape X_train:", X_train.shape)
print("Shape X_test:", X_test.shape)
print("Shape y_train:", y_train.shape)
print("Shape y_test:", y_test.shape)
print("Proporsi kelas di y_train:\n", y_train.value_counts(normalize=True))
print("Proporsi kelas di y_test:\n", y_test.value_counts(normalize=True))
```

Alasan Pembagian:

- 80% data untuk pelatihan dan 20% untuk evaluasi. Ini membantu model belajar dari sebagian besar data dan menguji kinerjanya di data yang tidak terlihat sebelumnya.
- Stratifikasi memastikan bahwa proporsi label churn dan tidak churn di kedua subset (train dan test) tetap seimbang, sehingga model dilatih untuk mengenali kedua kelas dengan baik.

### 5️⃣ Scaling Fitur Numerik

Fitur numerik dinormalisasi menggunakan StandardScaler agar berada dalam skala yang sama. 
```python
from sklearn.preprocessing import StandardScaler

numeric_cols = [
    'Account_Balance',
    'Transaction_Amount',
    'Loan_Amount',
    'Annual_Income',
    'Change_in_Account_Balance',
    'Credit_Score'
]

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

print("Contoh fitur numerik X_train setelah scaling:\n", X_train_scaled[numeric_cols].head())
print("Contoh fitur numerik X_test setelah scaling:\n", X_test_scaled[numeric_cols].head())
print("Scaler yang digunakan:\n", scaler)
```

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.
### 1. Logistic Regression
- **Parameter**: `class_weight='balanced'`  
- **Kelebihan**:
  - Sederhana dan cepat dilatih
  - Mudah diinterpretasikan
  - Dapat menangani data tidak seimbang dengan class_weight
- **Kekurangan**:
  - Hanya mampu menangkap hubungan linier antar fitur dan target
  - Kurang fleksibel untuk data yang kompleks dan non-linear

### 2. Support Vector Machine (SVM)
- **Kernel**: linear  
- **Kelebihan**:
  - Cocok untuk data berdimensi tinggi
  - Memiliki margin maksimal antara kelas
- **Kekurangan**:
  - Sensitif terhadap outlier
  - Training time lebih lama untuk data besar
  - Perlu normalisasi data

### 3. Random Forest Classifier
- **Parameter**: default  
- **Kelebihan**:
  - Mampu menangkap hubungan non-linear
  - Robust terhadap outlier dan overfitting
  - Memberikan informasi feature importance
- **Kekurangan**:
  - Kurang transparan dibanding logistic regression
  - Dapat mengalami bias terhadap kelas mayoritas jika data tidak seimbang

> 📌 Meskipun sempat dicoba dengan SMOTE, hasil menunjukkan bahwa model lebih optimal saat **dilatih pada data asli**, sehingga pendekatan akhir tidak menggunakan SMOTE untuk menghindari overfitting dan data leakage.



## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.
### ✅ Metrik Evaluasi yang Digunakan:
### ✅ Metrik Evaluasi yang Digunakan

1. **Accuracy**  
   Rumus: `(TP + TN) / (TP + TN + FP + FN)`  
   Mengukur proporsi prediksi yang benar dari keseluruhan data.

2. **Precision**  
   Rumus: `TP / (TP + FP)`  
   Mengukur seberapa akurat prediksi positif yang dibuat model.

3. **Recall (Sensitivity)**  
   Rumus: `TP / (TP + FN)`  
   Mengukur kemampuan model dalam menemukan seluruh kasus positif.

4. **F1-Score**  
   Rumus: `2 * (Precision * Recall) / (Precision + Recall)`  
   Kombinasi harmonik dari precision dan recall, penting ketika data tidak seimbang.

---

### 📈 Hasil Evaluasi Model (Berdasarkan 1.000 Data Uji)

#### 1. Logistic Regression
- **Kelas 0 (tidak churn)**:
  - Precision: 0.80
  - Recall: 0.65
- **Kelas 1 (churn)**:
  - Precision: 0.46
  - Recall: 0.64
- **Akurasi total**: 65%
- **Analisis**:  
  Model ini cukup baik dalam mengenali pelanggan churn (recall 0.64), tetapi precision rendah menunjukkan banyak false positive. Cocok jika ingin menangkap sebanyak mungkin pelanggan yang berisiko churn.

---

#### 2. SVM Classifier
- **Kelas 0**:
  - Precision: 0.79
  - Recall: 0.63
- **Kelas 1**:
  - Precision: 0.44
  - Recall: 0.64
- **Akurasi total**: 63%
- **Analisis**:  
  Performanya mirip Logistic Regression namun akurasinya sedikit lebih rendah. Precision dan F1-score untuk kelas churn masih rendah, menunjukkan tantangan dalam memprediksi pelanggan churn secara akurat.

---

#### 3. Random Forest Classifier
- **Kelas 0**:
  - Precision: 0.70
  - Recall: 0.91
- **Kelas 1**:
  - Precision: 0.47
  - Recall: 0.17
- **Akurasi total**: 68%
- **Analisis**:  
  Meskipun akurasi keseluruhan paling tinggi, model ini **sangat bias terhadap kelas mayoritas** (tidak churn). Recall kelas 1 yang rendah (0.17) menunjukkan model ini gagal mendeteksi sebagian besar pelanggan yang churn.

### 🏆 Model Terbaik

Berdasarkan metrik evaluasi dan tujuan proyek untuk **mendeteksi pelanggan yang akan churn**, **Logistic Regression** dipilih sebagai model terbaik karena:
- Memiliki **recall yang cukup tinggi (0.64)** untuk kelas churn, penting untuk mencegah kehilangan pelanggan.
- Lebih seimbang antara prediksi kelas 0 dan 1.
- Lebih transparan dan mudah dipahami untuk deployment di lingkungan bisnis.

**---Ini adalah bagian akhir laporan---**
