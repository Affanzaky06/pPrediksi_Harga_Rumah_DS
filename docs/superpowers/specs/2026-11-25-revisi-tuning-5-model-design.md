# Spesifikasi Desain: Revisi Hyperparameter Tuning 5 Kandidat Model & Sinkronisasi End-to-End (`prediksi.ipynb`, `main.py`, `src/`, `app.py`)

## 1. Latar Belakang & Tujuan (Intent & Why)

Berdasarkan hasil evaluasi presentasi Proyek Akhir Praktikum Data Science, terdapat tiga poin revisi utama pada tahap pemilihan model regresi serta satu kebutuhan sinkronisasi sistem:

1. **Hyperparameter Tuning untuk Seluruh Kandidat Model:** Semua 5 kandidat model regresi (`LinearRegression`, `Ridge`, `SVR`, `RandomForestRegressor`, `GradientBoostingRegressor`) harus di-*tuning* terlebih dahulu sampai ke tingkat hyperparameter menggunakan `GridSearchCV` + `GroupKFold(n_splits=5)` pada data latih (`X_train_final`), kemudian dibandingkan secara adil (*Before Tuning / Baseline Default* vs *After Tuning / Best Hyperparameters*).
2. **Kondisi `None` di Setiap Model:** Memasukkan nilai `None` ke dalam ruang pencarian (`param_grid`) pada hyperparameter yang menerima opsi `None` (kondisi tanpa batas / tanpa subsampling / iterasi otomatis bawaan), serta mendokumentasikan padanan kondisi `None` (tanpa pembatasan) pada model yang tidak memiliki argumen bertipe `None`.
3. **Alasan (Rasionalisasi) untuk Setiap Pilihan:** Menyediakan justifikasi teknis dan domain untuk:
   - Mengapa ke-5 kandidat algoritma tersebut dipilih.
   - Mengapa setiap hyperparameter dan rentang nilainya (termasuk `None`) dipilih di dalam `param_grid`.
   - Mengapa nilai hyperparameter terbaik (*winning hyperparameters*) terpilih pada masing-masing model.
   - Mengapa model terbaik keseluruhan (**Random Forest** hasil tuning) dipilih sebagai model final dibanding 4 kandidat lainnya.
4. **Sinkronisasi Penuh Tanpa Celah (`prediksi.ipynb` ↔ `src/house_price/` ↔ `main.py` ↔ `app.py` / `app_pages/` ↔ `README.md`):**
   - Artefak `model_artifacts/model_prediksi_rumah_jogja.joblib`, pipeline CLI `main.py`, kode paket `src/house_price/`, tampilan web `app.py` & `app_pages/`, serta `README.md` dan `tests/` wajib menggunakan model, hyperparameter, ukuran split, dan angka evaluasi yang **100% identik** dengan `prediksi.ipynb`.

---

## 2. Temuan Root Cause Inkonsistensi & Optimasi CPU

### 2.1 Root Cause Selisih Jumlah Baris Bersih (`1.383` vs `1.384`) pada Pandas 3.0
- Di `src/house_price/data/audit.py` (`flag_non_hunian`), kolom `deskripsi` sudah dikonversi ke `.astype(object)` sebelum pencocokan regex sehingga memakai mesin regex standar Python (`re`) dan menghasilkan **1.384 baris bersih** (`[2020, 2019, 1581, 1409, 1384]`).
- Di `prediksi.ipynb` Sel 33, kode `judul_asli = data["deskripsi"].str.lower()` belum memakai `.astype(object)`. Pada `pandas 3.0.5` (`.dsvenv`), kolom string memakai backend PyArrow (`re2`) yang mencocokkan `POLA_POTENSI` berbeda pada 1 baris (menghasilkan 1.383 baris bila dijalankan ulang).
- **Keputusan Desain:** Tambahkan `.astype(object)` pada Sel 33 `prediksi.ipynb` (`judul_asli = data["deskripsi"].astype(object).str.lower()`) agar output `prediksi.ipynb` selalu menghasilkan **1.384 baris bersih** yang identik dengan `src/house_price`.

### 2.2 Root Cause Selisih Ukuran Split Train/Test (`1.110 / 274` vs `1.107 / 277`)
- Pada environment aktif `.dsvenv` (`Python 3.11.7`, `scikit-learn 1.9.0`), `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` pada 1.384 baris bersih menghasilkan **1.107 data latih (80,0%) dan 277 data uji (20,0%)** dengan rincian data uji per wilayah: **Sleman 200, Bantul 44, Kota Yogyakarta 26, Kulon Progo/Gunung Kidul 7**.
- Modul `src/house_price/`, `app_pages/03_metodologi.py`, dan `tests/test_preprocessing_and_leakage.py` sudah menggunakan angka `1.107 / 277`, sedangkan teks narasi di `prediksi.ipynb` dan `README.md` masih mencantumkan angka lama `1.110 / 274`.
- **Keputusan Desain:** Seluruh sel `prediksi.ipynb` dijalankan ulang pada `.dsvenv` dan seluruh narasi markdown di `prediksi.ipynb` (Sel 70, 89, 94, 96, 110, 111) serta `README.md` diselaraskan ke **1.107 data latih dan 277 data uji**.

### 2.3 Root Cause Lonjakan Beban CPU (*CPU Thrashing*) & Solusinya
- `scikit-learn` berjalan murni di **CPU** (tidak menggunakan GPU).
- Ketika `RandomForestRegressor(n_jobs=-1)` dimasukkan ke dalam `GridSearchCV(n_jobs=-1)`, terjadi **nested parallelism** ($N \text{ proses CV} \times N \text{ thread pohon} = N^2$ worker bersaing di CPU) yang membuat laptop macet dan memicu ratusan `UserWarning: sklearn.utils.parallel.delayed`.
- **Keputusan Desain:**
  1. Saat melakukan `GridSearchCV`, base estimator `RandomForestRegressor` wajib disetel ke **`n_jobs=1`**, dan paralelisasi hanya dilakukan satu lapis di level `GridSearchCV(n_jobs=-1)` (atau `n_jobs=2` untuk menjaga responsivitas CPU). Setelah `best_params_` ditemukan, model final dilatih sekali dengan `n_jobs=-1` (lalu disetel `n_jobs=1` saat disimpan ke bundle `.joblib` untuk inferensi Streamlit yang bebas thread-overhead).
  2. Ruang pencarian (`param_grid`) dirancang fokus pada titik-titik hiperparameter yang bermakna secara teoritis (termasuk `None`) agar ke-5 model selesai di-*tuning* seluruhnya dalam waktu singkat tanpa mengorbankan kualitas solusi optimal.

---

## 3. Spesifikasi Ruang Pencarian (`param_grid`), Kondisi `None`, dan Rasionalisasi 5 Model

Seluruh model dievaluasi pada 6 fitur final (`X_train_final`: `log_luas_bangunan`, `log_luas_tanah`, `kamar_mandi`, `wilayah_Kota Yogyakarta`, `wilayah_Kulon Progo/Gunung Kidul`, `wilayah_Sleman`) dengan target `y_train = log10(harga)` menggunakan `GroupKFold(n_splits=5)` berbasis grup `train["deskripsi"].str.lower()` dan metrik utama `neg_root_mean_squared_error` (didampingi `neg_mean_absolute_error` dan `r2`).

### 3.1 `LinearRegression` (4 kombinasi × 5 fold = 20 fit)
- **`param_grid`:**
  - `fit_intercept`: `[True, False]` — Menguji apakah garis regresi membutuhkan konstanta dasar ($\beta_0$) ketika fitur numerik sudah distandarisasi (`mean=0`) dan wilayah menggunakan *one-hot encoding* dengan *baseline* Bantul.
  - `positive`: `[False, True]` — Menguji kondisi tanpa batasan tanda koefisien (`False`, ekuivalen kondisi `None`/unconstrained OLS) vs memaksa seluruh koefisien $\beta_i \ge 0$ (`True`). Потому bahwa *baseline* wilayah adalah Bantul sedangkan Kulon Progo/Gunung Kidul memiliki harga lebih rendah dari Bantul (koefisien negatif), pembatasan `positive=True` akan merusak prediksi wilayah tersebut.
- **Kondisi `None`:** Pada `LinearRegression` (OLS analitik), `copy_X=True` dan tidak ada parameter regularisasi/iterasi bertipe `None`; kondisi tanpa batasan (*unconstrained*) diwakili oleh `positive=False` dan `fit_intercept=True`.
- **Hasil Terbaik Terverifikasi:** `{'fit_intercept': True, 'positive': False}`
  - **CV RMSE (log10):** `0.1640` | **CV MAE (log10):** `0.1267` | **CV R²:** `0.7826` (sama dengan baseline default).

### 3.2 `Ridge` / Ridge Regression (20 kombinasi × 5 fold = 100 fit)
- **`param_grid`:**
  - `alpha`: `[0.01, 0.1, 1.0, 10.0, 100.0]` — Kekuatan penalti regularisasi L2 ($\lambda \sum \beta_j^2$) dalam skala logaritmik dari sangat lemah (`0.01`, mendekati OLS tanpa penalti) hingga sangat kuat (`100.0`).
  - `fit_intercept`: `[True, False]` — Menguji kebutuhan intersep dasar ($\beta_0$) untuk wilayah *baseline* (Bantul).
  - `max_iter`: **`[None, 1000]`** — **Kondisi `None`** membiarkan solver menentukan konvergensi analitik/otomatis tanpa batas pemotongan iterasi manual, dibandingkan batas eksplisit `1000` iterasi.
- **Hasil Terbaik Terverifikasi:** `{'alpha': 1.0, 'fit_intercept': True, 'max_iter': None}`
  - **CV RMSE (log10):** `0.1640` | **CV MAE (log10):** `0.1268` | **CV R²:** `0.7826`.
  - **Alasan:** Karena seleksi fitur (Bagian 8) sudah menekan multikolinearitas hingga seluruh $\text{VIF} < 5$, penalti L2 (`alpha=1.0`) memberikan hasil yang identik dengan OLS (`LinearRegression`).

### 3.3 `SVR` / Support Vector Regression (48 kombinasi × 5 fold = 240 fit)
- **`param_grid`:**
  - `kernel`: `['rbf', 'linear']` — Membandingkan pemetaan non-linear *Radial Basis Function* terhadap hyperplane linear pada ruang fitur terstandarisasi.
  - `C`: `[0.1, 1.0, 10.0]` — Parameter regularisasi penalti error di luar tabung $\epsilon$. Nilai kecil (`0.1`) menghasilkan margin sangat halus (*underfitting*), nilai besar (`10.0`) berisiko mengikuti pencilan harga (*overfitting*).
  - `epsilon`: `[0.01, 0.05, 0.1, 0.2]` — Lebar tabung toleransi tanpa penalti (*$\epsilon$-insensitive tube*) dalam satuan $\log_{10}(\text{harga})$.
  - `gamma`: `['scale', 'auto']` — Koefisien radius pengaruh kernel RBF (`'scale'` = $1 / (n_{\text{fitur}} \cdot \text{Var}(X))$ vs `'auto'` = $1 / n_{\text{fitur}}$).
- **Kondisi `None`:** Pada `SVR` (`LIBSVM`), parameter batas iterasi menggunakan `max_iter=-1` sebagai representasi kondisi tanpa batas (setara `None`), sedangkan `gamma` memakai `'scale'`/`'auto'`.
- **Hasil Terbaik Terverifikasi:** `{'C': 1.0, 'epsilon': 0.01, 'gamma': 'auto', 'kernel': 'rbf'}`
  - **CV RMSE (log10):** `0.1516` (membaik dari baseline default `0.1522`) | **CV MAE (log10):** `0.1123` | **CV R²:** `0.8138`.
  - **Alasan:** Kernel `rbf` menangkap kurva non-linear pada rumah berukuran ekstrem yang tidak tertangkap regresi linear, dan penyempitan tabung `epsilon` dari `0.1` ke `0.01` membuat vektor pendukung lebih presisi pada kepadatan data menengah.

### 3.4 `RandomForestRegressor` (72 kombinasi × 5 fold = 360 fit — atau 108 kombinasi bila `n_estimators: [100, 200, 300]`)
- **`param_grid`:**
  - `n_estimators`: `[100, 200, 300]` — Jumlah pohon keputusan dalam ensemble *bagging* untuk menstabilkan varians prediksi.
  - `max_depth`: **`[None, 10, 20]`** — **Kondisi `None`** mengizinkan pohon tumbuh penuh tanpa batas kedalaman kaku (berhenti hanya ketika terhalang `min_samples_split`/`min_samples_leaf`), dibandingkan batas eksplisit `10` dan `20`.
  - `max_features`: **`[None, 'sqrt', 0.5]`** — Jumlah fitur yang dipertimbangkan pada setiap *split* simpul. Untuk 6 fitur final:
    - **`None`** (setara `1.0`): menggunakan **seluruh 6 fitur** ($100\%$) pada tiap *split* (*bagging* penuh tanpa *feature subsampling*).
    - **`0.5`**: menggunakan $0.5 \times 6 =$ **3 fitur** acak per *split* ($50\%$).
    - **`'sqrt'`**: menggunakan $\lfloor\sqrt{6}\rfloor =$ **2 fitur** acak per *split* ($33,3\%$), yang mendekorrelasi pohon-pohon di dalam hutan sehingga tidak semuanya didominasi oleh `log_luas_bangunan` pada simpul akar.
  - `min_samples_split`: `[5, 10]` — Minimum sampel untuk memecah simpul internal (mencegah pemecahan pada kelompok data yang terlalu spesifik).
  - `min_samples_leaf`: `[2, 4]` — Minimum sampel di setiap daun (mencegah daun berisi 1 rumah tunggal / menghafal *outlier*).
  - `bootstrap`: `[True]` — Mengaktifkan *bootstrap aggregating* (sampel acak dengan pengembalian).
- **Hasil Terbaik Terverifikasi:** `{'bootstrap': True, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 300}`
  - **CV RMSE (log10):** `0.1430` (membaik signifikan dari baseline default `0.1512`) | **CV MAE (log10):** `0.1040` | **CV R²:** `0.8353`.
  - **Alasan Mengapa `max_depth=None` dan `max_features='sqrt'` Terpilih:** Ketika `min_samples_leaf=2` dan `min_samples_split=5` sudah menahan ukuran daun minimum, pembatasan kedalaman pohon (`max_depth=None`) tidak menyebabkan *overfitting* parah karena **`max_features='sqrt'` (2 fitur per split)** berhasil mendekorrelasi antar-pohon dan menurunkan varians ensemble secara drastis dibanding `max_features=None` (semua 6 fitur).

### 3.5 `GradientBoostingRegressor` (54 kombinasi × 5 fold = 270 fit)
- **`param_grid`:**
  - `n_estimators`: `[100, 200]` — Jumlah tahapan *boosting* berurutan.
  - `learning_rate`: `[0.03, 0.05, 0.1]` — Laju penyusutan (*shrinkage*) kontribusi tiap pohon terhadap residual.
  - `max_depth`: **`[None, 3, 5]`** — **Kondisi `None`** menguji pohon tanpa batas kedalaman vs pohon pembelajar lemah (*weak learner*) dangkal `3` dan `5`.
  - `max_features`: **`[None, 'sqrt', 0.5]`** — **Kondisi `None`** (semua 6 fitur) vs subsampling fitur `'sqrt'` (2 fitur) dan `0.5` (3 fitur).
  - `min_samples_split`: `[10]` (atau `[5, 10]`) — Minimum sampel pemecahan simpul.
  - `min_samples_leaf`: `[4]` (atau `[2, 4]`) — Minimum sampel daun untuk meredam sensitivitas *boosting* terhadap *outlier* harga.
- **Hasil Terbaik Terverifikasi:** `{'learning_rate': 0.03, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 100}`
  - **CV RMSE (log10):** `0.1456` (membaik signifikan dari baseline default `0.1523`) | **CV MAE (log10):** `0.1049` | **CV R²:** `0.8292`.
  - **Alasan:** Kombinasi *learning rate* kecil (`0.03`), daun yang lebih konservatif (`min_samples_leaf=4`, `min_samples_split=10`), dan `max_features='sqrt'` menjaga pohon dengan `max_depth=None` tetap terkendali dalam memperbaiki residual secara bertahap.

---

## 4. Tabel Perbandingan 5 Model (Baseline Default vs Setelah Hyperparameter Tuning)

| Model | Baseline CV RMSE (log) | Tuned CV RMSE (log) | $\Delta$ RMSE | Tuned CV MAE (log) | Tuned CV R² | Hyperparameter Terbaik (`best_params_`) |
|---|---:|---:|---:|---:|---:|---|
| **Random Forest** | 0,1512 | **0,1430** | **−0,0082** | **0,1040** | **0,8353** | `n_estimators=300, max_depth=None, max_features='sqrt', min_samples_split=5, min_samples_leaf=2, bootstrap=True` |
| **Gradient Boosting** | 0,1523 | 0,1456 | −0,0067 | 0,1049 | 0,8292 | `n_estimators=100, learning_rate=0.03, max_depth=None, max_features='sqrt', min_samples_split=10, min_samples_leaf=4` |
| **SVR (RBF)** | 0,1522 | 0,1516 | −0,0006 | 0,1123 | 0,8138 | `kernel='rbf', C=1.0, epsilon=0.01, gamma='auto'` |
| **Ridge Regression** | 0,1640 | 0,1640 | 0,0000 | 0,1268 | 0,7826 | `alpha=1.0, fit_intercept=True, max_iter=None` |
| **Linear Regression** | 0,1640 | 0,1640 | 0,0000 | 0,1267 | 0,7826 | `fit_intercept=True, positive=False` |

**Alasan Pemilihan Model Pemenang (`Random Forest`):**
1. **Akurasi Validasi Silang Terbaik:** Mencapai CV RMSE terendah (`0,1430` pada skala $\log_{10}$), CV MAE terendah (`0,1040`), dan CV R² tertinggi (`83,53%`), mengungguli Gradient Boosting (`0,1456`) dan SVR (`0,1516`).
2. **Dampak Tuning Paling Signifikan:** Mengalami penurunan error terbesar setelah tuning ($\Delta \text{RMSE} = -0,0082$) berkat dekorrelasi pohon melalui `max_features='sqrt'` dan penghalusan daun `min_samples_leaf=2`.
3. **Ketahanan terhadap Outlier & Interpretabilitas:** Sebagai metode *bagging* (rata-rata 300 pohon independen), Random Forest lebih tahan terhadap pencilan harga rumah mewah dibanding *boosting* yang berfokus pada residual ekstrem, serta menyediakan *feature importance* untuk menjawab pertanyaan bisnis #1.

---

## 5. Angka Evaluasi Baru yang Wajib Konsisten di Seluruh Berkas

Setelah `model_terbaik` (`RandomForestRegressor` hasil tuning) dilatih pada **1.107 data latih** dan diuji pada **277 data uji**:

### 5.1 Cek Overfitting & Learning Curve (Bagian 10.3)
- **Training MSE (log10):** `0,0098` (`0.009783`) | **Test MSE (log10):** `0,0162` (`0.016186`) | **Selisih:** `0,0064` (jauh lebih rapat dibanding model lama `0,0073` vs `0,0186`!).
- **Training R²:** `0,922` (`92,21%`) | **Test R²:** `0,871` (`87,14%`) | **Selisih:** `0,051` (`5,07%`, turun dari selisih lama `8,3%` — bukti kuat bahwa tuning dengan `max_features='sqrt'` berhasil menekan *overfitting*!).
- **Learning Curve (`GroupKFold` 5-fold):**
  - Training MSE: `[0.0155, 0.0117, 0.0107, 0.0096, 0.0099]` (mendatar di sekitar `0,010`).
  - Validation MSE: `[0.0280, 0.0234, 0.0224, 0.0211, 0.0207]` (turun dari `0,028` ke `0,0207`, jarak dengan error latih menyempit menjadi `±0,011`).

### 5.2 Metrik Global Data Uji (Bagian 11.1 & 12)
| Metrik | Hasil Baru | Hasil Lama (Sebagai Referensi Perubahan) | Target | Status |
|---|---|---|---|---|
| **R²** | **87,14%** | 85,77% | ≥ 75% | **Tercapai** |
| **MAE** | **Rp 354.905.653** (±Rp 354,9 juta) | Rp 422.640.860 | Sekecil mungkin | - |
| **RMSE** | **Rp 840.362.311** (±Rp 840,4 juta) | Rp 964.431.107 | Sekecil mungkin | - |
| **MdAPE** | **14,17%** | 15,57% | Pembanding MAPE | - |
| **MAPE** | **21,56%** | 23,23% | ≤ 20% | **Belum tercapai** (namun jauh lebih dekat ke 20%) |

### 5.3 Feature Importance (Bagian 11.3)
- `log_luas_bangunan`: **43,75%**
- `log_luas_tanah`: **29,48%**
- `kamar_mandi`: **24,14%**
- `wilayah_Kulon Progo/Gunung Kidul`: **1,19%**
- `wilayah_Kota Yogyakarta`: **0,74%**
- `wilayah_Sleman`: **0,70%**
*(Catatan menarik: dengan `max_features='sqrt'`, distribusi pentingnya fitur fisik menjadi lebih berimbang antara luas bangunan 43,8%, luas tanah 29,5%, dan kamar mandi 24,1%, dengan urutan peringkat yang tetap 100% konsisten dengan EDA dan drop-column CV!)*

### 5.4 Analisis Residual (Bagian 11.4)
- Pencilan *underprediction* terbesar (residual positif): rumah dengan harga aktual **Rp 16,00 miliar** yang diprediksi sekitar **Rp 6,84 miliar** (selisih **+Rp 9,16 miliar**).
- Pencilan *overprediction* terbesar (residual negatif): rumah dengan harga aktual **Rp 4,40 miliar** yang diprediksi sekitar **Rp 6,01 miliar** (selisih **−Rp 1,61 miliar**).

### 5.5 Evaluasi per Wilayah pada 277 Data Uji (Bagian 11.5 & 12)
| Wilayah | Sampel Uji | MdAPE | MAPE | MAE |
|---|---:|---:|---:|---:|
| **Kulon Progo/Gunung Kidul** | 7 | 10,89% | 7,27% | Rp 21.049.476 (±Rp 21,0 juta) |
| **Sleman** | 200 | 13,39% | 19,08% | Rp 348.371.269 (±Rp 348,4 juta) |
| **Kota Yogyakarta** | 26 | 19,55% | 32,57% | Rp 718.439.002 (±Rp 718,4 juta) |
| **Bantul** | 44 | 19,64% | 28,61% | Rp 222.905.622 (±Rp 222,9 juta) |

*(Temuan penting: di Sleman yang mencakup 72% data uji, MAPE sudah mencapai **19,08% (≤ 20%)** dan MdAPE **13,39%**! Di Kota Yogyakarta, MdAPE juga membaik drastis dari 26,16% menjadi **19,55%**, meskipun MAPE [32,57%] dan MAE [Rp 718,4 juta] tetap tertinggi karena pencilan harga rumah mewah di pusat kota).*
