# Ringkasan Revisi Proyek Prediksi Harga Rumah Yogyakarta

Dokumen ini merangkum seluruh perubahan hasil revisi presentasi agar mudah dibaca dan dipresentasikan kembali oleh tim, meliputi: **perbandingan sebelum vs sesudah revisi**, **alasan pemilihan 5 model**, **isi `param_grid` beserta peran kondisi `None`**, **alasan setiap optimasi & pengurangan ruang pencarian/CPU**, **hasil terbaru**, serta **daftar berkas yang disinkronkan**.

---

## 1. Inti Revisi & Perbandingan Sebelum vs Sesudah (`Before` vs `After`)

| Aspek | Sebelum Revisi | Sesudah Revisi |
|---|---|---|
| **Skema Perbandingan Model** | Hanya `Random Forest` yang di-*tuning* lewat `GridSearchCV` (16 kombinasi); 4 kandidat lain dievaluasi pada pengaturan *default* saja. | **Seluruh 5 kandidat model di-*tuning* terlebih dahulu** dengan `GridSearchCV` + `GroupKFold(n_splits=5)` pada skala $\log_{10}(\text{harga})$, lalu dibandingkan sebelum vs sesudah tuning (*Baseline* vs *Tuned*). |
| **Pengujian Kondisi `None`** | Kondisi `None` tidak ada di `param_grid` dan belum dijelaskan di setiap kandidat model. | **`None` diuji secara eksplisit** pada parameter yang menerimanya (`Ridge.max_iter`, `RandomForest.max_depth`, `RandomForest.max_features`, `GradientBoosting.max_depth`, `GradientBoosting.max_features`) dan dipadankan opsi *unconstrained* pada `LinearRegression` (`positive=False`) dan `SVR` (`max_iter=-1`). |
| **Konsistensi Grid & Parameter Terbaik** | Notebook `prediksi.ipynb` (`n_estimators=200`, `max_depth=20`, tanpa `max_features`), `config.py` (`n_estimators=100`, `max_depth=15`, `max_features=0.5`), dan `train.py` (`DEFAULT_RF_PARAM_GRID` 432 kombinasi) berbeda-beda. | **Satu sumber kebenaran seragam** di notebook, `src/`, `main.py`, Streamlit, dan `.joblib`: pemenang **Random Forest** (`n_estimators=300`, `max_depth=None`, `max_features='sqrt'`, `min_samples_split=5`, `min_samples_leaf=2`, `bootstrap=True`). |
| **Efisiensi CPU & Waktu Tuning** | `RandomForestRegressor(n_jobs=-1)` di dalam `GridSearchCV(n_jobs=-1)` memicu *nested parallelism* (CPU berat/macet & ratusan warning): **80 fit** saja butuh **130,2 detik**. | Base estimator RF disetel **`n_jobs=1`** di dalam `GridSearchCV(n_jobs=-1)` (`learning_curve(n_jobs=-1)`, serta bundle `.joblib`). **360 kombinasi (1.800 fit CV, 22,5× lebih banyak)** selesai hanya dalam **53,9 detik** di `prediksi.ipynb` (**75,0 detik** di `main.py --grid-search`). |
| **Pembersihan & Ukuran Split** | `prediksi.ipynb` Cell 33 belum memakai `.astype(object)` sebelum regex (berisiko jadi `1.383` baris di `pandas 3.0.5`), dan narasi masih menyebut `1.110 / 274`. | Cell 33 memakai `.astype(object)` sehingga konsisten **1.384 baris bersih**; split pada environment aktif `.dsvenv` seragam di **1.107 latih / 277 uji** di seluruh proyek. |
| **Performa Generalisasi Model Terpilih** | Test $R^2$ **85,77%**, MAE **Rp 422,6 juta**, RMSE **Rp 964,4 juta**, MdAPE **15,57%**, MAPE **23,23%**, selisih $R^2$ latih–uji **0,083** (`0,941` vs `0,858`). | Test $R^2$ naik menjadi **87,14%**, MAE turun ke **Rp 354,9 juta**, RMSE turun ke **Rp 840,4 juta**, MdAPE membaik ke **14,17%**, MAPE turun ke **21,56%**, dan selisih $R^2$ latih–uji menyempit ke **0,051** (`0,922` vs `0,871`). |

---

## 2. Dataset, Pembersihan Data, & Fitur Final

Alur data pada `prediksi.ipynb` dan modul `src/house_price/` menghasilkan dataset terverifikasi berikut:

- **Total data mentah -> bersih:** `2.020` listing mentah -> `1.384` listing bersih (melalui tahapan `[2020, 2019, 1581, 1409, 1384]`).
- **Mengapa ditambahkan `.astype(object)` di Cell 33 `prediksi.ipynb`?** Pada `pandas 3.0.5`, kolom string memakai backend PyArrow (`re2`) yang menafsirkan pola regex `POLA_POTENSI` berbeda pada 1 baris sehingga menyusutkan data bersih menjadi `1.383` baris saat dijalankan ulang. Menambahkan `.astype(object)` menyamakan mesin regex ke Python `re` sebagaimana `src/house_price/data/audit.py` agar jumlah baris bersih selalu identik (**1.384 baris**).
- **Pembagian data (*Grouped-Stratified Split*):** `1.107` data latih (**80,0%**) dan `277` data uji (**20,0%**), dikelompokkan berdasarkan judul listing (`deskripsi.str.lower()`) sehingga terdapat **0 kebocoran grup lintas train-test** (dibandingkan **5 overlap** jika memakai *random split* biasa).
- **Kualitas fitur final (6 fitur pada `X_train_final`):**
  - `log_luas_bangunan`, `log_luas_tanah`, `kamar_mandi`, `wilayah_Kota Yogyakarta`, `wilayah_Sleman`, `wilayah_Kulon Progo/Gunung Kidul` (*baseline*: Bantul).
  - **Multikolinearitas aman ($\text{VIF} < 5$):** tertinggi pada `log_luas_bangunan` (`3,9912`) dan `kamar_mandi` (`3,0925`).
  - **Fitur yang dieliminasi:** `kamar_tidur` (redundan terhadap luas bangunan & kamar mandi), `carport` & `carport_kosong` (dampak *drop-column RMSE* mendekati nol), serta `area` (**57 area unik** di train dengan **24 area memiliki < 5 listing / 58 listing**, berisiko memicu *overfitting* dan koefisien tidak stabil).

---

## 3. Ruang Hyperparameter (`param_grid`), Peran `None`, & Alasan Setiap Pilihan

### A. Mengapa 5 Model Ini yang Dipilih?
Kelima model dipilih untuk mewakili keluarga algoritma yang berbeda:
1. **Linear Regression (OLS):** *Baseline* parametrik paling sederhana dan transparan.
2. **Ridge Regression (Regularisasi L2):** Menguji apakah penyusutan (*shrinkage*) koefisien L2 masih membantu meredam sisa korelasi fitur dimensi rumah.
3. **SVR (*Support Vector Regression*):** Menangkap kurva non-linear pada segmen harga ekstrem sekaligus tahan terhadap pencilan lewat *margin* $\epsilon$ pada fitur yang sudah di-*scale*.
4. **Random Forest (*Bagging Ensemble*):** Menangkap interaksi antarfitur dan hubungan non-linear dengan rata-rata banyak pohon independen yang stabil terhadap *outlier* harga rumah mewah, plus menyediakan *feature importance*.
5. **Gradient Boosting (*Sequential Boosting Ensemble*):** Pembanding sesama *tree-based ensemble* yang memperbaiki *error residual* secara bertahap.

### B. Isi `param_grid` dan Peran Kondisi `None` Tiap Model

| Model | Kombinasi (Fit 5-Fold CV) | Hyperparameter yang Diuji (`param_grid`) | Peran Kondisi `None` (atau Padanan *Unconstrained*) & Alasannya |
|---|---:|---|---|
| **Linear Regression** | 4 (20 fit) | `fit_intercept`: `[True, False]`<br>`positive`: `[False, True]` | OLS analitik tidak punya argumen `None`. Kondisi bebas tanpa batasan diwakili **`positive=False`** + `fit_intercept=True`. `positive=True` diuji sebagai pembanding, namun karena wilayah `Kulon Progo/Gunung Kidul` berharga di bawah *baseline* Bantul, koefisien wilayah perlu bernilai negatif (`positive=False`). |
| **Ridge Regression** | 20 (100 fit) | `alpha`: `[0.01, 0.1, 1.0, 10.0, 100.0]`<br>`fit_intercept`: `[True, False]`<br>`max_iter`: **`[None, 1000]`** | **`max_iter=None`** membiarkan solver menentukan konvergensi analitik/otomatis tanpa batas pemotongan iterasi manual, dibandingkan batas eksplisit `1000`. Rentang `alpha` menguji regularisasi L2 lintas 5 orde besaran. |
| **SVR** | 48 (240 fit) | `kernel`: `['rbf', 'linear']`<br>`C`: `[0.1, 1.0, 10.0]`<br>`epsilon`: `[0.01, 0.05, 0.1, 0.2]`<br>`gamma`: `['scale', 'auto']` | Pada `SVR` (`LIBSVM`), kondisi tanpa batas iterasi adalah nilai bawaan **`max_iter=-1`** (ekuivalen `None`). `kernel` menguji non-linear `rbf` vs `linear`; `C` mengatur penalti pelanggaran margin; `epsilon` mengatur lebar tabung toleransi error; `gamma` mengatur radius pengaruh RBF. |
| **Random Forest** | 72 (360 fit) | `n_estimators`: `[200, 300]`<br>`max_depth`: **`[None, 10, 20]`**<br>`max_features`: **`[None, 'sqrt', 0.5]`**<br>`min_samples_split`: `[5, 10]`<br>`min_samples_leaf`: `[2, 4]`<br>`bootstrap`: `[True]` | **`max_depth=None`** mengizinkan pohon tumbuh penuh sampai tertahan aturan `min_samples_split`/`min_samples_leaf`, dibandingkan batas kedalaman kaku (`10`, `20`). **`max_features=None`** memakai **100% fitur (6 fitur)** (*bagging* murni), diadu dengan `0.5` (**3 fitur**) dan `'sqrt'` (**2 fitur**) untuk melihat manfaat dekorrelasi pohon. |
| **Gradient Boosting** | 216 (1.080 fit) | `n_estimators`: `[100, 200]`<br>`learning_rate`: `[0.03, 0.05, 0.1]`<br>`max_depth`: **`[None, 3, 5]`**<br>`max_features`: **`[None, 'sqrt', 0.5]`**<br>`min_samples_split`: `[5, 10]`<br>`min_samples_leaf`: `[2, 4]` | **`max_depth=None`** menguji pohon penuh terhadap *weak learner* dangkal standar boosting (`3`, `5`). **`max_features=None`** menguji seluruh 6 fitur terhadap subsampling `'sqrt'` (2 fitur) dan `0.5` (3 fitur). `learning_rate` + `n_estimators` menguji laju koreksi residual vs risiko *overfitting*. |

### C. Alasan Setiap Optimasi, Pengurangan Grid/Nilai `n`, dan Arsitektur CPU

Bagian ini menjawab pertanyaan yang sangat mungkin ditanyakan penguji mengenai **mengapa beberapa rentang parameter diperkecil/diubah dan mengapa CPU sempat kewalahan**:

1. **Mengapa Grid `Random Forest` Dirampingkan dari `432` Kombinasi (`DEFAULT_RF_PARAM_GRID` lama) Menjadi `72` Kombinasi (Pengurangan 83,3% Fit)?**
   - **`n_estimators` dikurangi dari `[100, 200, 300]` menjadi `[200, 300]` (`100` dihapus):**
     Pada algoritma *bagging* Random Forest, menambah jumlah pohon menurunkan varians tanpa memicu *overfitting*. Jumlah $\ge 200$ pohon sudah terbukti jauh lebih stabil dibanding `100` pohon (dan hasil akhir memang memilih `300`). Menghapus `100` memangkas biaya komputasi grid RF sebesar **33%** tanpa mengorbankan kualitas kandidat terbaik.
   - **`max_depth` diubah dari `[10, 15, 20, 30]` menjadi `[None, 10, 20]` (`15` dan `30` dihapus, diganti `None`):**
     `None` wajib dimasukkan sesuai revisi agar pohon bebas tumbuh hingga tertahan aturan daun/simpul. Untuk `1.107` baris latih dengan hanya `6` fitur, nilai `10` (moderat) dan `20` (dalam) sudah mengapit `15` dengan baik, sedangkan `30` menjadi redundan karena `None` sudah mencakup kedalaman tak terbatas bila memang diperlukan.
   - **`max_features` diubah dari `[0.5, 0.7, 0.8, 1.0]` menjadi `[None, 'sqrt', 0.5]` (`0.7` dan `0.8` dihapus, `'sqrt'` ditambah, `1.0` diganti `None`):**
     Dengan **6 fitur final**, fraksi `0.7` ($\lfloor 0{,}7 \times 6 \rfloor = 4$ fitur) dan `0.8` ($\lfloor 0{,}8 \times 6 \rfloor = 4$ fitur) secara matematis memilih **jumlah fitur per split yang sama persis (4 fitur)** — sehingga menguji keduanya adalah komputasi yang mubazir! Sementara itu, `1.0` ekuivalen dengan **`None`** (**6 fitur / 100%**). Menghapus titik redundan tersebut lalu menggantinya dengan **`'sqrt'`** ($\lfloor\sqrt{6}\rfloor = 2$ fitur atau **33,3%**), `0.5` (**3 fitur / 50%**), dan `None` (**6 fitur / 100%**) menghasilkan 3 rezim dekorrelasi yang benar-benar berbeda — dan `'sqrt'` menjadi kunci kemenangan Random Forest.
   - **`min_samples_split` dari `[2, 5, 10]` menjadi `[5, 10]` dan `min_samples_leaf` dari `[1, 2, 4]` menjadi `[2, 4]` (`2` dan `1` dihapus):**
     Pada `1.107` data yang mengandung *outlier* harga rumah mewah, `min_samples_leaf=1` membiarkan daun berisi hanya **1 rumah tunggal** (*menghafal outlier*) dan `min_samples_split=2` memecah kelompok yang terlalu kecil. Menghapus nilai ekstrem `1` dan `2` menekan *overfitting* sekaligus memangkas jumlah kombinasi secara drastis.

2. **Alasan Ukuran Grid & Batasan Parameter pada 4 Model Lainnya:**
   - **Gradient Boosting (`216` kombinasi = `1.080` fit):** `n_estimators` dibatasi ke **`[100, 200]`** (tidak sampai `300`/`500` seperti metode *bagging*) karena pohon dibangun **sekuensial** (tidak bisa diparalelkan antar-pohon dalam 1 fit) dan *boosting* dengan pohon terlalu banyak + `learning_rate` tinggi mudah menghafal *outlier*. Terbukti pemenangnya justru **`n_estimators=100` dengan `learning_rate=0.03`**. Selain itu, `min_samples_split` `[5, 10]` dan `min_samples_leaf` `[2, 4]` disamakan dengan grid Random Forest (dari rancangan awal 54 menjadi 216 kombinasi) agar perbandingan kedua algoritma *ensemble* pohon berlangsung adil/simetris.
   - **SVR (`48` kombinasi = `240` fit):** Kernel difokuskan ke **`['rbf', 'linear']`** (tanpa `poly` dan `sigmoid`) karena polinomial berderajat tinggi lambat di `LIBSVM` serta rentan meledak saat ekstrapolasi harga ekstrem, sedangkan `sigmoid` jarang stabil untuk regresi kontinu. Rentang `C` `[0.1, 1.0, 10.0]` + `epsilon` `[0.01, 0.05, 0.1, 0.2]` + `gamma` `['scale', 'auto']` memetakan *margin* sempit–lebar tanpa nilai `C=100+` yang memperlambat konvergensi solver kuadratik secara drastis.
   - **Ridge (`20` kombinasi = `100` fit):** `alpha` memilih **5 titik orde besaran logaritmik** (`[0.01, 0.1, 1.0, 10.0, 100.0]`) alih-alih deret rapat karena skala logaritmik sudah memadai untuk mengamati efek penyusutan L2 dari hampir-OLS hingga penalti kuat.
   - **Linear Regression (`4` kombinasi = `20` fit):** OLS dihitung secara analitik sehingga cukup menguji `fit_intercept` dan `positive`.

3. **Optimasi Arsitektur CPU (`n_jobs=1` pada Base Estimator RF) — Mengapa Tidak Pakai GPU?**
   - **Mengapa tidak menggunakan GPU?** Seluruh algoritma yang dipakai berasal dari `scikit-learn` (`RandomForestRegressor`, `GradientBoostingRegressor`, `SVR`, `Ridge`, `LinearRegression`, `GridSearchCV`), yang secara arsitektural berjalan **murni di CPU** (Cython/OpenMP/`joblib`) dan memang tidak memiliki dukungan CUDA/GPU.
   - **Penyebab CPU ngap-ngapan (*Nested Parallelism*):** Sebelumnya `RandomForestRegressor(n_jobs=-1)` dipasang di dalam `GridSearchCV(n_jobs=-1)` maupun `learning_curve(n_jobs=-1)`. Akibatnya setiap *worker* CV mencoba memicu *thread* RF sebanyak jumlah core lagi ($N \times N = N^2$ *thread* aktif berebut CPU bersamaan), menimbulkan *thrashing* dan ratusan `UserWarning`. Itulah mengapa di notebook lama hanya **16 kombinasi (80 fit)** memakan waktu **130,2 detik**.
   - **Solusi optimasi:** Estimator dasar `RandomForestRegressor` wajib disetel **`n_jobs=1`** ketika masuk ke `GridSearchCV(n_jobs=-1)` maupun `learning_curve(n_jobs=-1)` (paralelisme bersih **1 lapis** di level CV saja), dan pada artefak `.joblib` final juga disetel `n_jobs=1` agar prediksi 1 baris input di aplikasi Streamlit bebas overhead *thread spawning*.
   - **Dampak terukur:** Meskipun total pencarian meningkat **22,5× lipat** (dari **80 fit** di notebook lama menjadi **1.800 fit CV** across 5 model), waktu tuning justru turun menjadi **53,9 detik** di `prediksi.ipynb` (**75,0 detik** di `main.py --grid-search`).

4. **Efisiensi Pipeline (`StandardScaler` di luar CV loop & pemisahan `--grid-search` di `main.py`):**
   - `StandardScaler` dan *one-hot encoding* di-fit **hanya dari `X_train`** sebelum tahap modelling (Bagian 7 / `preprocessing.py`). Karena `X_train_final` sudah bebas *data leakage*, loop `GridSearchCV` tidak perlu membungkus `Pipeline` yang akan mengulang *scaling* sebanyak `1.800` kali.
   - Di `main.py`, eksekusi standar (`python main.py`) memakai `benchmark_models` (10 CV cepat untuk memverifikasi baseline vs tuned) dan `BEST_RF_PARAMS` agar pipeline & pengujian berjalan dalam beberapa detik, sementara `python main.py --grid-search` disediakan untuk mereplikasi `tune_all_candidate_models` penuh (360 kombinasi).

---

## 4. Hasil Tuning 5 Model & Alasan Random Forest Menang

Seluruh model dievaluasi dengan **GroupKFold 5-fold** pada `X_train_final` (**1.107 baris**), diurutkan berdasarkan **Tuned CV RMSE ($\log_{10}$)** terendah:

| Peringkat | Model | Baseline CV RMSE | Tuned CV RMSE | $\Delta$ RMSE | Tuned CV MAE | Tuned CV R² | Hyperparameter Terbaik (`best_params_`) |
|---:|---|---:|---:|---:|---:|---:|---|
| **1** | **Random Forest** | 0,1512 | **0,1430** | **−0,0082** | **0,1040** | **0,8353** | `n_estimators=300, max_depth=None, max_features='sqrt', min_samples_split=5, min_samples_leaf=2, bootstrap=True` |
| 2 | Gradient Boosting | 0,1523 | 0,1456 | −0,0067 | 0,1049 | 0,8292 | `n_estimators=100, learning_rate=0.03, max_depth=None, max_features='sqrt', min_samples_split=10, min_samples_leaf=4` |
| 3 | SVR (RBF) | 0,1522 | 0,1516 | −0,0006 | 0,1123 | 0,8138 | `kernel='rbf', C=1.0, epsilon=0.01, gamma='auto'` |
| 4 | Ridge Regression | 0,1640 | 0,1640 | 0,0000 | 0,1268 | 0,7826 | `alpha=1.0, fit_intercept=True, max_iter=None` |
| 5 | Linear Regression | 0,1640 | 0,1640 | 0,0000 | 0,1267 | 0,7826 | `fit_intercept=True, positive=False` |

### Mengapa Parameter Ini yang Menang (Siap untuk Tanya-Jawab Presentasi)

- **Mengapa Random Forest terbaik secara keseluruhan?**
  1. Unggul pada **seluruh metrik CV** setelah semua kandidat model sama-sama di-*tuning* (`CV RMSE 0,1430`, `CV MAE 0,1040`, `CV R² 83,53%`).
  2. Mencatat penurunan error terbesar (`−0,0082`).
  3. Sebagai *bagging* dari 300 pohon independen, ia lebih tahan terhadap pencilan harga rumah mewah dibanding *boosting* sekuensial, sekaligus menghasilkan *feature importance* yang jelas.
- **Mengapa `max_depth=None` menang di Random Forest & Gradient Boosting?**
  Karena pohon tidak perlu dipotong kaku berdasarkan kedalaman ketika pembatas simpul & daun (`min_samples_split=5`, `min_samples_leaf=2` pada RF; `min_samples_split=10`, `min_samples_leaf=4` pada GB) sudah menjaga agar tiap daun mewakili beberapa rumah dan tidak menghafal 1 listing tunggal.
- **Mengapa `max_features='sqrt'` mengalahkan `None` pada Random Forest & Gradient Boosting?**
  Jika `max_features=None` (semua 6 fitur dipakai di tiap split), pohon cenderung didominasi fitur terkuat (`log_luas_bangunan`) di simpul akar sehingga pohon-pohon menjadi seragam dan berkorelasi tinggi. Membatasi kandidat ke `'sqrt'` (**2 fitur acak per split**) mendekorrelasi antar-pohon sehingga agregasi ensemble jauh lebih akurat.
- **Mengapa `alpha=1.0` & `max_iter=None` pada Ridge hasilnya identik dengan Linear Regression (`0,1640`)?**
  Ini membuktikan seleksi fitur sebelumnya (seluruh fitur final memiliki $\text{VIF} < 5$) sudah membersihkan multikolinearitas parah sehingga penalti L2 tidak lagi banyak mengubah error linier.

---

## 5. Evaluasi Akhir Model Terpilih di Data Uji (`277` Rumah)

Model **Random Forest** terbaik dilatih ulang pada seluruh **1.107 data latih** lalu diuji pada **277 data uji** yang terisolasi (*hold-out*):

### Metrik Keseluruhan & Cek Overfitting (Before vs After)
| Metrik | Sebelum Revisi (274 uji) | Sesudah Revisi (277 uji) | Catatan |
|---|---:|---:|---|
| **Train MSE (log)** | `0,0073` | **`0,009783`** | Error latih tidak terlalu rendah ekstrem; memorisasi berkurang. |
| **Test MSE (log)** | `0,0186` | **`0,016186`** | Generalisasi membaik (selisih latih–uji turun dari `0,0113` -> **`0,0064`**). |
| **Train R² / Test R²** | `94,10%` / `85,77%` | **`92,21%` / `87,14%`** | Selisih latih–uji turun dari **8,3%** menjadi **5,1%**, target $\ge 75\%$ **Tercapai**. |
| **MAE (Rupiah)** | Rp 422.640.860 | **Rp 354.905.653** | Turun ±Rp 67,7 juta. |
| **RMSE (Rupiah)** | Rp 964.431.107 | **Rp 840.362.311** | Turun ±Rp 124,1 juta. |
| **MdAPE** | 15,57% | **14,17%** | Makin mendekati error tipikal ~14%. |
| **MAPE** | 23,23% | **21,56%** | Belum $\le 20\%$ global karena *outlier* rumah mewah, namun turun signifikan. |

- **Learning Curve (`GroupKFold` 5-fold):**
  - *Training MSE:* `[0.0155, 0.0117, 0.0107, 0.0096, 0.0099]`
  - *Validation MSE:* `[0.0280, 0.0234, 0.0224, 0.0211, 0.0207]` *(terus menurun seiring bertambahnya data)*

### Kontribusi Fitur (*Feature Importances*)
1. `log_luas_bangunan`: **43,75%**
2. `log_luas_tanah`: **29,48%**
3. `kamar_mandi`: **24,14%**
4. `wilayah_Kulon Progo/Gunung Kidul`: **1,19%**
5. `wilayah_Kota Yogyakarta`: **0,74%**
6. `wilayah_Sleman`: **0,70%**

### Performa per Wilayah pada Data Uji (`n = 277`)
| Wilayah | Jumlah Uji | MdAPE | MAPE | MAE (Rupiah) |
|---|---:|---:|---:|---:|
| **Sleman** | 200 | 13,39% | 19,08% | Rp 348.371.308 |
| **Bantul** | 44 | 19,64% | 28,61% | Rp 222.905.560 |
| **Kota Yogyakarta** | 26 | 19,55% | 32,57% | Rp 718.438.974 |
| **Kulon Progo/Gunung Kidul** | 7 | 10,89% | 7,27% | Rp 21.049.481 |

*(Catatan positif: di **Sleman** yang menyumbang ~72% data uji, MAPE kini mencapai **19,08% ($\le 20\%$)** dan MdAPE membaik ke **13,39%**, serta di **Kota Yogyakarta** MdAPE turun drastis dari **26,16%** menjadi **19,55%**).*

---

## 6. Daftar Berkas yang Diubah & Perannya

| Berkas | Perubahan Utama |
|---|---|
| `prediksi.ipynb` | Perbaikan Pandas 3.0 regex (`.astype(object)`) di Cell 33; sinkronisasi narasi seleksi fitur (`24 area < 5 listing`) di Cell 87; penambahan `param_grid` 5 model + kondisi `None` + `n_jobs=1` RF di Cell 90–91; tabel analisis, perbaikan LaTeX `\text{VIF}`, alasan pemenang, dan grafik Baseline vs Tuned di Cell 92–94; pembaruan seluruh narasi & eksekusi ulang penuh tanpa error (Cell 96–113). |
| `src/house_price/config.py` | Penyelarasan `BEST_RF_PARAMS` (`n_estimators=300`, `max_depth=None`, `max_features='sqrt'`, `min_samples_leaf=2`, `min_samples_split=5`) dan `DEFAULT_MDAPE_RATE` (`0.1417`). |
| `src/house_price/models/train.py` | Pembaruan `DEFAULT_RF_PARAM_GRID` (432 -> 72 kombinasi), penambahan `PARAM_GRIDS_5_MODELS` (360 kombinasi), `BEST_TUNED_PARAMS_5_MODELS`, `tune_all_candidate_models`, `benchmark_models` (termasuk `Delta RMSE` & `Best Hyperparameters`), pencegahan *nested parallelism* (`n_jobs=1` pada RF), dan memastikan `save_model_bundle` menyimpan model dengan `n_jobs=1`. |
| `main.py` | Penyelarasan mode `--grid-search` agar menjalankan `tune_all_candidate_models` pada seluruh 5 model dan memilih estimator teratas; mode default menampilkan `benchmark_models` (Baseline vs Tuned). |
| `model_artifacts/model_prediksi_rumah_jogja.joblib` | Artefak model hasil eksekusi terbaru (`RandomForestRegressor`, `n_estimators=300`, `max_depth=None`, `max_features='sqrt'`, `min_samples_split=5`, `min_samples_leaf=2`, `n_jobs=1`). |
| `app.py` | Menyesuaikan tampilan spesifikasi model secara dinamis (`n_trees`, `max_feat`, `max_depth`) dari metadata bundle hasil prediksi (`bundle["model"]`). |
| `app_pages/03_metodologi.py` | Memperbarui tabel perbandingan 5 model hasil tuning (termasuk `Baseline CV RMSE (log)`, `Delta RMSE`, `Best Hyperparameters`), diagram alur Mermaid, dan tabel spesifikasi/metrik agar sinkron dengan notebook & artefak model. |
| `src/house_price/visualization/web_charts.py` | Menyesuaikan *tooltip* dan label sumbu pada `make_benchmark_chart` agar menampilkan informasi Baseline vs Tuned 5 model beserta `Best Hyperparameters`. |
| `notebooks/04_modeling_evaluation.ipynb` | Disinkronkan dengan konfigurasi 5 model terbaru dan dieksekusi ulang agar output notebook modul tetap identik dengan `prediksi.ipynb`. |
| `README.md` | Pembaruan ringkasan metodologi tuning 5 model + kondisi `None`, ukuran split (`1.107 / 277`), dan metrik evaluasi terkini. |
| `tests/test_modeling_and_predict.py` | Menambah pengujian unit & integrasi untuk `param_grid` berisi `None`, `benchmark_models`, `tune_all_candidate_models` (jalur `n_jobs=2` dan `n_jobs=1` pada RF default), `AppTest` Streamlit (`app.py` dan `03_metodologi.py`), serta konsistensi markdown `prediksi.ipynb`. |
| `docs/superpowers/specs/2026-11-25-revisi-tuning-5-model-design.md` & `docs/superpowers/plans/2026-11-25-revisi-tuning-5-model-dan-sinkronisasi.md` | Spesifikasi desain revisi dan rencana implementasi 4 tugas yang menjadi acuan sinkronisasi. |

---

## 7. Cara Menjalankan & Memverifikasi Ulang (Lokal)

Pastikan menggunakan virtual environment Python proyek (`.dsvenv`):

- **Menjalankan seluruh unit & integration test (16 pengujian):**
  ```powershell
  python -m unittest discover -s tests -v
  ```
- **Melatih & mengevaluasi model dengan parameter terbaik (mode cepat/default):**
  ```powershell
  python main.py
  ```
- **Menjalankan ulang tuning penuh 5 model (`GridSearchCV` 360 kombinasi / 1.800 fit CV, ~75 detik via CLI):**
  ```powershell
  python main.py --grid-search
  ```
- **Menjalankan aplikasi web Streamlit:**
  ```powershell
  python -m streamlit run app.py
  ```
