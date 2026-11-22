# Prediksi Harga Rumah di Wilayah Yogyakarta Melalui Pemodelan Regresi Berbasis Data Listing Properti Daring

Proyek Data Science dengan kerangka **CRISP-DM** untuk memahami dan memprediksi **harga penawaran rumah** di wilayah Yogyakarta dari data iklan Rumah123.

- **Mata kuliah:** Data Science, Program Studi Informatika, Universitas Pembangunan Nasional "Veteran" Yogyakarta
- **Dosen pengampu:** Dhimas Arief Dharmawan
- **Tahun ajaran:** 2026/2027 (semester ganjil)

## Tim

Nomor Kelompok : 9

Nama Kelompok  : PERTALITE LANGKA 

| Nama | NIM |
|---|---|
| Muhammad Rifty Alfattah | 123240093 |
| Muhamad Atallah Alfa Dzaky | 123240105 |
| Nicolaus Narindra Lianto | 123240100 |
| Abyaz Affanzaky Shanahan | 123240130 |


## Daftar Isi

1. [Latar belakang](#latar-belakang)
2. [Status proyek](#status-proyek)
3. [Dataset](#dataset)
4. [Temuan sementara](#temuan-sementara)
5. [Alur pengolahan data](#alur-pengolahan-data)
6. [Cara menjalankan](#cara-menjalankan)
7. [Hasil pemodelan dan evaluasi](#hasil-pemodelan-dan-evaluasi)
8. [Keterbatasan](#keterbatasan)
9. [Lisensi, etika, dan sitasi](#lisensi-etika-dan-sitasi)
10. [Tim](#tim)

## Latar Belakang

Agen properti di Yogyakarta umumnya menentukan harga pasang berdasarkan pengalaman pribadi dan perbandingan manual dengan beberapa iklan di sekitar lokasi. Cara ini memakan waktu, hasilnya bisa berbeda antaragen, dan sulit dipertanggungjawabkan dengan data. Proyek ini memanfaatkan data iklan rumah untuk:

1. mengetahui **faktor yang paling memengaruhi harga** rumah,
2. mengukur **seberapa akurat** harga dapat diprediksi dari karakteristik fisik dan lokasi,
3. membandingkan model regresi untuk dijadikan **acuan awal harga pasang**.

**Klien utama:** agen properti. **Pemangku kepentingan lain:** penjual rumah, calon pembeli, dan platform properti.

## Status Proyek

| Fase CRISP-DM | Status | Keterangan |
|---|---|---|
| Business Understanding | Selesai | Masalah, stakeholder, tujuan SMART, metrik, risiko dan etika |
| Data Understanding | Selesai | Sumber dan lisensi, data dictionary, audit kualitas data, EDA, representativitas |
| Data Preparation | Selesai | Parsing tipe data, koreksi wilayah, cleaning berurutan (1.384 baris), split *grouped-stratified*, scaling & seleksi fitur (6 fitur) |
| Modeling | Selesai | Benchmark 5 model dengan *GroupKFold* (5-fold) & *hyperparameter tuning* `RandomForestRegressor` |
| Evaluation | Selesai | Evaluasi data uji ($R^2 = 87,27\%$, $\text{MdAPE} = 14,71\%$, $\text{MAE} = \text{Rp } 352,08\text{ Juta}$), analisis residual & evaluasi per wilayah |
| Deployment | Selesai | Aplikasi web multipage Streamlit (`app.py` + `app_pages/`), CLI runner (`main.py`), dan 14 *unit test* otomatis |

## Dataset

| Aspek | Keterangan |
|---|---|
| Sumber asli | Rumah123 (rumah123.com), diambil dengan *web scraping* |
| Distributor | Kaggle, pengguna `pramudyadika`: [Yogyakarta Housing Price (Indonesia)](https://www.kaggle.com/datasets/pramudyadika/yogyakarta-housing-price-ndonesia) |
| Periode pengambilan | 20 Maret sampai 21 Mei 2024 (tidak ada tanggal per listing) |
| Berkas | `rumah123_yogya_unfiltered.csv` |
| Ukuran mentah | 2.020 baris dan 9 kolom |
| Lisensi di Kaggle | Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) |


### Kamus data (data mentah)

| Kolom | Deskripsi | Tipe mentah | Satuan |
|---|---|---|---|
| `price` | Harga penawaran | String, mis. "Rp 1,79 Miliar" | Rupiah |
| `nav-link` | URL listing (memuat ID dan kategori wilayah portal) | String | - |
| `description` | Judul iklan singkat | String | - |
| `listing-location` | Lokasi berformat "Area, Kabupaten/Kota" | String | - |
| `bed` | Jumlah kamar tidur | Float | Kamar |
| `bath` | Jumlah kamar mandi | Float | Kamar |
| `carport` | Jumlah carport | Float | Unit |
| `surface_area` | Luas tanah | String, mis. "120 m²" | m² |
| `building_area` | Luas bangunan | String, mis. "110 m²" | m² |

## Temuan Sementara

Hasil EDA pada dataset analisis (data mentah dikurangi duplikat dan baris yang terkena aturan pengecualian, **1.384 listing**):

- **Harga sangat miring ke kanan** (skewness 3,42; median Rp 995 juta, rata-rata Rp 1,58 miliar). Transformasi logaritma menurunkan skewness menjadi 0,26.
- **Luas bangunan paling berkaitan dengan harga** (Spearman 0,87), diikuti kamar mandi (0,81), luas tanah (0,80), dan kamar tidur (0,74). Carport lemah (0,39).
- **Prediktor saling berkorelasi tinggi** (kamar tidur dan kamar mandi sekitar 0,80), sehingga multikolinearitas perlu diwaspadai pada model linear.
- **Data didominasi Sleman** (sekitar 72%), diikuti Bantul dan Kota Yogyakarta. Kulon Progo dan Gunung Kidul terlalu sedikit untuk dianalisis andal.
- **Kategori wilayah pada portal bukan batas administratif.** Sebagian listing berkategori "Yogyakarta" berada di Sleman, Bantul, atau Kulon Progo, sehingga dikoreksi melalui pemetaan manual yang masih perlu diverifikasi.
- **Unit kembar** (harga, lokasi, dan spesifikasi sama) mencapai 438 baris (21,7% data mentah) dan berisiko menimbulkan *data leakage*.

## Alur Pengolahan Data

1. **Parsing tipe data.** Harga dan luas diubah dari teks ("Rp 1,79 Miliar", "120 m²") menjadi angka dengan ekspresi reguler. Kolom jumlah memakai tipe `Int64` yang mengizinkan nilai kosong.
2. **Normalisasi lokasi.** Spasi ganda dibersihkan, lalu lokasi dipecah menjadi `area` dan `kab_label`.
3. **Penandaan anomali (bukan penghapusan langsung).** Setiap aturan menghasilkan kolom bendera True/False sehingga baris dapat ditinjau sebelum dikeluarkan:

   | Aturan | Terdeteksi |
   |---|---|
   | Luas bangunan > 5.000 m² | 2 |
   | Luas tanah atau bangunan ≤ 15 m² | 3 |
   | Kamar tidur > 10 | 91 |
   | Kamar mandi > 10 | 71 |
   | Harga < Rp 50 juta | 1 |
   | Harga > Rp 20 miliar | 10 |
   | Luas bangunan > 3 × luas tanah (hanya ditinjau) | 9 |
   | Judul menandakan kos, villa/homestay, ruko, gudang, atau tanah kosong | 183 |
   | Luas bangunan > luas tanah (dipertahankan, wajar untuk rumah bertingkat) | 495 |

   Ambang batas ini ditetapkan peneliti sebagai usulan kerja, bukan standar baku. Aturan jumlah kamar paling sensitif terhadap ambang dan perlu dikonfirmasi pelaku pasar.
4. **Penanganan duplikat.** Baris yang isinya identik (63) dihapus. Unit kembar lainnya dipertahankan dan dikelompokkan agar tidak terpisah antara data latih dan data uji.
5. **Koreksi wilayah.** Area berkategori "Yogyakarta" yang secara administratif berada di kabupaten lain dipetakan ulang secara manual.

## Cara Menjalankan

### Struktur Proyek Modular

```text
├── app.py                              # Entrypoint multipage Streamlit (`st.navigation`)
├── app_pages/                          # Halaman modular Streamlit (`01_kalkulator`, `02_eda`, `03_metodologi`)
├── main.py                             # Runner CLI end-to-end (build data -> train -> evaluasi -> simpan .joblib)
├── pyproject.toml                      # Konfigurasi paket Python (src layout)
├── src/house_price/
│   ├── config.py                       # Path, seed, peta koreksi wilayah, aturan domain, parameter model
│   ├── data/                           # parsing.py, audit.py, cleaning.py
│   ├── features/                       # preprocessing.py, selection.py
│   ├── models/                         # train.py, evaluate.py, predict.py, demo.py
│   ├── visualization/                  # plots.py (notebook), web_charts.py (Altair web)
│   └── web/                            # loaders.py (shared Streamlit cache & state)
├── notebooks/
│   ├── 01_data_understanding_cleaning.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_feature_selection.ipynb
│   └── 04_modeling_evaluation.ipynb
├── tests/                              # Unit test (unittest / pytest)
├── model_artifacts/
│   └── model_prediksi_rumah_jogja.joblib # Artefak model utama (diperbarui langsung saat training)
└── prediksi.ipynb                      # Arsip notebook monolitik lengkap
```

1. **Siapkan lingkungan Python** (disarankan 3.10 atau lebih baru)

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Jalankan pipeline end-to-end via CLI** (membangun checkpoint `data/processed/` dan menimpa model utama di `model_artifacts/model_prediksi_rumah_jogja.joblib` sesuai versi `scikit-learn` lokal):

   ```bash
   python main.py
   ```

3. **Jalankan notebook modular per fase (untuk eksplorasi & visualisasi)**

   Buka salah satu notebook di folder `notebooks/` (`01` hingga `04`). Setiap notebook dapat dijalankan per sel maupun berdiri sendiri karena otomatis memuat atau membangun checkpoint di `data/processed/`.

4. **Jalankan pengujian otomatis**

   ```bash
   python -m unittest discover -s tests -v
   ```

5. **Jalankan aplikasi web Streamlit**

   ```bash
   streamlit run app.py
   ```

   *(Jika dibuka di mesin dengan versi `scikit-learn` berbeda dari artefak tersimpan, `app.py` otomatis membangun ulang `model_artifacts/model_prediksi_rumah_jogja.joblib` agar kompatibel dan bebas warning.)*

### Catatan Pengembangan Lanjutan
- `GridSearchCV` dan `learning_curve` pada tahap modeling telah dilengkapi dengan `GroupKFold(n_splits=5)` per `deskripsi` serta tuning parameter `max_features` untuk mendekorelasi pohon-pohon pada Random Forest dan mencegah *data leakage* antar unit kembar.
- `KOLOM_SCALE` masih menyertakan `kamar_tidur` dan `carport` untuk kompatibilitas skema scaler; di UI Streamlit keduanya ditempatkan pada expander kompatibilitas opsional.

## Hasil Pemodelan dan Evaluasi

| Komponen | Implementasi & Hasil Aktual |
|---|---|
| Target | $\log_{10}(\text{harga penawaran})$, dikembalikan ke Rupiah dengan $10^{\hat{y}}$ |
| Fitur Final (6 fitur) | `log_luas_bangunan`, `log_luas_tanah`, `kamar_mandi`, dan 3 *dummy* `wilayah` (baseline: Bantul) |
| Validasi & Split | `StratifiedGroupKFold` (80% Train = 1.107 baris, 20% Test = 277 baris; 0 kebocoran judul iklan) |
| Kandidat Model Diuji | Linear Regression, Ridge Regression, SVR (RBF), Random Forest, Gradient Boosting |
| Model Terpilih | `RandomForestRegressor(n_estimators=100, max_depth=15, max_features=0.5, min_samples_leaf=2, min_samples_split=5, random_state=42)` |
| Skor Validasi Silang (Train) | 5-Fold `GroupKFold` CV RMSE ($\log_{10}$) = **0,1440** |
| Kinerja Data Uji ($n = 277$) | **$R^2 = 87,27\%$** (target $\ge 75\%$), **MdAPE = 14,71%** (target $\le 20\%$), **MAPE = 21,32%**, **MAE = Rp 352.079.279**, **RMSE = Rp 828.875.988** |
| Perbandingan Baseline | Melampaui *Baseline Median per Wilayah* ($R^2 = 14,76\%$, MdAPE $= 55,17\%$) maupun Regresi Linear ($R^2 = 82,10\%$, MdAPE $= 21,93\%$) |
| Deployment | Aplikasi web multipage Streamlit (`streamlit run app.py`) dengan simulasi harga, rentang wajar $\pm 14,71\%$, uji sampel nyata *test set*, EDA interaktif, dan *auto-rebuild* lintas-versi `scikit-learn` |

## Keterbatasan

- Target adalah **harga penawaran**, bukan harga transaksi.
- Data hanya dari **satu platform** (Rumah123) dan satu periode (Maret sampai Mei 2024), sehingga dapat berbeda dari kondisi pasar saat ini.
- Tidak tersedia tanggal listing, koordinat, tahun bangun, kondisi bangunan, maupun status sertifikat.
- Sebaran wilayah timpang, sehingga hasil **tidak digeneralisasi ke seluruh DIY**.
- Pembersihan properti non-hunian berbasis kata kunci pada judul dan dapat meleset.
- Pemetaan wilayah administratif dibuat manual dan perlu diverifikasi dengan data batas resmi.

## Lisensi, Etika, dan Sitasi

- **Data:** CC BY-NC 4.0 (penggunaan non-komersial dengan atribusi). Lisensi ini ditetapkan oleh pengunggah di Kaggle dan tidak mengalihkan hak atas konten asli Rumah123.
- **Penggunaan:** hanya untuk keperluan akademik. Prototipe tidak dipublikasikan sebagai layanan komersial.
- **Privasi:** kolom `description` tidak dipakai sebagai fitur model dan tidak ditampilkan mentah.
- **Kode:** digunakan untuk keperluan akademik mata kuliah Data Science UPN "Veteran" Yogyakarta.

Jika memakai dataset ini, sitasi sumbernya:

> pramudyadika. (2024). *Yogyakarta housing price (Indonesia)* [Data set]. Kaggle. https://www.kaggle.com/datasets/pramudyadika/yogyakarta-housing-price-ndonesia
