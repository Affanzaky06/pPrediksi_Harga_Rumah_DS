# Prediksi Harga Rumah di Wilayah Yogyakarta Melalui Pemodelan Regresi Berbasis Data Listing Properti Daring

Proyek Data Science dengan kerangka CRISP-DM untuk memahami dan memprediksi **harga penawaran rumah** di wilayah Yogyakarta dari data iklan Rumah123.

**Mata kuliah:** Data Science, Program Studi Informatika, Universitas Pembangunan Nasional "Veteran" Yogyakarta
**Dosen pengampu:** Dhimas Arief Dharmawan
**Tahun ajaran:** 2026/2027 (semester ganjil)

## Tim

**Kelompok 9 – PERTALITE LANGKA**

| Nama | NIM | Kontribusi |
|---|---|---|
| Muhammad Rifty Alfattah | 123240093 | Cleaning dan Preprocessing (audit kualitas data, cleaning berurutan, split anti-leakage, imputasi, encoding, scaling) serta seleksi fitur |
| Muhamad Atallah Alfa Dzaky | 123240105 | Exploratory Data Analysis (statistik deskriptif, visualisasi, korelasi, dan Konklusi) |
| Nicolaus Narindra Lianto | 123240100 | Modelling dan Evaluasi (metrik Rupiah dan persen, learning curve, residual, evaluasi per wilayah) |
| Abyaz Affanzaky Shanahan | 123240130 | Business Understanding, Data Gathering (sumber, lisensi, kamus data), penyusunan README dan pengelolaan repositori GitHub, dan  hyperparameter tuning

Setiap anggota juga ikut meninjau dan menguji bagian anggota lain agar seluruh alur dapat dipertanggungjawabkan bersama.

## Daftar Isi
1. [Latar belakang](#latar-belakang)
2. [Struktur repositori](#struktur-repositori)
3. [Status proyek](#status-proyek)
4. [Dataset](#dataset)
5. [Temuan EDA](#temuan-eda)
6. [Alur pengolahan data](#alur-pengolahan-data)
7. [Pemodelan dan hasil](#pemodelan-dan-hasil)
8. [Cara menjalankan](#cara-menjalankan)
9. [Keterbatasan](#keterbatasan)
10. [Lisensi, etika, dan sitasi](#lisensi-etika-dan-sitasi)

## Latar Belakang
Agen properti di Yogyakarta umumnya menentukan harga pasang berdasarkan pengalaman pribadi dan perbandingan manual dengan beberapa iklan di sekitar lokasi. Cara ini memakan waktu, hasilnya bisa berbeda antaragen, dan sulit dipertanggungjawabkan dengan data. Proyek ini memanfaatkan data iklan rumah untuk:

- mengetahui faktor yang paling memengaruhi harga rumah,
- mengukur seberapa akurat harga dapat diprediksi dari karakteristik fisik dan lokasi,
- membandingkan model regresi untuk dijadikan acuan awal harga pasang.

Klien utama: agen properti. Pemangku kepentingan lain: penjual rumah, calon pembeli, dan platform properti. Uraian lengkap Business Understanding ada di bagian awal notebook final.

## Struktur Repositori

| Berkas / folder | Isi |
|---|---|
| `prediksi.ipynb` | **Notebook final (acuan penilaian):** Business Understanding → Data Gathering → EDA → Cleaning dan Preprocessing → Modelling → Evaluasi → Konklusi |
| `rumah123_yogya_unfiltered.csv` | Dataset mentah (2.020 baris, 9 kolom) |
| `src/house_price/` | Kode pendukung: parsing, audit, cleaning, preprocessing, seleksi fitur, training, evaluasi, plot |
| `model_artifacts/` | Bundel model terlatih (`.joblib`) |
| `app.py`, `app_pages/` | Demo Streamlit (pelengkap, bukan bagian penilaian utama) |
| `requirements.txt` | Daftar pustaka Python |

Folder `data/processed/` dibuat otomatis saat pipeline dijalankan dan tidak disimpan di repositori. Seluruh angka di README ini mengacu pada notebook final.

## Status Proyek

| Fase CRISP-DM | Status | Keterangan |
|---|---|---|
| Business Understanding | Selesai | Masalah, stakeholder, pertanyaan, tujuan SMART, metrik, risiko dan etika |
| Data Understanding | Selesai | Sumber dan lisensi, kamus data, audit kualitas, EDA, representativitas |
| Data Preparation | Selesai | Parsing, cleaning berurutan, split anti-leakage, imputasi, log10, encoding, scaling, seleksi fitur |
| Modeling | Selesai | 5 model pembanding di-*tuning* seluruhnya (`GridSearchCV` + kondisi `None` + `GroupKFold`), Random Forest sebagai model final |
| Evaluation | Selesai | Metrik Rupiah dan persen, learning curve, residual, evaluasi per wilayah |
| Deployment | Pelengkap | Demo Streamlit (`app.py`) memakai bundel model |

## Dataset

| Aspek | Keterangan |
|---|---|
| Sumber asli | Rumah123 (rumah123.com), diambil dengan *web scraping* |
| Distributor | Kaggle, pengguna pramudyadika: [Yogyakarta Housing Price (Indonesia)](https://www.kaggle.com/datasets/pramudyadika/yogyakarta-housing-price-ndonesia) |
| Periode pengambilan | 20 Maret sampai 21 Mei 2024 (tidak ada tanggal per listing) |
| Berkas | `rumah123_yogya_unfiltered.csv` |
| Ukuran mentah | 2.020 baris dan 9 kolom |
| Ukuran setelah cleaning | 1.384 baris (68,5% dari data mentah): 1.107 data latih dan 277 data uji |
| Lisensi di Kaggle | Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) |

### Kamus data (data mentah)

| Kolom | Deskripsi | Tipe mentah | Satuan |
|---|---|---|---|
| `price` | Harga penawaran (target) | String, mis. "Rp 1,79 Miliar" | Rupiah |
| `nav-link` | URL listing (memuat ID dan kategori wilayah portal) | String | - |
| `description` | Judul iklan singkat | String | - |
| `listing-location` | Lokasi berformat "Area, Kabupaten/Kota" | String | - |
| `bed` | Jumlah kamar tidur | Float | Kamar |
| `bath` | Jumlah kamar mandi | Float | Kamar |
| `carport` | Jumlah carport | Float | Unit |
| `surface_area` | Luas tanah | String, mis. "120 m²" | m² |
| `building_area` | Luas bangunan | String, mis. "110 m²" | m² |

## Temuan EDA

Dataset analisis: 1.384 listing (data mentah setelah cleaning).

- Harga sangat miring ke kanan (skewness 3,41; median Rp 996,5 juta, rata-rata Rp 1,58 miliar). Transformasi log10 menurunkan skewness menjadi 0,27.
- Luas bangunan paling berkaitan dengan harga (Spearman 0,87), diikuti kamar mandi (0,81), luas tanah (0,80), dan kamar tidur (0,74). Carport lemah (0,38).
- Prediktor saling berkorelasi tinggi (luas bangunan, kamar tidur, kamar mandi sekitar 0,80), tetapi VIF pada skala log tetap di bawah 5.
- Data didominasi Sleman (sekitar 72%), diikuti Bantul (16%) dan Kota Yogyakarta (9%). Kulon Progo (28 listing) dan Gunung Kidul (5 listing) terlalu sedikit untuk dianalisis sendiri dan digabung saat pemodelan.
- Harga berbeda signifikan antarwilayah (Kruskal-Wallis H = 154,7; p ≈ 2e-32).
- Kategori wilayah pada portal bukan batas administratif: 154 dari 342 listing berkategori "Yogyakarta" ternyata berada di kabupaten lain, sehingga dikoreksi melalui pemetaan manual yang masih perlu diverifikasi.
- Unit kembar (harga, lokasi, dan spesifikasi sama) mencapai 438 baris (21,7% data mentah) dan berisiko menimbulkan *data leakage*.
- `carport` kosong (15,2%) bukan acak (tidak ada nilai 0 di data mentah) dan ditangani dengan indikator `carport_kosong`.

## Alur Pengolahan Data

1. **Parsing tipe data.** Harga dan luas diubah dari teks menjadi angka dengan ekspresi reguler.
2. **Normalisasi lokasi dan koreksi wilayah.** Spasi ganda dibersihkan, lokasi dipecah menjadi area dan wilayah, lalu kategori portal "Yogyakarta" dipetakan ulang ke kabupaten/kota administratif.
3. **Audit kualitas.** Missing value, pola MNAR carport, duplikat bertingkat, outlier IQR, dan aturan domain.
4. **Cleaning berurutan** (setiap langkah dicatat dalam *cleaning log*):

   | Langkah | Dibuang | Sisa |
   |---|---|---|
   | Data mentah | - | 2.020 |
   | 1. Luas tanah/bangunan kosong | 1 | 2.019 |
   | 2. Duplikat spesifikasi (unit kembar) | 438 | 1.581 |
   | 3. Non-hunian (kata kunci judul) | 172 | 1.409 |
   | 4. Nilai mustahil (aturan domain) | 25 | 1.384 |

   Aturan domain yang ditandai (jumlah terdeteksi sebelum dibuang): non-hunian 187, kamar tidur > 10: 91, kamar mandi > 10: 71, harga tidak wajar 11, luas ≤ 15 m² 3, luas bangunan > 5.000 m² 2, luas bangunan > 3x luas tanah 9, harga per m² tanah < Rp 1 juta 4. Luas bangunan > luas tanah (495 baris) dipertahankan karena wajar untuk rumah bertingkat. Ambang batas ini adalah usulan kerja peneliti, bukan standar baku.

5. **Preprocessing (anti-leakage).** Split 80/20 terstratifikasi menurut wilayah dan terkelompok menurut judul iklan; imputasi median kamar tidur dan kamar mandi dari data latih; log10 pada harga, luas tanah, dan luas bangunan; one-hot encoding wilayah (4 kelompok); *standard scaling* (hanya dipelajari dari data latih).
6. **Seleksi fitur.** VIF dan *drop-column* CV RMSE. Fitur final (6 kolom): `log_luas_bangunan`, `log_luas_tanah`, `kamar_mandi`, dan tiga *dummy* wilayah. Kamar tidur, carport, dan `carport_kosong` tidak dipakai karena kontribusinya hampir nol.

## Pemodelan dan Hasil

| Komponen | Keterangan |
|---|---|
| Target | log10(harga penawaran), dikembalikan ke Rupiah saat evaluasi |
| Fitur final | Luas bangunan (log10), luas tanah (log10), kamar mandi, wilayah (one-hot) |
| Model pembanding | Linear Regression, Ridge, SVR (RBF), Random Forest, Gradient Boosting |
| Validasi | **GroupKFold** (5-fold) menurut judul iklan; tidak memakai *time series split* karena tidak ada kolom tanggal |
| Metrik | R², MAE, RMSE, MAPE, MdAPE; evaluasi per wilayah |
| Target awal | MAPE ≤ 20% dan R² ≥ 0,75 pada data uji |

**Perbandingan model (validasi silang GroupKFold 5-fold, skala log10):** Seluruh 5 kandidat model di-*tuning* terlebih dahulu menggunakan `GridSearchCV` dengan `GroupKFold` (5-fold) serta menyertakan kondisi `None` pada ruang pencarian hyperparameter setiap model:

| Model | Baseline CV RMSE | Tuned CV RMSE | Δ RMSE | Tuned CV R² |
|---|---:|---:|---:|---:|
| **Random Forest** | 0,1512 | **0,1430** | **−0,0082** | **0,8353** |
| **Gradient Boosting** | 0,1523 | 0,1456 | −0,0067 | 0,8292 |
| **SVR (RBF)** | 0,1522 | 0,1516 | −0,0006 | 0,8138 |
| **Ridge Regression** | 0,1640 | 0,1640 | 0,0000 | 0,7826 |
| **Linear Regression** | 0,1640 | 0,1640 | 0,0000 | 0,7826 |

Random Forest dipilih sebagai model pemenang karena meraih skor terbaik di seluruh metrik validasi silang setelah *tuning*, memperoleh penurunan error terbesar, tahan terhadap outlier harga, dan mudah dijelaskan lewat *feature importance*. Rincian `param_grid` (termasuk kondisi `None`) serta alasan untuk setiap pilihan tercantum di bagian Modelling `prediksi.ipynb`.

**Model final:** Random Forest (`n_estimators=300`, `max_depth=None`, `max_features='sqrt'`, `min_samples_leaf=2`, `min_samples_split=5`, `bootstrap=True`), hasil `GridSearchCV` dengan `GroupKFold` (CV RMSE `0,1430` pada skala log10, turun dari baseline `0,1512`).

### Hasil pada data uji (277 listing)

| Metrik | Nilai | Target | Status |
|---|---|---|---|
| R² | 87,14% | ≥ 75% | Tercapai |
| MAE | Rp 354.905.653 | - | - |
| RMSE | Rp 840.362.311 | - | - |
| MdAPE | 14,17% | - | - |
| MAPE | 21,56% | ≤ 20% | Belum tercapai |

MAE sekitar Rp 354,9 juta berarti prediksi rata-rata meleset sekitar nilai tersebut dari harga penawaran sebenarnya. Pada rumah yang tipikal, prediksi meleset sekitar 14,2% (MdAPE).

**Kinerja per wilayah (MdAPE):** Sleman **13,39%** (200 sampel, MAPE 19,08%), Kota Yogyakarta **19,55%** (26 sampel, MAPE 32,57%), Bantul **19,64%** (44 sampel, MAPE 28,61%), dan Kulon Progo/Gunung Kidul **10,89%** (hanya 7 sampel sehingga tidak dapat disimpulkan). Model cenderung memprediksi terlalu rendah untuk rumah di atas sekitar Rp 4 miliar. Pembahasan lengkap ada di bagian Evaluasi dan Konklusi notebook final.

## Cara Menjalankan

1. Klon repositori
   ```bash
   git clone <url-repositori>
   cd <nama-folder>
   ```
2. Siapkan lingkungan Python (disarankan 3.10 atau lebih baru)
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Pastikan dataset `rumah123_yogya_unfiltered.csv` ada di folder utama repositori (unduh dari Kaggle melalui tautan di bagian Sitasi jika belum ada).
4. Jalankan **`prediksi.ipynb`** dari sel paling atas secara berurutan (*Restart & Run All*). Beberapa sel memakai variabel dari sel sebelumnya, sehingga menjalankan ulang sel di tengah dapat menyebabkan `KeyError`. Tahap *grid search* 5 model membutuhkan sekitar 1 menit pada CPU modern (lebih lama pada perangkat yang lambat). Atau jalankan pipeline CLI: `python main.py` (mode cepat dengan `BEST_RF_PARAMS`) / `python main.py --grid-search` (tuning penuh ke-5 kandidat model).
5. Periksa hasil: data mentah 2.020 baris, data bersih 1.384 baris (1.107 latih dan 277 uji), dan metrik Random Forest pada data uji sesuai tabel hasil (R² 87,14%, MAE Rp 354.905.653, MdAPE 14,17%). Seluruh proses acak memakai `random_state=42`, tetapi angka dapat berbeda tipis pada versi `scikit-learn` yang berbeda.
6. (Opsional) Demo web: `pip install streamlit altair`, lalu `streamlit run app.py`.

## Keterbatasan

- Target adalah harga penawaran, bukan harga transaksi.
- Data hanya dari satu platform (Rumah123) dan satu periode (Maret sampai Mei 2024), sehingga dapat berbeda dari kondisi pasar saat ini.
- Tidak tersedia tanggal listing, koordinat, tahun bangun, kondisi bangunan, maupun status sertifikat. Ini kemungkinan penyebab utama error tinggi (MAPE 32,57%) di Kota Yogyakarta, tempat harga bergantung pada lokasi mikro.
- Sebaran wilayah timpang, sehingga hasil tidak digeneralisasi ke seluruh DIY. Evaluasi Kulon Progo/Gunung Kidul hanya berdasar 7 sampel uji.
- Pembersihan properti non-hunian berbasis kata kunci pada judul dan dapat meleset. Ambang jumlah kamar paling sensitif dan perlu dikonfirmasi pelaku pasar.
- Pemetaan wilayah administratif dibuat manual dan perlu diverifikasi dengan data batas resmi.
- Model sedikit *overfitting* ringan (R² latih 0,922 vs uji 0,871) dan cenderung memprediksi terlalu rendah untuk rumah mahal.
- Target MAPE ≤ 20% secara global belum tercapai (21,56%), meskipun pada wilayah Sleman sudah mencapai 19,08%, dan error terbesar (MAPE 32,57%, MAE Rp 718,4 juta) terjadi di Kota Yogyakarta.
- Seluruh 5 kandidat model telah di-*tuning* dengan menyertakan kondisi `None` (360 kombinasi / 1.800 fit CV), sehingga keterbatasan akurasi terutama berasal dari absennya fitur lokasi mikro dan kondisi fisik bangunan, bukan kurangnya eksplorasi model.

## Lisensi, Etika, dan Sitasi

- **Data:** CC BY-NC 4.0 (penggunaan non-komersial dengan atribusi). Lisensi ini ditetapkan oleh pengunggah di Kaggle dan tidak mengalihkan hak atas konten asli Rumah123.
- **Penggunaan:** hanya untuk keperluan akademik. Prototipe tidak dipublikasikan sebagai layanan komersial.
- **Privasi:** kolom `description` tidak dipakai sebagai fitur model dan tidak ditampilkan mentah.
- **Kode:** digunakan untuk keperluan akademik mata kuliah Data Science UPN "Veteran" Yogyakarta.

Jika memakai dataset ini, sitasi sumbernya:

> pramudyadika. (2024). *Yogyakarta housing price (Indonesia)* [Data set]. Kaggle. https://www.kaggle.com/datasets/pramudyadika/yogyakarta-housing-price-ndonesia
