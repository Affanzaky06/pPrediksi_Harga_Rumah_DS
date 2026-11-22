# Prediksi Harga Rumah di Wilayah Yogyakarta Melalui Pemodelan Regresi Berbasis Data Listing Properti Daring

Proyek Data Science dengan kerangka CRISP-DM untuk memahami dan memprediksi harga penawaran rumah di wilayah Yogyakarta dari data iklan Rumah123.

**Mata kuliah:** Data Science, Program Studi Informatika, Universitas Pembangunan Nasional "Veteran" Yogyakarta
**Dosen pengampu:** Dhimas Arief Dharmawan
**Tahun ajaran:** 2026/2027 (semester ganjil)

## Tim

**Kelompok 9 – PERTALITE LANGKA**

| Nama | NIM |
|---|---|
| Muhammad Rifty Alfattah | 123240093 | 
| Muhamad Atallah Alfa Dzaky | 123240105 |
| Nicolaus Narindra Lianto | 123240100 | 
| Abyaz Affanzaky Shanahan | 123240130 |


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

Klien utama: agen properti. Pemangku kepentingan lain: penjual rumah, calon pembeli, dan platform properti. Uraian lengkap Business Understanding ada di **Bagian 1 notebook final**.

## Struktur Repositori
> Sesuaikan jika ada berkas atau folder yang berbeda di repositori.

| Berkas / folder | Isi |
|---|---|
| `Proyek_Akhir_Prediksi_Harga_Rumah_Yogyakarta.ipynb` | **Notebook final (acuan penilaian):** Business Understanding → Data Understanding & Cleaning → EDA → Preprocessing → Modeling → Evaluation → Konklusi |
| `01`–`04` (notebook modular) | Versi bertahap yang menjadi bahan notebook final (tidak lagi menjadi acuan) |
| `src/house_price/` | Kode pendukung: parsing, audit, cleaning, preprocessing, seleksi fitur, training, evaluasi, plot |
| `data/` | Dataset mentah dan hasil olahan (`data/processed/`) |
| `model_artifacts/` | Bundel model terlatih (`.joblib`) |
| `app.py` | Demo Streamlit (pelengkap) |

Seluruh angka di README ini mengacu pada notebook final.

## Status Proyek
| Fase CRISP-DM | Status | Keterangan |
|---|---|---|
| Business Understanding | Selesai | Masalah, stakeholder, pertanyaan, tujuan SMART, metrik, risiko dan etika |
| Data Understanding | Selesai | Sumber dan lisensi, kamus data, audit kualitas, EDA, representativitas |
| Data Preparation | Selesai | Parsing, cleaning berurutan, split anti-leakage, imputasi, log10, encoding, scaling, seleksi fitur |
| Modeling | Selesai | 5 model pembanding dengan GroupKFold, Random Forest sebagai model final |
| Evaluation | Selesai | Metrik Rupiah dan persen, learning curve, residual, evaluasi per wilayah |
| Deployment | Selesai | Demo Streamlit (`app.py`) memakai bundel model |

## Dataset
| Aspek | Keterangan |
|---|---|
| Sumber asli | Rumah123 (rumah123.com), diambil dengan web scraping |
| Distributor | Kaggle, pengguna pramudyadika: Yogyakarta Housing Price (Indonesia) |
| Periode pengambilan | 20 Maret sampai 21 Mei 2024 (tidak ada tanggal per listing) |
| Berkas | `rumah123_yogya_unfiltered.csv` |
| Ukuran mentah | 2.020 baris dan 9 kolom |
| Ukuran setelah cleaning | 1.384 baris (68,5% dari data mentah): 1.107 data latih dan 277 data uji |
| Lisensi di Kaggle | Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) |

### Kamus data (data mentah)
| Kolom | Deskripsi | Tipe mentah | Satuan |
|---|---|---|---|
| price | Harga penawaran (target) | String, mis. "Rp 1,79 Miliar" | Rupiah |
| nav-link | URL listing (memuat ID dan kategori wilayah portal) | String | - |
| description | Judul iklan singkat | String | - |
| listing-location | Lokasi berformat "Area, Kabupaten/Kota" | String | - |
| bed | Jumlah kamar tidur | Float | Kamar |
| bath | Jumlah kamar mandi | Float | Kamar |
| carport | Jumlah carport | Float | Unit |
| surface_area | Luas tanah | String, mis. "120 m²" | m² |
| building_area | Luas bangunan | String, mis. "110 m²" | m² |

## Temuan EDA
Dataset analisis: 1.384 listing (data mentah setelah cleaning).

- Harga sangat miring ke kanan (skewness 3,41; median Rp 996,5 juta, rata-rata Rp 1,58 miliar). Transformasi log10 menurunkan skewness menjadi 0,27.
- Luas bangunan paling berkaitan dengan harga (Spearman 0,87), diikuti kamar mandi (0,81), luas tanah (0,80), dan kamar tidur (0,74). Carport lemah (0,38).
- Prediktor saling berkorelasi tinggi (luas bangunan, kamar tidur, kamar mandi sekitar 0,80), tetapi VIF pada skala log tetap di bawah 4.
- Data didominasi Sleman (sekitar 72%), diikuti Bantul (16%) dan Kota Yogyakarta (9%). Kulon Progo (28 listing) dan Gunung Kidul (5 listing) terlalu sedikit untuk dianalisis sendiri dan digabung saat pemodelan.
- Harga berbeda signifikan antarwilayah (Kruskal-Wallis H = 154,7; p ≈ 2e-32).
- Kategori wilayah pada portal bukan batas administratif: 154 dari 342 listing berkategori "Yogyakarta" ternyata berada di kabupaten lain, sehingga dikoreksi melalui pemetaan manual yang masih perlu diverifikasi.
- Unit kembar (harga, lokasi, dan spesifikasi sama) mencapai 438 baris (21,7% data mentah) dan berisiko menimbulkan data leakage.
- `carport` kosong (15,2%) bukan acak (tidak ada nilai 0 di data mentah) dan ditangani dengan indikator `carport_kosong`.

## Alur Pengolahan Data
1. **Parsing tipe data.** Harga dan luas diubah dari teks menjadi angka dengan ekspresi reguler.
2. **Normalisasi lokasi dan koreksi wilayah.** Spasi ganda dibersihkan, lokasi dipecah menjadi area dan wilayah, lalu kategori portal "Yogyakarta" dipetakan ulang ke kabupaten/kota administratif.
3. **Audit kualitas.** Missing value, pola MNAR carport, duplikat bertingkat, outlier IQR, dan aturan domain.
4. **Cleaning berurutan** (setiap langkah dicatat dalam cleaning log):

| Langkah | Dibuang | Sisa |
|---|---|---|
| Data mentah | - | 2.020 |
| 1. Luas tanah/bangunan kosong | 1 | 2.019 |
| 2. Duplikat spesifikasi (unit kembar) | 438 | 1.581 |
| 3. Non-hunian (kata kunci judul) | 172 | 1.409 |
| 4. Nilai mustahil (aturan domain) | 25 | 1.384 |

Aturan domain yang ditandai (jumlah terdeteksi sebelum dibuang): non-hunian 187, kamar tidur > 10: 91, kamar mandi > 10: 71, harga tidak wajar 11, luas ≤ 15 m² 3, luas bangunan > 5.000 m² 2, luas bangunan > 3x luas tanah 9, harga per m² tanah < Rp 1 juta 4. Luas bangunan > luas tanah (495 baris) dipertahankan karena wajar untuk rumah bertingkat. Ambang batas ini adalah usulan kerja peneliti, bukan standar baku.

5. **Preprocessing (anti-leakage).** Split 80/20 terstratifikasi menurut wilayah dan terkelompok menurut judul iklan; imputasi median kamar tidur dan kamar mandi dari data latih; log10 pada harga, luas tanah, dan luas bangunan; one-hot encoding wilayah (4 kelompok); standard scaling (hanya dipelajari dari data latih).
6. **Seleksi fitur.** VIF dan drop-column CV RMSE. Fitur final (6 kolom): `log_luas_bangunan`, `log_luas_tanah`, `kamar_mandi`, dan tiga dummy wilayah. Kamar tidur, carport, dan carport_kosong tidak dipakai karena kontribusinya hampir nol.

## Pemodelan dan Hasil
| Komponen | Keterangan |
|---|---|
| Target | log10(harga penawaran), dikembalikan ke Rupiah saat evaluasi |
| Fitur final | Luas bangunan (log10), luas tanah (log10), kamar mandi, wilayah (one-hot) |
| Model pembanding | Linear Regression, Ridge, SVR (RBF), Random Forest, Gradient Boosting |
| Validasi | **GroupKFold** menurut judul iklan; tidak memakai time series split karena tidak ada kolom tanggal |
| Metrik | R², MAE, RMSE, MAPE, MdAPE; evaluasi per wilayah |
| Target awal | MAPE ≤ 20% dan R² ≥ 0,75 pada data uji |

### Perbandingan model (validasi silang GroupKFold, skala log10)
| Model | CV RMSE | CV MAE | CV R² |
|---|---|---|---|
| **Random Forest** | 0,14 | 0,10 | 0,83 |
| SVR (RBF) | 0,15 | 0,12 | 0,81 |
| Gradient Boosting | 0,15 | 0,11 | 0,81 |
| Ridge Regression | 0,16 | 0,13 | 0,78 |
| Linear Regression | 0,16 | 0,13 | 0,78 |

Model final: **Random Forest** (`n_estimators=100`, `max_depth=15`, `max_features=0.5`, `min_samples_leaf=2`, `min_samples_split=5`).

### Hasil pada data uji (277 listing)
| Metrik | Nilai | Target | Status |
|---|---|---|---|
| R² | 87,27% | ≥ 75% | Tercapai |
| MAE | Rp 352.079.279 | - | - |
| RMSE | Rp 828.875.988 | - | - |
| MdAPE | 14,71% | - | - |
| MAPE | 21,32% | ≤ 20% | Belum tercapai |

MAE sekitar Rp 352 juta berarti prediksi rata-rata meleset sekitar nilai tersebut dari harga penawaran sebenarnya. Kinerja per wilayah: Sleman paling akurat (MdAPE 13,86%), Bantul 18,99%, Kota Yogyakarta 18,13%; hasil Kulon Progo/Gunung Kidul (7 sampel uji) tidak dapat disimpulkan. Model cenderung memprediksi terlalu rendah untuk rumah di atas sekitar Rp 4 miliar. Pembahasan lengkap ada di Bagian 6 dan 7 notebook final.

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
   pip install pandas numpy scipy matplotlib seaborn scikit-learn jupyter
   ```
   Tambahkan `streamlit` dan `altair` jika ingin menjalankan demo dan unit test.
3. Pastikan dataset `rumah123_yogya_unfiltered.csv` ada di folder data (unduh dari Kaggle melalui tautan di bagian Sitasi jika belum ada).
4. Jalankan **notebook final** dari sel paling atas secara berurutan. Beberapa sel memakai variabel dari sel sebelumnya, sehingga menjalankan ulang sel di tengah dapat menyebabkan `KeyError`.
5. Periksa hasil: data mentah 2.020 baris, data bersih 1.384 baris (1.107 latih dan 277 uji), dan metrik Random Forest pada data uji sesuai tabel hasil (R² 87,27%, MAE Rp 352.079.279, MdAPE 14,71%).

## Keterbatasan
- Target adalah harga penawaran, bukan harga transaksi.
- Data hanya dari satu platform (Rumah123) dan satu periode (Maret sampai Mei 2024), sehingga dapat berbeda dari kondisi pasar saat ini.
- Tidak tersedia tanggal listing, koordinat, tahun bangun, kondisi bangunan, maupun status sertifikat.
- Sebaran wilayah timpang, sehingga hasil tidak digeneralisasi ke seluruh DIY. Evaluasi Kulon Progo/Gunung Kidul hanya berdasar 7 sampel uji.
- Pembersihan properti non-hunian berbasis kata kunci pada judul dan dapat meleset. Ambang jumlah kamar paling sensitif dan perlu dikonfirmasi pelaku pasar.
- Pemetaan wilayah administratif dibuat manual dan perlu diverifikasi dengan data batas resmi.
- Model sedikit overfitting (R² latih 0,93 vs uji 0,87) dan cenderung memprediksi terlalu rendah untuk rumah mahal.
- Target MAPE ≤ 20% belum tercapai (21,32%).

## Lisensi, Etika, dan Sitasi
- **Data:** CC BY-NC 4.0 (penggunaan non-komersial dengan atribusi). Lisensi ini ditetapkan oleh pengunggah di Kaggle dan tidak mengalihkan hak atas konten asli Rumah123.
- **Penggunaan:** hanya untuk keperluan akademik. Prototipe tidak dipublikasikan sebagai layanan komersial.
- **Privasi:** kolom `description` tidak dipakai sebagai fitur model dan tidak ditampilkan mentah.
- **Kode:** [PILIH: mis. MIT, lalu tambahkan berkas LICENSE]

Jika memakai dataset ini, sitasi sumbernya:

> pramudyadika. (2024). Yogyakarta housing price (Indonesia) [Data set]. Kaggle. https://www.kaggle.com/datasets/pramudyadika/yogyakarta-housing-price-ndonesia
