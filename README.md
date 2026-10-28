# Prediksi Harga Rumah di Wilayah Yogyakarta Melalui Pemodelan Regresi Berbasis Data Listing Properti Daring

Proyek Data Science dengan kerangka **CRISP-DM** untuk memahami dan memprediksi **harga penawaran rumah** di wilayah Yogyakarta dari data iklan Rumah123.

- **Mata kuliah:** Data Science, Program Studi Informatika, Universitas Pembangunan Nasional "Veteran" Yogyakarta
- **Dosen pengampu:** Dhimas Arief Dharmawan
- **Tahun ajaran:** 2026/2027 (semester ganjil)

## Daftar Isi

1. [Latar belakang](#latar-belakang)
2. [Status proyek](#status-proyek)
3. [Dataset](#dataset)
4. [Temuan sementara](#temuan-sementara)
5. [Alur pengolahan data](#alur-pengolahan-data)
6. [Cara menjalankan](#cara-menjalankan)
7. [Rencana pemodelan (UAS)](#rencana-pemodelan-uas)
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
| Business Understanding | Selesai (UTS) | Masalah, stakeholder, tujuan SMART, metrik, risiko dan etika |
| Data Understanding | Selesai (UTS) | Sumber dan lisensi, data dictionary, kualitas data, EDA, representativitas |
| Data Preparation | Berjalan | Konversi tipe data, penandaan anomali, koreksi wilayah; pipeline akhir menyusul |
| Modeling | Direncanakan (UAS) | Baseline, kandidat model, validasi *grouped k-fold* |
| Evaluation | Direncanakan (UAS) | Metrik, kinerja per wilayah dan segmen harga, *fairness* |
| Deployment | Direncanakan (UAS) | Notebook demo, skema I/O, rencana pemantauan *drift* |

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

Hasil EDA pada dataset analisis (data mentah dikurangi duplikat dan baris yang terkena aturan pengecualian, **1.383 listing**):

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

1. **Klon repositori**

   ```bash
   git clone <url-repositori-anda>
   cd <nama-folder>
   ```

2. **Siapkan lingkungan Python** (disarankan 3.10 atau lebih baru)

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. **Unduh dataset** dari Kaggle (tautan di bagian [Dataset](#dataset)).

4. **Jalankan notebook**

   Jalankan sel **berurutan dari atas**. Beberapa sel memakai variabel dari sel sebelumnya (mis. `df_process`), sehingga menjalankan ulang sel di tengah dapat menyebabkan galat `KeyError`.

5. **Periksa hasil.** Jumlah baris data mentah harus 2.020, dan jumlah yang terdeteksi pada tiap aturan anomali harus sama dengan tabel di atas.

## Rencana Pemodelan (UAS)

| Komponen | Rencana |
|---|---|
| Target | Harga penawaran |
| Fitur | Luas tanah, luas bangunan, kamar tidur, kamar mandi, carport, wilayah |
| Transformasi | Log pada harga dan luas untuk model linear; model berbasis pohon memakai skala asli |
| Baseline | Median harga per kecamatan, regresi linear |
| Kandidat model | Linear Regresion, Random Forest |
| Validasi | *Grouped k-fold* (unit kembar dalam satu lipatan); tidak memakai *time series split* karena tidak ada kolom tanggal |
| Metrik | MAE, RMSE, R², MAPE, MdAPE; kinerja per wilayah dan per segmen harga |
| Target awal | MAPE ≤ 20% dan R² ≥ 0,75 pada data uji (dapat disesuaikan setelah baseline) |
| Deployment | Notebook demo dengan skema I/O terdokumentasi, keluaran berupa rentang prediksi, rencana pemantauan *drift* dan *retraining* |

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
- **Kode:** [pilih lisensi untuk kode, mis. MIT, lalu tambahkan berkas `LICENSE`].

Jika memakai dataset ini, sitasi sumbernya:

> pramudyadika. (2024). *Yogyakarta housing price (Indonesia)* [Data set]. Kaggle. https://www.kaggle.com/datasets/pramudyadika/yogyakarta-housing-price-ndonesia

## Tim

| Nama | NIM |
|---|---|
| Muhammad Rifty Alfattah | 123240093 |
| Muhamad Atallah Alfa Dzaky | 123240105 |
| Nicolaus Narindra Lianto | 123240100 |
| Abyaz Affanzaky Shanahan | 123240130 |
