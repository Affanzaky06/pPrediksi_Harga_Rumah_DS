# Revisi Hyperparameter Tuning 5 Kandidat Model & Sinkronisasi End-to-End Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Melakukan hyperparameter tuning (`GridSearchCV` + `GroupKFold` 5-fold dengan opsi nilai `None` di setiap `param_grid`) pada seluruh 5 kandidat model regresi beserta justifikasi lengkapnya di `prediksi.ipynb`, lalu menyinkronkan model pemenang, metrik evaluasi, artefak `.joblib`, `src/house_price/`, `main.py`, tampilan Streamlit (`app.py`, `app_pages/`), `notebooks/04_modeling_evaluation.ipynb`, dan `README.md` hingga 100% bebas celah dan bebas inkonsistensi.

**Architecture:** `prediksi.ipynb` menjadi sumber kebenaran utama (*single source of truth*) yang menjalankan pipeline mandiri dari raw CSV hingga ekspor `model_artifacts/model_prediksi_rumah_jogja.joblib`. Paket modular `src/house_price/` (`config.py`, `models/train.py`), CLI `main.py`, dan aplikasi Streamlit (`app.py`, `app_pages/03_metodologi.py`) diselaraskan agar memakai konfigurasi hyperparameter pemenang (`BEST_RF_PARAMS`), tabel perbandingan 5 model (*Before vs After Tuning*), dan metrik evaluasi yang identik dengan `prediksi.ipynb`.

**Tech Stack:** Python 3.11 (`.dsvenv`), `pandas 3.0.5`, `numpy`, `scikit-learn 1.9.0`, `scipy`, `matplotlib`, `seaborn`, `altair`, `streamlit`, `joblib`, `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-25-revisi-tuning-5-model-design.md`

## Global Constraints

- **Python Environment:** Seluruh perintah Python, `unittest`, `main.py`, dan eksekusi ulang notebook WAJIB dijalankan menggunakan interpreter `D:\Dev\Python\Semester-5\DataScience\.dsvenv\Scripts\python.exe`.
- **Anti-CPU Thrashing (No Nested Parallelism):** Di dalam `GridSearchCV(..., n_jobs=-1)`, estimator `RandomForestRegressor` WAJIB diinisialisasi dengan `n_jobs=1` (bukan `n_jobs=-1`) untuk mencegah *nested multiprocessing oversubscription* ($N \times N$ thread) dan menghilangkan ratusan `UserWarning: sklearn.utils.parallel.delayed`.
- **Konsistensi Regex Pandas 3.0 (1.384 Baris Bersih):** Di `prediksi.ipynb` Sel 33, konversi `data["deskripsi"].astype(object).str.lower()` WAJIB digunakan (sama seperti `src/house_price/data/audit.py`) agar engine regex Python `re` menghasilkan tepat **1.384 baris bersih** (`1.107` data latih dan `277` data uji).
- **Konsistensi Kontrak Bundle `.joblib`:** Artefak `model_artifacts/model_prediksi_rumah_jogja.joblib` WAJIB memiliki tepat 6 kunci (`model`, `scaler`, `fitur_final`, `kolom_scale`, `median_train`, `kategori_wilayah`) dengan `model.n_jobs = 1` saat disimpan/dimuat untuk inferensi web bebas warning.
- **Konsistensi Angka di Seluruh Narasi:** Setiap angka di sel markdown `prediksi.ipynb`, `app.py`, `app_pages/03_metodologi.py`, dan `README.md` WAJIB cocok dengan output aktual:
  - Split: **1.384 listing bersih** = **1.107 data latih (80,0%)** dan **277 data uji (20,0%)** (Sleman 200, Bantul 44, Kota Yogyakarta 26, Kulon Progo/Gunung Kidul 7).
  - Pemenang: **Random Forest** (`n_estimators=300, max_depth=None, max_features='sqrt', min_samples_split=5, min_samples_leaf=2, bootstrap=True`), **CV RMSE = 0,1430**, **CV MAE = 0,1040**, **CV R² = 0,8353**.
  - Overfitting Check: **Train MSE = 0,0098** (`0.009783`), **Test MSE = 0,0162** (`0.016186`), **Train R² = 0,922** (`92,21%`), **Test R² = 0,871** (`87,14%`).
  - Evaluasi Data Uji (277 sampel): **R² = 87,14%**, **MAE = Rp 354.905.653**, **RMSE = Rp 840.362.311**, **MdAPE = 14,17%**, **MAPE = 21,56%**.

## Review Focus

1. **Nested `n_jobs=-1` di dalam `GridSearchCV`:** Jika `RandomForestRegressor(n_jobs=-1)` dipakai sebagai base estimator `GridSearchCV(n_jobs=-1)`, CPU akan hang dan muncul ratusan `UserWarning` dari `sklearn.utils.parallel`. Test di Task 1 memverifikasi tidak ada warning paralelisasi dan waktu eksekusi cepat.
2. **Ketidaksinkronan `judul_asli` di `prediksi.ipynb` Sel 33 pada Pandas 3.0:** Tanpa `.astype(object)`, `df_clean` di `prediksi.ipynb` menjadi 1.383 baris (bukan 1.384 baris). Verifikasi di Task 2 memastikan `len(df_clean) == 1384`, `len(train) == 1107`, dan `len(test) == 277`.
3. **Nilai `None` Hilang dari `param_grid` atau Tabel Dokumentasi:** Penguji meminta eksplisit kondisi `None` di setiap model. Task 1 dan Task 2 memastikan `None` ada di `param_grid` (`Ridge.max_iter`, `RandomForest.max_depth`, `RandomForest.max_features`, `GradientBoosting.max_depth`, `GradientBoosting.max_features`) dan penjelasan padanan `None` tertulis untuk `LinearRegression` dan `SVR`.
4. **Hardcoded String Lama di UI Streamlit (`100 trees`, `max_depth=15`, `max_features=0.5`, `274 data uji`):** Task 3 memeriksa dan memperbarui seluruh badge, diagram Mermaid, dan tabel spesifikasi di `app.py` serta `app_pages/03_metodologi.py`.
5. **Selisih Angka antara `prediksi.ipynb` dan `README.md`:** Task 4 menyinkronkan setiap tabel dan paragraf di `README.md` dan `notebooks/04_modeling_evaluation.ipynb` dengan output `prediksi.ipynb`.

---

## File Structure

- **Modify:** `src/house_price/config.py` — Memperbarui `BEST_RF_PARAMS` (`n_estimators=300, max_depth=None, max_features='sqrt', min_samples_leaf=2, min_samples_split=5, bootstrap=True`) dan `DEFAULT_MDAPE_RATE = 0.1417`.
- **Modify:** `src/house_price/models/train.py` — Menambahkan `PARAM_GRIDS_5_MODELS` (dengan opsi `None`), `BEST_TUNED_PARAMS_5_MODELS`, fungsi `tune_all_candidate_models()`, serta memperbarui `benchmark_models()` agar mengembalikan perbandingan *Baseline Default vs Tuned* untuk ke-5 model (bebas nested `n_jobs`).
- **Modify:** `main.py` — Memastikan opsi `--grid-search` dan mode default sama-sama menghasilkan model terbaik yang sinkron dengan `prediksi.ipynb`.
- **Modify:** `tests/test_modeling_and_predict.py` — Menambahkan unit test untuk memastikan `PARAM_GRIDS_5_MODELS` memuat nilai `None`, `benchmark_models()` mengembalikan 5 model terurut dengan `Random Forest` di peringkat 1 (`CV RMSE < 0.144`), dan metrik test set sesuai (`R² > 0.87`, `MdAPE < 14.5`).
- **Modify:** `prediksi.ipynb` — Memperbaiki Sel 33 (`.astype(object)`), merombak Sel 89–98 (tuning 5 kandidat model dengan `None`, tabel perbandingan Baseline vs Tuned, alasan setiap pilihan hyperparameter & algoritma, cek overfitting, learning curve), memperbarui narasi Sel 70, 101, 104, 106, 108, 110, 111, dan mengeksekusi ulang seluruh sel dari awal sampai akhir (*Restart & Run All*) hingga menyimpan `model_artifacts/model_prediksi_rumah_jogja.joblib`.
- **Modify:** `app.py` & `app_pages/03_metodologi.py` — Memperbarui badge jumlah pohon (`300 trees`), tabel perbandingan benchmark 5 model hasil tuning, diagram Mermaid CRISP-DM, dan tabel spesifikasi teknis pipeline.
- **Modify:** `notebooks/04_modeling_evaluation.ipynb` & `README.md` — Menyinkronkan output notebook modular `04` dan seluruh angka di `README.md`.

---

### Task 1: Sinkronisasi Konfigurasi, Fungsi Tuning 5 Model, dan Unit Test di `src/house_price/` & `main.py`

**Files:**
- Modify: `src/house_price/config.py:118-129`
- Modify: `src/house_price/models/train.py:40-138`
- Modify: `main.py:43-54`
- Test: `tests/test_modeling_and_predict.py`

**Interfaces:**
- Consumes: `PreprocessedData` dari `house_price.features.preprocessing`
- Produces:
  - `house_price.config.BEST_RF_PARAMS: dict[str, object]`
  - `house_price.config.DEFAULT_MDAPE_RATE: float` (`0.1417`)
  - `house_price.models.train.PARAM_GRIDS_5_MODELS: dict[str, dict[str, list[object]]]`
  - `house_price.models.train.BEST_TUNED_PARAMS_5_MODELS: dict[str, dict[str, object]]`
  - `house_price.models.train.tune_all_candidate_models(X_train_final, y_train, groups, random_state=RANDOM_STATE) -> tuple[pd.DataFrame, dict[str, object]]`
  - `house_price.models.train.benchmark_models(X_train_final, y_train, groups, random_state=RANDOM_STATE) -> pd.DataFrame` (mengembalikan kolom `["Model", "Baseline CV RMSE (log)", "CV RMSE (log)", "CV MAE (log)", "CV R2", "Best Hyperparameters"]`)

- [ ] **Step 1: Tambahkan unit test yang menguji keberadaan opsi `None` di `PARAM_GRIDS_5_MODELS`, hasil `benchmark_models()`, dan metrik `BEST_RF_PARAMS` di `tests/test_modeling_and_predict.py`**

Tambahkan import `PARAM_GRIDS_5_MODELS`, `BEST_TUNED_PARAMS_5_MODELS`, `benchmark_models` dari `house_price.models.train` dan tambahkan method test berikut di dalam `TestModelingAndPredict`:

```python
    def test_param_grids_include_none_and_benchmark_tuned_models(self):
        self.assertEqual(
            set(PARAM_GRIDS_5_MODELS.keys()),
            {"Linear Regression", "Ridge Regression", "SVR (RBF)", "Random Forest", "Gradient Boosting"},
        )
        self.assertIn(None, PARAM_GRIDS_5_MODELS["Ridge Regression"]["max_iter"])
        self.assertIn(None, PARAM_GRIDS_5_MODELS["Random Forest"]["max_depth"])
        self.assertIn(None, PARAM_GRIDS_5_MODELS["Random Forest"]["max_features"])
        self.assertIn(None, PARAM_GRIDS_5_MODELS["Gradient Boosting"]["max_depth"])
        self.assertIn(None, PARAM_GRIDS_5_MODELS["Gradient Boosting"]["max_features"])

        groups = self.prep.train_df["deskripsi"].str.lower()
        df_bench = benchmark_models(self.prep.X_train_final, self.prep.y_train, groups=groups)
        self.assertEqual(len(df_bench), 5)
        self.assertEqual(df_bench.iloc[0]["Model"], "Random Forest")
        self.assertLess(float(df_bench.iloc[0]["CV RMSE (log)"]), 0.144)
        self.assertIn("Baseline CV RMSE (log)", df_bench.columns)
        self.assertIn("Best Hyperparameters", df_bench.columns)
```

Dan di `test_evaluate_metrics_and_wilayah_breakdown`, perbarui asersi untuk memverifikasi peningkatan performa model baru:
```python
        self.assertAlmostEqual(metrics["r2"], 0.8714, places=3)
        self.assertAlmostEqual(metrics["mdape"], 14.17, places=1)
```

- [ ] **Step 2: Jalankan test untuk memverifikasi kegagalan awal (RED)**

Run: `D:\Dev\Python\Semester-5\DataScience\.dsvenv\Scripts\python.exe -m unittest tests/test_modeling_and_predict.py -v`
Expected: FAIL / ImportError karena `PARAM_GRIDS_5_MODELS` belum didefinisikan dan `BEST_RF_PARAMS` masih memakai parameter lama (`R2 = 0.8698`, `MdAPE = 14.71`).

- [ ] **Step 3: Perbarui `src/house_price/config.py`, `src/house_price/models/train.py`, dan `main.py`**

1. Di `src/house_price/config.py` (baris 118–129), ubah `BEST_RF_PARAMS` dan `DEFAULT_MDAPE_RATE` menjadi:
```python
BEST_RF_PARAMS: dict[str, object] = {
    "bootstrap": True,
    "max_depth": None,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "min_samples_split": 5,
    "n_estimators": 300,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

DEFAULT_MDAPE_RATE: float = 0.1417
```

2. Di `src/house_price/models/train.py`:
   - Perbarui `DEFAULT_RF_PARAM_GRID` dan tambahkan `PARAM_GRIDS_5_MODELS` serta `BEST_TUNED_PARAMS_5_MODELS`:
```python
DEFAULT_RF_PARAM_GRID: dict[str, list[object]] = {
    "n_estimators": [200, 300],
    "max_depth": [None, 10, 20],
    "max_features": [None, "sqrt", 0.5],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "bootstrap": [True],
}

PARAM_GRIDS_5_MODELS: dict[str, dict[str, list[object]]] = {
    "Linear Regression": {
        "fit_intercept": [True, False],
        "positive": [False, True],
    },
    "Ridge Regression": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "fit_intercept": [True, False],
        "max_iter": [None, 1000],
    },
    "SVR (RBF)": {
        "kernel": ["rbf", "linear"],
        "C": [0.1, 1.0, 10.0],
        "epsilon": [0.01, 0.05, 0.1, 0.2],
        "gamma": ["scale", "auto"],
    },
    "Random Forest": DEFAULT_RF_PARAM_GRID,
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [None, 3, 5],
        "max_features": [None, "sqrt", 0.5],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
    },
}

BEST_TUNED_PARAMS_5_MODELS: dict[str, dict[str, object]] = {
    "Linear Regression": {"fit_intercept": True, "positive": False},
    "Ridge Regression": {"alpha": 1.0, "fit_intercept": True, "max_iter": None},
    "SVR (RBF)": {"C": 1.0, "epsilon": 0.01, "gamma": "auto", "kernel": "rbf"},
    "Random Forest": {
        "bootstrap": True,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "min_samples_split": 5,
        "n_estimators": 300,
    },
    "Gradient Boosting": {
        "learning_rate": 0.03,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 4,
        "min_samples_split": 10,
        "n_estimators": 100,
    },
}
```
   - Perbarui `benchmark_models()` agar mengevaluasi kondisi **Baseline (Default)** dan **Tuned (`BEST_TUNED_PARAMS_5_MODELS`)** menggunakan `GroupKFold(n_splits=5)` dengan `n_jobs=1` pada estimator agar cepat (~1 detik) dan deterministik untuk halaman Metodologi Streamlit, serta tambahkan fungsi `tune_all_candidate_models()` untuk menjalankan `GridSearchCV` penuh pada ke-5 model:
```python
def _build_default_candidates(random_state: int = RANDOM_STATE) -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=random_state),
        "SVR (RBF)": SVR(),
        "Random Forest": RandomForestRegressor(random_state=random_state, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }


def _build_tuned_candidates(random_state: int = RANDOM_STATE) -> dict[str, object]:
    rf_params = dict(BEST_TUNED_PARAMS_5_MODELS["Random Forest"])
    rf_params.update({"random_state": random_state, "n_jobs": 1})
    gbr_params = dict(BEST_TUNED_PARAMS_5_MODELS["Gradient Boosting"])
    gbr_params["random_state"] = random_state
    ridge_params = dict(BEST_TUNED_PARAMS_5_MODELS["Ridge Regression"])
    ridge_params["random_state"] = random_state
    return {
        "Linear Regression": LinearRegression(**BEST_TUNED_PARAMS_5_MODELS["Linear Regression"]),
        "Ridge Regression": Ridge(**ridge_params),
        "SVR (RBF)": SVR(**BEST_TUNED_PARAMS_5_MODELS["SVR (RBF)"]),
        "Random Forest": RandomForestRegressor(**rf_params),
        "Gradient Boosting": GradientBoostingRegressor(**gbr_params),
    }


def benchmark_models(
    X_train_final: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Membandingkan 5 kandidat model (Baseline Default vs Setelah Hyperparameter Tuning) dengan GroupKFold 5-fold CV."""
    defaults = _build_default_candidates(random_state=random_state)
    tuned = _build_tuned_candidates(random_state=random_state)
    cv = GroupKFold(n_splits=5)
    hasil: list[dict[str, object]] = []
    for nama in defaults:
        base_scores = cross_validate(
            defaults[nama],
            X_train_final,
            y_train,
            groups=groups,
            cv=cv,
            scoring=["neg_root_mean_squared_error"],
        )
        tuned_scores = cross_validate(
            tuned[nama],
            X_train_final,
            y_train,
            groups=groups,
            cv=cv,
            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
        )
        base_rmse = -base_scores["test_neg_root_mean_squared_error"].mean()
        tuned_rmse = -tuned_scores["test_neg_root_mean_squared_error"].mean()
        params_str = ", ".join(f"{k}={v!r}" for k, v in BEST_TUNED_PARAMS_5_MODELS[nama].items())
        hasil.append({
            "Model": nama,
            "Baseline CV RMSE (log)": float(base_rmse),
            "CV RMSE (log)": float(tuned_rmse),
            "Delta RMSE": float(tuned_rmse - base_rmse),
            "CV MAE (log)": float(-tuned_scores["test_neg_mean_absolute_error"].mean()),
            "CV R2": float(tuned_scores["test_r2"].mean()),
            "Best Hyperparameters": params_str,
        })
    return pd.DataFrame(hasil).sort_values(by="CV RMSE (log)").reset_index(drop=True)
```
   - Di `run_rf_grid_search()`, ubah `base_rf = RandomForestRegressor(random_state=random_state, n_jobs=1)` agar tidak terjadi nested parallelism saat `GridSearchCV(..., n_jobs=-1)` dijalankan.
   - Tambahkan fungsi `tune_all_candidate_models()` di `src/house_price/models/train.py` yang menjalankan `GridSearchCV` pada ke-5 model di `PARAM_GRIDS_5_MODELS` dengan `GroupKFold(n_splits=5)` dan mengembalikan `(df_perbandingan, dict_best_estimators)`.

3. Di `main.py`, jalankan `python main.py` untuk memperbarui `model_artifacts/model_prediksi_rumah_jogja.joblib`.

- [ ] **Step 4: Jalankan `main.py` dan seluruh unit test untuk memverifikasi kelulusan (GREEN)**

Run: `D:\Dev\Python\Semester-5\DataScience\.dsvenv\Scripts\python.exe main.py && D:\Dev\Python\Semester-5\DataScience\.dsvenv\Scripts\python.exe -m unittest discover -s tests -v`
Expected: `main.py` mencetak `R2 Score: 87.14%`, `MAE Rupiah: Rp 354,905,653`, `RMSE Rupiah: Rp 840,362,311`, `MdAPE: 14.17%`, `MAPE: 21.56%`, dan seluruh unit test di `tests/` berstatus `OK`.

---

### Task 2: Revisi `prediksi.ipynb` (Tuning 5 Model + Kondisi `None` + Alasan Lengkap + Sinkronisasi Seluruh Narasi & Output)

**Files:**
- Modify: `prediksi.ipynb` (Sel 33, 70, 89, 90–98, 101, 104, 106, 108, 110, 111)

**Interfaces:**
- Consumes: `rumah123_yogya_unfiltered.csv`
- Produces:
  - Notebook `prediksi.ipynb` yang telah dieksekusi penuh dari Sel 0 hingga آخر (*Restart & Run All*) tanpa error dan tanpa `UserWarning`.
  - Artefak `model_artifacts/model_prediksi_rumah_jogja.joblib` dari eksekusi Sel terakhir `prediksi.ipynb`.

- [ ] **Step 1: Perbaiki Sel 33 di `prediksi.ipynb` agar regex kompatibel dengan Pandas 3.0 dan menghasilkan tepat 1.384 baris bersih**

Di Sel 33 (code), ubah baris pertama:
```python
# Sebelum:
judul_asli = data["deskripsi"].str.lower()

# Sesudah:
judul_asli = data["deskripsi"].astype(object).str.lower()
```

- [ ] **Step 2: Perbarui narasi jumlah split di Sel 70/89 dan rencana Modelling di Sel 89 `prediksi.ipynb`**

Di Sel 89 (markdown `## 9. Ringkasan Data Preparation dan Transisi ke Modelling`), pastikan jumlah baris data latih dan uji tertulis **`1.107 data latih, 277 data uji`**, dan ubah sub-bagian **Rencana Modelling (bagian 10)** menjadi:
```markdown
**Rencana Modelling (bagian 10):**
1. Mendefinisikan 5 kandidat model regresi yang mewakili keluarga algoritma berbeda: Linear Regression, Ridge Regression, SVR, Random Forest, dan Gradient Boosting, lengkap dengan alasan pemilihannya.
2. Melakukan *hyperparameter tuning* (`GridSearchCV` dengan `GroupKFold` 5-fold pada data latih) untuk **seluruh 5 kandidat model**, dengan menyertakan kondisi **`None`** (tanpa batas / *default* tak-terbatas) pada parameter setiap model serta alasan teknis pemilihan setiap nilai di dalam `param_grid`.
3. Membandingkan kinerja kelima model sebelum (*baseline default*) dan sesudah *tuning* berdasarkan CV RMSE, CV MAE, dan CV R² pada skala $\log_{10}$, menganalisis alasan terpilihnya hyperparameter terbaik di tiap model, lalu memilih 1 model terbaik keseluruhan.
4. Melatih model pemenang pada seluruh data latih, memeriksa potensi *overfitting* serta *learning curve*, lalu mengevaluasinya pada data uji (bagian 11) dan menyimpulkan temuan (bagian 12).
```

- [ ] **Step 3: Rombak Bagian 10 (`## 10. Modelling`, Sel 90–98) di `prediksi.ipynb` untuk menjalankan Hyperparameter Tuning pada ke-5 Model (termasuk kondisi `None`) beserta alasan lengkapnya**

Susun Bagian 10 di `prediksi.ipynb` dengan struktur sel berikut:

1. **Sel 90 (Markdown — `## 10. Modelling` & `### 10.1 Kandidat Model, Ruang Hyperparameter, dan Kondisi None`):**
   - Tabel alasan pemilihan 5 kandidat model (`Linear Regression`, `Ridge Regression`, `SVR`, `Random Forest`, `Gradient Boosting`).
   - Penjelasan skema validasi `GroupKFold(n_splits=5)` berbasis grup `train["deskripsi"].str.lower()` dan alasan penggunaan RMSE ($\log_{10}$) sebagai metrik utama `refit`.
   - **Tabel Lengkap Ruang Hyperparameter (`param_grid`), Peran Kondisi `None`, dan Alasan Pemilihan Nilai untuk Kelima Model** (mengacu persis pada Bagian 3 di `docs/superpowers/specs/2026-11-25-revisi-tuning-5-model-design.md`):
     - `LinearRegression`: `fit_intercept: [True, False]`, `positive: [False, True]` (jelaskan bahwa OLS analitik tidak memiliki parameter regularisasi/struktur bertipe `None`; kondisi tanpa batasan/*unconstrained* diwakili oleh `positive=False` dan `fit_intercept=True`).
     - `Ridge`: `alpha: [0.01, 0.1, 1.0, 10.0, 100.0]`, `fit_intercept: [True, False]`, `max_iter: [None, 1000]` (jelaskan mengapa `max_iter=None` diuji: membiarkan solver menentukan konvergensi otomatis tanpa pemotongan iterasi manual, serta rentang `alpha` 5 orde besaran dari hampir tanpa penalti hingga penalti kuat).
     - `SVR`: `kernel: ['rbf', 'linear']`, `C: [0.1, 1.0, 10.0]`, `epsilon: [0.01, 0.05, 0.1, 0.2]`, `gamma: ['scale', 'auto']` (jelaskan alasan setiap nilai dan bahwa pada `SVR`, kondisi tanpa batas iterasi diwakili oleh default `max_iter=-1` yang ekuivalen dengan `None`).
     - `RandomForestRegressor`: `n_estimators: [200, 300]`, `max_depth: [None, 10, 20]`, `max_features: [None, 'sqrt', 0.5]`, `min_samples_split: [5, 10]`, `min_samples_leaf: [2, 4]`, `bootstrap: [True]` (jelaskan secara matematis untuk 6 fitur final: `max_features=None` memakai seluruh **6 fitur** [100%], `0.5` memakai **3 fitur** [50%], dan `'sqrt'` memakai $\lfloor\sqrt{6}\rfloor =$ **2 fitur** [33,3%] per *split* untuk mendekorrelasi pohon; serta `max_depth=None` menguji pohon tumbuh penuh hingga tertahan `min_samples_leaf`/`min_samples_split`).
     - `GradientBoostingRegressor`: `n_estimators: [100, 200]`, `learning_rate: [0.03, 0.05, 0.1]`, `max_depth: [None, 3, 5]`, `max_features: [None, 'sqrt', 0.5]`, `min_samples_split: [5, 10]`, `min_samples_leaf: [2, 4]` (jelaskan kontras antara `max_depth=None` [pohon dalam] vs *weak learner* dangkal `3` dan `5`, serta *trade-off* antara `learning_rate` dan `n_estimators`).

2. **Sel 91 (Code — Eksekusi Baseline Default & `GridSearchCV` untuk Kelima Kandidat Model):**
   Gunakan kode bersih berikut (dengan `n_jobs=1` pada base estimator agar tidak terjadi nested parallelism):
```python
import time
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_validate

grup_cv = train["deskripsi"].str.lower()
cv = GroupKFold(n_splits=5)

# 1. Kandidat Model Baseline (Default bawaan scikit-learn) & Ruang Hyperparameter (dengan kondisi None)
konfigurasi_model = {
    "Linear Regression": (
        LinearRegression(),
        {
            "fit_intercept": [True, False],
            "positive": [False, True],
        },
    ),
    "Ridge Regression": (
        Ridge(random_state=RANDOM_STATE),
        {
            "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "fit_intercept": [True, False],
            "max_iter": [None, 1000],
        },
    ),
    "SVR (RBF)": (
        SVR(),
        {
            "kernel": ["rbf", "linear"],
            "C": [0.1, 1.0, 10.0],
            "epsilon": [0.01, 0.05, 0.1, 0.2],
            "gamma": ["scale", "auto"],
        },
    ),
    "Random Forest": (
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
        {
            "n_estimators": [200, 300],
            "max_depth": [None, 10, 20],
            "max_features": [None, "sqrt", 0.5],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [2, 4],
            "bootstrap": [True],
        },
    ),
    "Gradient Boosting": (
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        {
            "n_estimators": [100, 200],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [None, 3, 5],
            "max_features": [None, "sqrt", 0.5],
            "min_samples_split": [5, 10],
            "min_samples_leaf": [2, 4],
        },
    ),
}

hasil_perbandingan = []
grid_results = {}
start_total = time.time()

for nama, (estimator_dasar, param_grid) in konfigurasi_model.items():
    t0 = time.time()
    # Evaluasi Baseline (Default sebelum tuning)
    cv_base = cross_validate(
        estimator_dasar,
        X_train_final,
        y_train,
        groups=grup_cv,
        cv=cv,
        scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
        n_jobs=-1,
    )
    base_rmse = -cv_base["test_neg_root_mean_squared_error"].mean()

    # Hyperparameter Tuning dengan GridSearchCV + GroupKFold
    gs = GridSearchCV(
        estimator=estimator_dasar,
        param_grid=param_grid,
        cv=cv,
        scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
        refit="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    gs.fit(X_train_final, y_train, groups=grup_cv)
    grid_results[nama] = gs

    idx = gs.best_index_
    tuned_rmse = -gs.cv_results_["mean_test_neg_root_mean_squared_error"][idx]
    tuned_mae = -gs.cv_results_["mean_test_neg_mean_absolute_error"][idx]
    tuned_r2 = gs.cv_results_["mean_test_r2"][idx]
    durasi = time.time() - t0

    hasil_perbandingan.append({
        "Model": nama,
        "Kombinasi Diuji": len(gs.cv_results_["params"]),
        "Baseline CV RMSE (log)": base_rmse,
        "Tuned CV RMSE (log)": tuned_rmse,
        "Delta RMSE": tuned_rmse - base_rmse,
        "Tuned CV MAE (log)": tuned_mae,
        "Tuned CV R2": tuned_r2,
        "Waktu (s)": round(durasi, 1),
        "Best Hyperparameters": gs.best_params_,
    })
    print(f"[{nama}] ({len(gs.cv_results_['params'])} kombinasi, {durasi:.1f}s) "
          f"| Baseline RMSE: {base_rmse:.4f} -> Tuned RMSE: {tuned_rmse:.4f} "
          f"| Best: {gs.best_params_}")

print(f"\nTotal waktu tuning 5 model: {time.time() - start_total:.1f} detik")

df_hasil = pd.DataFrame(hasil_perbandingan).sort_values(by="Tuned CV RMSE (log)").reset_index(drop=True)
display(df_hasil.style.format({
    "Baseline CV RMSE (log)": "{:.4f}",
    "Tuned CV RMSE (log)": "{:.4f}",
    "Delta RMSE": "{:+.4f}",
    "Tuned CV MAE (log)": "{:.4f}",
    "Tuned CV R2": "{:.4f}",
    "Waktu (s)": "{:.1f}",
}))

grid_search = grid_results["Random Forest"]
best_rf_grid = grid_search.best_estimator_
```

3. **Sel 92–94 (Markdown — `### 10.2 Analisis Hasil Tuning Kelima Model dan Alasan Pemilihan Model Terbaik`):**
   - Sajian tabel ringkasan dan alasan mengapa masing-masing parameter menang pada tiap model:
     1. **Random Forest (`CV RMSE 0,1430`, turun `0,0082` dari baseline `0,1512`; `CV R² 0,8353`):**
        - `max_depth=None` terpilih bersama `min_samples_split=5` dan `min_samples_leaf=2`: pohon tidak perlu dipangkas dengan kedalaman kaku (`10` atau `20`) karena ukuran minimum daun (`2` rumah) dan pemecahan simpul (`5` rumah) sudah mencegah daun menghafal satu rumah tunggal.
        - `max_features='sqrt'` (2 dari 6 fitur per *split*) mengalahkan `None` (semua 6 fitur) dan `0.5` (3 fitur): jika `max_features=None` dipakai, hampir setiap pohon selalu memilih `log_luas_bangunan` di simpul akar sehingga pohon-pohon menjadi sangat mirip (berkorelasi tinggi). Dengan `'sqrt'`, tiap *split* hanya melihat 2 fitur acak, sehingga pohon-pohon saling independen (*dekorrelasi*) dan rata-rata ensemble menjadi jauh lebih akurat.
        - `n_estimators=300` memberikan rata-rata yang sedikit lebih stabil dibanding `200`.
     2. **Gradient Boosting (`CV RMSE 0,1456`, turun `0,0067` dari baseline `0,1523`; `CV R² 0,8292`):**
        - `learning_rate=0.03` dengan `n_estimators=100`, `max_depth=None`, `max_features='sqrt'`, `min_samples_split=10`, dan `min_samples_leaf=4` terpilih: laju belajar yang lambat (`0.03`) dikombinasikan dengan syarat daun yang ketat (`min_samples_leaf=4`, `min_samples_split=10`) dan subsampling fitur (`'sqrt'`) menahan *boosting* agar tidak *overfitting* meskipun `max_depth=None`.
     3. **SVR (`CV RMSE 0,1516`, turun `0,0006` dari baseline `0,1522`; `CV R² 0,8138`):**
        - `kernel='rbf'` jauh mengungguli `'linear'`, membuktikan adanya kelengkungan non-linear pada segmen harga/luas ekstrem. `C=1.0`, `gamma='auto'` ($1/6$), dan `epsilon=0.01` mempersempit tabung toleransi error sehingga kurva regresi lebih rapat pada data menengah.
     4. **Ridge Regression (`CV RMSE 0,1640`; `CV R² 0,7826`) & Linear Regression (`CV RMSE 0,1640`; `CV R² 0,7826`):**
        - `fit_intercept=True` wajib karena target $\log_{10}(\text{harga})$ tidak berpusat di nol (berpusat di sekitar $\approx 9,0$ atau Rp 1 miliar untuk baseline wilayah Bantul).
        - Pada Linear Regression, `positive=False` (kondisi tanpa batasan tanda) menang karena dummy `wilayah_Kulon Progo/Gunung Kidul` memiliki harga di bawah baseline Bantul (membutuhkan koefisien negatif).
        - Pada Ridge, `alpha=1.0` dan `max_iter=None` memberikan skor yang identik dengan OLS (`0,1640`) karena multikolinearitas sudah dieliminasi pada tahap seleksi fitur ($\text{VIF} < 5$).
   - **Alasan Pemilihan `Random Forest` sebagai Model Final:**
     1. Skor validasi silang terbaik di seluruh metrik setelah semua kandidat di-*tuning* secara adil (**CV RMSE `0,1430`**, **CV MAE `0,1040`**, **CV R² `83,53%`**).
     2. Penurunan error terbesar dari proses *tuning* ($\Delta \text{RMSE} = -0,0082$).
     3. Mekanisme *bagging* lebih tahan terhadap *outlier* harga rumah mewah dibandingkan *boosting* sekuensial, serta menghasilkan *feature importance* yang mudah diinterpretasikan oleh agen properti.

4. **Sel 95–98 (Training Model Final, Cek Overfitting, & Learning Curve):**
   - Sel 95 (code) melatih `model_terbaik = RandomForestRegressor(**grid_search.best_params_, n_jobs=-1, random_state=RANDOM_STATE)` pada `X_train_final, y_train`.
   - Sel 96 (markdown) menampilkan tabel perbandingan MSE & R² latih vs uji yang baru:
     - **MSE (log10):** Data latih `0,0098` | Data uji `0,0162` | Selisih `0,0064`
     - **R²:** Data latih `0,922` | Data uji `0,871` | Selisih `0,051`
     - Interpretasi: selisih R² latih vs uji hanya `5,1%` (`0,922` vs `0,871` pada 1.107 data latih dan 277 data uji), jauh lebih sehat dan menunjukkan *overfitting* yang sangat ringan berkat regularisasi `max_features='sqrt'` dan `min_samples_leaf=2`.
   - Sel 97 (code) & Sel 98 (markdown) menampilkan *learning curve* (`GroupKFold` 5-fold, dengan `n_jobs=1` pada estimator atau `learning_curve` agar bebas warning):
     - Error latih mendatar di sekitar `0,010`, sedangkan error validasi turun dari `±0,028` ke `0,0207` (jarak menyempit menjadi `±0,011`).

- [ ] **Step 4: Perbarui Seluruh Narasi Evaluasi (Bagian 11, Sel 101, 104, 106, 108, 110) dan Konklusi (Bagian 12, Sel 111) di `prediksi.ipynb`**

Perbarui angka pada sel-sel markdown berikut agar 100% sesuai dengan output kode:
1. **Sel 101 (Tabel & Interpretasi Metrik Global pada 277 Data Uji):**
   - **R²:** `87,14%` (Target ≥ 75% → **Tercapai**)
   - **MAPE:** `21,56%` (Target ≤ 20% → **Belum tercapai**, namun tinggal berselisih 1,56 poin persentase)
   - **MdAPE:** `14,17%` (pada rumah tipikal, prediksi meleset sekitar 14,2%)
   - **MAE:** `Rp 354,9 juta` (`Rp 354.905.653`)
   - **RMSE:** `Rp 840,4 juta` (`Rp 840.362.311`)
2. **Sel 104 (Harga Aktual vs Prediksi):**
   - R² = `87,14%` (`0,87`).
3. **Sel 106 (Faktor yang Paling Memengaruhi Harga / Feature Importance):**
   - Urutan tetap konsisten: **Luas bangunan** (`43,75%`), diikuti **luas tanah** (`29,48%`) dan **jumlah kamar mandi** (`24,14%`), sedangkan ketiga dummy **wilayah** menyumbang total `2,63%` (`wilayah_Kulon Progo/Gunung Kidul` `1,19%`, `wilayah_Kota Yogyakarta` `0,74%`, `wilayah_Sleman` `0,70%`). Jelaskan bahwa penggunaan `max_features='sqrt'` membuat pembagian peran antara luas bangunan, luas tanah, dan kamar mandi lebih proporsional tanpa mengubah urutan peringkatnya.
4. **Sel 108 (Analisis Residual):**
   - Sebutkan pencilan aktual dari data uji: satu rumah mewah dengan harga aktual **Rp 16,00 miliar** yang diprediksi sekitar **Rp 6,84 miliar** (residual positif **+Rp 9,16 miliar**) dan satu rumah dengan harga aktual **Rp 4,40 miliar** yang diprediksi sekitar **Rp 6,01 miliar** (residual negatif **−Rp 1,61 miliar**). Pencilan inilah yang menarik RMSE (`Rp 840,4 juta`) di atas MAE (`Rp 354,9 juta`).
5. **Sel 110 (Evaluasi per Wilayah pada 277 Data Uji):**
   - Tabel hasil per wilayah:
     | Wilayah | Sampel | MdAPE | MAPE | MAE |
     |---|---|---|---|---|
     | Kulon Progo/Gunung Kidul | 7 | 10,89% | 7,27% | Rp 21,0 juta |
     | Sleman | 200 | 13,39% | 19,08% | Rp 348,4 juta |
     | Kota Yogyakarta | 26 | 19,55% | 32,57% | Rp 718,4 juta |
     | Bantul | 44 | 19,64% | 28,61% | Rp 222,9 juta |
   - Temuan: seluruh wilayah kini memiliki **MdAPE di bawah 20%**! **Sleman (72% data uji, 200 sampel)** menjadi segmen paling solid dengan MdAPE `13,39%` dan MAPE `19,08%` (sudah memenuhi target ≤ 20%). **Kota Yogyakarta (26 sampel)** dan **Bantul (44 sampel)** memiliki MdAPE sekitar `19,6%`, namun Kota Yogyakarta memiliki MAPE (`32,57%`) dan MAE (`Rp 718,4 juta`) tertinggi karena variasi harga lokasi mikro di pusat kota. **Kulon Progo/Gunung Kidul** hanya memiliki `7` sampel uji sehingga tidak dapat digeneralisasi.
6. **Sel 111 (`## 12. Konklusi`):**
   - Perbarui jumlah data uji menjadi **277 data uji** (dan **1.107 data latih** dari **1.384 listing bersih**).
   - Perbarui tabel metrik di Jawaban #2 (`R² 87,14%`, `MdAPE 14,17%`, `MAPE 21,56%`, `MAE Rp 354,9 juta`, `RMSE Rp 840,4 juta`, serta angka MdAPE wilayah: Sleman `13,39%`, Kota Yogyakarta `19,55%`, Bantul `19,64%`, Kulon Progo/Gunung Kidul `10,89%` pada 7 sampel).
   - Perbarui Jawaban #3: kelima kandidat model telah di-*tuning* seluruhnya dengan `GridSearchCV` + `GroupKFold` 5-fold (termasuk menguji kondisi `None` pada setiap model). **Random Forest** (`n_estimators=300, max_depth=None, max_features='sqrt', min_samples_split=5, min_samples_leaf=2, bootstrap=True`) keluar sebagai pemenang dengan CV RMSE `0,1430` (unggul atas Gradient Boosting `0,1456`, SVR `0,1516`, Ridge `0,1640`, dan Linear Regression `0,1640`).
7. **Sel 113 (Penyimpanan Bundle `.joblib`):**
   - Pastikan sebelum `joblib.dump(bundle_model, 'model_artifacts/model_prediksi_rumah_jogja.joblib')`, `model_terbaik.n_jobs = 1` disetel agar artefak yang disimpan dari `prediksi.ipynb` 100% kompatibel dengan pengujian bebas-warning di Streamlit.

- [ ] **Step 5: Eksekusi ulang seluruh `prediksi.ipynb` dari awal sampai akhir dan verifikasi outputnya**

Run: `D:\Dev\Python\Semester-5\DataScience\.dsvenv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace prediksi.ipynb`
*(Jika `jupyter nbconvert` tidak terpasang, jalankan script Python pengeksekusi sel berurutan yang memperbarui `outputs` dan `execution_count` di `prediksi.ipynb`)*.
Expected: Seluruh sel `prediksi.ipynb` tereksekusi bersih tanpa error, menghasilkan `model_artifacts/model_prediksi_rumah_jogja.joblib` dengan `R² = 87.14%`.

---

### Task 3: Sinkronisasi Tampilan Aplikasi Web Streamlit (`app.py` & `app_pages/03_metodologi.py`)

**Files:**
- Modify: `app.py:91-109`
- Modify: `app_pages/03_metodologi.py:79-159`
- Modify: `src/house_price/visualization/web_charts.py:184-211`

**Interfaces:**
- Consumes: `get_model_bundle()`, `get_pipeline_evaluation()`, `get_model_benchmark()` dari `house_price.web.loaders`
- Produces: Tampilan Streamlit yang secara dinamis menampilkan parameter model terlatih (`300 trees, max_depth=None, max_features='sqrt'`), metrik baru (`R² Test: 87.14%`, `MdAPE: 14.17%`), dan tabel perbandingan 5 model hasil *hyperparameter tuning* (*Baseline vs Tuned* + *Best Hyperparameters*).

- [ ] **Step 1: Perbarui `app.py` agar menampilkan informasi jumlah pohon & parameter model secara dinamis dari `bundle["model"]`**

Di `app.py` (baris 91–109), ganti teks statis `"100 trees, tuned"` dengan pembacaan langsung dari `bundle["model"]`:
```python
n_trees = getattr(bundle["model"], "n_estimators", 300)
max_feat = getattr(bundle["model"], "max_features", "sqrt")
```
- Di sidebar (`st.caption` baris 93–96):
  ```python
  st.caption(
      f"Artefak: `{artifact_name}`\n"
      f"Algoritma: `RandomForestRegressor` ({n_trees} trees, max_features={max_feat!r}, max_depth=None)\n"
      "Fitur final: 6 fitur (log-luas, kamar mandi, dummy wilayah)"
  )
  ```
- Di badge header (baris 105):
  ```python
  st.badge(f"Random Forest ({n_trees} trees, tuned)", icon=":material/layers:", color="blue")
  ```

- [ ] **Step 2: Perbarui `app_pages/03_metodologi.py` dan `src/house_price/visualization/web_charts.py` untuk menampilkan perbandingan 5 model yang telah di-tuning**

1. Di `app_pages/03_metodologi.py`:
   - Pada bagian perbandingan 5 kandidat model (baris 80–100), ubah subheader menjadi:
     `st.subheader("Perbandingan 5 kandidat model setelah hyperparameter tuning (GroupKFold 5-Fold)", icon=":material/analytics:")`
   - Tambahkan `st.caption("Seluruh 5 kandidat model di-tuning menggunakan GridSearchCV + GroupKFold (5-fold) dengan menyertakan kondisi None di setiap ruang pencarian.")`
   - Konfigurasikan `st.dataframe(df_bench, ...)` dengan `column_config` untuk menampilkan `"Baseline CV RMSE (log)"`, `"CV RMSE (log)"` (label `"Tuned CV RMSE"`), `"Delta RMSE"`, `"CV MAE (log)"`, `"CV R2"`, dan `"Best Hyperparameters"`.
   - Pada diagram Mermaid (baris 104–113), ubah node `G` menjadi:
     `F --> G["Tuning 5 Model (GridSearchCV + Kondisi None) → Random Forest (300 trees, max_depth=None, max_features='sqrt')"]`
   - Pada tabel `spec_df` (baris 150–158), perbarui baris `"Algoritma model"` menjadi:
     `{"Komponen": "Pemilihan & tuning model", "Implementasi": "5 kandidat di-tuning via GridSearchCV (GroupKFold 5-fold, termasuk opsi None)"}` dan
     `{"Komponen": "Model final terpilih", "Implementasi": "RandomForestRegressor (n_estimators=300, max_depth=None, max_features='sqrt', min_samples_leaf=2, min_samples_split=5, bootstrap=True)"}`

2. Di `src/house_price/visualization/web_charts.py` (`make_benchmark_chart`), tambahkan `Baseline CV RMSE (log)` dan `Best Hyperparameters` ke dalam `tooltip` grafik Altair bila kolom tersebut tersedia di `df_bench`.

- [ ] **Step 3: Jalankan test integrasi Streamlit (`AppTest`) untuk memverifikasi halaman kalkulator, EDA, dan metodologi**

Run: `D:\Dev\Python\Semester-5\DataScience\.dsvenv\Scripts\python.exe -m unittest tests/test_modeling_and_predict.py -v`
Expected: PASS (`AppTest` merender seluruh 3 halaman tanpa exception dan seluruh grafik Altair terbentuk valid).

---

### Task 4: Sinkronisasi `README.md` dan `notebooks/04_modeling_evaluation.ipynb` serta Verifikasi Akhir

**Files:**
- Modify: `README.md:76, 132-154, 170, 178-184`
- Modify: `notebooks/04_modeling_evaluation.ipynb`

**Interfaces:**
- Consumes: Hasil evaluasi baru dari `prediksi.ipynb` dan `main.py`
- Produces: Dokumentasi `README.md` dan notebook pendukung `04_modeling_evaluation.ipynb` yang 100% selaras dengan `prediksi.ipynb`.

- [ ] **Step 1: Perbarui seluruh angka di `README.md` agar identik dengan `prediksi.ipynb`**

Perbarui bagian-bagian berikut di `README.md`:
1. **Baris 76 (Tabel Dataset):** `1.384 baris (68,5% dari data mentah): 1.107 data latih dan 277 data uji`.
2. **Baris 137–139 (Perbandingan model & Model final):**
   - Jelaskan bahwa seluruh **5 kandidat model regresi** (`Linear Regression`, `Ridge`, `SVR`, `Random Forest`, `Gradient Boosting`) di-*tuning* terlebih dahulu menggunakan `GridSearchCV` dengan `GroupKFold` (5-fold) dan menyertakan kondisi `None` pada ruang pencarian hyperparameter setiap model.
   - Cantumkan perbandingan CV RMSE setelah tuning: **Random Forest (`0,1430`)**, **Gradient Boosting (`0,1456`)**, **SVR (`0,1516`)**, **Ridge (`0,1640`)**, dan **Linear Regression (`0,1640`)**.
   - **Model final:** Random Forest (`n_estimators=300`, `max_depth=None`, `max_features='sqrt'`, `min_samples_leaf=2`, `min_samples_split=5`, `bootstrap=True`), hasil `GridSearchCV` dengan `GroupKFold` (CV RMSE `0,1430` pada skala log10, turun dari baseline `0,1512`).
3. **Baris 141–153 (Hasil pada data uji — 277 listing):**
   - `R²`: **87,14%** (Target ≥ 75% → Tercapai)
   - `MAE`: **Rp 354.905.653** (±Rp 354,9 juta)
   - `RMSE`: **Rp 840.362.311** (±Rp 840,4 juta)
   - `MdAPE`: **14,17%**
   - `MAPE`: **21,56%** (Target ≤ 20% → Belum tercapai)
   - **Kinerja per wilayah (MdAPE):** Sleman **13,39%** (200 sampel, MAPE 19,08%), Kota Yogyakarta **19,55%** (26 sampel, MAPE 32,57%), Bantul **19,64%** (44 sampel, MAPE 28,61%), dan Kulon Progo/Gunung Kidul **10,89%** (hanya 7 sampel sehingga tidak dapat disimpulkan).
4. **Baris 170 & 178–184 (Cara Menjalankan & Keterbatasan):**
   - Baris 170: `1.384 baris (1.107 latih dan 277 uji)`, `R² 87,14%, MAE Rp 354.905.653, MdAPE 14,17%`.
   - Baris 178: `Evaluasi Kulon Progo/Gunung Kidul hanya berdasar 7 sampel uji.`
   - Baris 181: `Model sedikit overfitting ringan (R² latih 0,92 vs uji 0,87) dan cenderung memprediksi terlalu rendah untuk rumah mahal.`
   - Baris 182: `Target MAPE ≤ 20% secara global belum tercapai (21,56%), meskipun pada wilayah Sleman sudah mencapai 19,08%.`

- [ ] **Step 2: Eksekusi ulang `notebooks/04_modeling_evaluation.ipynb` agar output selnya sinkron dengan `src/house_price/` yang baru**

Jalankan ulang sel-sel `notebooks/04_modeling_evaluation.ipynb` menggunakan interpreter `.dsvenv` sehingga tabel benchmark, metrik evaluasi (`87,14%`), dan evaluasi per wilayah di dalamnya ikut terbarui.

- [ ] **Step 3: Jalankan verifikasi menyeluruh (Tests + Audit Konsistensi Lintas Berkas)**

Run:
```powershell
D:\Dev\Python\Semester-5\DataScience\.dsvenv\Scripts\python.exe main.py
D:\Dev\Python\Semester-5\DataScience\.dsvenv\Scripts\python.exe -m unittest discover -s tests -v
```
Dan periksa dengan script audit cepat bahwa tidak ada lagi string angka lama (`85,77`, `422.640.860`, `964.431.107`, `15,57`, `23,23`, `0,1426`, `1.110`, `274`) yang tertinggal di `prediksi.ipynb`, `README.md`, `app.py`, `app_pages/`, maupun `src/house_price/`.
Expected: 0 kemunculan angka usang, `main.py` sukses, dan seluruh unit test `OK`.
