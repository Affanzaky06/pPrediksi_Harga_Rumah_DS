"""Preprocessing fitur: stratified group split, imputasi train-only, log10, encoding, scaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from house_price.config import (
    BASELINE_WILAYAH,
    FITUR_FINAL,
    KOLOM_SCALE,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    WILAYAH_MERGE_MAP,
)
from house_price.data.cleaning import load_or_build_clean_data


@dataclass
class PreprocessedData:
    """Kontainer hasil preprocessing lengkap untuk training, seleksi fitur, dan evaluasi."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    X_train_unscaled: pd.DataFrame
    X_test_unscaled: pd.DataFrame
    X_train_final: pd.DataFrame
    X_test_final: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: StandardScaler
    median_train: pd.Series
    kategori_wilayah: list[str]
    fitur_final: list[str]
    kolom_scale: list[str]


def add_wilayah_model(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Menambahkan kolom wilayah_model dengan menggabungkan kategori wilayah langka."""
    out = df_clean.copy()
    out["wilayah_model"] = out["wilayah"].replace(WILAYAH_MERGE_MAP)
    return out


def split_train_test(
    df_clean: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Membagi data menjadi train/test (80/20) terstratifikasi dan dikelompokkan per deskripsi."""
    df_ready = add_wilayah_model(df_clean) if "wilayah_model" not in df_clean.columns else df_clean
    grup_split = df_ready["deskripsi"].str.lower()
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    idx_train, idx_test = next(sgkf.split(df_ready, df_ready["wilayah_model"], groups=grup_split))
    return df_ready.iloc[idx_train].copy(), df_ready.iloc[idx_test].copy()


def impute_missing_from_train(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Mengisi missing kamar_tidur/kamar_mandi dengan median train dan carport dengan 0 + flag."""
    train_out = train.copy()
    test_out = test.copy()

    median_train = train_out[["kamar_tidur", "kamar_mandi"]].median()
    for d in (train_out, test_out):
        d[["kamar_tidur", "kamar_mandi"]] = d[["kamar_tidur", "kamar_mandi"]].fillna(median_train)
        d["carport_kosong"] = d["carport"].isna().astype(int)
        d["carport"] = d["carport"].fillna(0)

    return train_out, test_out, median_train


def build_feature_matrix(d: pd.DataFrame) -> pd.DataFrame:
    """Membangun matriks fitur kandidat (termasuk transformasi log10 luas)."""
    x = pd.DataFrame(index=d.index)
    x["log_luas_tanah"] = np.log10(d["luas_tanah"])
    x["log_luas_bangunan"] = np.log10(d["luas_bangunan"])
    x["kamar_tidur"] = d["kamar_tidur"]
    x["kamar_mandi"] = d["kamar_mandi"]
    x["carport"] = d["carport"]
    x["carport_kosong"] = d["carport_kosong"]
    x["wilayah"] = d["wilayah_model"]
    return x


def encode_wilayah(
    x: pd.DataFrame,
    kategori_wilayah: list[str],
    baseline: str = BASELINE_WILAYAH,
) -> pd.DataFrame:
    """Melakukan one-hot encoding pada kolom wilayah dengan drop baseline."""
    out = x.copy()
    out["wilayah"] = pd.Categorical(out["wilayah"], categories=kategori_wilayah)
    dummies = pd.get_dummies(out["wilayah"], prefix="wilayah", dtype=int).drop(
        columns=f"wilayah_{baseline}"
    )
    return pd.concat([out.drop(columns="wilayah"), dummies], axis=1)


def run_preprocessing_pipeline(
    df_clean: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> PreprocessedData:
    """Menjalankan seluruh tahap preprocessing tanpa data leakage."""
    train_split, test_split = split_train_test(df_clean, random_state=random_state)
    train_imp, test_imp, median_train = impute_missing_from_train(train_split, test_split)

    X_train_raw = build_feature_matrix(train_imp)
    X_test_raw = build_feature_matrix(test_imp)
    y_train = np.log10(train_imp["harga"])
    y_test = np.log10(test_imp["harga"])

    kategori_wilayah = sorted(train_imp["wilayah_model"].unique())
    X_train_unscaled = encode_wilayah(X_train_raw, kategori_wilayah=kategori_wilayah)
    X_test_unscaled = encode_wilayah(X_test_raw, kategori_wilayah=kategori_wilayah)

    scaler = StandardScaler().fit(X_train_unscaled[KOLOM_SCALE])
    X_train = X_train_unscaled.copy()
    X_test = X_test_unscaled.copy()
    X_train[KOLOM_SCALE] = scaler.transform(X_train[KOLOM_SCALE])
    X_test[KOLOM_SCALE] = scaler.transform(X_test[KOLOM_SCALE])

    X_train_final = X_train[FITUR_FINAL].copy()
    X_test_final = X_test[FITUR_FINAL].copy()

    return PreprocessedData(
        train_df=train_imp,
        test_df=test_imp,
        X_train=X_train,
        X_test=X_test,
        X_train_unscaled=X_train_unscaled,
        X_test_unscaled=X_test_unscaled,
        X_train_final=X_train_final,
        X_test_final=X_test_final,
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
        median_train=median_train,
        kategori_wilayah=kategori_wilayah,
        fitur_final=list(FITUR_FINAL),
        kolom_scale=list(KOLOM_SCALE),
    )


def save_preprocessed_artifacts(
    prep: PreprocessedData,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> Path:
    """Menyimpan hasil preprocessing ke disk sebagai checkpoint antar-notebook."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "preprocessing_artifacts.joblib"
    joblib.dump(prep, artifact_path)
    return artifact_path


def load_or_build_preprocessed_data(
    processed_dir: Path = PROCESSED_DATA_DIR,
    raw_path: Path = RAW_DATA_PATH,
    force_rebuild: bool = False,
) -> PreprocessedData:
    """Memuat artefak preprocessing atau membangunnya ulang dari data bersih."""
    artifact_path = processed_dir / "preprocessing_artifacts.joblib"
    if artifact_path.exists() and not force_rebuild:
        prep = joblib.load(artifact_path)
        if isinstance(prep, PreprocessedData) and list(prep.X_train_final.columns) == FITUR_FINAL:
            return prep

    df_clean = load_or_build_clean_data(
        processed_dir=processed_dir,
        raw_path=raw_path,
        force_rebuild=force_rebuild,
    )
    prep = run_preprocessing_pipeline(df_clean)
    save_preprocessed_artifacts(prep, output_dir=processed_dir)
    return prep
