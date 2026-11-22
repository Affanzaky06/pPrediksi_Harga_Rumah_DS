"""Pembersihan data berurutan, log cleaning, dan checkpoint data bersih."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from house_price.config import KOLOM_SPEK, NUM_COLUMNS, PROCESSED_DATA_DIR, RAW_DATA_PATH
from house_price.data.audit import build_domain_rules, tabel_outlier
from house_price.data.parsing import load_raw_data, parse_and_normalize

_REQUIRED_CLEAN_COLUMNS: set[str] = {
    "listing_id",
    "kategori_portal",
    "deskripsi",
    "lokasi",
    "area",
    "harga",
    "harga_juta",
    "luas_tanah",
    "luas_bangunan",
    "kamar_tidur",
    "kamar_mandi",
    "carport",
    "wilayah",
}


def clean_listings(
    data: pd.DataFrame,
    aturan: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Menjalankan 4 tahap cleaning berurutan secara pure dan mengembalikan (df_clean, log_df)."""
    if aturan is None:
        aturan = build_domain_rules(data)

    bersih = data.copy()
    log_cleaning: list[tuple[str, int, int]] = [("Data mentah", len(bersih), 0)]

    langkah_aturan = [
        ("1. Luas tanah/bangunan kosong", data[["luas_tanah", "luas_bangunan"]].isna().any(axis=1)),
        ("2. Duplikat spesifikasi", data.duplicated(KOLOM_SPEK)),
        ("3. Non-hunian (judul)", aturan["non-hunian (judul)"]),
        ("4. Nilai mustahil (aturan domain)", aturan.drop(columns="non-hunian (judul)").any(axis=1)),
    ]

    for nama, mask_buang in langkah_aturan:
        mask_aktif = mask_buang.loc[bersih.index]
        n_buang = int(mask_aktif.sum())
        bersih = bersih.loc[~mask_aktif]
        log_cleaning.append((nama, len(bersih), n_buang))

    df_clean = bersih.copy()
    df_clean["harga_juta"] = df_clean["harga"] / 1e6

    log_df = pd.DataFrame(log_cleaning, columns=["langkah", "sisa_baris", "dibuang"])
    log_df["dibuang_kumulatif_%"] = (1 - log_df["sisa_baris"] / len(data)) * 100
    return df_clean, log_df


def _profil_statistik(d: pd.DataFrame, label: str) -> pd.DataFrame:
    return pd.DataFrame({
        f"n_{label}": d[NUM_COLUMNS].notna().sum(),
        f"missing_{label}": d[NUM_COLUMNS].isna().sum(),
        f"median_{label}": d[NUM_COLUMNS].median(),
        f"mean_{label}": d[NUM_COLUMNS].mean(),
        f"max_{label}": d[NUM_COLUMNS].max(),
        f"skew_{label}": d[NUM_COLUMNS].skew(),
        f"outlierIQR_{label}": tabel_outlier(d, NUM_COLUMNS)["n_outlier"],
    })


def compare_before_after(data_before: pd.DataFrame, df_clean: pd.DataFrame) -> pd.DataFrame:
    """Menghasilkan tabel perbandingan statistik sebelum vs sesudah cleaning."""
    banding = pd.concat(
        [_profil_statistik(data_before, "sebelum"), _profil_statistik(df_clean, "sesudah")],
        axis=1,
    )
    urutan = [
        f"{m}_{w}"
        for m in ["n", "missing", "median", "mean", "max", "skew", "outlierIQR"]
        for w in ["sebelum", "sesudah"]
    ]
    return banding[urutan]


def save_clean_artifacts(
    data_parsed: pd.DataFrame,
    df_clean: pd.DataFrame,
    output_dir: Path = PROCESSED_DATA_DIR,
) -> tuple[Path, Path]:
    """Menyimpan data parsed & bersih dengan raw_index agar index asli tidak hilang."""
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_path = output_dir / "listings_parsed.csv"
    clean_path = output_dir / "listings_clean.csv"
    data_parsed.to_csv(parsed_path, index=True, index_label="raw_index")
    df_clean.to_csv(clean_path, index=True, index_label="raw_index")
    return parsed_path, clean_path


def load_or_build_clean_data(
    processed_dir: Path = PROCESSED_DATA_DIR,
    raw_path: Path = RAW_DATA_PATH,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Memuat checkpoint listings_clean.csv atau membangunnya ulang dari data mentah."""
    clean_path = processed_dir / "listings_clean.csv"
    if clean_path.exists() and not force_rebuild:
        df_clean = pd.read_csv(clean_path, index_col="raw_index")
        df_clean.index.name = None
        if _REQUIRED_CLEAN_COLUMNS.issubset(df_clean.columns):
            return df_clean

    df_raw = load_raw_data(raw_path)
    data_parsed = parse_and_normalize(df_raw)
    df_clean, _ = clean_listings(data_parsed)
    save_clean_artifacts(data_parsed, df_clean, output_dir=processed_dir)
    return df_clean
