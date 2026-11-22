"""Audit kualitas data: missing value, duplikat, outlier IQR, dan aturan domain."""

from __future__ import annotations

import numpy as np
import pandas as pd

from house_price.config import (
    KOLOM_SPEK,
    MAX_HARGA,
    MAX_KAMAR,
    MAX_LUAS_BANGUNAN,
    MAX_RASIO_LB_LT,
    MIN_HARGA,
    MIN_HARGA_PER_M2_TANAH,
    MIN_LUAS,
    POLA_AWAL,
    POLA_KOS,
    POLA_POTENSI,
    POLA_USAHA,
)


def batas_iqr(s: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Menghitung batas bawah dan atas outlier menggunakan metode IQR (Tukey)."""
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(q1 - k * iqr), float(q3 + k * iqr)


def tabel_outlier(data: pd.DataFrame, kolom: list[str]) -> pd.DataFrame:
    """Ringkasan outlier IQR per kolom: batas, rentang, jumlah, dan persentase."""
    baris = []
    for k in kolom:
        s = data[k].dropna()
        bawah, atas = batas_iqr(s)
        n = int(((s < bawah) | (s > atas)).sum())
        baris.append({
            "kolom": k,
            "batas_bawah": bawah,
            "batas_atas": atas,
            "min": s.min(),
            "max": s.max(),
            "n_outlier": n,
            "persen_%": n / len(s) * 100,
        })
    return pd.DataFrame(baris).set_index("kolom")


def audit_missing(data: pd.DataFrame) -> pd.DataFrame:
    """Menghitung jumlah dan persentase missing value per kolom."""
    return pd.DataFrame({
        "missing": data.isna().sum(),
        "persen_%": data.isna().mean() * 100,
    })


def audit_carport_mnar(data: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Menganalisis pola Missing Not At Random (MNAR) pada kolom carport."""
    grup_carport = np.where(
        data["carport"].isna(),
        "kosong",
        data["carport"].clip(upper=3).astype("Int64").astype(str),
    )
    grup_series = pd.Series(grup_carport, index=data.index).replace({"3": "3+"})
    ringkasan = (
        data.assign(grup_carport=grup_series)
        .groupby("grup_carport")
        .agg(
            listing=("harga", "size"),
            median_harga_juta=("harga", lambda s: s.median() / 1e6),
            median_LB=("luas_bangunan", "median"),
            median_LT=("luas_tanah", "median"),
        )
    )
    ada_nol = bool((data["carport"] == 0).any())
    return ringkasan, ada_nol


def audit_duplicates(df_raw: pd.DataFrame, data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Memeriksa duplikasi bertingkat dari ID listing hingga deskripsi iklan."""
    dup = pd.DataFrame(
        {
            "jumlah_kopian": [
                int(data["listing_id"].duplicated().sum()),
                int(df_raw.drop(columns="nav-link").duplicated().sum()),
                int(data.duplicated(KOLOM_SPEK).sum()),
                int(data["deskripsi"].duplicated().sum()),
            ]
        },
        index=["ID listing sama", "Identik selain URL", "Spesifikasi sama", "Deskripsi sama"],
    )
    dup["persen_%"] = dup["jumlah_kopian"] / len(data) * 100

    desk_sama = data[data["deskripsi"].duplicated(keep=False)]
    n_beda = int(
        desk_sama.groupby("deskripsi")
        .apply(lambda g: g[KOLOM_SPEK].drop_duplicates().shape[0] > 1)
        .sum()
    )
    return dup, n_beda


def flag_non_hunian(deskripsi: pd.Series, strip_potensi: bool = True) -> pd.Series:
    """Menandai iklan properti non-hunian berdasarkan pola judul iklan."""
    # Paksa dtype=object agar semantik regex Python (`re`) konsisten dengan atau tanpa pyarrow terpasang.
    judul = deskripsi.fillna("").astype(object).str.lower()
    if strip_potensi:
        judul = judul.str.replace(POLA_POTENSI, "", regex=True)

    tanah_saja = judul.str.contains("tanah", regex=False) & ~judul.str.contains(r"rumah|rmh", regex=True)
    result = (
        judul.str.contains(POLA_AWAL, regex=True)
        | judul.str.contains(POLA_KOS, regex=True)
        | judul.str.contains(POLA_USAHA, regex=True)
        | tanah_saja
    )
    return result.fillna(False).astype(bool)


def build_domain_rules(data: pd.DataFrame) -> pd.DataFrame:
    """Menghasilkan DataFrame boolean untuk 8 aturan anomali domain dan non-hunian."""
    harga_per_m2_tanah = data["harga"] / data["luas_tanah"]
    aturan = pd.DataFrame(
        {
            "harga tidak wajar": (data["harga"] < MIN_HARGA) | (data["harga"] > MAX_HARGA),
            "kamar tidur > 10": data["kamar_tidur"] > MAX_KAMAR,
            "kamar mandi > 10": data["kamar_mandi"] > MAX_KAMAR,
            "luas <= 15 m2": (data["luas_tanah"] <= MIN_LUAS) | (data["luas_bangunan"] <= MIN_LUAS),
            "luas bangunan > 5000 m2": data["luas_bangunan"] > MAX_LUAS_BANGUNAN,
            "LB > 3x LT": data["luas_bangunan"] > MAX_RASIO_LB_LT * data["luas_tanah"],
            "harga/m2 tanah < 1 jt": harga_per_m2_tanah < MIN_HARGA_PER_M2_TANAH,
            "non-hunian (judul)": flag_non_hunian(data["deskripsi"], strip_potensi=True),
        },
        index=data.index,
    )
    return aturan.fillna(False).astype(bool)
