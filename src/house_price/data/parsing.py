"""Pemuatan data mentah, parsing teks harga/luas, dan normalisasi lokasi."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from house_price.config import KOREKSI_AREA, NAMA_WILAYAH, RAW_DATA_PATH

_SATUAN_HARGA: dict[str, float] = {
    "miliar": 1e9,
    "juta": 1e6,
    "ribu": 1e3,
}


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Memuat dataset mentah CSV listing properti."""
    return pd.read_csv(path)


def ubah_harga(teks: str) -> float:
    """Mengubah teks harga Indonesia (mis. 'Rp 1,79 Miliar') menjadi float Rupiah."""
    if not isinstance(teks, str) or not teks.strip():
        raise ValueError(f"Format harga tidak valid (kosong atau bukan string): {teks!r}")

    bagian = teks.lower().replace("rp", "").strip().split()
    if not bagian:
        raise ValueError(f"Format harga tidak memuat angka: {teks!r}")

    angka_str = bagian[0].replace(".", "").replace(",", ".")
    try:
        angka = float(angka_str)
    except ValueError as exc:
        raise ValueError(f"Tidak dapat mem-parse angka harga dari {teks!r}") from exc

    satuan = bagian[1] if len(bagian) > 1 else ""
    return angka * _SATUAN_HARGA.get(satuan, 1.0)


def ubah_luas(kolom: pd.Series) -> pd.Series:
    """Mengubah teks luas ('120 m²') menjadi float; nilai kosong tetap NaN."""
    teks = (
        kolom.astype("string")
        .str.replace(r"[^\d,\.]", "", regex=True)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(teks, errors="coerce")


def summarize_raw_dictionary(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Menghasilkan ringkasan kamus data mentah (tipe, terisi, kosong, unik, contoh)."""
    return pd.DataFrame({
        "tipe_pandas": df_raw.dtypes.astype(str),
        "terisi": df_raw.notna().sum(),
        "kosong": df_raw.isna().sum(),
        "nilai_unik": df_raw.nunique(),
        "contoh": df_raw.iloc[0],
    })


def parse_and_normalize(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Mem-parse kolom teks, menormalisasi spasi lokasi, dan mengoreksi wilayah administratif."""
    data = pd.DataFrame(index=df_raw.index)
    data["listing_id"] = df_raw["nav-link"].str.extract(r"(hos\d+)")[0]
    data["kategori_portal"] = df_raw["nav-link"].str.extract(r"properti/([^/]+)/")[0].str.lower()
    data["deskripsi"] = df_raw["description"].str.replace(r"\s+", " ", regex=True).str.strip()

    lokasi = df_raw["listing-location"].str.replace(r"\s+", " ", regex=True).str.strip()
    data["lokasi"] = lokasi
    data["area"] = lokasi.str.split(",").str[0].str.strip()

    data["harga"] = df_raw["price"].apply(ubah_harga)
    data["luas_tanah"] = ubah_luas(df_raw["surface_area"])
    data["luas_bangunan"] = ubah_luas(df_raw["building_area"])
    data["kamar_tidur"] = df_raw["bed"]
    data["kamar_mandi"] = df_raw["bath"]
    data["carport"] = df_raw["carport"]

    data["wilayah"] = data["kategori_portal"].map(NAMA_WILAYAH)
    mask_yogya = data["kategori_portal"].eq("yogyakarta")
    data.loc[mask_yogya, "wilayah"] = (
        data.loc[mask_yogya, "area"].map(KOREKSI_AREA).fillna("Kota Yogyakarta")
    )
    return data


def compare_wilayah_correction(data: pd.DataFrame) -> pd.DataFrame:
    """Membandingkan distribusi wilayah versi portal vs setelah koreksi manual."""
    perbandingan = pd.DataFrame({
        "versi_portal": data["kategori_portal"].map(NAMA_WILAYAH).value_counts(),
        "setelah_koreksi": data["wilayah"].value_counts(),
    })
    perbandingan["selisih"] = perbandingan["setelah_koreksi"] - perbandingan["versi_portal"]
    return perbandingan
