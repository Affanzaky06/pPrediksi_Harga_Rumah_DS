"""Fungsi pembantu pure untuk analisis dan simulasi demo pada aplikasi Streamlit."""

from __future__ import annotations

import pandas as pd

from house_price.config import WILAYAH_MERGE_MAP
from house_price.models.predict import predict_house_price

_FEATURE_LABELS: dict[str, str] = {
    "log_luas_bangunan": "Luas bangunan (log10)",
    "log_luas_tanah": "Luas tanah (log10)",
    "kamar_mandi": "Jumlah kamar mandi",
    "wilayah_Kota Yogyakarta": "Wilayah: Kota Yogyakarta",
    "wilayah_Sleman": "Wilayah: Sleman",
    "wilayah_Kulon Progo/Gunung Kidul": "Wilayah: Kulon Progo / Gunung Kidul",
}

DEMO_PRESETS: dict[str, dict[str, object]] = {
    "Tipe 45/90 (Bantul)": {
        "wilayah": "Bantul",
        "luas_tanah": 90.0,
        "luas_bangunan": 45.0,
        "kamar_mandi": 1,
        "kamar_tidur": 2,
        "carport": 1,
    },
    "Keluarga (Sleman)": {
        "wilayah": "Sleman",
        "luas_tanah": 120.0,
        "luas_bangunan": 100.0,
        "kamar_mandi": 2,
        "kamar_tidur": 3,
        "carport": 1,
    },
    "Mewah (Kota Jogja)": {
        "wilayah": "Kota Yogyakarta",
        "luas_tanah": 250.0,
        "luas_bangunan": 300.0,
        "kamar_mandi": 4,
        "kamar_tidur": 5,
        "carport": 2,
    },
    "Minimalis (KP / GK)": {
        "wilayah": "Kulon Progo/Gunung Kidul",
        "luas_tanah": 100.0,
        "luas_bangunan": 54.0,
        "kamar_mandi": 1,
        "kamar_tidur": 2,
        "carport": 1,
    },
}


def simulate_wilayah_comparison(
    bundle: dict,
    luas_tanah: float,
    luas_bangunan: float,
    kamar_mandi: int | float,
    kamar_tidur: int | float,
    carport: int | float,
    selected_wilayah: str | None = None,
) -> pd.DataFrame:
    """Mensimulasikan prediksi harga untuk spesifikasi identik di seluruh kategori wilayah."""
    rows: list[dict[str, object]] = []
    for wil in bundle["kategori_wilayah"]:
        pred_rp, _, _ = predict_house_price(
            bundle=bundle,
            luas_tanah=luas_tanah,
            luas_bangunan=luas_bangunan,
            kamar_mandi=kamar_mandi,
            kamar_tidur=kamar_tidur,
            carport=carport,
            wilayah=wil,
        )
        rows.append({
            "Wilayah": wil,
            "Estimasi (Rp)": pred_rp,
            "Estimasi (Miliar Rp)": pred_rp / 1e9,
            "Estimasi (Juta Rp)": pred_rp / 1e6,
            "Harga / m² LB (Juta)": (pred_rp / luas_bangunan) / 1e6,
            "Status": "Dipilih" if wil == selected_wilayah else "Pembanding",
        })
    return pd.DataFrame(rows).sort_values("Estimasi (Rp)", ascending=False).reset_index(drop=True)


def build_feature_importance_df(bundle: dict) -> pd.DataFrame:
    """Mengonversi bobot feature_importances_ model ke tabel terurut yang mudah dibaca."""
    fitur_final = list(bundle["fitur_final"])
    importances = bundle["model"].feature_importances_
    df_imp = pd.DataFrame({
        "Fitur": [_FEATURE_LABELS.get(col, col) for col in fitur_final],
        "Kolom model": fitur_final,
        "Kontribusi (%)": importances * 100.0,
    })
    return df_imp.sort_values("Kontribusi (%)", ascending=False).reset_index(drop=True)


def summarize_market_by_wilayah(df_clean: pd.DataFrame) -> pd.DataFrame:
    """Menyusun ringkasan statistik pasar per wilayah model dari dataset bersih."""
    df_w = df_clean.copy()
    df_w["wilayah_model"] = df_w["wilayah"].replace(WILAYAH_MERGE_MAP)
    df_w["harga_m2_lb_juta"] = (df_w["harga"] / df_w["luas_bangunan"]) / 1e6

    summary = (
        df_w.groupby("wilayah_model")
        .agg(
            Jumlah_Listing=("harga", "size"),
            Median_Harga_Juta=("harga_juta", "median"),
            Median_LT=("luas_tanah", "median"),
            Median_LB=("luas_bangunan", "median"),
            Median_Harga_m2_LB_Juta=("harga_m2_lb_juta", "median"),
        )
        .reset_index()
        .rename(columns={"wilayah_model": "Wilayah"})
        .sort_values("Median_Harga_Juta", ascending=False)
        .reset_index(drop=True)
    )
    return summary
