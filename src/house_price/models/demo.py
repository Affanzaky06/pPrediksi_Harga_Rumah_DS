"""Fungsi pembantu pure untuk analisis EDA, simulasi demo, dan ringkasan di Streamlit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from house_price.config import FITUR_NUM, WILAYAH_MERGE_MAP
from house_price.models.predict import predict_house_price

_FEATURE_LABELS: dict[str, str] = {
    "log_luas_bangunan": "Luas bangunan (log10)",
    "log_luas_tanah": "Luas tanah (log10)",
    "kamar_mandi": "Jumlah kamar mandi",
    "wilayah_Kota Yogyakarta": "Wilayah: Kota Yogyakarta",
    "wilayah_Sleman": "Wilayah: Sleman",
    "wilayah_Kulon Progo/Gunung Kidul": "Wilayah: Kulon Progo / Gunung Kidul",
}

_EDA_COL_LABELS: dict[str, str] = {
    "harga_juta": "Harga (Juta Rp)",
    "harga": "Harga",
    "luas_tanah": "Luas tanah (m²)",
    "luas_bangunan": "Luas bangunan (m²)",
    "kamar_tidur": "Kamar tidur",
    "kamar_mandi": "Kamar mandi",
    "carport": "Carport",
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
    summary["Porsi (%)"] = summary["Jumlah_Listing"] / summary["Jumlah_Listing"].sum() * 100.0
    return summary


def compute_descriptive_stats(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Menghitung statistik deskriptif lengkap (persentil, skewness, missing) untuk EDA."""
    cols = ["harga_juta", "luas_tanah", "luas_bangunan", "kamar_tidur", "kamar_mandi", "carport"]
    desk = df_filtered[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    desk["skewness"] = df_filtered[cols].skew()
    desk["missing"] = df_filtered[cols].isna().sum()
    desk.index = [_EDA_COL_LABELS.get(c, c) for c in desk.index]
    return desk.reset_index().rename(columns={"index": "Variabel"})


def compute_carport_mnar_eda(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Menganalisis profil harga & ukuran pada kelompok carport terisi vs kosong (MNAR)."""
    grup = df_filtered["carport"].map(
        lambda v: "Kosong" if pd.isna(v) else ("3+" if v >= 3 else str(int(v)))
    )
    order_map = {"Kosong": 0, "1": 1, "2": 2, "3+": 3}
    df_cp = df_filtered.assign(
        Grup_Carport=grup,
        harga_m2_lb_juta=(df_filtered["harga"] / df_filtered["luas_bangunan"]) / 1e6,
    )
    res = (
        df_cp.groupby("Grup_Carport")
        .agg(
            Listing=("harga", "size"),
            Median_Harga_Juta=("harga_juta", "median"),
            Median_LB=("luas_bangunan", "median"),
            Median_LT=("luas_tanah", "median"),
            Median_Harga_m2_LB_Juta=("harga_m2_lb_juta", "median"),
        )
        .reset_index()
    )
    res["_ord"] = res["Grup_Carport"].map(lambda x: order_map.get(x, 99))
    return res.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)


def compute_correlation_tables(df_filtered: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Menghasilkan matriks korelasi Spearman (long format untuk heatmap) dan tabel korelasi target."""
    cols = FITUR_NUM + ["harga"]
    kor = df_filtered[cols].corr(method="spearman")

    kor_labeled = kor.copy()
    kor_labeled.index = [_EDA_COL_LABELS.get(c, c) for c in kor_labeled.index]
    kor_labeled.columns = [_EDA_COL_LABELS.get(c, c) for c in kor_labeled.columns]

    long_df = (
        kor_labeled.stack()
        .reset_index()
        .rename(columns={"level_0": "Variabel_X", "level_1": "Variabel_Y", 0: "Korelasi"})
    )

    log_df = df_filtered.assign(
        log_luas_tanah=np.log10(df_filtered["luas_tanah"]),
        log_luas_bangunan=np.log10(df_filtered["luas_bangunan"]),
        log_harga=np.log10(df_filtered["harga"]),
    )
    kor_target = pd.DataFrame({
        "Spearman": kor["harga"].drop("harga"),
        "Pearson (asli)": df_filtered[cols].corr()["harga"].drop("harga"),
        "Pearson (log harga)": log_df[FITUR_NUM + ["log_harga"]].corr()["log_harga"].drop("log_harga"),
    })
    kor_target.loc[["luas_tanah", "luas_bangunan"], "Pearson (log-log)"] = [
        log_df[["log_luas_tanah", "log_harga"]].corr().iloc[0, 1],
        log_df[["log_luas_bangunan", "log_harga"]].corr().iloc[0, 1],
    ]
    kor_target = kor_target.sort_values("Spearman", ascending=False)
    kor_target.index = [_EDA_COL_LABELS.get(c, c) for c in kor_target.index]
    return long_df, kor_target.reset_index().rename(columns={"index": "Fitur"})


def compute_price_segment_distribution(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """Menghitung persentase komposisi segmen harga per wilayah untuk analisis bias sampel."""
    df_s = df_filtered.copy()
    if "wilayah_model" not in df_s.columns:
        df_s["wilayah_model"] = df_s["wilayah"].replace(WILAYAH_MERGE_MAP)

    segmen_labels = ["< 500 jt", "500 jt–1 M", "1–2 M", "2–5 M", "> 5 M"]
    df_s["Segmen Harga"] = pd.cut(
        df_s["harga"],
        bins=[0, 5e8, 1e9, 2e9, 5e9, np.inf],
        labels=segmen_labels,
    )
    ct = pd.crosstab(df_s["wilayah_model"], df_s["Segmen Harga"], normalize="index") * 100.0
    long_df = (
        ct.reset_index()
        .melt(id_vars="wilayah_model", var_name="Segmen Harga", value_name="Porsi (%)")
        .rename(columns={"wilayah_model": "Wilayah"})
    )
    return long_df


def compute_kruskal_wilayah(df_filtered: pd.DataFrame) -> tuple[float, float] | None:
    """Menjalankan uji Kruskal-Wallis perbedaan harga antarwilayah (>= 2 grup non-kosong)."""
    col = "wilayah_model" if "wilayah_model" in df_filtered.columns else "wilayah"
    groups = [g["harga"].values for _, g in df_filtered.groupby(col) if len(g) > 0]
    if len(groups) < 2:
        return None
    h_stat, p_val = stats.kruskal(*groups)
    return float(h_stat), float(p_val)
