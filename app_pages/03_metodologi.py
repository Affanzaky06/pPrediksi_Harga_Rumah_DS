"""Halaman 3: Metodologi, benchmark model, & bukti visual evaluasi."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from house_price.visualization.web_charts import (
    make_actual_vs_pred_chart,
    make_benchmark_chart,
    make_residual_hist_chart,
)
from house_price.web.loaders import (
    get_model_benchmark,
    get_model_bundle,
    get_pipeline_evaluation,
)

bundle, _ = get_model_bundle()
eval_data = get_pipeline_evaluation(bundle)
metrics = eval_data["metrics"]
eval_wilayah = eval_data["eval_wilayah"]
cleaning_log = eval_data["cleaning_log"]
test_eval_df = eval_data["test_eval_df"]
is_dark = st.context.theme.type == "dark"

st.subheader("Kinerja model pada unseen test set", icon=":material/fact_check:")
st.caption(
    f"Evaluasi langsung pada **{eval_data['n_test']}** data uji "
    f"(dipisahkan dari **{eval_data['n_train']}** data latih dengan Stratified Group Split, 0 kebocoran judul)."
)

with st.grid(4, gap="medium"):
    st.metric(
        label="R² score (test set)",
        value=f"{metrics['r2'] * 100:.2f}%",
        border=True,
        height="stretch",
    )
    st.metric(
        label="MdAPE (median error)",
        value=f"{metrics['mdape']:.2f}%",
        border=True,
        height="stretch",
    )
    st.metric(
        label="MAPE (mean error)",
        value=f"{metrics['mape']:.2f}%",
        border=True,
        height="stretch",
    )
    st.metric(
        label="MAE rata-rata",
        value=f"Rp {metrics['mae_rupiah'] / 1e6:,.0f} Juta",
        border=True,
        height="stretch",
    )

# Bukti visual evaluasi model pada unseen test set
col_v1, col_v2 = st.columns(2, gap="large")
with col_v1:
    with st.container(border=True):
        st.subheader("Harga aktual vs prediksi (data test)", icon=":material/scatter_plot:")
        st.altair_chart(
            make_actual_vs_pred_chart(test_eval_df, is_dark=is_dark),
            alt="Scatter plot harga aktual terhadap harga prediksi dengan garis referensi y=x",
        )
        st.caption("Semakin dekat titik ke garis merah putus-putus ($y = x$), semakin akurat estimasi model.")

with col_v2:
    with st.container(border=True):
        st.subheader("Distribusi error residual (skala log10)", icon=":material/bar_chart:")
        st.altair_chart(
            make_residual_hist_chart(test_eval_df, is_dark=is_dark),
            alt="Histogram distribusi error residual log10 yang terpusat di sekitar nol",
        )
        st.caption("Residual terpusat simetris di sekitar `0`, mengindikasikan tidak ada bias sistematis.")

# Perbandingan benchmark 5 kandidat model
with st.container(border=True):
    st.subheader("Perbandingan benchmark 5 kandidat model (GroupKFold 5-Fold)", icon=":material/analytics:")
    df_bench = get_model_benchmark()
    b_col1, b_col2 = st.columns([3, 2], gap="large")
    with b_col1:
        st.altair_chart(
            make_benchmark_chart(df_bench, is_dark=is_dark),
            alt="Perbandingan CV RMSE lima kandidat algoritma regresi",
        )
    with b_col2:
        st.dataframe(
            df_bench,
            hide_index=True,
            column_config={
                "CV RMSE (log)": st.column_config.NumberColumn("CV RMSE", format="%.4f"),
                "CV MAE (log)": st.column_config.NumberColumn("CV MAE", format="%.4f"),
                "CV R2": st.column_config.NumberColumn("CV R²", format="%.4f"),
            },
            alt="Tabel metrik cross-validation lima kandidat model regresi",
        )

with st.container(border=True):
    st.subheader("Alur pipeline CRISP-DM modular", icon=":material/layers:")
    st.mermaid_chart(
        """
        flowchart LR
            A["Data Mentah (2.020 baris)"] --> B["Parsing & Koreksi Wilayah (154 listing)"]
            B --> C["Cleaning Berurutan (1.384 baris bersih)"]
            C --> D["StratifiedGroupKFold (1.107 Train / 277 Test)"]
            D --> E["Imputasi Train-Only + Log10 + StandardScaler"]
            E --> F["Seleksi Fitur: VIF & Drop-Column CV (6 Fitur)"]
            F --> G["Random Forest (100 trees, max_features=0.5) + Estimasi Streamlit"]
        """,
        alt="Diagram alur pipeline Data Science dari data mentah hingga prediksi Streamlit",
    )

col_eval, col_clean = st.columns(2, gap="large")
with col_eval:
    with st.container(border=True):
        st.subheader("Evaluasi tersegmen per wilayah", icon=":material/location_on:")
        st.dataframe(
            eval_wilayah,
            hide_index=True,
            column_config={
                "wilayah_model": "Wilayah",
                "Jumlah_Sampel": st.column_config.NumberColumn("Sampel uji", format="%d"),
                "MAE_Rupiah": st.column_config.NumberColumn("MAE (Rp)", format="Rp %,.0f"),
                "MdAPE_Persen": st.column_config.NumberColumn("MdAPE (%)", format="%.2f%%"),
                "MAPE_Persen": st.column_config.NumberColumn("MAPE (%)", format="%.2f%%"),
            },
            alt="Tabel evaluasi MAE, MdAPE, dan MAPE per segmen wilayah pada test set",
        )

with col_clean:
    with st.container(border=True):
        st.subheader("Log pembersihan data eksplisit", icon=":material/dataset:")
        st.dataframe(
            cleaning_log,
            hide_index=True,
            column_config={
                "langkah": "Tahap cleaning",
                "sisa_baris": st.column_config.NumberColumn("Sisa baris", format="%d"),
                "dibuang": st.column_config.NumberColumn("Dibuang", format="%d"),
                "dibuang_kumulatif_%": st.column_config.NumberColumn("Kumulatif (%)", format="%.2f%%"),
            },
            alt="Log bertahap pembersihan data dari 2.020 baris menjadi 1.384 baris bersih",
        )

with st.container(border=True):
    st.subheader("Spesifikasi teknis pipeline", icon=":material/menu_book:")
    spec_df = pd.DataFrame([
        {"Komponen": "Sumber data", "Implementasi": "Rumah123 Yogyakarta (Kaggle, CC BY-NC 4.0, Mar–Mei 2024)"},
        {"Komponen": "Parsing & lokasi", "Implementasi": "Ekstraksi regex harga/luas + koreksi 154 area portal 'yogyakarta'"},
        {"Komponen": "Split anti-leakage", "Implementasi": "StratifiedGroupKFold (80/20) dikelompokkan per judul deskripsi"},
        {"Komponen": "Imputasi & encoding", "Implementasi": "Median train-only (KT=3, KM=2), OHE wilayah (baseline: Bantul)"},
        {"Komponen": "Transformasi & scaling", "Implementasi": "Log10 target & luas; StandardScaler fit hanya pada data train"},
        {"Komponen": "Algoritma model", "Implementasi": "RandomForestRegressor (n_estimators=100, max_depth=15, max_features=0.5, min_samples_leaf=2, min_samples_split=5)"},
    ])
    st.table(spec_df, alt="Ringkasan spesifikasi teknis tiap tahap pipeline machine learning")

st.info(
    "Model memprediksi harga penawaran (asking price) listing rumah tinggal wajar di D.I. Yogyakarta. "
    "Harga transaksi riil dapat berbeda bergantung pada legalitas sertifikat (SHM/HGB), lebar akses jalan, "
    "kualitas material bangunan, dan hasil negosiasi.",
    title="Batasan interpretasi estimasi",
    icon=":material/info:",
)
