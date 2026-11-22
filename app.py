"""Entrypoint utama Aplikasi Web Demo Prediksi Harga Rumah D.I. Yogyakarta.

Menggunakan arsitektur multipage modern (`st.navigation` + `app_pages/`)
dengan logika bisnis & visualisasi terpisah di `src/house_price/`.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st

from house_price.web.loaders import (
    get_clean_dataset,
    get_model_bundle,
    get_pipeline_evaluation,
    init_calculator_state,
)

st.set_page_config(
    page_title="Prediksi harga rumah D.I. Yogyakarta",
    page_icon=":material/real_estate_agent:",
    layout="wide",
    initial_sidebar_state="expanded",
)

bundle, artifact_name = get_model_bundle()
df_clean = get_clean_dataset()
eval_data = get_pipeline_evaluation(bundle)
metrics = eval_data["metrics"]

init_calculator_state(
    kategori_wilayah=list(bundle["kategori_wilayah"]),
    median_train=bundle["median_train"],
)

pages = [
    st.Page(
        "app_pages/01_kalkulator.py",
        title="Simulasi & kalkulator",
        icon=":material/calculate:",
        default=True,
        help=(
            "Simulasi estimasi harga pasar rumah di DIY lengkap dengan preset demo, "
            "uji sampel nyata dari data test, dan perbandingan antar-wilayah."
        ),
    ),
    st.Page(
        "app_pages/02_eda.py",
        title="Eksplorasi & EDA",
        icon=":material/analytics:",
        help=(
            "Eksplorasi interaktif 1.384 data listing bersih: distribusi harga dan fitur, "
            "korelasi log-log, anomali carport MNAR, dan bias wilayah."
        ),
    ),
    st.Page(
        "app_pages/03_metodologi.py",
        title="Metodologi & evaluasi",
        icon=":material/menu_book:",
        help=(
            "Evaluasi model pada 277 data uji, benchmark 5 algoritma, grafik Aktual vs Prediksi, "
            "analisis residual, dan alur pipeline CRISP-DM."
        ),
    ),
]

current_page = st.navigation(pages, position="top")

with st.sidebar:
    st.subheader("Proyek Data Science", icon=":material/real_estate_agent:")
    st.badge("Model aktif & siap demo", icon=":material/check_circle:", color="green")
    st.caption(
        "Prediksi harga penawaran rumah di wilayah D.I. Yogyakarta berbasis data Rumah123."
    )

    st.markdown("**Tim — Kelompok 9 (PERTALITE LANGKA)**")
    st.caption(
        "Informatika UPN \"Veteran\" Yogyakarta · 2026/2027\n"
        "- Muhammad Rifty Alfattah (`123240093`)\n"
        "- Muhamad Atallah Alfa Dzaky (`123240105`)\n"
        "- Nicolaus Narindra Lianto (`123240100`)\n"
        "- Abyaz Affanzaky Shanahan (`123240130`)"
    )

    st.markdown("**Ringkasan artefak**")
    st.caption(
        f"Artefak: `{artifact_name}`\n"
        "Algoritma: `RandomForestRegressor` (100 trees, tuned)\n"
        "Fitur final: 6 fitur (log-luas, kamar mandi, dummy wilayah)"
    )

st.title("Prediksi harga rumah D.I. Yogyakarta", icon=":material/real_estate_agent:")
st.caption(
    "Dashboard demo interaktif estimasi harga pasar properti residensial menggunakan "
    "Random Forest Regressor teroptimasi dengan Stratified Group Split."
)

with st.container(horizontal=True):
    st.badge("Random Forest (100 trees, tuned)", icon=":material/layers:", color="blue")
    st.badge(f"R² Test: {metrics['r2'] * 100:.2f}%", icon=":material/analytics:", color="green")
    st.badge(f"MdAPE: {metrics['mdape']:.2f}%", icon=":material/fact_check:", color="violet")
    st.badge(f"{len(df_clean):,} listing bersih", icon=":material/dataset:", color="gray")

current_page.run()

st.caption(
    f"Proyek Data Science UPN \"Veteran\" Yogyakarta · Paket modular `src/house_price/` · Artefak aktif: `{artifact_name}`"
)
