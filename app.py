"""Aplikasi Web Demo Prediksi Harga Rumah D.I. Yogyakarta.

Berbasis Machine Learning (Random Forest Regressor) menggunakan paket modular
`src/house_price/` serta artefak model terlatih.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from house_price.config import (
    DEFAULT_MDAPE_RATE,
    MODEL_ARTIFACT_PATH,
    WILAYAH_MERGE_MAP,
)
from house_price.data.cleaning import clean_listings, load_or_build_clean_data
from house_price.data.parsing import load_raw_data, parse_and_normalize
from house_price.features.preprocessing import load_or_build_preprocessed_data
from house_price.models.demo import (
    DEMO_PRESETS,
    build_feature_importance_df,
    compute_carport_mnar_eda,
    compute_correlation_tables,
    compute_descriptive_stats,
    compute_kruskal_wilayah,
    compute_price_segment_distribution,
    simulate_wilayah_comparison,
    summarize_market_by_wilayah,
)
from house_price.models.evaluate import compute_regression_metrics, evaluate_by_wilayah
from house_price.models.predict import format_rupiah, predict_batch_log, predict_house_price
from house_price.models.train import load_model_bundle

st.set_page_config(
    page_title="Prediksi harga rumah D.I. Yogyakarta",
    page_icon=":material/real_estate_agent:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(max_entries=2)
def get_model_bundle() -> tuple[dict, str]:
    """Memuat satu sumber artefak model; otomatis rebuild bila versi scikit-learn berbeda."""
    try:
        bundle = load_model_bundle(path=MODEL_ARTIFACT_PATH, auto_rebuild=True)
        return bundle, MODEL_ARTIFACT_PATH.name
    except FileNotFoundError:
        st.error(
            f"Artefak model tidak ditemukan di `{MODEL_ARTIFACT_PATH}` dan data mentah "
            "tidak tersedia untuk rebuild otomatis. Jalankan `python main.py` terlebih dahulu.",
            icon=":material/error:",
        )
        st.stop()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gagal memuat artefak model: {exc}", icon=":material/error:")
        st.stop()


@st.cache_data(ttl="1h", max_entries=2)
def get_clean_dataset() -> pd.DataFrame:
    """Memuat dataset listing bersih (1.384 baris) beserta kolom wilayah_model."""
    df_clean = load_or_build_clean_data().copy()
    df_clean["wilayah_model"] = df_clean["wilayah"].replace(WILAYAH_MERGE_MAP)
    return df_clean


@st.cache_data(ttl="1h", max_entries=2)
def get_pipeline_evaluation(_bundle: dict) -> dict[str, object]:
    """Menghitung metrik test set aktual, evaluasi per wilayah, dan log cleaning."""
    prep = load_or_build_preprocessed_data()
    y_pred_log = predict_batch_log(_bundle["model"], prep.X_test_final)
    metrics = compute_regression_metrics(prep.y_test, y_pred_log)
    eval_wilayah = evaluate_by_wilayah(prep.test_df, metrics["y_pred_rupiah"])

    df_raw = load_raw_data()
    data_parsed = parse_and_normalize(df_raw)
    _, cleaning_log = clean_listings(data_parsed)

    return {
        "metrics": metrics,
        "eval_wilayah": eval_wilayah,
        "cleaning_log": cleaning_log,
        "n_train": len(prep.train_df),
        "n_test": len(prep.test_df),
    }


def _apply_preset() -> None:
    preset_name = st.session_state.get("demo_preset")
    if not preset_name or preset_name not in DEMO_PRESETS:
        return
    preset = DEMO_PRESETS[preset_name]
    st.session_state["inp_wilayah"] = preset["wilayah"]
    st.session_state["inp_lt"] = float(preset["luas_tanah"])
    st.session_state["inp_lb"] = float(preset["luas_bangunan"])
    st.session_state["inp_km"] = int(preset["kamar_mandi"])
    st.session_state["inp_kt"] = int(preset["kamar_tidur"])
    st.session_state["inp_cp"] = int(preset["carport"])


def _init_session_state(kategori_wilayah: list[str], median_train: pd.Series) -> None:
    default_wilayah = "Sleman" if "Sleman" in kategori_wilayah else kategori_wilayah[0]
    defaults: dict[str, object] = {
        "demo_preset": "Keluarga (Sleman)",
        "inp_wilayah": default_wilayah,
        "inp_lt": 120.0,
        "inp_lb": 100.0,
        "inp_km": int(median_train.get("kamar_mandi", 2)),
        "inp_kt": int(median_train.get("kamar_tidur", 3)),
        "inp_cp": 1,
        "demo_comparisons": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


bundle, artifact_name = get_model_bundle()
kategori_wilayah: list[str] = list(bundle["kategori_wilayah"])
median_train: pd.Series = bundle["median_train"]
_init_session_state(kategori_wilayah, median_train)

# ---------------------------------------------------------
# Sidebar: Identitas Proyek & Status Artefak (Ringkas)
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("Proyek Sains Data", icon=":material/real_estate_agent:")
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
        "Algoritma: `RandomForestRegressor` (300 trees)\n"
        "Fitur final: 6 fitur (log-luas, kamar mandi, dummy wilayah)"
    )

# ---------------------------------------------------------
# Header Utama
# ---------------------------------------------------------
st.title("Prediksi harga rumah D.I. Yogyakarta", icon=":material/real_estate_agent:")
st.caption(
    "Dashboard demo interaktif estimasi harga pasar properti residensial menggunakan "
    "Random Forest Regressor teroptimasi dengan Stratified Group Split."
)

with st.container(horizontal=True):
    st.badge("Random Forest (300 trees)", icon=":material/layers:", color="blue")
    st.badge("R² Test: 86,51%", icon=":material/analytics:", color="green")
    st.badge("MdAPE: 15,65%", icon=":material/fact_check:", color="violet")
    st.badge("1.384 listing bersih", icon=":material/dataset:", color="gray")

# ---------------------------------------------------------
# Navigasi Tab (Dynamic rendering agar rerun ringan)
# ---------------------------------------------------------
tab_prediksi, tab_analisis, tab_metodologi = st.tabs(
    [
        ":material/calculate: Simulasi & kalkulator harga",
        ":material/analytics: Eksplorasi data & EDA",
        ":material/menu_book: Metodologi & evaluasi model",
    ],
    key="nav_tabs",
    on_change="rerun",
)

# =========================================================
# TAB 1: SIMULASI & KALKULATOR HARGA
# =========================================================
if tab_prediksi.open:
    with tab_prediksi:
        with st.container(border=True):
            st.markdown("**Skenario cepat untuk demo presentasi**")
            st.pills(
                "Pilih contoh profil rumah",
                options=list(DEMO_PRESETS.keys()),
                selection_mode="single",
                key="demo_preset",
                on_change=_apply_preset,
                label_visibility="collapsed",
                help="Klik salah satu skenario untuk mengisi parameter spesifikasi secara otomatis.",
            )

        with st.container(border=True):
            col_loc, col_room = st.columns(2, gap="large")

            with col_loc:
                st.subheader("Lokasi & dimensi properti", icon=":material/location_on:")
                wilayah_pilihan = st.segmented_control(
                    "Wilayah administratif",
                    options=kategori_wilayah,
                    selection_mode="single",
                    required=True,
                    key="inp_wilayah",
                    width="stretch",
                    help="Pilih kabupaten atau kota administratif di D.I. Yogyakarta.",
                )
                if wilayah_pilihan is None:
                    wilayah_pilihan = kategori_wilayah[0]

                luas_tanah = st.number_input(
                    "Luas tanah (m²)",
                    min_value=20.0,
                    max_value=3000.0,
                    step=5.0,
                    key="inp_lt",
                    help="Total luas bidang tanah dalam meter persegi.",
                )
                luas_bangunan = st.number_input(
                    "Luas bangunan (m²)",
                    min_value=20.0,
                    max_value=2000.0,
                    step=5.0,
                    key="inp_lb",
                    help="Total luas lantai bangunan rumah dalam meter persegi.",
                )
                if luas_bangunan > luas_tanah:
                    st.caption(
                        ":material/info: Luas bangunan lebih besar dari luas tanah "
                        "(umum pada rumah bertingkat 2–3 lantai)."
                    )

            with col_room:
                st.subheader("Fasilitas & ruangan", icon=":material/bed:")
                def_km = int(median_train.get("kamar_mandi", 2))
                def_kt = int(median_train.get("kamar_tidur", 3))

                kamar_mandi = st.number_input(
                    "Jumlah kamar mandi",
                    min_value=1,
                    max_value=15,
                    step=1,
                    key="inp_km",
                    help=f"Nilai median data training: {def_km}. Digunakan sebagai salah satu fitur final.",
                )
                kamar_tidur = st.number_input(
                    "Jumlah kamar tidur",
                    min_value=1,
                    max_value=20,
                    step=1,
                    key="inp_kt",
                    help=f"Nilai median data training: {def_kt}. Dipertahankan untuk kompatibilitas scaler.",
                )
                carport = st.number_input(
                    "Kapasitas carport (mobil)",
                    min_value=0,
                    max_value=10,
                    step=1,
                    key="inp_cp",
                    help="Dipertahankan untuk kompatibilitas skema scaler; tidak memengaruhi bobot fitur final.",
                )
                st.caption(
                    "Catatan seleksi fitur: `kamar_tidur` dan `carport` tidak masuk 6 fitur final "
                    "karena sudah terwakili oleh luas bangunan & kamar mandi."
                )

        pred_rupiah, pred_log, df_fitur = predict_house_price(
            bundle=bundle,
            luas_tanah=float(luas_tanah),
            luas_bangunan=float(luas_bangunan),
            kamar_mandi=int(kamar_mandi),
            kamar_tidur=int(kamar_tidur),
            carport=int(carport),
            wilayah=wilayah_pilihan,
        )

        str_rp, str_ringkas = format_rupiah(pred_rupiah)
        batas_bawah = pred_rupiah * (1 - DEFAULT_MDAPE_RATE)
        batas_atas = pred_rupiah * (1 + DEFAULT_MDAPE_RATE)
        _, ringkas_bawah = format_rupiah(batas_bawah)
        _, ringkas_atas = format_rupiah(batas_atas)
        harga_per_m2_lt = pred_rupiah / float(luas_tanah)
        harga_per_m2_lb = pred_rupiah / float(luas_bangunan)

        with st.container(border=True):
            header_col, action_col = st.columns([3, 2], vertical_alignment="center")
            with header_col:
                st.subheader("Hasil estimasi nilai pasar properti", icon=":material/payments:")
                st.caption(
                    f"Profil: **{wilayah_pilihan}** · LT **{luas_tanah:,.0f} m²** · "
                    f"LB **{luas_bangunan:,.0f} m²** · **{kamar_mandi} KM**"
                )
            with action_col:
                with st.container(horizontal=True, horizontal_alignment="right"):
                    if st.button(
                        "Simpan ke perbandingan",
                        type="primary",
                        icon=":material/add_chart:",
                        key="btn_save_scenario",
                    ):
                        st.session_state["demo_comparisons"].append({
                            "Wilayah": wilayah_pilihan,
                            "LT (m²)": int(luas_tanah),
                            "LB (m²)": int(luas_bangunan),
                            "KM": int(kamar_mandi),
                            "Estimasi (Rp)": round(pred_rupiah),
                            "Estimasi Ringkas": f"Rp {str_ringkas}" if str_ringkas else str_rp,
                            "Harga / m² LB (Rp)": round(harga_per_m2_lb),
                        })
                        st.toast("Skenario ditambahkan ke tabel perbandingan.", icon=":material/check_circle:")

                    if st.session_state["demo_comparisons"] and st.button(
                        "Reset",
                        icon=":material/restart_alt:",
                        key="btn_reset_scenarios",
                    ):
                        st.session_state["demo_comparisons"] = []

            with st.grid(4, gap="medium"):
                st.metric(
                    label="Estimasi harga pasar",
                    value=f"Rp {str_ringkas}" if str_ringkas else str_rp,
                    delta=str_rp,
                    delta_color="off",
                    delta_arrow="off",
                    icon=":material/sell:",
                    border=True,
                    height="stretch",
                )
                st.metric(
                    label="Rentang wajar (±15,65% MdAPE)",
                    value=f"{ringkas_bawah} – {ringkas_atas}",
                    delta="Berbasis error median data test",
                    delta_color="off",
                    delta_arrow="off",
                    icon=":material/compare_arrows:",
                    border=True,
                    height="stretch",
                )
                st.metric(
                    label="Estimasi harga / m² tanah",
                    value=f"Rp {harga_per_m2_lt:,.0f}".replace(",", "."),
                    delta=f"LT {luas_tanah:,.0f} m²",
                    delta_color="off",
                    delta_arrow="off",
                    icon=":material/straighten:",
                    border=True,
                    height="stretch",
                )
                st.metric(
                    label="Estimasi harga / m² bangunan",
                    value=f"Rp {harga_per_m2_lb:,.0f}".replace(",", "."),
                    delta=f"LB {luas_bangunan:,.0f} m²",
                    delta_color="off",
                    delta_arrow="off",
                    icon=":material/architecture:",
                    border=True,
                    height="stretch",
                )

        if st.session_state["demo_comparisons"]:
            with st.container(border=True):
                st.subheader("Perbandingan skenario demo tersimpan", icon=":material/table_chart:")
                df_comp = pd.DataFrame(st.session_state["demo_comparisons"])
                st.dataframe(
                    df_comp,
                    hide_index=True,
                    column_config={
                        "Estimasi (Rp)": st.column_config.NumberColumn("Estimasi (Rp)", format="Rp %,.0f"),
                        "Harga / m² LB (Rp)": st.column_config.NumberColumn("Harga / m² LB", format="Rp %,.0f"),
                    },
                    alt="Tabel perbandingan simulasi skenario harga rumah yang disimpan",
                )

        df_clean = get_clean_dataset()
        col_sim, col_mkt = st.columns(2, gap="large")

        with col_sim:
            with st.container(border=True):
                st.subheader("Simulasi premi lokasi antar-wilayah", icon=":material/compare_arrows:")
                st.caption("Jika spesifikasi rumah yang sama persis berada di kabupaten/kota berbeda:")
                df_sim = simulate_wilayah_comparison(
                    bundle=bundle,
                    luas_tanah=float(luas_tanah),
                    luas_bangunan=float(luas_bangunan),
                    kamar_mandi=int(kamar_mandi),
                    kamar_tidur=int(kamar_tidur),
                    carport=int(carport),
                    selected_wilayah=wilayah_pilihan,
                )
                chart_sim = (
                    alt.Chart(df_sim)
                    .mark_bar(cornerRadiusEnd=6)
                    .encode(
                        x=alt.X("Estimasi (Miliar Rp):Q", title="Estimasi harga (Miliar Rp)"),
                        y=alt.Y("Wilayah:N", sort="-x", title=""),
                        color=alt.Color(
                            "Status:N",
                            scale=alt.Scale(domain=["Dipilih", "Pembanding"], range=["#1D4ED8", "#94A3B8"]),
                            legend=alt.Legend(title="", orient="bottom"),
                        ),
                        tooltip=[
                            alt.Tooltip("Wilayah:N"),
                            alt.Tooltip("Estimasi (Miliar Rp):Q", format=",.2f", title="Estimasi (Miliar Rp)"),
                            alt.Tooltip("Harga / m² LB (Juta):Q", format=",.2f", title="Harga/m² LB (Juta)"),
                        ],
                    )
                    .properties(height=260)
                )
                st.altair_chart(
                    chart_sim,
                    alt="Perbandingan estimasi harga untuk spesifikasi yang sama di seluruh wilayah DIY",
                )

        with col_mkt:
            with st.container(border=True):
                st.subheader(f"Posisi terhadap pasar {wilayah_pilihan}", icon=":material/scatter_plot:")
                st.caption("Membandingkan hasil prediksi (titik merah) dengan listing nyata di wilayah terpilih:")
                df_wil_market = df_clean[df_clean["wilayah_model"] == wilayah_pilihan][
                    ["luas_bangunan", "luas_tanah", "harga_juta", "area"]
                ].copy()
                df_wil_market["Kategori"] = "Listing pasar"

                df_point = pd.DataFrame([{
                    "luas_bangunan": float(luas_bangunan),
                    "luas_tanah": float(luas_tanah),
                    "harga_juta": pred_rupiah / 1e6,
                    "area": wilayah_pilihan,
                    "Kategori": "Simulasi Anda",
                }])

                base_pts = (
                    alt.Chart(df_wil_market)
                    .mark_circle(size=55, opacity=0.45)
                    .encode(
                        x=alt.X("luas_bangunan:Q", title="Luas bangunan (m²)"),
                        y=alt.Y("harga_juta:Q", title="Harga penawaran (juta Rp)"),
                        color=alt.value("#64748B"),
                        tooltip=[
                            alt.Tooltip("area:N", title="Area"),
                            alt.Tooltip("luas_bangunan:Q", title="LB (m²)"),
                            alt.Tooltip("luas_tanah:Q", title="LT (m²)"),
                            alt.Tooltip("harga_juta:Q", format=",.0f", title="Harga (Juta Rp)"),
                        ],
                    )
                )
                sim_pt = (
                    alt.Chart(df_point)
                    .mark_point(shape="diamond", size=240, filled=True)
                    .encode(
                        x="luas_bangunan:Q",
                        y="harga_juta:Q",
                        color=alt.value("#DC2626"),
                        tooltip=[
                            alt.Tooltip("Kategori:N"),
                            alt.Tooltip("luas_bangunan:Q", title="LB (m²)"),
                            alt.Tooltip("harga_juta:Q", format=",.0f", title="Prediksi (Juta Rp)"),
                        ],
                    )
                )
                st.altair_chart(
                    (base_pts + sim_pt).properties(height=260).interactive(),
                    alt=f"Sebaran harga listing di {wilayah_pilihan} dibandingkan titik simulasi",
                )

        exp_tech = st.expander(
            "Rincian transformasi fitur model (internal preprocessing)",
            icon=":material/tune:",
            key="exp_fitur",
            on_change="rerun",
        )
        if exp_tech.open:
            with exp_tech:
                st.write("Vektor fitur akhir yang dimasukkan ke model Random Forest (`X_final`):")
                st.dataframe(
                    df_fitur.round(4),
                    hide_index=True,
                    alt="Vektor fitur terstandarisasi yang dikirim ke Random Forest",
                )
                st.caption(
                    f"Log10 prediksi target: **{pred_log:.4f}** ➔ Inversi ($10^{{{pred_log:.4f}}}$) = "
                    f"**Rp {pred_rupiah:,.0f}**".replace(",", ".")
                )

# =========================================================
# TAB 2: EKSPLORASI DATA & EDA LENGKAP
# =========================================================
if tab_analisis.open:
    with tab_analisis:
        df_clean = get_clean_dataset()

        with st.container(border=True):
            ctrl1, ctrl2 = st.columns([3, 3], gap="large")
            with ctrl1:
                wil_filter = st.pills(
                    "Filter wilayah analisis",
                    options=kategori_wilayah,
                    selection_mode="multi",
                    default=kategori_wilayah,
                    key="eda_wilayah",
                )
            with ctrl2:
                eda_view = st.segmented_control(
                    "Fokus analisis EDA",
                    options=[
                        "Ringkasan & distribusi",
                        "Hubungan & korelasi",
                        "Wilayah, bias & fitur model",
                    ],
                    selection_mode="single",
                    default="Ringkasan & distribusi",
                    required=True,
                    key="eda_subview",
                    width="stretch",
                )

        if not wil_filter:
            wil_filter = kategori_wilayah
        if eda_view is None:
            eda_view = "Ringkasan & distribusi"

        df_filtered = df_clean[df_clean["wilayah_model"].isin(wil_filter)].copy()
        df_filtered["log_harga"] = np.log10(df_filtered["harga"])

        # KPI Ringkasan Sampel EDA
        med_harga_rp = float(df_filtered["harga"].median())
        mean_harga_rp = float(df_filtered["harga"].mean())
        _, med_ringkas = format_rupiah(med_harga_rp)
        _, mean_ringkas = format_rupiah(mean_harga_rp)
        porsi_1m = float((df_filtered["harga"] <= 1e9).mean() * 100.0)
        skew_asli = float(df_filtered["harga"].skew())
        skew_log = float(df_filtered["log_harga"].skew())

        with st.grid(4, gap="medium"):
            st.metric(
                label="Listing bersih terfilter",
                value=f"{len(df_filtered):,}",
                delta=f"dari {len(df_clean):,} total bersih",
                delta_color="off",
                delta_arrow="off",
                icon=":material/dataset:",
                border=True,
                height="stretch",
            )
            st.metric(
                label="Median harga penawaran",
                value=f"Rp {med_ringkas}",
                delta=f"Mean: Rp {mean_ringkas} ({mean_harga_rp / med_harga_rp:.2f}x)",
                delta_color="off",
                delta_arrow="off",
                icon=":material/payments:",
                border=True,
                height="stretch",
            )
            st.metric(
                label="Rumah tipikal (median LB / LT)",
                value=f"{df_filtered['luas_bangunan'].median():.0f} / {df_filtered['luas_tanah'].median():.0f} m²",
                delta=f"{df_filtered['kamar_tidur'].median():.0f} KT · {df_filtered['kamar_mandi'].median():.0f} KM",
                delta_color="off",
                delta_arrow="off",
                icon=":material/home_work:",
                border=True,
                height="stretch",
            )
            st.metric(
                label="Segmen <= Rp 1 miliar",
                value=f"{porsi_1m:.1f}%",
                delta=f"Skewness log10: {skew_log:.2f} (asli {skew_asli:.2f})",
                delta_color="off",
                delta_arrow="off",
                icon=":material/analytics:",
                border=True,
                height="stretch",
            )

        # -----------------------------------------------------
        # SUB-VIEW 1: RINGKASAN & DISTRIBUSI
        # -----------------------------------------------------
        if eda_view == "Ringkasan & distribusi":
            col_t1, col_t2 = st.columns(2, gap="large")
            with col_t1:
                with st.container(border=True):
                    st.subheader("Distribusi target harga (skala asli)", icon=":material/bar_chart:")
                    hist_raw = (
                        alt.Chart(df_filtered)
                        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                        .encode(
                            x=alt.X("harga_juta:Q", bin=alt.Bin(maxbins=35), title="Harga penawaran (juta Rp)"),
                            y=alt.Y("count():Q", title="Jumlah listing"),
                            tooltip=[alt.Tooltip("count():Q", title="Jumlah listing")],
                        )
                        .properties(height=250)
                    )
                    st.altair_chart(
                        hist_raw,
                        alt="Histogram distribusi harga penawaran rumah pada skala asli yang miring ke kanan",
                    )
                    st.caption(
                        f"Sangat miring ke kanan (skewness = **{skew_asli:.2f}**); ekor panjang rumah mewah menarik mean jauh di atas median."
                    )

            with col_t2:
                with st.container(border=True):
                    st.subheader("Distribusi target setelah log10", icon=":material/bar_chart:")
                    hist_log = (
                        alt.Chart(df_filtered)
                        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#059669")
                        .encode(
                            x=alt.X("log_harga:Q", bin=alt.Bin(maxbins=35), title="log10(Harga Rp)"),
                            y=alt.Y("count():Q", title="Jumlah listing"),
                            tooltip=[alt.Tooltip("count():Q", title="Jumlah listing")],
                        )
                        .properties(height=250)
                    )
                    st.altair_chart(
                        hist_log,
                        alt="Histogram distribusi log10 harga rumah yang jauh lebih simetris mendekati normal",
                    )
                    st.caption(
                        f"Transformasi $\\log_{{10}}$ menurunkan skewness menjadi **{skew_log:.2f}** dan menstabilkan error relatif (%)."
                    )

            col_f1, col_f2 = st.columns(2, gap="large")
            with col_f1:
                with st.container(border=True):
                    st.subheader("Distribusi fitur prediktor", icon=":material/insights:")
                    feat_choice = st.segmented_control(
                        "Pilih fitur",
                        options=["Luas bangunan", "Luas tanah", "Kamar tidur", "Kamar mandi", "Carport"],
                        selection_mode="single",
                        default="Luas bangunan",
                        required=True,
                        key="eda_dist_feat",
                    )
                    if feat_choice in ("Luas bangunan", "Luas tanah"):
                        col_map = {"Luas bangunan": "luas_bangunan", "Luas tanah": "luas_tanah"}[feat_choice]
                        use_log_feat = st.toggle("Tampilkan skala log10", value=True, key="eda_feat_log")
                        df_plot_f = df_filtered[[col_map]].copy()
                        if use_log_feat:
                            df_plot_f["nilai"] = np.log10(df_plot_f[col_map])
                            x_title = f"log10({feat_choice})"
                        else:
                            df_plot_f["nilai"] = df_plot_f[col_map]
                            x_title = f"{feat_choice} (m²)"
                        chart_feat = (
                            alt.Chart(df_plot_f)
                            .mark_bar(color="#D97706")
                            .encode(
                                x=alt.X("nilai:Q", bin=alt.Bin(maxbins=30), title=x_title),
                                y=alt.Y("count():Q", title="Jumlah listing"),
                                tooltip=[alt.Tooltip("count():Q", title="Listing")],
                            )
                            .properties(height=235)
                        )
                    else:
                        col_map = {
                            "Kamar tidur": "kamar_tidur",
                            "Kamar mandi": "kamar_mandi",
                            "Carport": "carport",
                        }[feat_choice]
                        vc = (
                            df_filtered[col_map]
                            .map(lambda v: "Kosong" if pd.isna(v) else str(int(v)))
                            .value_counts()
                            .reset_index()
                        )
                        vc.columns = ["Nilai", "Jumlah"]
                        vc["Status"] = vc["Nilai"].eq("Kosong").map({True: "Kosong (MNAR)", False: "Terisi"})
                        chart_feat = (
                            alt.Chart(vc)
                            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                            .encode(
                                x=alt.X("Nilai:N", sort=None, title=feat_choice),
                                y=alt.Y("Jumlah:Q", title="Jumlah listing"),
                                color=alt.Color(
                                    "Status:N",
                                    scale=alt.Scale(domain=["Terisi", "Kosong (MNAR)"], range=["#64748B", "#DC2626"]),
                                    legend=alt.Legend(title="", orient="bottom"),
                                ),
                                tooltip=["Nilai:N", "Jumlah:Q"],
                            )
                            .properties(height=275)
                        )
                    st.altair_chart(
                        chart_feat,
                        alt=f"Distribusi fitur {feat_choice} pada dataset rumah bersih Yogyakarta",
                    )

            with col_f2:
                with st.container(border=True):
                    st.subheader("Bukti anomali carport (MNAR)", icon=":material/fact_check:")
                    df_mnar = compute_carport_mnar_eda(df_filtered)
                    chart_mnar = (
                        alt.Chart(df_mnar)
                        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                        .encode(
                            x=alt.X("Grup_Carport:N", sort=["Kosong", "1", "2", "3+"], title="Kapasitas carport"),
                            y=alt.Y("Median_Harga_Juta:Q", title="Median harga (juta Rp)"),
                            color=alt.condition(
                                alt.datum.Grup_Carport == "Kosong",
                                alt.value("#DC2626"),
                                alt.value("#1D4ED8"),
                            ),
                            tooltip=[
                                alt.Tooltip("Grup_Carport:N", title="Carport"),
                                alt.Tooltip("Listing:Q"),
                                alt.Tooltip("Median_Harga_Juta:Q", format=",.0f", title="Median harga (Jt)"),
                                alt.Tooltip("Median_LB:Q", format=",.0f", title="Median LB (m²)"),
                                alt.Tooltip("Median_Harga_m2_LB_Juta:Q", format=",.2f", title="Harga/m² LB (Jt)"),
                            ],
                        )
                        .properties(height=180)
                    )
                    st.altair_chart(
                        chart_mnar,
                        alt="Perbandingan median harga kelompok carport kosong versus terisi 1, 2, dan 3+",
                    )
                    st.dataframe(
                        df_mnar,
                        hide_index=True,
                        column_config={
                            "Grup_Carport": "Carport",
                            "Median_Harga_Juta": st.column_config.NumberColumn("Median (Jt)", format="Rp %,.0f"),
                            "Median_LB": st.column_config.NumberColumn("LB (m²)", format="%.0f"),
                            "Median_LT": st.column_config.NumberColumn("LT (m²)", format="%.0f"),
                            "Median_Harga_m2_LB_Juta": st.column_config.NumberColumn("Harga/m² LB", format="Rp %.2f Jt"),
                        },
                        alt="Tabel ringkasan median harga dan luas bangunan menurut status carport",
                    )

            exp_desk = st.expander(
                "Tabel statistik deskriptif lengkap (persentil, skewness, missing)",
                icon=":material/table_chart:",
                key="exp_deskriptif",
                on_change="rerun",
            )
            if exp_desk.open:
                with exp_desk:
                    df_desk = compute_descriptive_stats(df_filtered)
                    st.dataframe(
                        df_desk,
                        hide_index=True,
                        column_config={
                            "count": st.column_config.NumberColumn("Count", format="%d"),
                            "mean": st.column_config.NumberColumn("Mean", format="%.2f"),
                            "std": st.column_config.NumberColumn("Std", format="%.2f"),
                            "min": st.column_config.NumberColumn("Min", format="%.1f"),
                            "5%": st.column_config.NumberColumn("P5", format="%.1f"),
                            "25%": st.column_config.NumberColumn("Q1", format="%.1f"),
                            "50%": st.column_config.NumberColumn("Median", format="%.1f"),
                            "75%": st.column_config.NumberColumn("Q3", format="%.1f"),
                            "95%": st.column_config.NumberColumn("P95", format="%.1f"),
                            "max": st.column_config.NumberColumn("Max", format="%.1f"),
                            "skewness": st.column_config.NumberColumn("Skew", format="%.2f"),
                            "missing": st.column_config.NumberColumn("Missing", format="%d"),
                        },
                        alt="Statistik deskriptif lengkap seluruh variabel numerik dataset bersih",
                    )

        # -----------------------------------------------------
        # SUB-VIEW 2: HUBUNGAN & KORELASI
        # -----------------------------------------------------
        elif eda_view == "Hubungan & korelasi":
            with st.container(border=True):
                st.subheader("Hubungan ukuran luas vs harga penawaran", icon=":material/scatter_plot:")
                sc_c1, sc_c2 = st.columns(2)
                with sc_c1:
                    x_feat_label = st.segmented_control(
                        "Sumbu X (ukuran properti)",
                        options=["Luas bangunan", "Luas tanah"],
                        selection_mode="single",
                        default="Luas bangunan",
                        required=True,
                        key="eda_scatter_x",
                    )
                with sc_c2:
                    scale_mode = st.segmented_control(
                        "Skala tampilan",
                        options=["Skala log-log (linear)", "Skala asli (corong)"],
                        selection_mode="single",
                        default="Skala log-log (linear)",
                        required=True,
                        key="eda_scatter_scale",
                    )

                x_col = "luas_bangunan" if x_feat_label == "Luas bangunan" else "luas_tanah"
                use_log_scale = scale_mode == "Skala log-log (linear)"
                if use_log_scale:
                    r_val = float(np.corrcoef(np.log10(df_filtered[x_col]), df_filtered["log_harga"])[0, 1])
                    x_enc = alt.X(f"{x_col}:Q", scale=alt.Scale(type="log"), title=f"{x_feat_label} (m², log)")
                    y_enc = alt.Y("harga_juta:Q", scale=alt.Scale(type="log"), title="Harga (juta Rp, log)")
                else:
                    r_val = float(df_filtered[[x_col, "harga"]].corr().iloc[0, 1])
                    x_enc = alt.X(f"{x_col}:Q", title=f"{x_feat_label} (m²)")
                    y_enc = alt.Y("harga_juta:Q", title="Harga (juta Rp)")

                scatter_rel = (
                    alt.Chart(df_filtered)
                    .mark_circle(size=45, opacity=0.55)
                    .encode(
                        x=x_enc,
                        y=y_enc,
                        color=alt.Color("wilayah_model:N", title="Wilayah"),
                        tooltip=[
                            alt.Tooltip("wilayah_model:N", title="Wilayah"),
                            alt.Tooltip("area:N", title="Area"),
                            alt.Tooltip("luas_bangunan:Q", title="LB (m²)"),
                            alt.Tooltip("luas_tanah:Q", title="LT (m²)"),
                            alt.Tooltip("harga_juta:Q", format=",.0f", title="Harga (Jt)"),
                        ],
                    )
                    .properties(height=300)
                    .interactive()
                )
                st.altair_chart(
                    scatter_rel,
                    alt=f"Scatter plot {x_feat_label} versus harga pada {scale_mode}",
                )
                st.caption(
                    f"Korelasi Pearson ({scale_mode}) untuk **{x_feat_label}**: **r = {r_val:.2f}** — "
                    "transformasi log-log menghilangkan pola corong heteroskedastis dan meluruskan tren."
                )

            col_b1, col_b2 = st.columns(2, gap="large")
            with col_b1:
                with st.container(border=True):
                    st.subheader("Sebaran harga menurut jumlah kamar", icon=":material/bed:")
                    room_choice = st.segmented_control(
                        "Variabel kamar",
                        options=["Kamar mandi", "Kamar tidur"],
                        selection_mode="single",
                        default="Kamar mandi",
                        required=True,
                        key="eda_room_col",
                    )
                    room_col = "kamar_mandi" if room_choice == "Kamar mandi" else "kamar_tidur"
                    df_room = df_filtered[df_filtered[room_col].between(1, 7)].copy()
                    df_room["Jumlah"] = df_room[room_col].astype(int).astype(str)

                    box_room = (
                        alt.Chart(df_room)
                        .mark_boxplot(size=28)
                        .encode(
                            x=alt.X("Jumlah:O", title=room_choice),
                            y=alt.Y("harga_juta:Q", scale=alt.Scale(type="log"), title="Harga (juta Rp, log)"),
                            color=alt.value("#0284C7"),
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(
                        box_room,
                        alt=f"Boxplot sebaran harga berdasarkan jumlah {room_choice}",
                    )

            with col_b2:
                with st.container(border=True):
                    st.subheader("Heatmap korelasi Spearman", icon=":material/analytics:")
                    long_corr, target_corr_df = compute_correlation_tables(df_filtered)
                    base_heat = alt.Chart(long_corr).encode(
                        x=alt.X("Variabel_X:N", title=""),
                        y=alt.Y("Variabel_Y:N", title=""),
                    )
                    rects = base_heat.mark_rect().encode(
                        color=alt.Color(
                            "Korelasi:Q",
                            scale=alt.Scale(scheme="blueorange", domain=[0, 1]),
                            legend=alt.Legend(title="Spearman"),
                        ),
                        tooltip=[
                            alt.Tooltip("Variabel_X:N", title="Variabel 1"),
                            alt.Tooltip("Variabel_Y:N", title="Variabel 2"),
                            alt.Tooltip("Korelasi:Q", format=".2f"),
                        ],
                    )
                    labels = base_heat.mark_text(fontWeight="bold", fontSize=11).encode(
                        text=alt.Text("Korelasi:Q", format=".2f"),
                        color=alt.condition(
                            alt.datum.Korelasi > 0.72,
                            alt.value("#FFFFFF"),
                            alt.value("#0F172A"),
                        ),
                    )
                    st.altair_chart(
                        (rects + labels).properties(height=320),
                        alt="Heatmap matriks korelasi Spearman antar fitur numerik dan target harga",
                    )

            with st.container(border=True):
                st.subheader("Perbandingan korelasi fitur terhadap harga", icon=":material/table_chart:")
                st.dataframe(
                    target_corr_df,
                    hide_index=True,
                    column_config={
                        "Spearman": st.column_config.NumberColumn("Spearman", format="%.2f"),
                        "Pearson (asli)": st.column_config.NumberColumn("Pearson (asli)", format="%.2f"),
                        "Pearson (log harga)": st.column_config.NumberColumn("Pearson (log harga)", format="%.2f"),
                        "Pearson (log-log)": st.column_config.NumberColumn("Pearson (log-log)", format="%.2f"),
                    },
                    alt="Perbandingan koefisien korelasi Spearman dan Pearson sebelum serta sesudah log transformasi",
                )

        # -----------------------------------------------------
        # SUB-VIEW 3: WILAYAH, BIAS & FITUR MODEL
        # -----------------------------------------------------
        else:
            df_market = summarize_market_by_wilayah(df_filtered)
            kruskal_res = compute_kruskal_wilayah(df_filtered)

            col_w1, col_w2 = st.columns([3, 2], gap="large")
            with col_w1:
                with st.container(border=True):
                    st.subheader("Sebaran harga per wilayah administratif", icon=":material/location_on:")
                    box_wil = (
                        alt.Chart(df_filtered)
                        .mark_boxplot(size=36)
                        .encode(
                            x=alt.X("wilayah_model:N", sort=list(df_market["Wilayah"]), title=""),
                            y=alt.Y("harga_juta:Q", scale=alt.Scale(type="log"), title="Harga (juta Rp, log)"),
                            color=alt.Color("wilayah_model:N", legend=None),
                        )
                        .properties(height=270)
                    )
                    st.altair_chart(
                        box_wil,
                        alt="Boxplot distribusi harga rumah per wilayah administratif Yogyakarta pada skala log",
                    )
                    if kruskal_res is not None:
                        h_stat, p_val = kruskal_res
                        st.caption(
                            f"Uji Kruskal-Wallis: **H = {h_stat:.1f}**, **p-value = {p_val:.2e}** → "
                            "disparitas harga antarwilayah signifikan secara statistik."
                        )
                    else:
                        st.caption("Pilih minimal 2 wilayah pada filter di atas untuk menjalankan uji Kruskal-Wallis.")

            with col_w2:
                with st.container(border=True):
                    st.subheader("Statistik median pasar per wilayah", icon=":material/table_chart:")
                    st.dataframe(
                        df_market,
                        hide_index=True,
                        column_config={
                            "Jumlah_Listing": st.column_config.NumberColumn("Listing", format="%d"),
                            "Porsi (%)": st.column_config.NumberColumn("Porsi", format="%.1f%%"),
                            "Median_Harga_Juta": st.column_config.NumberColumn("Median (Jt)", format="Rp %,.0f"),
                            "Median_LT": st.column_config.NumberColumn("LT (m²)", format="%.0f"),
                            "Median_LB": st.column_config.NumberColumn("LB (m²)", format="%.0f"),
                            "Median_Harga_m2_LB_Juta": st.column_config.NumberColumn("Harga/m² LB", format="Rp %.1f Jt"),
                        },
                        alt="Ringkasan jumlah listing, porsi, median harga, dan ukuran properti per wilayah",
                    )

            col_b, col_imp = st.columns(2, gap="large")
            with col_b:
                with st.container(border=True):
                    st.subheader("Komposisi segmen harga & bias sampel", icon=":material/bar_chart:")
                    df_seg = compute_price_segment_distribution(df_filtered)
                    seg_order = ["< 500 jt", "500 jt–1 M", "1–2 M", "2–5 M", "> 5 M"]
                    chart_seg = (
                        alt.Chart(df_seg)
                        .mark_bar()
                        .encode(
                            x=alt.X("Porsi (%):Q", stack="normalize", axis=alt.Axis(format="%"), title="Proporsi listing"),
                            y=alt.Y("Wilayah:N", sort=list(df_market["Wilayah"]), title=""),
                            color=alt.Color(
                                "Segmen Harga:N",
                                sort=seg_order,
                                legend=alt.Legend(title="Segmen", orient="bottom"),
                            ),
                            tooltip=[
                                alt.Tooltip("Wilayah:N"),
                                alt.Tooltip("Segmen Harga:N"),
                                alt.Tooltip("Porsi (%):Q", format=".1f", title="Porsi (%)"),
                            ],
                        )
                        .properties(height=250)
                    )
                    st.altair_chart(
                        chart_seg,
                        alt="Grafik batang bertumpuk proporsi segmen harga per wilayah",
                    )
                    st.caption(
                        "Sleman mendominasi ~72% sampel; Kulon Progo & Gunung Kidul berjumlah kecil sehingga digabung."
                    )

            with col_imp:
                with st.container(border=True):
                    st.subheader("Bobot kepentingan fitur (Random Forest)", icon=":material/insights:")
                    df_imp = build_feature_importance_df(bundle)
                    chart_imp = (
                        alt.Chart(df_imp)
                        .mark_bar(cornerRadiusEnd=6)
                        .encode(
                            x=alt.X("Kontribusi (%):Q", title="Kontribusi (%)"),
                            y=alt.Y("Fitur:N", sort="-x", title=""),
                            tooltip=[
                                alt.Tooltip("Fitur:N"),
                                alt.Tooltip("Kontribusi (%):Q", format=".2f"),
                            ],
                        )
                        .properties(height=180)
                    )
                    st.altair_chart(
                        chart_imp,
                        alt="Grafik batang horizontal bobot kepentingan fitur model Random Forest",
                    )
                    st.dataframe(
                        df_imp[["Fitur", "Kontribusi (%)"]],
                        hide_index=True,
                        column_config={
                            "Kontribusi (%)": st.column_config.ProgressColumn(
                                "Kontribusi (%)",
                                min_value=0.0,
                                max_value=100.0,
                                format="%.1f%%",
                            )
                        },
                        alt="Tabel bobot kepentingan fitur dalam persen",
                    )

            with st.container(border=True):
                st.subheader("5 insight utama EDA → keputusan modeling", icon=":material/fact_check:")
                st.markdown(
                    "1. **Ukuran fisik dominan:** `luas_bangunan` ($r_s = 0.87$) dan `luas_tanah` ($r_s = 0.80$) menjadi prediktor terkuat.\n"
                    "2. **Transformasi log-log:** menurunkan skewness harga ke $0.27$, meluruskan hubungan luas–harga, dan menstabilkan varians error.\n"
                    "3. **Multikolinearitas dimensi:** `kamar_tidur` ($r_s = 0.80$ terhadap LB & KM) terbukti redundan via VIF & drop-column CV → dieliminasi demi parsimoni.\n"
                    "4. **Anomali carport MNAR:** kelompok kosong punya harga/m² hampir identik dengan carport terisi (`Rp 10,69` vs `Rp 10,83 jt/m²`), sehingga variasi harga sudah dijelaskan oleh luas bangunan.\n"
                    "5. **Disparitas wilayah & bias sampel:** lokasi memberi premi harga nyata ($p \\ll 0.05$), namun sampel sangat timpang → ditangani dengan *StratifiedGroupKFold* dan penggabungan wilayah langka."
                )

# =========================================================
# TAB 3: METODOLOGI & EVALUASI MODEL
# =========================================================
if tab_metodologi.open:
    with tab_metodologi:
        eval_data = get_pipeline_evaluation(bundle)
        metrics = eval_data["metrics"]
        eval_wilayah = eval_data["eval_wilayah"]
        cleaning_log = eval_data["cleaning_log"]

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
                    F --> G["Random Forest (300 trees) + Estimasi Streamlit"]
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
                {"Komponen": "Algoritma model", "Implementasi": "RandomForestRegressor (n_estimators=300, max_depth=20, min_samples_leaf=2, min_samples_split=10)"},
            ])
            st.table(spec_df, alt="Ringkasan spesifikasi teknis tiap tahap pipeline machine learning")

        st.info(
            "Model memprediksi harga penawaran (asking price) listing rumah tinggal wajar di D.I. Yogyakarta. "
            "Harga transaksi riil dapat berbeda bergantung pada legalitas sertifikat (SHM/HGB), lebar akses jalan, "
            "kualitas material bangunan, dan hasil negosiasi.",
            title="Batasan interpretasi estimasi",
            icon=":material/info:",
        )

st.caption(
    f"Proyek Sains Data UPN \"Veteran\" Yogyakarta · Paket modular `src/house_price/` · Artefak aktif: `{artifact_name}`"
)
