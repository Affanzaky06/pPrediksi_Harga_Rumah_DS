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
import pandas as pd
import streamlit as st

from house_price.config import (
    DEFAULT_MDAPE_RATE,
    LEGACY_MODEL_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    WILAYAH_MERGE_MAP,
)
from house_price.data.cleaning import clean_listings, load_or_build_clean_data
from house_price.data.parsing import load_raw_data, parse_and_normalize
from house_price.features.preprocessing import load_or_build_preprocessed_data
from house_price.models.demo import (
    DEMO_PRESETS,
    build_feature_importance_df,
    simulate_wilayah_comparison,
    summarize_market_by_wilayah,
)
from house_price.models.evaluate import compute_regression_metrics, evaluate_by_wilayah
from house_price.models.predict import format_rupiah, predict_house_price
from house_price.models.train import load_model_bundle

st.set_page_config(
    page_title="Prediksi harga rumah D.I. Yogyakarta",
    page_icon=":material/real_estate_agent:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(max_entries=2)
def get_model_bundle() -> tuple[dict, str]:
    """Memuat artefak model modular dengan fallback otomatis ke artefak legacy."""
    try:
        bundle = load_model_bundle(
            path=MODEL_ARTIFACT_PATH,
            fallback_path=LEGACY_MODEL_ARTIFACT_PATH,
        )
        source_label = (
            MODEL_ARTIFACT_PATH.name
            if MODEL_ARTIFACT_PATH.exists()
            else LEGACY_MODEL_ARTIFACT_PATH.name
        )
        return bundle, source_label
    except FileNotFoundError:
        st.error(
            f"Artefak model tidak ditemukan di `{MODEL_ARTIFACT_PATH}` maupun "
            f"`{LEGACY_MODEL_ARTIFACT_PATH}`. Jalankan `python main.py` terlebih dahulu.",
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
    y_pred_log = _bundle["model"].predict(prep.X_test_final)
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
        ":material/analytics: Eksplorasi pasar & fitur",
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

        # Visualisasi Konteks Demo: Premi Lokasi & Posisi di Pasar Wilayah Terpilih
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
# TAB 2: EKSPLORASI PASAR & FITUR
# =========================================================
if tab_analisis.open:
    with tab_analisis:
        df_clean = get_clean_dataset()
        df_imp = build_feature_importance_df(bundle)

        col_chart, col_stat = st.columns([3, 2], gap="large")
        with col_chart:
            with st.container(border=True):
                st.subheader("Bobot kepentingan fitur (Random Forest)", icon=":material/bar_chart:")
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
                    .properties(height=260)
                )
                st.altair_chart(
                    chart_imp,
                    alt="Grafik batang horizontal bobot kepentingan fitur model Random Forest",
                )

        with col_stat:
            with st.container(border=True):
                st.subheader("Ringkasan kontribusi fitur", icon=":material/insights:")
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
                    alt="Tabel bobot kepentingan tiap fitur dalam persen",
                )
                st.caption(
                    "Luas bangunan dan luas tanah menyumbang lebih dari 94% keputusan prediksi harga; "
                    "kamar mandi bertindak sebagai proksi kelas kemewahan rumah."
                )

        st.subheader("Sebaran pasar properti bersih (1.384 listing)", icon=":material/dataset:")
        wil_filter = st.pills(
            "Filter wilayah yang ditampilkan",
            options=kategori_wilayah,
            selection_mode="multi",
            default=kategori_wilayah,
            key="eda_wilayah",
        )
        if not wil_filter:
            wil_filter = kategori_wilayah

        df_filtered = df_clean[df_clean["wilayah_model"].isin(wil_filter)]
        df_market = summarize_market_by_wilayah(df_filtered)

        col_eda1, col_eda2 = st.columns([3, 2], gap="large")
        with col_eda1:
            with st.container(border=True):
                st.markdown("**Hubungan luas bangunan vs harga penawaran (skala log-log)**")
                scatter_eda = (
                    alt.Chart(df_filtered)
                    .mark_circle(size=45, opacity=0.55)
                    .encode(
                        x=alt.X("luas_bangunan:Q", scale=alt.Scale(type="log"), title="Luas bangunan (m², skala log)"),
                        y=alt.Y("harga_juta:Q", scale=alt.Scale(type="log"), title="Harga (juta Rp, skala log)"),
                        color=alt.Color("wilayah_model:N", title="Wilayah"),
                        tooltip=[
                            alt.Tooltip("wilayah_model:N", title="Wilayah"),
                            alt.Tooltip("area:N", title="Area"),
                            alt.Tooltip("luas_bangunan:Q", title="LB (m²)"),
                            alt.Tooltip("luas_tanah:Q", title="LT (m²)"),
                            alt.Tooltip("harga_juta:Q", format=",.0f", title="Harga (Juta Rp)"),
                        ],
                    )
                    .properties(height=320)
                    .interactive()
                )
                st.altair_chart(
                    scatter_eda,
                    alt="Scatter plot log-log hubungan luas bangunan dan harga menurut wilayah",
                )

        with col_eda2:
            with st.container(border=True):
                st.markdown("**Statistik median pasar per wilayah**")
                st.dataframe(
                    df_market,
                    hide_index=True,
                    column_config={
                        "Jumlah_Listing": st.column_config.NumberColumn("Listing", format="%d"),
                        "Median_Harga_Juta": st.column_config.NumberColumn("Median harga (Jt)", format="Rp %,.0f"),
                        "Median_LT": st.column_config.NumberColumn("Median LT", format="%.0f m²"),
                        "Median_LB": st.column_config.NumberColumn("Median LB", format="%.0f m²"),
                        "Median_Harga_m2_LB_Juta": st.column_config.NumberColumn("Harga/m² LB", format="Rp %.1f Jt"),
                    },
                    alt="Ringkasan jumlah listing, median harga, dan ukuran properti per wilayah",
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
