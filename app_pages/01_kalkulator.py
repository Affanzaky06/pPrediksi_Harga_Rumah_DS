"""Halaman 1: Simulasi & kalkulator harga properti."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from house_price.config import DEFAULT_MDAPE_RATE
from house_price.models.demo import DEMO_PRESETS, simulate_wilayah_comparison
from house_price.models.predict import format_rupiah, predict_house_price
from house_price.visualization.web_charts import (
    make_market_position_chart,
    make_wilayah_sim_chart,
)
from house_price.web.loaders import (
    apply_demo_preset,
    get_clean_dataset,
    get_model_bundle,
    get_pipeline_evaluation,
)

bundle, _ = get_model_bundle()
kategori_wilayah: list[str] = list(bundle["kategori_wilayah"])
median_train: pd.Series = bundle["median_train"]
df_clean = get_clean_dataset()
eval_data = get_pipeline_evaluation(bundle)
test_eval_df: pd.DataFrame = eval_data["test_eval_df"]
is_dark = st.context.theme.type == "dark"


def _pick_random_test_sample() -> None:
    n_rows = len(test_eval_df)
    current = st.session_state.get("sampled_test_idx")
    rng = np.random.default_rng()
    idx = int(rng.integers(0, n_rows))
    if n_rows > 1 and idx == current:
        idx = (idx + 1) % n_rows
    st.session_state["sampled_test_idx"] = idx


def _load_sample_into_calculator() -> None:
    idx = st.session_state.get("sampled_test_idx")
    if idx is None:
        return
    row = test_eval_df.iloc[int(idx)]
    st.session_state["demo_preset"] = None
    st.session_state["inp_wilayah"] = str(row["wilayah_model"])
    st.session_state["inp_lt"] = float(np.clip(row["luas_tanah"], 20.0, 3000.0))
    st.session_state["inp_lb"] = float(np.clip(row["luas_bangunan"], 20.0, 2000.0))
    st.session_state["inp_km"] = int(np.clip(row["kamar_mandi"], 1, 15))
    st.session_state["inp_kt"] = int(np.clip(row["kamar_tidur"], 1, 20))
    st.session_state["inp_cp"] = int(np.clip(row["carport"], 0, 10))


# Preset skenario cepat
with st.container(border=True):
    st.markdown("**Skenario cepat untuk demo presentasi**")
    st.pills(
        "Pilih contoh profil rumah",
        options=list(DEMO_PRESETS.keys()),
        selection_mode="single",
        key="demo_preset",
        on_change=apply_demo_preset,
        label_visibility="collapsed",
        help="Klik salah satu skenario untuk mengisi parameter spesifikasi secara otomatis.",
    )

# Panel input utama (hanya 4 fitur aktif yang memengaruhi prediksi)
with st.container(border=True):
    st.subheader("Spesifikasi utama properti", icon=":material/location_on:")
    col_loc, col_dim = st.columns(2, gap="large")

    with col_loc:
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

        kamar_mandi = st.number_input(
            "Jumlah kamar mandi",
            min_value=1,
            max_value=15,
            step=1,
            key="inp_km",
            help=f"Median data training: {int(median_train.get('kamar_mandi', 2))}. Fitur aktif dalam model.",
        )

    with col_dim:
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

    exp_compat = st.expander(
        "Input kompatibilitas scaler (opsional — tidak memengaruhi estimasi)",
        icon=":material/tune:",
        key="exp_compat",
    )
    with exp_compat:
        st.caption(
            "Hasil seleksi fitur (VIF & drop-column CV) mengeliminasi `kamar_tidur` dan `carport` "
            "karena redundan terhadap luas bangunan dan kamar mandi. Nilai di bawah hanya disimpan "
            "untuk kompatibilitas skema `StandardScaler`."
        )
        c_kt, c_cp = st.columns(2)
        with c_kt:
            kamar_tidur = st.number_input(
                "Jumlah kamar tidur",
                min_value=1,
                max_value=20,
                step=1,
                key="inp_kt",
            )
        with c_cp:
            carport = st.number_input(
                "Kapasitas carport (mobil)",
                min_value=0,
                max_value=10,
                step=1,
                key="inp_cp",
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

# Kartu hasil estimasi
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

# Uji sampel nyata dari unseen test set
with st.container(border=True):
    test_hdr, test_btn_col = st.columns([3, 2], vertical_alignment="center")
    with test_hdr:
        st.subheader("Uji rumah nyata dari data test (unseen)", icon=":material/fact_check:")
        st.caption(
            f"Validasi langsung dengan mengambil 1 listing acak dari **{len(test_eval_df)}** data uji "
            "yang tidak pernah dilihat saat training."
        )
    with test_btn_col:
        with st.container(horizontal=True, horizontal_alignment="right"):
            st.button(
                "Ambil sampel acak test set",
                icon=":material/bolt:",
                key="btn_sample_test",
                on_click=_pick_random_test_sample,
            )
            if st.session_state.get("sampled_test_idx") is not None:
                st.button(
                    "Muat ke kalkulator",
                    icon=":material/calculate:",
                    key="btn_load_sample_to_calc",
                    on_click=_load_sample_into_calculator,
                )

    sampled_idx = st.session_state.get("sampled_test_idx")
    if sampled_idx is not None:
        sample = test_eval_df.iloc[int(sampled_idx)]
        _, act_short = format_rupiah(float(sample["harga"]))
        _, pred_short = format_rupiah(float(sample["harga_prediksi"]))
        st.caption(
            f"Sampel #{int(sampled_idx) + 1}: **{sample['area']}, {sample['wilayah_model']}** · "
            f"LT **{sample['luas_tanah']:.0f} m²** · LB **{sample['luas_bangunan']:.0f} m²** · "
            f"**{int(sample['kamar_mandi'])} KM**"
        )
        with st.grid(3, gap="medium"):
            st.metric(
                label="Harga aktual di iklan",
                value=f"Rp {act_short}",
                delta=format_rupiah(float(sample["harga"]))[0],
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
            )
            st.metric(
                label="Prediksi model",
                value=f"Rp {pred_short}",
                delta=format_rupiah(float(sample["harga_prediksi"]))[0],
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
            )
            st.metric(
                label="Selisih error absolut",
                value=f"{sample['error_persen']:.2f}%",
                delta="Pada unseen test set",
                delta_color="off",
                delta_arrow="off",
                border=True,
                height="stretch",
            )

# Dua visualisasi kontekstual berdampingan
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
        st.altair_chart(
            make_wilayah_sim_chart(df_sim, is_dark=is_dark),
            alt="Perbandingan estimasi harga untuk spesifikasi yang sama di seluruh wilayah DIY",
        )

with col_mkt:
    with st.container(border=True):
        st.subheader(f"Posisi terhadap pasar {wilayah_pilihan}", icon=":material/scatter_plot:")
        st.caption("Membandingkan hasil prediksi (titik merah) dengan listing nyata di wilayah terpilih:")
        df_wil_market = df_clean[df_clean["wilayah_model"] == wilayah_pilihan][
            ["luas_bangunan", "luas_tanah", "harga_juta", "area"]
        ].copy()
        st.altair_chart(
            make_market_position_chart(
                df_wil_market,
                luas_bangunan=float(luas_bangunan),
                pred_rupiah=pred_rupiah,
                wilayah=wilayah_pilihan,
                is_dark=is_dark,
            ),
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
