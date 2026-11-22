"""Halaman 2: Eksplorasi data & EDA interaktif."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from house_price.models.demo import (
    build_feature_importance_df,
    compute_carport_mnar_eda,
    compute_correlation_tables,
    compute_descriptive_stats,
    compute_kruskal_wilayah,
    compute_price_segment_distribution,
    summarize_market_by_wilayah,
)
from house_price.models.predict import format_rupiah
from house_price.visualization.web_charts import (
    make_correlation_heatmap,
    make_histogram_chart,
)
from house_price.web.loaders import get_clean_dataset, get_model_bundle

bundle, _ = get_model_bundle()
kategori_wilayah: list[str] = list(bundle["kategori_wilayah"])
df_clean = get_clean_dataset()
is_dark = st.context.theme.type == "dark"

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

if eda_view == "Ringkasan & distribusi":
    col_t1, col_t2 = st.columns(2, gap="large")
    with col_t1:
        with st.container(border=True):
            st.subheader("Distribusi target harga (skala asli)", icon=":material/bar_chart:")
            st.altair_chart(
                make_histogram_chart(
                    df_filtered,
                    field="harga_juta",
                    x_title="Harga penawaran (juta Rp)",
                    color="#60A5FA" if is_dark else "#1D4ED8",
                ),
                alt="Histogram distribusi harga penawaran rumah pada skala asli yang miring ke kanan",
            )
            st.caption(
                f"Sangat miring ke kanan (skewness = **{skew_asli:.2f}**); ekor panjang rumah mewah menarik mean jauh di atas median."
            )

    with col_t2:
        with st.container(border=True):
            st.subheader("Distribusi target setelah log10", icon=":material/bar_chart:")
            st.altair_chart(
                make_histogram_chart(
                    df_filtered,
                    field="log_harga",
                    x_title="log10(Harga Rp)",
                    color="#34D399" if is_dark else "#059669",
                ),
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
                chart_feat = make_histogram_chart(
                    df_plot_f,
                    field="nilai",
                    x_title=x_title,
                    color="#FBBF24" if is_dark else "#D97706",
                    maxbins=30,
                    height=235,
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
                            scale=alt.Scale(
                                domain=["Terisi", "Kosong (MNAR)"],
                                range=["#94A3B8" if is_dark else "#64748B", "#F87171" if is_dark else "#DC2626"],
                            ),
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
                        alt.value("#F87171" if is_dark else "#DC2626"),
                        alt.value("#60A5FA" if is_dark else "#1D4ED8"),
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
                    color=alt.value("#38BDF8" if is_dark else "#0284C7"),
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
            st.altair_chart(
                make_correlation_heatmap(long_corr, is_dark=is_dark),
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
