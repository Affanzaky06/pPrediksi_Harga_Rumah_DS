"""Kumpulan builder chart Altair untuk aplikasi web Streamlit (theme-aware)."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd


def make_wilayah_sim_chart(df_sim: pd.DataFrame, is_dark: bool = False) -> alt.Chart:
    """Bar chart perbandingan estimasi harga lintas wilayah untuk spek yang sama."""
    active_color = "#60A5FA" if is_dark else "#1D4ED8"
    muted_color = "#475569" if is_dark else "#94A3B8"
    return (
        alt.Chart(df_sim)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X("Estimasi (Miliar Rp):Q", title="Estimasi harga (Miliar Rp)"),
            y=alt.Y("Wilayah:N", sort="-x", title=""),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(domain=["Dipilih", "Pembanding"], range=[active_color, muted_color]),
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


def make_market_position_chart(
    df_wil_market: pd.DataFrame,
    luas_bangunan: float,
    pred_rupiah: float,
    wilayah: str,
    is_dark: bool = False,
) -> alt.LayerChart:
    """Scatter pasar wilayah terpilih beserta titik posisi simulasi properti saat ini."""
    df_point = pd.DataFrame([{
        "luas_bangunan": float(luas_bangunan),
        "harga_juta": pred_rupiah / 1e6,
        "area": wilayah,
        "Kategori": "Simulasi Anda",
    }])
    market_color = "#94A3B8" if is_dark else "#64748B"
    highlight_color = "#F87171" if is_dark else "#DC2626"

    base_pts = (
        alt.Chart(df_wil_market)
        .mark_circle(size=55, opacity=0.45)
        .encode(
            x=alt.X("luas_bangunan:Q", title="Luas bangunan (m²)"),
            y=alt.Y("harga_juta:Q", title="Harga penawaran (juta Rp)"),
            color=alt.value(market_color),
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
            color=alt.value(highlight_color),
            tooltip=[
                alt.Tooltip("Kategori:N"),
                alt.Tooltip("luas_bangunan:Q", title="LB (m²)"),
                alt.Tooltip("harga_juta:Q", format=",.0f", title="Prediksi (Juta Rp)"),
            ],
        )
    )
    return (base_pts + sim_pt).properties(height=260).interactive()


def make_histogram_chart(
    df: pd.DataFrame,
    field: str,
    x_title: str,
    color: str = "#1D4ED8",
    maxbins: int = 35,
    height: int = 250,
) -> alt.Chart:
    """Histogram Altair serbaguna."""
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=color)
        .encode(
            x=alt.X(f"{field}:Q", bin=alt.Bin(maxbins=maxbins), title=x_title),
            y=alt.Y("count():Q", title="Jumlah listing"),
            tooltip=[alt.Tooltip("count():Q", title="Jumlah listing")],
        )
        .properties(height=height)
    )


def make_correlation_heatmap(long_corr: pd.DataFrame, is_dark: bool = False) -> alt.LayerChart:
    """Heatmap matriks korelasi Spearman dengan anotasi angka di setiap sel."""
    low_text = "#F1F5F9" if is_dark else "#0F172A"
    base = alt.Chart(long_corr).encode(
        x=alt.X("Variabel_X:N", title=""),
        y=alt.Y("Variabel_Y:N", title=""),
    )
    rects = base.mark_rect().encode(
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
    labels = base.mark_text(fontWeight="bold", fontSize=11).encode(
        text=alt.Text("Korelasi:Q", format=".2f"),
        color=alt.condition(
            alt.datum.Korelasi > 0.72,
            alt.value("#FFFFFF"),
            alt.value(low_text),
        ),
    )
    return (rects + labels).properties(height=320)


def make_actual_vs_pred_chart(test_eval_df: pd.DataFrame, is_dark: bool = False) -> alt.LayerChart:
    """Scatter plot harga aktual vs prediksi pada data test beserta garis diagonal y=x."""
    maks = float(max(test_eval_df["harga_aktual_miliar"].max(), test_eval_df["harga_prediksi_miliar"].max()))
    diag_df = pd.DataFrame({"x": [0.0, maks], "y": [0.0, maks]})

    pts = (
        alt.Chart(test_eval_df)
        .mark_circle(size=55, opacity=0.6)
        .encode(
            x=alt.X("harga_aktual_miliar:Q", title="Harga aktual (Miliar Rp)"),
            y=alt.Y("harga_prediksi_miliar:Q", title="Harga prediksi (Miliar Rp)"),
            color=alt.Color("wilayah_model:N", title="Wilayah"),
            tooltip=[
                alt.Tooltip("wilayah_model:N", title="Wilayah"),
                alt.Tooltip("area:N", title="Area"),
                alt.Tooltip("harga_aktual_miliar:Q", format=",.2f", title="Aktual (Miliar Rp)"),
                alt.Tooltip("harga_prediksi_miliar:Q", format=",.2f", title="Prediksi (Miliar Rp)"),
                alt.Tooltip("error_persen:Q", format=".1f", title="Error (%)"),
            ],
        )
    )
    line_color = "#F87171" if is_dark else "#DC2626"
    diag = (
        alt.Chart(diag_df)
        .mark_line(strokeDash=[6, 4], strokeWidth=2, color=line_color)
        .encode(x="x:Q", y="y:Q")
    )
    return (pts + diag).properties(height=280).interactive()


def make_residual_hist_chart(test_eval_df: pd.DataFrame, is_dark: bool = False) -> alt.LayerChart:
    """Histogram distribusi residual skala log10 dengan garis referensi nol."""
    bar_color = "#34D399" if is_dark else "#059669"
    line_color = "#F87171" if is_dark else "#DC2626"
    hist = (
        alt.Chart(test_eval_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=bar_color)
        .encode(
            x=alt.X("residual_log:Q", bin=alt.Bin(maxbins=30), title="Residual log10 (Aktual - Prediksi)"),
            y=alt.Y("count():Q", title="Jumlah sampel uji"),
            tooltip=[alt.Tooltip("count():Q", title="Sampel")],
        )
    )
    zero_rule = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(
        strokeDash=[5, 4], strokeWidth=2, color=line_color
    ).encode(x="x:Q")
    return (hist + zero_rule).properties(height=280)


def make_benchmark_chart(df_bench: pd.DataFrame, is_dark: bool = False) -> alt.Chart:
    """Bar chart perbandingan CV RMSE (log10) antar 5 kandidat model."""
    best_color = "#34D399" if is_dark else "#059669"
    other_color = "#60A5FA" if is_dark else "#94A3B8"
    df_plot = df_bench.copy()
    best_model = df_plot.sort_values("CV RMSE (log)").iloc[0]["Model"]
    df_plot["Status"] = np.where(df_plot["Model"] == best_model, "Terpilih", "Pembanding")

    tooltips = [alt.Tooltip("Model:N")]
    if "Baseline CV RMSE (log)" in df_plot.columns:
        tooltips.append(alt.Tooltip("Baseline CV RMSE (log):Q", format=".4f", title="Baseline RMSE"))
    tooltips.extend([
        alt.Tooltip("CV RMSE (log):Q", format=".4f", title="Tuned CV RMSE"),
        alt.Tooltip("CV MAE (log):Q", format=".4f", title="Tuned CV MAE"),
        alt.Tooltip("CV R2:Q", format=".4f", title="Tuned CV R²"),
    ])
    if "Best Hyperparameters" in df_plot.columns:
        tooltips.append(alt.Tooltip("Best Hyperparameters:N", title="Best Params"))

    return (
        alt.Chart(df_plot)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X("CV RMSE (log):Q", title="Tuned CV RMSE (skala log10 — makin kecil makin baik)"),
            y=alt.Y("Model:N", sort="x", title=""),
            color=alt.Color(
                "Status:N",
                scale=alt.Scale(domain=["Terpilih", "Pembanding"], range=[best_color, other_color]),
                legend=None,
            ),
            tooltip=tooltips,
        )
        .properties(height=220)
    )
