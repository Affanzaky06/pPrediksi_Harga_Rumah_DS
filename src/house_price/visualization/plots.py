"""Fungsi visualisasi modular untuk Audit Data, EDA, dan Evaluasi Model."""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns


def format_juta(ax, axis: str = "y") -> None:
    """Memformat sumbu angka agar mudah dibaca dengan pemisah ribuan."""
    formatter = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(formatter)


def plot_missing_bars(missing_df: pd.DataFrame) -> plt.Figure:
    """Plot horizontal bar persentase missing value per kolom."""
    fig, ax = plt.subplots(figsize=(7, 3))
    sub = missing_df.loc[missing_df["missing"] > 0, "persen_%"].sort_values()
    sub.plot.barh(ax=ax, color="#c44e52")
    ax.set(title="Persentase missing value per kolom", xlabel="% baris kosong")
    for p in ax.patches:
        ax.annotate(
            f"{p.get_width():.1f}%",
            (p.get_width(), p.get_y() + p.get_height() / 2),
            va="center",
            fontsize=9,
        )
    fig.tight_layout()
    return fig


def plot_raw_boxplots(data: pd.DataFrame, num_cols: list[str]) -> plt.Figure:
    """Plot boxplot skala log10 untuk mendeteksi outlier IQR pada data mentah."""
    fig, axes = plt.subplots(1, len(num_cols), figsize=(18, 4))
    for ax, k in zip(axes, num_cols):
        sns.boxplot(y=data[k], ax=ax, color="#8172b3")
        ax.set_yscale("log")
        ax.set_title(k)
        ax.set_ylabel("")
    fig.suptitle("Boxplot data mentah (skala log) — titik di luar whisker = outlier IQR", y=1.02)
    fig.tight_layout()
    return fig


def plot_before_after_histograms(
    data_before: pd.DataFrame,
    df_clean: pd.DataFrame,
    num_cols: list[str],
) -> plt.Figure:
    """Plot perbandingan distribusi log10 sebelum vs sesudah cleaning."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, k in zip(axes.ravel(), num_cols):
        s0, s1 = data_before[k].dropna(), df_clean[k].dropna()
        bins = np.histogram_bin_edges(np.log10(s0.clip(lower=1)), bins=40)
        ax.hist(np.log10(s0.clip(lower=1)), bins=bins, alpha=0.5, label=f"sebelum (n={len(s0)})", color="#c44e52")
        ax.hist(np.log10(s1.clip(lower=1)), bins=bins, alpha=0.6, label=f"sesudah (n={len(s1)})", color="#4c72b0")
        ax.set(title=k, xlabel=f"log10({k})")
        ax.legend(fontsize=8)
    fig.suptitle("Distribusi sebelum vs sesudah cleaning (skala log10 agar nilai ekstrem terlihat)", y=1.01)
    fig.tight_layout()
    return fig


def plot_target_distribution(df_clean: pd.DataFrame) -> plt.Figure:
    """Plot distribusi harga asli, log10(harga), dan Q-Q plot normalitas."""
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    sns.histplot(df_clean["harga_juta"], bins=50, ax=ax[0], color="#55a868")
    ax[0].axvline(df_clean["harga_juta"].median(), color="k", ls="--", label="median")
    ax[0].axvline(df_clean["harga_juta"].mean(), color="r", ls=":", label="mean")
    ax[0].set(title="Harga (skala asli)", xlabel="Harga (juta Rp)")
    ax[0].legend()
    format_juta(ax[0], "x")

    log_harga = np.log10(df_clean["harga"])
    sns.histplot(log_harga, bins=40, ax=ax[1], color="#55a868", kde=True)
    ax[1].set(title="Harga setelah log10", xlabel="log10(harga Rp)")

    stats.probplot(log_harga, dist="norm", plot=ax[2])
    ax[2].set_title("Q-Q plot log10(harga) vs distribusi normal")
    fig.tight_layout()
    return fig


def plot_feature_distributions(df_clean: pd.DataFrame) -> plt.Figure:
    """Plot distribusi luas tanah/bangunan (asli & log10) serta kamar & carport."""
    fig, ax = plt.subplots(2, 3, figsize=(17, 8))
    sns.histplot(df_clean["luas_tanah"], bins=50, ax=ax[0, 0], color="#dd8452")
    ax[0, 0].set_title("Luas tanah (asli)")
    sns.histplot(np.log10(df_clean["luas_tanah"]), bins=40, ax=ax[0, 1], color="#dd8452")
    ax[0, 1].set_title("log10(luas tanah)")
    sns.histplot(np.log10(df_clean["luas_bangunan"]), bins=40, ax=ax[0, 2], color="#dd8452")
    ax[0, 2].set_title("log10(luas bangunan)")

    for a, k in zip(ax[1], ["kamar_tidur", "kamar_mandi", "carport"]):
        vc = df_clean[k].value_counts(dropna=False).sort_index()
        vc.index = ["kosong" if pd.isna(i) else str(int(i)) for i in vc.index]
        vc.plot.bar(ax=a, color=["#c44e52" if i == "kosong" else "#8c8c8c" for i in vc.index])
        a.set(title=f"Jumlah {k.replace('_', ' ')}", xlabel="", ylabel="listing")
    fig.tight_layout()
    return fig


def plot_area_vs_price_loglog(df_clean: pd.DataFrame) -> plt.Figure:
    """Plot perbandingan scatter skala asli vs log-log untuk luas bangunan dan luas tanah."""
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    for j, k in enumerate(["luas_bangunan", "luas_tanah"]):
        sns.scatterplot(data=df_clean, x=k, y="harga_juta", ax=ax[0, j], s=12, alpha=0.5)
        ax[0, j].set(title=f"{k} vs harga (skala asli)", ylabel="Harga (juta Rp)")
        r_asli = df_clean[[k, "harga"]].corr().iloc[0, 1]
        ax[0, j].text(0.02, 0.93, f"Pearson r = {r_asli:.2f}", transform=ax[0, j].transAxes, fontsize=11, weight="bold")

        lx, ly = np.log10(df_clean[k]), np.log10(df_clean["harga"])
        sns.regplot(x=lx, y=ly, ax=ax[1, j], scatter_kws={"s": 10, "alpha": 0.4}, line_kws={"color": "r"})
        ax[1, j].set(title=f"log10({k}) vs log10(harga)", xlabel=f"log10({k})", ylabel="log10(harga)")
        ax[1, j].text(
            0.02,
            0.93,
            f"Pearson r = {np.corrcoef(lx, ly)[0, 1]:.2f}",
            transform=ax[1, j].transAxes,
            fontsize=11,
            weight="bold",
        )
    fig.tight_layout()
    return fig


def plot_room_carport_boxplots(df_clean: pd.DataFrame) -> plt.Figure:
    """Plot boxplot harga menurut jumlah kamar tidur, kamar mandi, dan carport."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for a, k in zip(ax, ["kamar_tidur", "kamar_mandi"]):
        sub = df_clean[df_clean[k].between(1, 7)]
        sns.boxplot(data=sub, x=k, y="harga_juta", ax=a, color="#64b5cd", showfliers=False)
        a.set_yscale("log")
        a.set(title=f"Harga menurut {k.replace('_', ' ')} (1–7)", ylabel="Harga (juta Rp, log)")

    cp = df_clean.assign(
        carport_grup=df_clean["carport"].map(
            lambda v: "kosong" if pd.isna(v) else ("4+" if v >= 4 else str(int(v)))
        )
    )
    sns.boxplot(
        data=cp,
        x="carport_grup",
        y="harga_juta",
        order=["kosong", "1", "2", "3", "4+"],
        ax=ax[2],
        color="#64b5cd",
        showfliers=False,
    )
    ax[2].set_yscale("log")
    ax[2].set(title="Harga menurut carport", xlabel="carport", ylabel="Harga (juta Rp, log)")
    fig.tight_layout()
    return fig


def plot_wilayah_boxplot(df_clean: pd.DataFrame, wil_summary: pd.DataFrame) -> plt.Figure:
    """Plot boxplot sebaran harga per wilayah administratif."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(data=df_clean, x="wilayah", y="harga_juta", order=wil_summary.index, ax=ax, color="#937860")
    ax.set_yscale("log")
    ax.set(title="Harga per wilayah administratif", xlabel="", ylabel="Harga (juta Rp, log)")
    for i, w in enumerate(wil_summary.index):
        ax.text(i, ax.get_ylim()[1] * 0.6, f"n={wil_summary.loc[w, 'listing']}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(kor_df: pd.DataFrame) -> plt.Figure:
    """Plot heatmap segitiga bawah korelasi Spearman."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(
        kor_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        ax=ax,
        mask=np.triu(np.ones_like(kor_df, bool), 1),
    )
    ax.set_title("Korelasi Spearman (data bersih)")
    fig.tight_layout()
    return fig


def plot_sample_bias(df_clean: pd.DataFrame, wil_summary: pd.DataFrame) -> plt.Figure:
    """Plot komposisi porsi wilayah dan segmen harga per wilayah."""
    segmen = pd.cut(
        df_clean["harga"],
        [0, 5e8, 1e9, 2e9, 5e9, np.inf],
        labels=["< 500 jt", "500 jt–1 M", "1–2 M", "2–5 M", "> 5 M"],
    )
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    wil_summary["porsi_%"].sort_values().plot.barh(ax=ax[0], color="#4c72b0")
    ax[0].set(title="Porsi listing per wilayah (%)", xlabel="%")
    (pd.crosstab(df_clean["wilayah"], segmen, normalize="index") * 100).plot.barh(
        stacked=True,
        ax=ax[1],
        colormap="viridis",
    )
    ax[1].set(title="Komposisi segmen harga per wilayah (%)", xlabel="%", ylabel="")
    ax[1].legend(fontsize=8, bbox_to_anchor=(1, 1))
    fig.tight_layout()
    return fig


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_mean: np.ndarray,
    test_mean: np.ndarray,
) -> plt.Figure:
    """Plot learning curve MSE training vs cross-validation."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training error")
    ax.plot(train_sizes, test_mean, "o-", color="green", label="Cross-validation error")
    ax.set(title="Learning Curve", xlabel="Training Set Size", ylabel="MSE")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(
    y_test_rupiah: np.ndarray,
    y_pred_rupiah: np.ndarray,
    r2: float,
) -> plt.Figure:
    """Plot scatter harga aktual vs prediksi dalam satuan Miliar Rupiah."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.scatter(y_test_rupiah / 1e9, y_pred_rupiah / 1e9, alpha=0.5, color="royalblue", edgecolor="k")
    maks = max(float(np.max(y_test_rupiah)), float(np.max(y_pred_rupiah))) / 1e9
    ax.plot([0, maks], [0, maks], "r--", lw=2, label="Prediksi Sempurna (y=x)")
    ax.set(
        xlabel="Harga Aktual (Miliar Rp)",
        ylabel="Harga Prediksi (Miliar Rp)",
        title=f"Harga Aktual vs Prediksi — R² = {r2:.2f}",
    )
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list[str]) -> plt.Figure:
    """Plot horizontal bar bobot feature importance Random Forest."""
    importansi = pd.Series(model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    importansi.plot(kind="barh", color="teal", ax=ax)
    ax.set(title="Faktor Paling Berpengaruh Terhadap Harga (Random Forest)", xlabel="Importance Score")
    fig.tight_layout()
    return fig


def plot_residuals(
    y_pred_rupiah: np.ndarray,
    residual_rupiah: np.ndarray,
    residual_log: pd.Series | np.ndarray,
) -> plt.Figure:
    """Plot residual vs prediksi (skala Miliar Rp) dan histogram error residual (log10)."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    sns.scatterplot(x=y_pred_rupiah / 1e9, y=residual_rupiah / 1e9, alpha=0.5, color="crimson", ax=ax[0])
    ax[0].axhline(0, color="black", linestyle="--", lw=1.5)
    ax[0].set(
        title="Residual vs Prediksi (Miliar Rp)",
        xlabel="Harga Prediksi (Miliar Rp)",
        ylabel="Residual: Aktual - Prediksi (Miliar Rp)",
    )

    sns.histplot(residual_log, kde=True, color="teal", ax=ax[1])
    ax[1].axvline(0, color="red", linestyle="--", lw=1.5)
    ax[1].set(title="Distribusi Error Residual (Skala Log10)", xlabel="Residual (Log10)")
    fig.tight_layout()
    return fig
