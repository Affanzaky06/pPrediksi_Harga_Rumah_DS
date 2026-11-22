"""Seleksi fitur: Variance Inflation Factor (VIF), drop-column CV, dan uji redundansi."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, cross_val_score

from house_price.config import RANDOM_STATE


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Menghitung Variance Inflation Factor (VIF) untuk setiap kolom fitur."""
    vif: dict[str, float] = {}
    for kol in X.columns:
        lain = X.drop(columns=kol)
        r2 = LinearRegression().fit(lain, X[kol]).score(lain, X[kol])
        vif[kol] = np.inf if r2 >= 1 else 1.0 / (1.0 - r2)
    return pd.Series(vif, name="VIF").sort_values(ascending=False)


def rmse_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    kolom: list[str],
    groups: pd.Series,
    cv: GroupKFold,
) -> float:
    """Menghitung rata-rata RMSE cross-validation untuk subset kolom tertentu."""
    return float(
        -cross_val_score(
            model,
            X[kolom],
            y,
            groups=groups,
            cv=cv,
            scoring="neg_root_mean_squared_error",
        ).mean()
    )


def evaluate_drop_column_cv(
    X_train: pd.DataFrame,
    X_train_unscaled: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Menghitung drop-column importance (kenaikan RMSE CV 5-fold) pada Linear & Random Forest."""
    cv = GroupKFold(n_splits=5)
    models = {
        "Linear": (LinearRegression(), X_train),
        "RandomForest": (
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=random_state,
            ),
            X_train_unscaled,
        ),
    }
    grup_fitur = {
        "luas bangunan": ["log_luas_bangunan"],
        "luas tanah": ["log_luas_tanah"],
        "kamar mandi": ["kamar_mandi"],
        "kamar tidur": ["kamar_tidur"],
        "carport": ["carport"],
        "carport_kosong": ["carport_kosong"],
        "carport + carport_kosong": ["carport", "carport_kosong"],
        "wilayah (one-hot)": [c for c in X_train.columns if c.startswith("wilayah_")],
    }

    semua = list(X_train.columns)
    hasil: dict[str, dict[str, float]] = {}
    dasar: dict[str, float] = {}
    for nama_m, (m, X_mat) in models.items():
        dasar[nama_m] = rmse_cv(m, X_mat, y_train, semua, groups=groups, cv=cv)
        hasil[nama_m] = {
            g: rmse_cv(m, X_mat, y_train, [c for c in semua if c not in kol], groups=groups, cv=cv)
            - dasar[nama_m]
            for g, kol in grup_fitur.items()
        }

    drop_imp = pd.DataFrame(hasil).add_prefix("delta_RMSE_")
    drop_imp["delta_%error_Linear"] = (10 ** drop_imp["delta_RMSE_Linear"] - 1) * 100
    drop_imp = drop_imp.sort_values("delta_RMSE_Linear", ascending=False)
    return drop_imp, dasar


def verify_log_ratio_redundancy(X_train_unscaled: pd.DataFrame) -> float:
    """Menguji R^2 regresi log(LB/LT) terhadap log(LB) dan log(LT) untuk bukti kolinearitas."""
    rasio_log = X_train_unscaled["log_luas_bangunan"] - X_train_unscaled["log_luas_tanah"]
    cols = ["log_luas_bangunan", "log_luas_tanah"]
    return float(LinearRegression().fit(X_train_unscaled[cols], rasio_log).score(X_train_unscaled[cols], rasio_log))
