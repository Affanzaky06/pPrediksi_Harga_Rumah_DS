"""Evaluasi model regresi: metrik Rupiah/persentase, learning curve, dan per wilayah."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, learning_curve


def compute_regression_metrics(
    y_test_log: pd.Series | np.ndarray,
    y_pred_log: np.ndarray,
) -> dict[str, object]:
    """Menghitung metrik skala log dan skala Rupiah (MAE, RMSE, MAPE, MdAPE, R2)."""
    y_test_arr = np.asarray(y_test_log, dtype=float)
    y_pred_arr = np.asarray(y_pred_log, dtype=float)

    y_test_rupiah = 10.0 ** y_test_arr
    y_pred_rupiah = 10.0 ** y_pred_arr

    persen_error = np.abs((y_test_rupiah - y_pred_rupiah) / y_test_rupiah) * 100.0
    return {
        "r2": float(r2_score(y_test_arr, y_pred_arr)),
        "mse_log": float(mean_squared_error(y_test_arr, y_pred_arr)),
        "mae_rupiah": float(mean_absolute_error(y_test_rupiah, y_pred_rupiah)),
        "rmse_rupiah": float(np.sqrt(mean_squared_error(y_test_rupiah, y_pred_rupiah))),
        "mape": float(np.mean(persen_error)),
        "mdape": float(np.median(persen_error)),
        "y_test_rupiah": y_test_rupiah,
        "y_pred_rupiah": y_pred_rupiah,
    }


def evaluate_by_wilayah(test_df: pd.DataFrame, y_pred_rupiah: np.ndarray) -> pd.DataFrame:
    """Menghitung performa MAE, MdAPE, dan MAPE per segmen wilayah pada data test."""
    df_eval = test_df.copy()
    df_eval["harga_prediksi"] = np.asarray(y_pred_rupiah, dtype=float)
    df_eval["error_rupiah"] = np.abs(df_eval["harga"] - df_eval["harga_prediksi"])
    df_eval["persen_error"] = (df_eval["error_rupiah"] / df_eval["harga"]) * 100.0

    return (
        df_eval.groupby("wilayah_model")
        .agg(
            Jumlah_Sampel=("harga", "count"),
            MAE_Rupiah=("error_rupiah", "mean"),
            MdAPE_Persen=("persen_error", "median"),
            MAPE_Persen=("persen_error", "mean"),
        )
        .reset_index()
    )


def compute_learning_curve_stats(
    estimator,
    X_train_final: pd.DataFrame,
    y_train: pd.Series,
    cv: int | object = 5,
    groups: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Menghitung ukuran set, training MSE, dan validation MSE untuk learning curve."""
    cv_splitter = GroupKFold(n_splits=cv) if (groups is not None and isinstance(cv, int)) else cv
    train_sizes, train_scores, test_scores = learning_curve(
        estimator,
        X_train_final,
        y_train,
        groups=groups,
        cv=cv_splitter,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    train_mean = -np.mean(train_scores, axis=1)
    test_mean = -np.mean(test_scores, axis=1)
    return train_sizes, train_mean, test_mean
