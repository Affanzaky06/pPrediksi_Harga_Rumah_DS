"""Fungsi inferensi tunggal, batch aman-thread, dan pemformatan harga Rupiah."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


def format_rupiah(angka: float) -> tuple[str, str]:
    """Memformat angka float ke format Rupiah standar Indonesia dan ringkasannya."""
    if angka >= 1_000_000_000:
        return f"Rp {angka:,.0f}".replace(",", "."), f"{angka / 1_000_000_000:.2f} Miliar"
    if angka >= 1_000_000:
        return f"Rp {angka:,.0f}".replace(",", "."), f"{angka / 1_000_000:.1f} Juta"
    return f"Rp {angka:,.0f}".replace(",", "."), ""


def _validate_positive_finite(value: float, name: str) -> float:
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise ValueError(f"Parameter '{name}' harus berupa angka finite > 0, diterima: {value!r}")
    return val


def _validate_non_negative_finite(value: float, name: str) -> float:
    val = float(value)
    if not math.isfinite(val) or val < 0.0:
        raise ValueError(f"Parameter '{name}' harus berupa angka finite >= 0, diterima: {value!r}")
    return val


def predict_batch_log(model, X_df: pd.DataFrame) -> np.ndarray:
    """Menjalankan prediksi skala log10 tanpa memicu joblib.Parallel pada inferensi web.

    Untuk RandomForestRegressor, merata-ratakan prediksi tiap decision tree secara langsung
    menghindari overhead threadpool dan race condition `warnings.resetwarnings()` di
    `sklearn.utils.parallel` pada Python 3.11 / Streamlit.
    """
    if hasattr(model, "estimators_") and getattr(model, "n_outputs_", 1) == 1:
        expected_cols = getattr(model, "feature_names_in_", None)
        if expected_cols is not None:
            X_ordered = X_df[list(expected_cols)]
        else:
            X_ordered = X_df
        X_arr = X_ordered.to_numpy(dtype=np.float32)
        if not np.all(np.isfinite(X_arr)):
            raise ValueError("Input fitur mengandung nilai non-finite (NaN/inf).")
        preds = np.mean([est.predict(X_arr, check_input=False) for est in model.estimators_], axis=0)
        return np.asarray(preds, dtype=float)
    return np.asarray(model.predict(X_df), dtype=float)


def predict_house_price(
    bundle: dict,
    luas_tanah: float,
    luas_bangunan: float,
    kamar_mandi: int | float,
    kamar_tidur: int | float,
    carport: int | float,
    wilayah: str,
) -> tuple[float, float, pd.DataFrame]:
    """Memvalidasi input dan menjalankan pipeline inferensi yang identik dengan training."""
    lt = _validate_positive_finite(luas_tanah, "luas_tanah")
    lb = _validate_positive_finite(luas_bangunan, "luas_bangunan")
    km = _validate_non_negative_finite(kamar_mandi, "kamar_mandi")
    kt = _validate_non_negative_finite(kamar_tidur, "kamar_tidur")
    cp = _validate_non_negative_finite(carport, "carport")

    kategori_wilayah = list(bundle["kategori_wilayah"])
    if wilayah not in kategori_wilayah:
        raise ValueError(f"Wilayah tidak dikenal: {wilayah!r}. Pilihan valid: {kategori_wilayah}")

    model = bundle["model"]
    scaler = bundle["scaler"]
    fitur_final = list(bundle["fitur_final"])
    kolom_scale = list(bundle["kolom_scale"])

    df_scale_input = pd.DataFrame([{
        "log_luas_tanah": np.log10(lt),
        "log_luas_bangunan": np.log10(lb),
        "kamar_tidur": kt,
        "kamar_mandi": km,
        "carport": cp,
    }])[kolom_scale]

    scaled_array = scaler.transform(df_scale_input)
    df_scaled = pd.DataFrame(scaled_array, columns=kolom_scale)

    row_features: dict[str, float | int] = {
        "log_luas_bangunan": float(df_scaled["log_luas_bangunan"].iloc[0]),
        "log_luas_tanah": float(df_scaled["log_luas_tanah"].iloc[0]),
        "kamar_mandi": float(df_scaled["kamar_mandi"].iloc[0]),
    }

    for col in fitur_final:
        if col.startswith("wilayah_"):
            nama_wil = col.replace("wilayah_", "")
            row_features[col] = 1 if wilayah == nama_wil else 0

    df_final = pd.DataFrame([row_features])[fitur_final]
    pred_log = float(predict_batch_log(model, df_final)[0])
    pred_rupiah = float(10.0 ** pred_log)
    return pred_rupiah, pred_log, df_final
