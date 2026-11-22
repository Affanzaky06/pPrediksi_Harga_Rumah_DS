"""Pelatihan model, benchmark cross-validation, hyperparameter tuning, dan penyimpanan artefak."""

from __future__ import annotations

import logging
from pathlib import Path
import time
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.svm import SVR

from house_price.config import (
    BEST_RF_PARAMS,
    LEGACY_MODEL_ARTIFACT_PATH,
    MODEL_ARTIFACT_PATH,
    RANDOM_STATE,
)
from house_price.features.preprocessing import PreprocessedData

logger = logging.getLogger(__name__)

REQUIRED_BUNDLE_KEYS: set[str] = {
    "model",
    "scaler",
    "fitur_final",
    "kolom_scale",
    "median_train",
    "kategori_wilayah",
}

DEFAULT_RF_PARAM_GRID: dict[str, list[object]] = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}


def benchmark_models(
    X_train_final: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Membandingkan 5 kandidat model menggunakan GroupKFold 5-fold CV."""
    kandidat = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=random_state),
        "SVR (RBF)": SVR(C=1.0, epsilon=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }
    cv = GroupKFold(n_splits=5)
    hasil: list[dict[str, object]] = []
    for nama, model in kandidat.items():
        cv_scores = cross_validate(
            model,
            X_train_final,
            y_train,
            groups=groups,
            cv=cv,
            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
        )
        hasil.append({
            "Model": nama,
            "CV RMSE (log)": -cv_scores["test_neg_root_mean_squared_error"].mean(),
            "CV MAE (log)": -cv_scores["test_neg_mean_absolute_error"].mean(),
            "CV R2": cv_scores["test_r2"].mean(),
        })
    return pd.DataFrame(hasil).sort_values(by="CV RMSE (log)")


def run_rf_grid_search(
    X_train_final: pd.DataFrame,
    y_train: pd.Series,
    X_test_final: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    param_grid: dict[str, list[object]] | None = None,
    cv: int | object = 3,
    groups: pd.Series | None = None,
    random_state: int = RANDOM_STATE,
    verbose: int = 0,
) -> tuple[RandomForestRegressor, dict[str, object], float]:
    """Menjalankan GridSearchCV untuk RandomForestRegressor."""
    start = time.time()
    base_rf = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid or DEFAULT_RF_PARAM_GRID,
        cv=cv,
        n_jobs=-1,
        verbose=verbose,
    )
    fit_kwargs = {"groups": groups} if groups is not None else {}
    grid.fit(X_train_final, y_train, **fit_kwargs)
    best_model = grid.best_estimator_

    test_mse = float("nan")
    if X_test_final is not None and y_test is not None:
        y_pred = best_model.predict(X_test_final)
        test_mse = float(mean_squared_error(y_test, y_pred))

    elapsed = time.time() - start
    logger.info("GridSearchCV selesai dalam %.2f detik | test_mse=%.4f", elapsed, test_mse)
    return best_model, dict(grid.best_params_), test_mse


def train_final_model(
    X_train_final: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, object] | None = None,
) -> RandomForestRegressor:
    """Melatih model final RandomForestRegressor dengan hyperparameter terpilih."""
    rf_params = dict(BEST_RF_PARAMS)
    if params is not None:
        rf_params.update(params)
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train_final, y_train)
    return model


def save_model_bundle(
    model: RandomForestRegressor,
    prep: PreprocessedData,
    path: Path = MODEL_ARTIFACT_PATH,
) -> Path:
    """Menyimpan model terlatih dan artefak preprocessing dalam kontrak 6 kunci."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "scaler": prep.scaler,
        "fitur_final": prep.fitur_final,
        "kolom_scale": prep.kolom_scale,
        "median_train": prep.median_train,
        "kategori_wilayah": prep.kategori_wilayah,
    }
    joblib.dump(bundle, path)
    return path


def _validate_bundle(bundle: object, source_path: Path) -> dict:
    if not isinstance(bundle, dict):
        raise ValueError(f"Artefak model di {source_path} bukan dictionary.")
    missing = REQUIRED_BUNDLE_KEYS - set(bundle.keys())
    if missing:
        raise ValueError(f"Artefak model di {source_path} kekurangan kunci wajib: {sorted(missing)}")
    return bundle


def load_model_bundle(
    path: Path = MODEL_ARTIFACT_PATH,
    fallback_path: Path | None = LEGACY_MODEL_ARTIFACT_PATH,
) -> dict:
    """Memuat artefak model dengan fallback otomatis ke artefak legacy bila gagal."""
    candidates = [path]
    if fallback_path is not None and fallback_path != path:
        candidates.append(fallback_path)

    last_error: Exception | None = None
    for idx, candidate in enumerate(candidates):
        if not candidate.exists():
            last_error = FileNotFoundError(f"File model tidak ditemukan: {candidate}")
            if idx == 0 and len(candidates) > 1:
                logger.warning("Artefak utama %s tidak ditemukan; mencoba fallback %s.", candidate, candidates[1])
            continue
        try:
            bundle = joblib.load(candidate)
            return _validate_bundle(bundle, candidate)
        except Exception as exc:  # noqa: BLE001 — fallback lintas versi/korup
            last_error = exc
            if idx == 0 and len(candidates) > 1:
                logger.warning(
                    "Gagal memuat artefak utama %s (%s); beralih ke fallback %s.",
                    candidate,
                    exc,
                    candidates[1],
                )
            else:
                raise

    assert last_error is not None
    raise last_error
