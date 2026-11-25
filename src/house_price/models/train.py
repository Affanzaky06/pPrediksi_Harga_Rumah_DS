"""Pelatihan model, benchmark cross-validation, hyperparameter tuning, dan penyimpanan artefak."""

from __future__ import annotations

import logging
from pathlib import Path
import time
import warnings
import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold, cross_validate
from sklearn.svm import SVR

from house_price.config import (
    BEST_RF_PARAMS,
    MODEL_ARTIFACT_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
)
from house_price.data.cleaning import clean_listings
from house_price.data.parsing import load_raw_data, parse_and_normalize
from house_price.features.preprocessing import PreprocessedData, run_preprocessing_pipeline

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
    "n_estimators": [200, 300],
    "max_depth": [None, 10, 20],
    "max_features": [None, "sqrt", 0.5],
    "min_samples_split": [5, 10],
    "min_samples_leaf": [2, 4],
    "bootstrap": [True],
}

PARAM_GRIDS_5_MODELS: dict[str, dict[str, list[object]]] = {
    "Linear Regression": {
        "fit_intercept": [True, False],
        "positive": [False, True],
    },
    "Ridge Regression": {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "fit_intercept": [True, False],
        "max_iter": [None, 1000],
    },
    "SVR (RBF)": {
        "kernel": ["rbf", "linear"],
        "C": [0.1, 1.0, 10.0],
        "epsilon": [0.01, 0.05, 0.1, 0.2],
        "gamma": ["scale", "auto"],
    },
    "Random Forest": DEFAULT_RF_PARAM_GRID,
    "Gradient Boosting": {
        "n_estimators": [100, 200],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [None, 3, 5],
        "max_features": [None, "sqrt", 0.5],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
    },
}

BEST_TUNED_PARAMS_5_MODELS: dict[str, dict[str, object]] = {
    "Linear Regression": {"fit_intercept": True, "positive": False},
    "Ridge Regression": {"alpha": 1.0, "fit_intercept": True, "max_iter": None},
    "SVR (RBF)": {"C": 1.0, "epsilon": 0.01, "gamma": "auto", "kernel": "rbf"},
    "Random Forest": {
        "bootstrap": True,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "min_samples_split": 5,
        "n_estimators": 300,
    },
    "Gradient Boosting": {
        "learning_rate": 0.03,
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 4,
        "min_samples_split": 10,
        "n_estimators": 100,
    },
}


def _build_default_candidates(random_state: int = RANDOM_STATE) -> dict[str, object]:
    """Membangun 5 kandidat estimator dengan parameter default (n_jobs=1 untuk mencegah nested parallelism)."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=random_state),
        "SVR (RBF)": SVR(),
        "Random Forest": RandomForestRegressor(random_state=random_state, n_jobs=1),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }


def _build_tuned_candidates(random_state: int = RANDOM_STATE) -> dict[str, object]:
    """Membangun 5 kandidat estimator menggunakan hyperparameter terbaik hasil GridSearchCV."""
    rf_params = dict(BEST_TUNED_PARAMS_5_MODELS["Random Forest"])
    rf_params.update({"random_state": random_state, "n_jobs": 1})
    gbr_params = dict(BEST_TUNED_PARAMS_5_MODELS["Gradient Boosting"])
    gbr_params["random_state"] = random_state
    ridge_params = dict(BEST_TUNED_PARAMS_5_MODELS["Ridge Regression"])
    ridge_params["random_state"] = random_state
    return {
        "Linear Regression": LinearRegression(**BEST_TUNED_PARAMS_5_MODELS["Linear Regression"]),
        "Ridge Regression": Ridge(**ridge_params),
        "SVR (RBF)": SVR(**BEST_TUNED_PARAMS_5_MODELS["SVR (RBF)"]),
        "Random Forest": RandomForestRegressor(**rf_params),
        "Gradient Boosting": GradientBoostingRegressor(**gbr_params),
    }


def _format_params(params: dict[str, object]) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in params.items())


def benchmark_models(
    X_train_final: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Membandingkan 5 kandidat model (Baseline Default vs Setelah Hyperparameter Tuning) dengan GroupKFold 5-fold CV."""
    defaults = _build_default_candidates(random_state=random_state)
    tuned = _build_tuned_candidates(random_state=random_state)
    cv = GroupKFold(n_splits=5)
    hasil: list[dict[str, object]] = []
    for nama in defaults:
        base_scores = cross_validate(
            defaults[nama],
            X_train_final,
            y_train,
            groups=groups,
            cv=cv,
            scoring=["neg_root_mean_squared_error"],
        )
        tuned_scores = cross_validate(
            tuned[nama],
            X_train_final,
            y_train,
            groups=groups,
            cv=cv,
            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
        )
        base_rmse = float(-base_scores["test_neg_root_mean_squared_error"].mean())
        tuned_rmse = float(-tuned_scores["test_neg_root_mean_squared_error"].mean())
        hasil.append({
            "Model": nama,
            "Baseline CV RMSE (log)": base_rmse,
            "CV RMSE (log)": tuned_rmse,
            "Delta RMSE": float(tuned_rmse - base_rmse),
            "CV MAE (log)": float(-tuned_scores["test_neg_mean_absolute_error"].mean()),
            "CV R2": float(tuned_scores["test_r2"].mean()),
            "Best Hyperparameters": _format_params(BEST_TUNED_PARAMS_5_MODELS[nama]),
        })
    return pd.DataFrame(hasil).sort_values(by="CV RMSE (log)").reset_index(drop=True)


def tune_all_candidate_models(
    X_train_final: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    param_grids: dict[str, dict[str, list[object]]] | None = None,
    cv: int | object | None = None,
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Menjalankan GridSearchCV dengan GroupKFold (termasuk opsi None) pada seluruh 5 kandidat model."""
    defaults = _build_default_candidates(random_state=random_state)
    grids = param_grids or PARAM_GRIDS_5_MODELS
    cv_splitter: int | object = GroupKFold(n_splits=5) if cv is None else cv

    hasil: list[dict[str, object]] = []
    best_estimators: dict[str, object] = {}

    for nama, estimator_dasar in defaults.items():
        t0 = time.time()
        cv_base = cross_validate(
            estimator_dasar,
            X_train_final,
            y_train,
            groups=groups,
            cv=cv_splitter,
            scoring=["neg_root_mean_squared_error"],
            n_jobs=n_jobs,
        )
        base_rmse = float(-cv_base["test_neg_root_mean_squared_error"].mean())

        gs = GridSearchCV(
            estimator=estimator_dasar,
            param_grid=grids[nama],
            cv=cv_splitter,
            scoring=["neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"],
            refit="neg_root_mean_squared_error",
            n_jobs=n_jobs,
        )
        gs.fit(X_train_final, y_train, groups=groups)

        idx = int(gs.best_index_)
        tuned_rmse = float(-gs.cv_results_["mean_test_neg_root_mean_squared_error"][idx])
        tuned_mae = float(-gs.cv_results_["mean_test_neg_mean_absolute_error"][idx])
        tuned_r2 = float(gs.cv_results_["mean_test_r2"][idx])
        durasi = time.time() - t0

        best_estimators[nama] = gs.best_estimator_
        hasil.append({
            "Model": nama,
            "Kombinasi Diuji": len(gs.cv_results_["params"]),
            "Baseline CV RMSE (log)": base_rmse,
            "CV RMSE (log)": tuned_rmse,
            "Delta RMSE": float(tuned_rmse - base_rmse),
            "CV MAE (log)": tuned_mae,
            "CV R2": tuned_r2,
            "Waktu (s)": round(durasi, 1),
            "Best Hyperparameters": _format_params(dict(gs.best_params_)),
        })

    df_result = pd.DataFrame(hasil).sort_values(by="CV RMSE (log)").reset_index(drop=True)
    return df_result, best_estimators


def run_rf_grid_search(
    X_train_final: pd.DataFrame,
    y_train: pd.Series,
    X_test_final: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
    param_grid: dict[str, list[object]] | None = None,
    cv: int | object | None = None,
    groups: pd.Series | None = None,
    random_state: int = RANDOM_STATE,
    verbose: int = 0,
) -> tuple[RandomForestRegressor, dict[str, object], float]:
    """Menjalankan GridSearchCV untuk RandomForestRegressor (GroupKFold bila groups diberikan)."""
    start = time.time()
    if cv is None:
        cv_splitter: int | object = GroupKFold(n_splits=5) if groups is not None else 5
    else:
        cv_splitter = cv
    base_rf = RandomForestRegressor(random_state=random_state, n_jobs=1)
    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid or DEFAULT_RF_PARAM_GRID,
        cv=cv_splitter,
        scoring="neg_root_mean_squared_error",
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
    model,
    prep: PreprocessedData,
    path: Path = MODEL_ARTIFACT_PATH,
) -> Path:
    """Menyimpan model terlatih dan artefak preprocessing dalam kontrak 6 kunci."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
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


def build_and_save_bundle(
    output_path: Path = MODEL_ARTIFACT_PATH,
    raw_path: Path = RAW_DATA_PATH,
    save_to_disk: bool = True,
) -> dict:
    """Membangun ulang model dari data mentah sesuai versi scikit-learn lokal."""
    df_raw = load_raw_data(raw_path)
    data_parsed = parse_and_normalize(df_raw)
    df_clean, _ = clean_listings(data_parsed)
    prep = run_preprocessing_pipeline(df_clean)
    model = train_final_model(prep.X_train_final, prep.y_train)

    bundle = {
        "model": model,
        "scaler": prep.scaler,
        "fitur_final": prep.fitur_final,
        "kolom_scale": prep.kolom_scale,
        "median_train": prep.median_train,
        "kategori_wilayah": prep.kategori_wilayah,
    }

    if save_to_disk:
        try:
            save_model_bundle(model, prep, path=output_path)
        except OSError as exc:
            logger.warning("Tidak dapat menyimpan ulang artefak ke %s: %s", output_path, exc)

    if hasattr(bundle["model"], "n_jobs"):
        bundle["model"].n_jobs = 1
    return bundle


def load_model_bundle(
    path: Path = MODEL_ARTIFACT_PATH,
    auto_rebuild: bool = True,
    raw_path: Path = RAW_DATA_PATH,
) -> dict:
    """Memuat artefak model tunggal; otomatis rebuild jika hilang atau beda versi scikit-learn."""
    load_error: Exception | None = None

    if not path.exists():
        load_error = FileNotFoundError(f"File model tidak ditemukan: {path}")
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", InconsistentVersionWarning)
                bundle = joblib.load(path)
            validated = _validate_bundle(bundle, path)
            if hasattr(validated["model"], "n_jobs"):
                validated["model"].n_jobs = 1
            return validated
        except (InconsistentVersionWarning, Exception) as exc:  # noqa: BLE001
            load_error = exc

    if auto_rebuild and raw_path.exists():
        logger.warning(
            "Artefak model %s tidak tersedia atau tidak kompatibel (%s); "
            "membangun ulang otomatis untuk scikit-learn %s.",
            path,
            load_error,
            sklearn.__version__,
        )
        return build_and_save_bundle(output_path=path, raw_path=raw_path, save_to_disk=True)

    assert load_error is not None
    raise load_error
