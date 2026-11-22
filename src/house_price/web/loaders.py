"""Shared Streamlit cached loaders dan inisialisasi session state."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from house_price.config import MODEL_ARTIFACT_PATH, RANDOM_STATE, WILAYAH_MERGE_MAP
from house_price.data.cleaning import clean_listings, load_or_build_clean_data
from house_price.data.parsing import load_raw_data, parse_and_normalize
from house_price.features.preprocessing import load_or_build_preprocessed_data
from house_price.models.demo import DEMO_PRESETS
from house_price.models.evaluate import compute_regression_metrics, evaluate_by_wilayah
from house_price.models.predict import predict_batch_log
from house_price.models.train import benchmark_models, load_model_bundle


@st.cache_resource(max_entries=2)
def get_model_bundle() -> tuple[dict, str]:
    """Memuat satu sumber artefak model dengan auto-rebuild lintas-versi scikit-learn."""
    try:
        bundle = load_model_bundle(path=MODEL_ARTIFACT_PATH, auto_rebuild=True)
        return bundle, MODEL_ARTIFACT_PATH.name
    except FileNotFoundError:
        st.error(
            f"Artefak model tidak ditemukan di `{MODEL_ARTIFACT_PATH}` dan data mentah "
            "tidak tersedia untuk rebuild otomatis. Jalankan `python main.py` terlebih dahulu.",
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
    """Menghitung metrik test set aktual, tabel prediksi test, evaluasi wilayah, dan log cleaning."""
    prep = load_or_build_preprocessed_data()
    y_pred_log = predict_batch_log(_bundle["model"], prep.X_test_final)
    metrics = compute_regression_metrics(prep.y_test, y_pred_log)
    eval_wilayah = evaluate_by_wilayah(prep.test_df, metrics["y_pred_rupiah"])

    df_raw = load_raw_data()
    data_parsed = parse_and_normalize(df_raw)
    _, cleaning_log = clean_listings(data_parsed)

    test_eval_df = prep.test_df[
        ["wilayah_model", "area", "luas_tanah", "luas_bangunan", "kamar_tidur", "kamar_mandi", "carport", "harga"]
    ].copy()
    test_eval_df["harga_aktual_miliar"] = test_eval_df["harga"] / 1e9
    test_eval_df["harga_prediksi"] = metrics["y_pred_rupiah"]
    test_eval_df["harga_prediksi_miliar"] = metrics["y_pred_rupiah"] / 1e9
    test_eval_df["residual_log"] = np.asarray(prep.y_test, dtype=float) - y_pred_log
    test_eval_df["error_persen"] = (
        np.abs(test_eval_df["harga"] - test_eval_df["harga_prediksi"]) / test_eval_df["harga"] * 100.0
    )

    return {
        "metrics": metrics,
        "eval_wilayah": eval_wilayah,
        "cleaning_log": cleaning_log,
        "test_eval_df": test_eval_df.reset_index(drop=True),
        "n_train": len(prep.train_df),
        "n_test": len(prep.test_df),
    }


@st.cache_data(ttl="1h", max_entries=2)
def get_model_benchmark() -> pd.DataFrame:
    """Menjalankan benchmark 5 kandidat model (GroupKFold 5-fold CV) untuk halaman metodologi."""
    prep = load_or_build_preprocessed_data()
    groups = prep.train_df["deskripsi"].str.lower()
    return benchmark_models(prep.X_train_final, prep.y_train, groups=groups, random_state=RANDOM_STATE)


def apply_demo_preset() -> None:
    """Callback untuk mengisi parameter kalkulator berdasarkan preset demo."""
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


def init_calculator_state(kategori_wilayah: list[str], median_train: pd.Series) -> None:
    """Menginisialisasi default session_state di satu tempat yang konsisten."""
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
        "sampled_test_idx": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
