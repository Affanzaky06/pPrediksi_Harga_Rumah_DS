"""Runner CLI end-to-end untuk membangun ulang dataset bersih, model, dan artefak joblib."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from house_price.config import MODEL_ARTIFACT_PATH, PROCESSED_DATA_DIR, RAW_DATA_PATH
from house_price.data.cleaning import clean_listings, save_clean_artifacts
from house_price.data.parsing import load_raw_data, parse_and_normalize
from house_price.features.preprocessing import (
    run_preprocessing_pipeline,
    save_preprocessed_artifacts,
)
from house_price.models.evaluate import compute_regression_metrics, evaluate_by_wilayah
from house_price.models.train import (
    run_rf_grid_search,
    save_model_bundle,
    train_final_model,
)


def run_pipeline(
    use_grid_search: bool = False,
    output_model: Path = MODEL_ARTIFACT_PATH,
    raw_path: Path = RAW_DATA_PATH,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> dict[str, object]:
    """Menjalankan pipeline lengkap dari raw CSV hingga penyimpanan artefak model."""
    df_raw = load_raw_data(raw_path)
    data_parsed = parse_and_normalize(df_raw)
    df_clean, log_df = clean_listings(data_parsed)
    save_clean_artifacts(data_parsed, df_clean, output_dir=processed_dir)

    prep = run_preprocessing_pipeline(df_clean)
    save_preprocessed_artifacts(prep, output_dir=processed_dir)

    if use_grid_search:
        model, best_params, _ = run_rf_grid_search(
            prep.X_train_final,
            prep.y_train,
            X_test_final=prep.X_test_final,
            y_test=prep.y_test,
        )
        print(f"Best parameters (GridSearchCV): {best_params}")
    else:
        model = train_final_model(prep.X_train_final, prep.y_train)

    y_test_pred_log = model.predict(prep.X_test_final)
    metrics = compute_regression_metrics(prep.y_test, y_test_pred_log)
    eval_wilayah = evaluate_by_wilayah(prep.test_df, metrics["y_pred_rupiah"])
    saved_path = save_model_bundle(model, prep, path=output_model)

    print("\n=== CLEANING LOG ===")
    print(log_df.to_string(index=False))
    print("\n=== EVALUASI TEST SET ===")
    print(f"R2 Score   : {metrics['r2'] * 100:.2f}%")
    print(f"MAE Rupiah : Rp {metrics['mae_rupiah']:,.0f}")
    print(f"RMSE Rupiah: Rp {metrics['rmse_rupiah']:,.0f}")
    print(f"MdAPE      : {metrics['mdape']:.2f}%")
    print(f"MAPE       : {metrics['mape']:.2f}%")
    print("\n=== EVALUASI PER WILAYAH ===")
    print(eval_wilayah.to_string(index=False))
    print(f"\nArtefak model tersimpan di: {saved_path}")

    return {"log_df": log_df, "metrics": metrics, "eval_wilayah": eval_wilayah, "saved_path": saved_path}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline training prediksi harga rumah Yogyakarta.")
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Jalankan GridSearchCV penuh (default: langsung gunakan BEST_RF_PARAMS).",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=MODEL_ARTIFACT_PATH,
        help="Path output artefak bundle .joblib.",
    )
    args = parser.parse_args()
    run_pipeline(use_grid_search=args.grid_search, output_model=args.output_model)


if __name__ == "__main__":
    main()
