"""Unit tests untuk training, evaluasi, fallback bundle, prediksi, dan kompatibilitas app.py."""

import py_compile
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from house_price.config import FITUR_FINAL
from house_price.data.cleaning import clean_listings
from house_price.data.parsing import load_raw_data, parse_and_normalize
from house_price.features.preprocessing import run_preprocessing_pipeline
from house_price.models.demo import (
    build_feature_importance_df,
    simulate_wilayah_comparison,
    summarize_market_by_wilayah,
)
from house_price.models.evaluate import compute_regression_metrics, evaluate_by_wilayah
from house_price.models.predict import format_rupiah, predict_house_price
from house_price.models.train import (
    REQUIRED_BUNDLE_KEYS,
    load_model_bundle,
    save_model_bundle,
    train_final_model,
)


class TestModelingAndPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df_raw = load_raw_data()
        data = parse_and_normalize(df_raw)
        df_clean, _ = clean_listings(data)
        cls.prep = run_preprocessing_pipeline(df_clean)
        cls.model = train_final_model(cls.prep.X_train_final, cls.prep.y_train)

    def test_save_and_load_bundle_keys_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bundle_path = tmp / "model_modular.joblib"
            fallback_path = tmp / "model_legacy.joblib"
            corrupt_path = tmp / "corrupt.joblib"

            save_model_bundle(self.model, self.prep, path=bundle_path)
            save_model_bundle(self.model, self.prep, path=fallback_path)
            corrupt_path.write_bytes(b"not-a-valid-joblib-pickle")

            loaded = load_model_bundle(path=bundle_path, fallback_path=None)
            self.assertEqual(set(loaded.keys()), REQUIRED_BUNDLE_KEYS)

            # Missing primary falls back to legacy
            loaded_missing = load_model_bundle(path=tmp / "missing.joblib", fallback_path=fallback_path)
            self.assertEqual(set(loaded_missing.keys()), REQUIRED_BUNDLE_KEYS)

            # Corrupt/incompatible primary falls back to legacy
            with self.assertLogs("house_price.models.train", level="WARNING"):
                loaded_corrupt = load_model_bundle(path=corrupt_path, fallback_path=fallback_path)
            self.assertEqual(set(loaded_corrupt.keys()), REQUIRED_BUNDLE_KEYS)

    def test_evaluate_metrics_and_wilayah_breakdown(self):
        y_pred_log = self.model.predict(self.prep.X_test_final)
        metrics = compute_regression_metrics(self.prep.y_test, y_pred_log)

        self.assertGreater(metrics["r2"], 0.85)
        self.assertLess(metrics["mdape"], 20.0)

        eval_wil = evaluate_by_wilayah(self.prep.test_df, metrics["y_pred_rupiah"])
        self.assertEqual(int(eval_wil["Jumlah_Sampel"].sum()), 277)

    def test_predict_house_price_valid_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "bundle.joblib"
            save_model_bundle(self.model, self.prep, path=bundle_path)
            bundle = load_model_bundle(path=bundle_path, fallback_path=None)

        pred_rp, pred_log, df_final = predict_house_price(
            bundle=bundle,
            luas_tanah=120.0,
            luas_bangunan=100.0,
            kamar_mandi=2,
            kamar_tidur=3,
            carport=1,
            wilayah="Sleman",
        )
        self.assertGreater(pred_rp, 0.0)
        self.assertAlmostEqual(10 ** pred_log, pred_rp, places=4)
        self.assertEqual(list(df_final.columns), FITUR_FINAL)

        rp_text, short_text = format_rupiah(pred_rp)
        self.assertTrue(rp_text.startswith("Rp "))
        self.assertTrue(bool(short_text))

        # Validation failures
        with self.assertRaises(ValueError):
            predict_house_price(bundle, 0.0, 100.0, 2, 3, 1, "Sleman")
        with self.assertRaises(ValueError):
            predict_house_price(bundle, 120.0, -5.0, 2, 3, 1, "Sleman")
        with self.assertRaises(ValueError):
            predict_house_price(bundle, np.nan, 100.0, 2, 3, 1, "Sleman")
        with self.assertRaises(ValueError):
            predict_house_price(bundle, 120.0, 100.0, -1, 3, 1, "Sleman")
        with self.assertRaises(ValueError):
            predict_house_price(bundle, 120.0, 100.0, 2, 3, 1, "Jakarta Selatan")

    def test_demo_helpers_and_streamlit_app_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "bundle.joblib"
            save_model_bundle(self.model, self.prep, path=bundle_path)
            bundle = load_model_bundle(path=bundle_path, fallback_path=None)

        sim_df = simulate_wilayah_comparison(
            bundle=bundle,
            luas_tanah=120.0,
            luas_bangunan=100.0,
            kamar_mandi=2,
            kamar_tidur=3,
            carport=1,
            selected_wilayah="Sleman",
        )
        self.assertEqual(len(sim_df), 4)
        self.assertEqual(int((sim_df["Status"] == "Dipilih").sum()), 1)

        imp_df = build_feature_importance_df(bundle)
        self.assertAlmostEqual(float(imp_df["Kontribusi (%)"].sum()), 100.0, places=4)

        df_raw = load_raw_data()
        df_clean, _ = clean_listings(parse_and_normalize(df_raw))
        mkt_df = summarize_market_by_wilayah(df_clean)
        self.assertEqual(int(mkt_df["Jumlah_Listing"].sum()), 1384)

        app_path = PROJECT_ROOT / "app.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            py_compile.compile(str(app_path), cfile=str(Path(tmpdir) / "app.pyc"), doraise=True)

        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(app_path), default_timeout=20).run()
        self.assertFalse(at.exception)
        self.assertGreaterEqual(len(at.metric), 4)

        # Save scenario button appends to comparison table
        at.button(key="btn_save_scenario").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state["demo_comparisons"]), 1)

        # Reset button clears comparisons
        at.button(key="btn_reset_scenarios").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state["demo_comparisons"]), 0)

    def test_plots_render_headless(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from house_price.visualization.plots import (
            plot_actual_vs_predicted,
            plot_feature_importance,
            plot_residuals,
        )

        y_pred_log = self.model.predict(self.prep.X_test_final)
        metrics = compute_regression_metrics(self.prep.y_test, y_pred_log)
        residual_log = self.prep.y_test - y_pred_log
        residual_rupiah = metrics["y_test_rupiah"] - metrics["y_pred_rupiah"]

        figs = [
            plot_actual_vs_predicted(metrics["y_test_rupiah"], metrics["y_pred_rupiah"], metrics["r2"]),
            plot_feature_importance(self.model, FITUR_FINAL),
            plot_residuals(metrics["y_pred_rupiah"], residual_rupiah, residual_log),
        ]
        for fig in figs:
            self.assertGreater(len(fig.axes), 0)
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
