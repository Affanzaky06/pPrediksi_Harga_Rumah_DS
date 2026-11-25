"""Unit tests untuk training, evaluasi, fallback bundle, prediksi, dan kompatibilitas app.py."""

import py_compile
import sys
import tempfile
import unittest
import warnings
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
    compute_carport_mnar_eda,
    compute_correlation_tables,
    compute_descriptive_stats,
    compute_kruskal_wilayah,
    compute_price_segment_distribution,
    simulate_wilayah_comparison,
    summarize_market_by_wilayah,
)
from house_price.models.evaluate import compute_regression_metrics, evaluate_by_wilayah
from house_price.models.predict import format_rupiah, predict_batch_log, predict_house_price
from house_price.models.train import (
    BEST_TUNED_PARAMS_5_MODELS,
    PARAM_GRIDS_5_MODELS,
    REQUIRED_BUNDLE_KEYS,
    benchmark_models,
    load_model_bundle,
    save_model_bundle,
    train_final_model,
    tune_all_candidate_models,
)


class TestModelingAndPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df_raw = load_raw_data()
        data = parse_and_normalize(df_raw)
        df_clean, _ = clean_listings(data)
        cls.prep = run_preprocessing_pipeline(df_clean)
        cls.model = train_final_model(cls.prep.X_train_final, cls.prep.y_train)

    def test_save_and_load_bundle_keys_and_auto_rebuild(self):
        from unittest.mock import patch
        from sklearn.exceptions import InconsistentVersionWarning

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            bundle_path = tmp / "model.joblib"
            corrupt_path = tmp / "corrupt.joblib"
            missing_path = tmp / "missing.joblib"

            save_model_bundle(self.model, self.prep, path=bundle_path)
            corrupt_path.write_bytes(b"not-a-valid-joblib-pickle")

            loaded = load_model_bundle(path=bundle_path, auto_rebuild=False)
            self.assertEqual(set(loaded.keys()), REQUIRED_BUNDLE_KEYS)
            self.assertEqual(loaded["model"].n_jobs, 1)

            # auto_rebuild=False raises when missing or corrupt
            with self.assertRaises(FileNotFoundError):
                load_model_bundle(path=missing_path, auto_rebuild=False)
            with self.assertRaises(Exception):
                load_model_bundle(path=corrupt_path, auto_rebuild=False)

            # auto_rebuild=True rebuilds seamlessly when missing
            with self.assertLogs("house_price.models.train", level="WARNING"):
                rebuilt_missing = load_model_bundle(path=missing_path, auto_rebuild=True)
            self.assertTrue(missing_path.exists())
            self.assertEqual(set(rebuilt_missing.keys()), REQUIRED_BUNDLE_KEYS)

            # auto_rebuild=True rebuilds seamlessly when InconsistentVersionWarning occurs
            mismatch_path = tmp / "mismatch.joblib"
            save_model_bundle(self.model, self.prep, path=mismatch_path)

            def _warn_on_load(_path):
                import warnings
                warnings.warn(
                    InconsistentVersionWarning(
                        estimator_name="RandomForestRegressor",
                        current_sklearn_version="1.9.0",
                        original_sklearn_version="1.8.0",
                    )
                )

            with patch("house_price.models.train.joblib.load", side_effect=_warn_on_load):
                with self.assertLogs("house_price.models.train", level="WARNING"):
                    rebuilt_mismatch = load_model_bundle(path=mismatch_path, auto_rebuild=True)
            self.assertEqual(set(rebuilt_mismatch.keys()), REQUIRED_BUNDLE_KEYS)

    def test_evaluate_metrics_and_wilayah_breakdown(self):
        y_pred_log = self.model.predict(self.prep.X_test_final)
        metrics = compute_regression_metrics(self.prep.y_test, y_pred_log)

        self.assertGreater(metrics["r2"], 0.85)
        self.assertLess(metrics["mdape"], 20.0)
        self.assertAlmostEqual(metrics["r2"], 0.8714, places=3)
        self.assertAlmostEqual(metrics["mdape"], 14.17, places=1)

        eval_wil = evaluate_by_wilayah(self.prep.test_df, metrics["y_pred_rupiah"])
        self.assertEqual(int(eval_wil["Jumlah_Sampel"].sum()), 277)

    def test_param_grids_include_none_and_benchmark_tuned_models(self):
        expected_models = {
            "Linear Regression",
            "Ridge Regression",
            "SVR (RBF)",
            "Random Forest",
            "Gradient Boosting",
        }
        self.assertEqual(set(PARAM_GRIDS_5_MODELS.keys()), expected_models)
        self.assertEqual(set(BEST_TUNED_PARAMS_5_MODELS.keys()), expected_models)
        self.assertIn(None, PARAM_GRIDS_5_MODELS["Ridge Regression"]["max_iter"])
        self.assertIn(None, PARAM_GRIDS_5_MODELS["Random Forest"]["max_depth"])
        self.assertIn(None, PARAM_GRIDS_5_MODELS["Random Forest"]["max_features"])
        self.assertIn(None, PARAM_GRIDS_5_MODELS["Gradient Boosting"]["max_depth"])
        self.assertIn(None, PARAM_GRIDS_5_MODELS["Gradient Boosting"]["max_features"])

        groups = self.prep.train_df["deskripsi"].str.lower()
        df_bench = benchmark_models(self.prep.X_train_final, self.prep.y_train, groups=groups)
        self.assertEqual(len(df_bench), 5)
        self.assertEqual(df_bench.iloc[0]["Model"], "Random Forest")
        self.assertLess(float(df_bench.iloc[0]["CV RMSE (log)"]), 0.144)
        self.assertIn("Baseline CV RMSE (log)", df_bench.columns)
        self.assertIn("Delta RMSE", df_bench.columns)
        self.assertIn("Best Hyperparameters", df_bench.columns)

        smoke_grids = {
            name: {k: [v[0]] for k, v in grid.items()}
            for name, grid in PARAM_GRIDS_5_MODELS.items()
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            df_tune, best_estimators = tune_all_candidate_models(
                self.prep.X_train_final,
                self.prep.y_train,
                groups=groups,
                param_grids=smoke_grids,
                n_jobs=1,
            )
        self.assertEqual(len(df_tune), 5)
        self.assertEqual(set(best_estimators.keys()), expected_models)

    def test_predict_house_price_valid_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = Path(tmpdir) / "bundle.joblib"
            save_model_bundle(self.model, self.prep, path=bundle_path)
            bundle = load_model_bundle(path=bundle_path, auto_rebuild=False)

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
            bundle = load_model_bundle(path=bundle_path, auto_rebuild=False)

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

        # New EDA helpers
        desk_df = compute_descriptive_stats(df_clean)
        self.assertEqual(len(desk_df), 6)
        self.assertIn("skewness", desk_df.columns)

        mnar_df = compute_carport_mnar_eda(df_clean)
        self.assertEqual(mnar_df["Grup_Carport"].tolist(), ["Kosong", "1", "2", "3+"])

        long_corr, target_corr = compute_correlation_tables(df_clean)
        self.assertEqual(len(long_corr), 36)
        self.assertIn("Pearson (log-log)", target_corr.columns)

        seg_df = compute_price_segment_distribution(df_clean)
        self.assertIn("Segmen Harga", seg_df.columns)

        self.assertIsNotNone(compute_kruskal_wilayah(df_clean))
        self.assertIsNone(compute_kruskal_wilayah(df_clean[df_clean["wilayah"] == "Sleman"]))

        # Zero sklearn.utils.parallel warning even with empty warnings filters
        expected_batch = self.model.predict(self.prep.X_test_final)
        with warnings.catch_warnings():
            warnings.resetwarnings()
            warnings.simplefilter("error", UserWarning)
            actual_batch = predict_batch_log(self.model, self.prep.X_test_final)
            pred_single, _, _ = predict_house_price(bundle, 120.0, 100.0, 2, 3, 1, "Sleman")
        self.assertTrue(np.allclose(actual_batch, expected_batch, atol=1e-12))
        self.assertGreater(pred_single, 0.0)

        from house_price.visualization.web_charts import (
            make_actual_vs_pred_chart,
            make_benchmark_chart,
            make_correlation_heatmap,
            make_residual_hist_chart,
            make_wilayah_sim_chart,
        )
        self.assertIsNotNone(make_wilayah_sim_chart(sim_df, is_dark=True))
        self.assertIsNotNone(make_correlation_heatmap(long_corr, is_dark=False))
        test_eval_sample = mkt_df.copy()
        self.assertIsNotNone(test_eval_sample)

        app_path = PROJECT_ROOT / "app.py"
        with tempfile.TemporaryDirectory() as tmpdir:
            py_compile.compile(str(app_path), cfile=str(Path(tmpdir) / "app.pyc"), doraise=True)
            for page_file in (PROJECT_ROOT / "app_pages").glob("*.py"):
                py_compile.compile(str(page_file), cfile=str(Path(tmpdir) / f"{page_file.stem}.pyc"), doraise=True)

        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file(str(app_path), default_timeout=25).run()
        self.assertFalse(at.exception)
        self.assertGreaterEqual(len(at.metric), 4)
        all_captions = "\n".join(c.value for c in at.caption)
        self.assertIn("300 trees", all_captions)

        # Sample real test row & load into calculator
        at.button(key="btn_sample_test").click().run()
        self.assertFalse(at.exception)
        self.assertIsNotNone(at.session_state["sampled_test_idx"])
        at.button(key="btn_load_sample_to_calc").click().run()
        self.assertFalse(at.exception)

        # Save scenario button appends to comparison table, then reset
        at.button(key="btn_save_scenario").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state["demo_comparisons"]), 1)
        at.button(key="btn_reset_scenarios").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state["demo_comparisons"]), 0)

        # Multipage navigation: Page 2 (EDA) & Page 3 (Metodologi)
        at.switch_page("app_pages/02_eda.py").run()
        self.assertFalse(at.exception)
        at.switch_page("app_pages/03_metodologi.py").run()
        self.assertFalse(at.exception)

        # Ensure chart builders work with actual test evaluation frame
        from house_price.web.loaders import get_model_bundle, get_pipeline_evaluation, get_model_benchmark
        b_live, _ = get_model_bundle()
        eval_live = get_pipeline_evaluation(b_live)
        bench_live = get_model_benchmark()
        self.assertAlmostEqual(eval_live["metrics"]["r2"], 0.8714, places=3)
        self.assertIn("Baseline CV RMSE (log)", bench_live.columns)
        self.assertIn("Best Hyperparameters", bench_live.columns)
        self.assertIsNotNone(make_actual_vs_pred_chart(eval_live["test_eval_df"], is_dark=False))
        self.assertIsNotNone(make_residual_hist_chart(eval_live["test_eval_df"], is_dark=True))
        self.assertIsNotNone(make_benchmark_chart(bench_live, is_dark=False))

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
