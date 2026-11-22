"""Unit tests untuk preprocessing, pencegahan data leakage, dan feature selection."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from house_price.config import FITUR_FINAL
from house_price.data.cleaning import clean_listings
from house_price.data.parsing import load_raw_data, parse_and_normalize
from house_price.features.preprocessing import (
    add_wilayah_model,
    impute_missing_from_train,
    load_or_build_preprocessed_data,
    run_preprocessing_pipeline,
    save_preprocessed_artifacts,
    split_train_test,
)
from house_price.features.selection import (
    compute_vif,
    evaluate_drop_column_cv,
    verify_log_ratio_redundancy,
)


class TestPreprocessingAndLeakage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df_raw = load_raw_data()
        data = parse_and_normalize(df_raw)
        cls.df_clean, _ = clean_listings(data)

    def test_split_prevents_deskripsi_leakage_and_preserves_counts(self):
        df_with_model = add_wilayah_model(self.df_clean)
        train, test = split_train_test(df_with_model)

        self.assertEqual(len(train), 1107)
        self.assertEqual(len(test), 277)

        overlap = test["deskripsi"].str.lower().isin(train["deskripsi"].str.lower()).sum()
        self.assertEqual(int(overlap), 0)

    def test_imputation_learned_from_train_only(self):
        df_with_model = add_wilayah_model(self.df_clean)
        train, test = split_train_test(df_with_model)

        train_imp, test_imp, median_train = impute_missing_from_train(train, test)
        self.assertEqual(median_train.to_dict(), {"kamar_tidur": 3.0, "kamar_mandi": 2.0})
        self.assertEqual(int(train_imp[["kamar_tidur", "kamar_mandi", "carport"]].isna().sum().sum()), 0)
        self.assertEqual(int(test_imp[["kamar_tidur", "kamar_mandi", "carport"]].isna().sum().sum()), 0)
        self.assertIn("carport_kosong", train_imp.columns)
        self.assertIn("carport_kosong", test_imp.columns)

    def test_full_preprocessing_invariants_and_roundtrip(self):
        prep = run_preprocessing_pipeline(self.df_clean)

        self.assertEqual(prep.X_train_final.shape, (1107, 6))
        self.assertEqual(prep.X_test_final.shape, (277, 6))
        self.assertEqual(list(prep.X_train_final.columns), FITUR_FINAL)
        self.assertEqual(list(prep.X_test_final.columns), FITUR_FINAL)

        # Scaled train mean ~ 0, std ~ 1
        self.assertTrue(np.allclose(prep.X_train[prep.kolom_scale].mean().values, 0.0, atol=1e-12))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            save_preprocessed_artifacts(prep, output_dir=tmp_path)
            loaded = load_or_build_preprocessed_data(processed_dir=tmp_path)
            self.assertEqual(loaded.X_train_final.shape, (1107, 6))
            self.assertEqual(loaded.X_test_final.shape, (277, 6))
            self.assertEqual(loaded.kategori_wilayah, prep.kategori_wilayah)

    def test_feature_selection_metrics(self):
        prep = run_preprocessing_pipeline(self.df_clean)
        vif = compute_vif(prep.X_train_unscaled.astype(float))
        self.assertTrue((vif < 5.0).all())

        r2_ratio = verify_log_ratio_redundancy(prep.X_train_unscaled)
        self.assertAlmostEqual(r2_ratio, 1.0, places=6)

        drop_imp, dasar = evaluate_drop_column_cv(
            prep.X_train,
            prep.X_train_unscaled,
            prep.y_train,
            groups=prep.train_df["deskripsi"].str.lower(),
        )
        self.assertIn("Linear", dasar)
        self.assertIn("RandomForest", dasar)
        self.assertIn("luas bangunan", drop_imp.index)


if __name__ == "__main__":
    unittest.main()
