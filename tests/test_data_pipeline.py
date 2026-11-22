"""Unit tests untuk parsing, audit kualitas, dan cleaning data."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from house_price.data.audit import (
    audit_carport_mnar,
    audit_duplicates,
    audit_missing,
    batas_iqr,
    build_domain_rules,
    flag_non_hunian,
    tabel_outlier,
)
from house_price.data.cleaning import (
    clean_listings,
    compare_before_after,
    load_or_build_clean_data,
    save_clean_artifacts,
)
from house_price.data.parsing import (
    compare_wilayah_correction,
    load_raw_data,
    parse_and_normalize,
    summarize_raw_dictionary,
    ubah_harga,
    ubah_luas,
)


class TestDataPipeline(unittest.TestCase):
    def test_ubah_harga_valid_and_invalid(self):
        self.assertEqual(ubah_harga("Rp 1,79 Miliar"), 1_790_000_000.0)
        self.assertEqual(ubah_harga("Rp 800 Juta"), 800_000_000.0)
        self.assertEqual(ubah_harga("Rp 500 Ribu"), 500_000.0)
        with self.assertRaises(ValueError):
            ubah_harga("")
        with self.assertRaises(ValueError):
            ubah_harga("Rp tidak-ada-angka")
        with self.assertRaises(ValueError):
            ubah_harga(None)  # type: ignore[arg-type]

    def test_ubah_luas(self):
        s = pd.Series(["120 m²", "1.200 m²", None])
        res = ubah_luas(s)
        self.assertEqual(res.iloc[0], 120.0)
        self.assertEqual(res.iloc[1], 1200.0)
        self.assertTrue(pd.isna(res.iloc[2]))

    def test_flag_non_hunian_strips_potensi_phrase(self):
        judul = pd.Series([
            "Rumah Tinggal Nyaman Cocok Untuk Homestay",
            "Dijual Tanah Kavling Strategis",
            "Kost Putri Eksklusif Dekat UGM",
            "Rumah Minimalis Dekat Kantor",
        ])
        flags = flag_non_hunian(judul, strip_potensi=True)
        self.assertEqual(flags.tolist(), [False, True, True, False])

    def test_end_to_end_raw_parse_audit_clean_invariants(self):
        df_raw = load_raw_data()
        self.assertEqual(df_raw.shape, (2020, 9))

        raw_dict = summarize_raw_dictionary(df_raw)
        self.assertEqual(len(raw_dict), 9)

        data = parse_and_normalize(df_raw)
        self.assertEqual(len(data), 2020)
        self.assertEqual(data["area"].str.contains(r"\s{2,}").sum(), 0)

        mask_yogya = data["kategori_portal"].eq("yogyakarta")
        self.assertEqual(int(mask_yogya.sum()), 342)
        moved = int((mask_yogya & data["wilayah"].ne("Kota Yogyakarta")).sum())
        self.assertEqual(moved, 154)

        perbandingan_wilayah = compare_wilayah_correction(data)
        self.assertEqual(int(perbandingan_wilayah.loc["Kota Yogyakarta", "setelah_koreksi"]), 188)

        missing_df = audit_missing(data)
        self.assertIn("missing", missing_df.columns)

        carport_tbl, has_zero = audit_carport_mnar(data)
        self.assertFalse(has_zero)
        self.assertIn("kosong", carport_tbl.index)

        dup_tbl, n_desc_diff_spec = audit_duplicates(df_raw, data)
        self.assertEqual(int(dup_tbl.loc["Spesifikasi sama", "jumlah_kopian"]), 438)
        self.assertGreater(n_desc_diff_spec, 0)

        out_tbl = tabel_outlier(data, ["harga", "luas_tanah"])
        self.assertIn("n_outlier", out_tbl.columns)
        bawah, atas = batas_iqr(pd.Series([1, 2, 3, 4, 5]))
        self.assertLess(bawah, atas)

        aturan = build_domain_rules(data)
        self.assertEqual(aturan.shape[0], 2020)

        data_snapshot = data.copy()
        df_clean, log_df = clean_listings(data, aturan=aturan)

        # Ensure input data is not mutated
        pd.testing.assert_frame_equal(data, data_snapshot)

        self.assertEqual(len(df_clean), 1384)
        self.assertEqual(log_df["sisa_baris"].tolist(), [2020, 2019, 1581, 1409, 1384])
        self.assertEqual(log_df["dibuang"].tolist(), [0, 1, 438, 172, 25])

        banding = compare_before_after(data, df_clean)
        self.assertIn("n_sebelum", banding.columns)

    def test_save_and_load_clean_artifacts_preserves_raw_index(self):
        df_raw = load_raw_data()
        data = parse_and_normalize(df_raw)
        df_clean, _ = clean_listings(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            save_clean_artifacts(data, df_clean, output_dir=tmp_path)
            loaded = load_or_build_clean_data(processed_dir=tmp_path)

            self.assertEqual(loaded.index.tolist(), df_clean.index.tolist())
            self.assertEqual(list(loaded.columns), list(df_clean.columns))
            self.assertNotIn("Unnamed: 0", loaded.columns)


if __name__ == "__main__":
    unittest.main()
