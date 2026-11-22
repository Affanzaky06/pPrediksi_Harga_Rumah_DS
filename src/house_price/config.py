"""Konfigurasi terpusat untuk pipeline prediksi harga rumah Yogyakarta."""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DATA_PATH: Path = PROJECT_ROOT / "rumah123_yogya_unfiltered.csv"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
MODEL_ARTIFACTS_DIR: Path = PROJECT_ROOT / "model_artifacts"
MODEL_ARTIFACT_PATH: Path = MODEL_ARTIFACTS_DIR / "model_prediksi_rumah_jogja.joblib"

RANDOM_STATE: int = 42

NAMA_WILAYAH: dict[str, str] = {
    "sleman": "Sleman",
    "bantul": "Bantul",
    "yogyakarta": "Kota Yogyakarta",
    "kulon-progo": "Kulon Progo",
    "gunung-kidul": "Gunung Kidul",
}

# Area yang tercatat di kategori portal 'yogyakarta' tetapi administratifnya di kabupaten lain
KOREKSI_AREA: dict[str, str] = {
    # Sleman
    "Maguwoharjo": "Sleman",
    "Sleman": "Sleman",
    "Kaliurang": "Sleman",
    "Pogung": "Sleman",
    "Seturan": "Sleman",
    "Caturtunggal": "Sleman",
    "Cebongan": "Sleman",
    "Purwomartani": "Sleman",
    "Nologaten": "Sleman",
    "Sekip": "Sleman",
    # Bantul
    "Bantul": "Bantul",
    "Imogiri": "Bantul",
    # Kulon Progo
    "Kulonprogo": "Kulon Progo",
}

WILAYAH_MERGE_MAP: dict[str, str] = {
    "Kulon Progo": "Kulon Progo/Gunung Kidul",
    "Gunung Kidul": "Kulon Progo/Gunung Kidul",
}

BASELINE_WILAYAH: str = "Bantul"

NUM_COLUMNS: list[str] = [
    "harga",
    "luas_tanah",
    "luas_bangunan",
    "kamar_tidur",
    "kamar_mandi",
    "carport",
]

KOLOM_SPEK: list[str] = [
    "harga",
    "lokasi",
    "kamar_tidur",
    "kamar_mandi",
    "carport",
    "luas_tanah",
    "luas_bangunan",
]

FITUR_NUM: list[str] = [
    "luas_tanah",
    "luas_bangunan",
    "kamar_tidur",
    "kamar_mandi",
    "carport",
]

KOLOM_SCALE: list[str] = [
    "log_luas_tanah",
    "log_luas_bangunan",
    "kamar_tidur",
    "kamar_mandi",
    "carport",
]

FITUR_FINAL: list[str] = [
    "log_luas_bangunan",
    "log_luas_tanah",
    "kamar_mandi",
    "wilayah_Kota Yogyakarta",
    "wilayah_Kulon Progo/Gunung Kidul",
    "wilayah_Sleman",
]

# Pola teks properti non-hunian
POLA_POTENSI: str = (
    r"\b(?:cocok|bisa|ideal|potensial)\b"
    r"(?:\s+(?:untuk|utk|buat|jadi|dijadikan|dibuat|sebagai))?\s.*$"
)
POLA_AWAL: str = (
    r"^\s*(?:dijual\s+|jual\s+)?(?:tanah|kavling|kapling|ruko|gudang|pabrik|"
    r"kost|kos|hotel|villa|vila|homestay|resto|restoran)\b"
)
POLA_KOS: str = (
    r"\b(?:kost|kos)\s*(?:putri|putra|eksklusif|ekslusif|exclusive|-?an|aktif|full|campur|[0-9]+ ?kamar)"
    r"|\b(?:rumah|bangunan)\s+(?:kost|kos)\b|\bkoskosan\b|\bkost\b"
)
POLA_USAHA: str = (
    r"\b(?:ruko|gudang|pabrik|homestay|guest ?house|penginapan|cafe|kafe|rumah kantor)\b"
)

# Ambang batas domain cleaning (usulan kerja peneliti)
MIN_HARGA: float = 5e7
MAX_HARGA: float = 2e10
MAX_KAMAR: int = 10
MIN_LUAS: float = 15.0
MAX_LUAS_BANGUNAN: float = 5000.0
MAX_RASIO_LB_LT: float = 3.0
MIN_HARGA_PER_M2_TANAH: float = 1e6

BEST_RF_PARAMS: dict[str, object] = {
    "bootstrap": True,
    "max_depth": 15,
    "max_features": 0.5,
    "min_samples_leaf": 2,
    "min_samples_split": 5,
    "n_estimators": 100,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

DEFAULT_MDAPE_RATE: float = 0.1471
