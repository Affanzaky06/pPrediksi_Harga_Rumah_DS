"""
Aplikasi Web Prediksi Harga Rumah D.I. Yogyakarta
Berbasis Machine Learning (Random Forest Regressor)
Dataset: Rumah123 Yogyakarta (Kaggle)
Pipeline & Model: progress2.ipynb
"""

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ---------------------------------------------------------
# 1. Konfigurasi Halaman & Tema
# ---------------------------------------------------------
st.set_page_config(
    page_title="Prediksi Harga Rumah D.I. Yogyakarta",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan modern & elegan
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F0F9FF 0%, #E0F2FE 100%);
        border: 1px solid #BAE6FD;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #0369A1;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #0C4A6E;
        margin: 5px 0;
    }
    .metric-sub {
        font-size: 0.85rem;
        color: #0284C7;
    }
    .info-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #334155;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# 2. Fungsi Load Artefak Model (dengan Cache Streamlit)
# ---------------------------------------------------------
@st.cache_resource
def load_bundle_model():
    model_path = os.path.join("model_artifacts", "model_prediksi_rumah_jogja.joblib")
    if not os.path.exists(model_path):
        st.error(f"File model tidak ditemukan di: `{model_path}`. Pastikan file joblib sudah diekspor dari notebook progress2.")
        st.stop()
    bundle = joblib.load(model_path)
    return bundle

bundle = load_bundle_model()
model = bundle['model']
scaler = bundle['scaler']
fitur_final = bundle['fitur_final']
kolom_scale = bundle['kolom_scale']
median_train = bundle['median_train']
kategori_wilayah = bundle['kategori_wilayah']


# ---------------------------------------------------------
# 3. Helper Functions
# ---------------------------------------------------------
def format_rupiah(angka):
    """Format float ke format mata uang Rupiah standar Indonesia."""
    if angka >= 1_000_000_000:
        return f"Rp {angka:,.0f}".replace(",", "."), f"{angka / 1_000_000_000:.2f} Miliar"
    elif angka >= 1_000_000:
        return f"Rp {angka:,.0f}".replace(",", "."), f"{angka / 1_000_000:.1f} Juta"
    else:
        return f"Rp {angka:,.0f}".replace(",", "."), ""

def predict_pipeline(luas_tanah, luas_bangunan, kamar_mandi, kamar_tidur, carport, wilayah):
    """
    Menjalankan pipeline preprocessing yang sama persis dengan progress2.ipynb:
    1. Log10 transformation pada luas_tanah & luas_bangunan
    2. Scaling menggunakan StandardScaler dari training set
    3. Ekstraksi fitur final & dummy encoding wilayah (baseline: Bantul)
    4. Prediksi log-target -> inverse log10 ke Rupiah
    """
    log_lt = np.log10(luas_tanah)
    log_lb = np.log10(luas_bangunan)

    # 1. Scaling kolom numerik sesuai kolom_scale
    df_scale_input = pd.DataFrame([{
        'log_luas_tanah': log_lt,
        'log_luas_bangunan': log_lb,
        'kamar_tidur': float(kamar_tidur),
        'kamar_mandi': float(kamar_mandi),
        'carport': float(carport)
    }])[kolom_scale]

    scaled_array = scaler.transform(df_scale_input)
    df_scaled = pd.DataFrame(scaled_array, columns=kolom_scale)

    # 2. Susun baris fitur_final
    row_features = {
        'log_luas_bangunan': df_scaled['log_luas_bangunan'].iloc[0],
        'log_luas_tanah': df_scaled['log_luas_tanah'].iloc[0],
        'kamar_mandi': df_scaled['kamar_mandi'].iloc[0]
    }

    # One-Hot Encoding dummy wilayah (Bantul adalah baseline 0 0 0)
    for col in fitur_final:
        if col.startswith('wilayah_'):
            nama_wil = col.replace('wilayah_', '')
            row_features[col] = 1 if wilayah == nama_wil else 0

    df_final = pd.DataFrame([row_features])[fitur_final]

    # 3. Prediksi model (log10 harga)
    pred_log = model.predict(df_final)[0]
    pred_rupiah = 10 ** pred_log

    return pred_rupiah, pred_log, df_final


# ---------------------------------------------------------
# 4. Header Aplikasi
# ---------------------------------------------------------
st.markdown('<div class="main-header">🏡 Prediksi Harga Rumah D.I. Yogyakarta</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Estimasi harga pasar properti residensial berbasis Machine Learning (Random Forest) '
    'yang telah dioptimasi dengan Hyperparameter Tuning & Group Cross-Validation.'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# 5. Navigasi Tab
# ---------------------------------------------------------
tab_prediksi, tab_analisis, tab_metodologi = st.tabs([
    "🎯 Kalkulator Prediksi",
    "📊 Analisis & Feature Importance",
    "📚 Metodologi & Evaluasi Model"
])


# =========================================================
# TAB 1: KALKULATOR PREDIKSI
# =========================================================
with tab_prediksi:
    st.markdown("### Masukkan Spesifikasi Properti")
    st.write("Silakan lengkapi parameter rumah di bawah ini untuk mengestimasikan harga pasar yang realistis:")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📍 Lokasi & Dimensi Properti")
        
        wilayah_pilihan = st.selectbox(
            "Wilayah / Kabupaten-Kota:",
            options=kategori_wilayah,
            index=kategori_wilayah.index("Sleman") if "Sleman" in kategori_wilayah else 0,
            help="Pilih kabupaten atau kota administratif di D.I. Yogyakarta."
        )

        luas_tanah = st.number_input(
            "Luas Tanah (m²):",
            min_value=20.0,
            max_value=3000.0,
            value=120.0,
            step=5.0,
            help="Total luas bidang tanah dalam meter persegi."
        )

        luas_bangunan = st.number_input(
            "Luas Bangunan (m²):",
            min_value=20.0,
            max_value=2000.0,
            value=100.0,
            step=5.0,
            help="Total luas lantai bangunan rumah dalam meter persegi."
        )

        if luas_bangunan > luas_tanah:
            st.info("ℹ️ Luas bangunan lebih besar dari luas tanah (umum pada rumah 2-3 lantai).")

    with col2:
        st.subheader("🛋️ Fasilitas & Ruangan")

        def_km = int(median_train.get('kamar_mandi', 2))
        def_kt = int(median_train.get('kamar_tidur', 3))

        kamar_mandi = st.number_input(
            "Jumlah Kamar Mandi:",
            min_value=1,
            max_value=15,
            value=def_km,
            step=1,
            help=f"Nilai median pada data training: {def_km}"
        )

        kamar_tidur = st.number_input(
            "Jumlah Kamar Tidur:",
            min_value=1,
            max_value=20,
            value=def_kt,
            step=1,
            help=f"Nilai median pada data training: {def_kt}"
        )

        carport = st.number_input(
            "Kapasitas Garasi / Carport (Mobil):",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
            help="Kapasitas kendaraan roda empat di carport/garasi."
        )

    st.markdown("---")

    btn_hitung = st.button("🚀 Prediksi Harga Rumah", type="primary", use_container_width=True)

    if btn_hitung:
        with st.spinner("Memproses pipeline fitur & menghitung estimasi harga..."):
            pred_rupiah, pred_log, df_fitur = predict_pipeline(
                luas_tanah=luas_tanah,
                luas_bangunan=luas_bangunan,
                kamar_mandi=kamar_mandi,
                kamar_tidur=kamar_tidur,
                carport=carport,
                wilayah=wilayah_pilihan
            )

        str_rp, str_ringkas = format_rupiah(pred_rupiah)

        # Rentang estimasi wajar berbasis MdAPE (~15.65%)
        MDAPE_RATE = 0.1565
        batas_bawah = pred_rupiah * (1 - MDAPE_RATE)
        batas_atas = pred_rupiah * (1 + MDAPE_RATE)
        str_bawah, _ = format_rupiah(batas_bawah)
        str_atas, _ = format_rupiah(batas_atas)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Estimasi Nilai Pasar Properti</div>
            <div class="metric-value">{str_rp}</div>
            <div class="metric-sub">(Setara sekitar <b>{str_ringkas}</b>)</div>
        </div>
        """, unsafe_allow_html=True)

        res1, res2, res3 = st.columns(3)
        with res1:
            st.metric(
                label="Rentang Wajar (±15.65% MdAPE)",
                value=f"{str_bawah} - {str_atas}",
                help="Berdasarkan Median Absolute Percentage Error model pada unseen data."
            )
        with res2:
            harga_per_m2_lt = pred_rupiah / luas_tanah
            st.metric(
                label="Estimasi Harga / m² Tanah",
                value=f"Rp {harga_per_m2_lt:,.0f}/m²".replace(",", ".")
            )
        with res3:
            harga_per_m2_lb = pred_rupiah / luas_bangunan
            st.metric(
                label="Estimasi Harga / m² Bangunan",
                value=f"Rp {harga_per_m2_lb:,.0f}/m²".replace(",", ".")
            )

        with st.expander("🔍 Rincian Transformasi Fitur Model (Internal Preprocessing)"):
            st.write("Vektor fitur akhir yang dimasukkan ke model Random Forest (`X_final`):")
            st.dataframe(df_fitur.round(4))
            st.caption(f"Log10 Prediksi Target: **{pred_log:.4f}** ➔ Inversi ($10^{{{pred_log:.4f}}}$) = **Rp {pred_rupiah:,.0f}**")


# =========================================================
# TAB 2: ANALISIS & FEATURE IMPORTANCE
# =========================================================
with tab_analisis:
    st.subheader("Bobot Kepentingan Fitur (Feature Importance)")
    st.write("Kontribusi masing-masing fitur dalam memprediksi harga properti menurut model Random Forest Regressor:")

    importances = model.feature_importances_
    df_importance = pd.DataFrame({
        'Fitur': [
            'Luas Bangunan (Log)',
            'Luas Tanah (Log)',
            'Jumlah Kamar Mandi',
            'Wilayah: Kota Yogyakarta',
            'Wilayah: Sleman',
            'Wilayah: Kulon Progo / Gunung Kidul'
        ],
        'Tingkat Kepentingan (%)': importances * 100
    }).sort_values(by='Tingkat Kepentingan (%)', ascending=True)

    col_chart, col_stat = st.columns([3, 2], gap="large")

    with col_chart:
        st.bar_chart(
            df_importance.set_index('Fitur'),
            x_label="Persentase (%)",
            y_label="Fitur",
            horizontal=True
        )

    with col_stat:
        st.markdown("#### Temuan Kunci Model:")
        st.markdown("""
        1. **Luas Bangunan (~81.1%)** merupakan prediktor paling dominan dalam menentukan harga properti di Yogyakarta.
        2. **Luas Tanah (~13.8%)** menjadi faktor kedua terpenting, terutama untuk kavling luas di area suburban/rural.
        3. **Kamar Mandi (~4.1%)** lebih informatif daripada kamar tidur karena berkorelasi langsung dengan kelas rumah/finishing mewah.
        4. **Efek Wilayah (~1%)**: Baseline perbandingan adalah **Kabupaten Bantul**. Wilayah Kota Yogyakarta dan Sleman memberikan premi lokasi positif terhadap harga listing.
        """)


# =========================================================
# TAB 3: METODOLOGI & EVALUASI MODEL
# =========================================================
with tab_metodologi:
    st.subheader("Spesifikasi & Kinerja Model")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="R² Score (Test Set)", value="86.52%")
    with kpi2:
        st.metric(label="MdAPE (Median Error)", value="15.65%")
    with kpi3:
        st.metric(label="MAPE (Mean Error)", value="22.26%")
    with kpi4:
        st.metric(label="MAE Rata-rata", value="Rp 365 Juta")

    st.markdown("---")
    st.markdown("#### Detail Pipeline Data Science (`progress2.ipynb`)")

    st.markdown("""
    | Komponen | Implementasi Teknis |
    | :--- | :--- |
    | **Sumber Data** | Rumah123 Yogyakarta (Kaggle Dataset) |
    | **Data Cleansing** | Filter outlier ekstrem (Z-Score & batas kuantil), normalisasi format teks |
    | **Data Splitting** | Stratified Group Split (80% Train, 20% Test) dikelompokkan berdasarkan kemiripan deskripsi listing untuk mencegah *data leakage* |
    | **Imputasi Nilai Hilang** | Median training set (`kamar_tidur=3`, `kamar_mandi=2`), Carport default = 0 |
    | **Feature Engineering** | Log10 transformasi pada variabel skala luas (`luas_tanah`, `luas_bangunan`) & target `harga` untuk menstabilkan variansi residual |
    | **Encoding & Scaling** | One-Hot Encoding dummy wilayah (Drop baseline = Bantul), `StandardScaler` pada fitur kontinu & diskrit |
    | **Algoritma Model** | `RandomForestRegressor` dengan Hyperparameter Tuning (`max_depth=20`, `min_samples_leaf=2`, `min_samples_split=10`, `n_estimators=300`) |
    """)

    st.markdown("""
    > **Catatan Penggunaan:** Model ini dirancang untuk memprediksi harga penawaran listing rumah wajar di wilayah D.I. Yogyakarta. 
    > Harga riil transaksi dapat bervariasi bergantung pada legalitas (SHM/HGB), lebar akses jalan, material finishing interior, dan kondisi negosiasi penjual-pembeli.
    """)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.caption("Proyek Sains Data - Prediksi Harga Properti D.I. Yogyakarta | Model terkompilasi dari artefak `model_prediksi_rumah_jogja.joblib`")
