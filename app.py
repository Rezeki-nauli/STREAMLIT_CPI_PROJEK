import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Konfigurasi Halaman (PENTING: letakkan di baris paling pertama)
st.set_page_config(page_title="Forecasting IHK Medan", layout="wide", page_icon="📈")

# --- CUSTOM CSS UNTUK HEADER, METRIC CARDS, DAN RESPONSIVE LAYOUT ---
st.markdown("""
<style>
    /* ============================================================
       BASE STYLES (Desktop - default)
       ============================================================ */

    /* Styling untuk KPI Cards */
    div[data-testid="metric-container"] {
        background-color: #1a2235;
        border: 1px solid rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.2s ease;
    }

    /* Menyembunyikan elemen header default Streamlit */
    header {visibility: hidden;}

    /* Styling untuk Custom Top Header ala IHKCAST */
    .custom-header {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        gap: 12px;
        background-color: #121826;
        padding: 15px 25px;
        border-radius: 8px;
        border: 1px solid #1e293b;
        margin-bottom: 25px;
        margin-top: -50px;
    }
    .header-left {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 15px;
    }
    .logo-text {
        color: #00c9a7;
        font-weight: 800;
        font-size: 1.2rem;
        letter-spacing: 1px;
        display: flex;
        align-items: center;
        gap: 8px;
        white-space: nowrap;
    }
    .logo-dot {
        width: 10px;
        height: 10px;
        background-color: #00c9a7;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }
    .header-title {
        color: #94a3b8;
        font-size: 1.05rem;
        border-left: 1px solid #334155;
        padding-left: 15px;
    }
    .header-right .badge {
        background-color: rgba(0, 201, 167, 0.1);
        color: #00c9a7;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(0, 201, 167, 0.2);
        white-space: nowrap;
    }

    /* ============================================================
       RESPONSIVE: Gambar/Grafik matplotlib selalu mengikuti lebar
       kontainer agar tidak overflow di layar sempit.
       ============================================================ */
    div[data-testid="stImage"] img,
    .element-container img {
        max-width: 100% !important;
        height: auto !important;
    }

    /* Tabel & dataframe agar bisa discroll horizontal di mobile
       daripada memaksa layout melebar */
    div[data-testid="stDataFrame"] {
        overflow-x: auto;
    }

    /* ============================================================
       TABLET (<= 1024px): rapatkan padding & font sedikit,
       KPI cards jadi 2 kolom per baris.
       ============================================================ */
    @media (max-width: 1024px) {
        .custom-header {
            margin-top: -30px;
            padding: 12px 18px;
        }
        .header-title {
            font-size: 0.95rem;
        }
        div[data-testid="metric-container"] {
            padding: 12px;
        }
        div[data-testid="column"] {
            min-width: 45% !important;
            flex: 1 1 45% !important;
        }
    }

    /* ============================================================
       MOBILE (<= 768px): semua kolom Streamlit ditumpuk vertikal
       (1 kolom penuh), header jadi vertikal, font diperkecil.
       ============================================================ */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 2rem !important;
        }

        .custom-header {
            flex-direction: column;
            align-items: flex-start;
            margin-top: 0;
            padding: 12px 15px;
        }
        .header-left {
            flex-direction: column;
            align-items: flex-start;
            gap: 8px;
        }
        .header-title {
            border-left: none;
            padding-left: 0;
            font-size: 0.9rem;
        }
        .header-right .badge {
            font-size: 0.75rem;
            padding: 5px 10px;
        }
        .logo-text {
            font-size: 1.05rem;
        }

        /* Paksa semua kolom Streamlit (KPI cards, grafik berdampingan,
           panel kiri/kanan) menjadi satu kolom penuh agar mudah dibaca
           dan tidak berdesakan di layar kecil */
        div[data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }

        div[data-testid="metric-container"] {
            padding: 10px;
        }
        div[data-testid="metric-container"] label {
            font-size: 0.75rem !important;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            font-size: 1.3rem !important;
        }

        /* Judul section (##### ...) sedikit lebih kecil di mobile */
        h5 {
            font-size: 1rem !important;
        }
    }

    /* ============================================================
       MOBILE KECIL (<= 480px): penyesuaian ekstra untuk HP layar
       sempit (mis. lebar < 400px).
       ============================================================ */
    @media (max-width: 480px) {
        .logo-text {
            font-size: 0.95rem;
        }
        .header-title {
            font-size: 0.8rem;
        }
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# 2. Memuat Paket Model (format DICT hasil train_model.py yang sudah diperbaiki)
# Dict berisi: model, nama_model, order, seasonal_order, data_historis, data_train,
# data_test, tabel_grid_search, mae_test, mape_test, prediksi_test_mean/ci,
# ramalan_masa_depan_mean/ci, tabel_ramalan.
MODEL_PATH = "src/models/sarima_ihk_medan.pkl"


@st.cache_resource
def load_paket():
    with open(MODEL_PATH, "rb") as f:
        paket = pickle.load(f)

    model = paket["model"]
    if hasattr(model, "model") and hasattr(model.model, "init_kwds"):
        if "dates" in model.model.init_kwds:
            del model.model.init_kwds["dates"]

    return paket


try:
    paket = load_paket()
except FileNotFoundError:
    st.error(
        f"Model belum tersedia! Pastikan file '{MODEL_PATH}' ada (jalankan train_model.py "
        "terlebih dahulu, atau sesuaikan MODEL_PATH di atas)."
    )
    st.stop()

fit_final = paket["model"]
nama_model_final = paket.get("nama_model", "SARIMA")
mae_test = paket.get("mae_test", np.nan)
mape_test = paket.get("mape_test", np.nan)
df_grid = paket.get("tabel_grid_search")

# Data historis diambil langsung dari paket (hasil backward splicing), bukan dari
# fit_final.data.endog, supaya tidak bergantung pada representasi internal statsmodels.
ihk_series_bawaan = paket["data_historis"]["IHK_Spliced"].copy()
ihk_series_bawaan.index = pd.DatetimeIndex(ihk_series_bawaan.index)

# --- MEMBUAT CUSTOM HEADER IHKCAST ---
bulan_terakhir_data = ihk_series_bawaan.index[-1].strftime('%b %Y')
st.markdown(f"""
<div class="custom-header">
    <div class="header-left">
        <div class="logo-text"><span class="logo-dot"></span> CPI - Customer Price Index</div>
        <div class="header-title">Peramalan IHK Kota Medan - Badan Pusat Statistik (BPS)</div>
    </div>
    <div class="header-right">
        <div class="badge">● Data s.d. {bulan_terakhir_data} &nbsp;|&nbsp; Model: {nama_model_final}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- RAMALAN SINGKAT (pakai hasil yang sudah dihitung di train_model.py bila tersedia) ---
if "ramalan_masa_depan_mean" in paket and "ramalan_masa_depan_ci" in paket:
    fc_mean_overview = paket["ramalan_masa_depan_mean"]
    fc_ci_overview = paket["ramalan_masa_depan_ci"]
else:
    forecast_overview = fit_final.get_forecast(steps=12)
    fc_mean_overview = forecast_overview.predicted_mean
    fc_ci_overview = forecast_overview.conf_int(alpha=0.05)
    last_date_ov = ihk_series_bawaan.index[-1]
    fc_index_ov = pd.date_range(start=last_date_ov + pd.DateOffset(months=1), periods=12, freq='ME')
    fc_mean_overview.index = fc_index_ov
    fc_ci_overview.index = fc_index_ov


def get_fig_size(default=(10, 5), mobile=(6.5, 4)):
    """
    Streamlit murni server-side tidak tahu lebar layar client secara
    langsung tanpa komponen JS tambahan. Sebagai solusi praktis dan ringan,
    figsize dasar dibuat lebih ramping (rasio lebih persegi) supaya saat
    di-scale oleh CSS (max-width:100%) ke layar sempit, proporsi teks/garis
    tetap enak dibaca dibanding figsize yang sangat lebar (mis. 12x4).
    """
    return default


# 3. TOP NAVIGATION TABS
tab_ringkasan, tab_ramalan, tab_model, tab_simulasi = st.tabs([
    "📋 Ringkasan",
    "🚀 Ramalan",
    "⚙️ Hasil Latih Model",
    "🎛️ Simulasi Kebijakan BPS"
])

# ==========================================
# TAB 1: RINGKASAN
# ==========================================
with tab_ringkasan:
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    nilai_terakhir = ihk_series_bawaan.iloc[-1]
    bulan_terakhir = ihk_series_bawaan.index[-1].strftime('%b %Y')

    nilai_ramalan_akhir = fc_mean_overview.iloc[-1]
    bulan_ramalan_akhir = fc_mean_overview.index[-1].strftime('%b %Y')

    delta_terkini = nilai_terakhir - ihk_series_bawaan.iloc[-2]

    with kpi1:
        st.metric(label=f"IHK TERKINI ({bulan_terakhir})", value=f"{nilai_terakhir:.2f}", delta=f"{delta_terkini:+.2f} vs Bulan Lalu", delta_color="inverse")
    with kpi2:
        delta_ramalan = nilai_ramalan_akhir - nilai_terakhir
        st.metric(label=f"RAMALAN {bulan_ramalan_akhir.upper()}", value=f"{nilai_ramalan_akhir:.2f}", delta=f"{delta_ramalan:+.2f} Proyeksi", delta_color="inverse")
    with kpi3:
        st.metric(label="AKURASI (MAE)", value=f"{mae_test:.2f}", delta="Data Testing", delta_color="off")
    with kpi4:
        st.metric(label="AKURASI (MAPE)", value=f"{mape_test:.2f}%", delta="Data Testing", delta_color="off")
    with kpi5:
        st.metric(label="TARGET INFLASI", value="2.5 ±1%", delta="Dalam Kisaran Aman", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)

    col_tren, col_musim = st.columns([2, 1])

    with col_tren:
        st.markdown("##### TREN IHK HISTORIS & RAMALAN")
        fig_hist, ax_hist = plt.subplots(figsize=get_fig_size())

        ihk_hist_plot = ihk_series_bawaan  # seluruh histori hasil backward splicing (2014-sekarang)
        ax_hist.plot(ihk_hist_plot.index, ihk_hist_plot.values, color='#00c9a7', linewidth=1.5, marker='o', markersize=2, label='IHK Aktual')
        ax_hist.plot(fc_mean_overview.index, fc_mean_overview.values, color='#3b82f6', linewidth=2.5, linestyle='--', marker='o', markersize=4, label='Ramalan SARIMA')
        ax_hist.fill_between(fc_ci_overview.index, fc_ci_overview.iloc[:, 0], fc_ci_overview.iloc[:, 1], color='#3b82f6', alpha=0.15, label='CI atas 95%')

        ax_hist.set_facecolor('#1a2235')
        fig_hist.patch.set_facecolor('#0e1117')
        ax_hist.tick_params(colors='white')

        ax_hist.grid(color='white', linestyle='-', linewidth=0.2, alpha=0.3)
        ax_hist.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, frameon=False, labelcolor='white')
        ax_hist.xaxis.set_major_locator(mdates.YearLocator())
        ax_hist.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        st.pyplot(fig_hist, use_container_width=True)

    with col_musim:
        st.markdown("##### POLA MUSIMAN RATA-RATA")
        fig_sea, ax_sea = plt.subplots(figsize=(6, 5))

        try:
            dec = seasonal_decompose(ihk_series_bawaan, model='additive', period=12)
            seasonal_comp = dec.seasonal
            monthly_seasonality = seasonal_comp.groupby(seasonal_comp.index.month).mean()

            months_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Ags', 'Sep', 'Okt', 'Nov', 'Des']
            bar_colors = ['#eab308' if val > 0.8 else ('#ef4444' if val < 0 else '#00c9a7') for val in monthly_seasonality]

            ax_sea.bar(months_labels, monthly_seasonality, color=bar_colors, width=0.7)

            ax_sea.set_facecolor('#1a2235')
            fig_sea.patch.set_facecolor('#0e1117')
            ax_sea.tick_params(colors='white')
            ax_sea.axhline(0, color='gray', linewidth=0.8, linestyle='--')
            ax_sea.grid(color='white', linestyle='-', linewidth=0.2, alpha=0.3, axis='y')

            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig_sea, use_container_width=True)

        except Exception:
            st.error("Data tidak memiliki rentang waktu yang cukup.")

# ==========================================
# TAB 2: RAMALAN DENGAN UPLOAD DATA
# ==========================================
with tab_ramalan:
    col_kiri, col_kanan = st.columns([1, 1])

    with col_kiri:
        st.subheader("1. Sumber Data Historis")
        st.info("Secara default, prediksi menggunakan data saat model dilatih. Anda bisa mengunggah data IHK terbaru (CSV) untuk meramalkan dari titik terakhir data baru tersebut.")
        uploaded_file = st.file_uploader("Upload Data IHK Terbaru (.csv delimiter ';')", type=['csv'])

    with col_kanan:
        st.subheader("2. Pengaturan Prediksi")
        n_forecast = st.slider("Horizon Waktu (Bulan ke depan):", min_value=1, max_value=36, value=12)
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Jalankan Prediksi 🚀", use_container_width=True)

    if run_btn:
        with st.spinner("Memproses data dan menghitung proyeksi..."):
            try:
                dinamis_mae = mae_test
                dinamis_mape = mape_test
                is_uploaded = False

                if uploaded_file is not None:
                    is_uploaded = True
                    df_new = pd.read_csv(uploaded_file, sep=';')
                    col_tgl = df_new.columns[0]
                    col_val = df_new.columns[1]

                    df_new[col_tgl] = pd.to_datetime(df_new[col_tgl], dayfirst=True, errors='coerce')
                    df_new = df_new.dropna(subset=[col_tgl, col_val])
                    df_new.set_index(col_tgl, inplace=True)

                    ihk_series_aktif = df_new[col_val].astype(float)
                    ihk_series_aktif.index = pd.DatetimeIndex(ihk_series_aktif.index).to_period('M').to_timestamp()

                    current_model = fit_final.apply(ihk_series_aktif)

                    fitted_aktif = current_model.fittedvalues
                    start_idx = 12 if len(ihk_series_aktif) > 24 else 0

                    aktual_calc = ihk_series_aktif[start_idx:]
                    fitted_calc = fitted_aktif[start_idx:]

                    dinamis_mae = np.mean(np.abs(aktual_calc - fitted_calc))
                    dinamis_mape = np.mean(np.abs((aktual_calc - fitted_calc) / aktual_calc)) * 100

                else:
                    current_model = fit_final
                    ihk_series_aktif = ihk_series_bawaan

                st.divider()
                if is_uploaded:
                    st.success("✅ Berhasil memetakan model ke data unggahan terbaru!")
                    st.markdown("##### 🎯 Evaluasi Akurasi pada Data yang Diunggah:")
                    col_acc1, col_acc2, col_acc3 = st.columns([1, 1, 2])
                    col_acc1.metric("MAE (Data Baru)", f"{dinamis_mae:.2f}", "Nilai Error Absolut", delta_color="off")
                    col_acc2.metric("MAPE (Data Baru)", f"{dinamis_mape:.2f}%", "Persentase Error", delta_color="off")
                else:
                    st.info("ℹ️ Menggunakan data historis bawaan dari model (BPS).")

                st.markdown("<br>", unsafe_allow_html=True)

                forecast_obj = current_model.get_forecast(steps=n_forecast)
                forecast_mean = forecast_obj.predicted_mean
                forecast_ci = forecast_obj.conf_int(alpha=0.05)

                last_date = ihk_series_aktif.index[-1]
                forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_forecast, freq='ME')
                forecast_mean.index = forecast_index
                forecast_ci.index = forecast_index

                df_forecast = pd.DataFrame({
                    'Bulan': forecast_index.strftime('%B %Y'),
                    'Nilai IHK (Proyeksi)': forecast_mean.values.round(2),
                    'Batas Bawah (95%)': forecast_ci.iloc[:, 0].values.round(2),
                    'Batas Atas (95%)': forecast_ci.iloc[:, 1].values.round(2)
                })

                col_chart, col_table = st.columns([2, 1])
                with col_chart:
                    st.markdown("##### 📊 Grafik Tren dan Interval Kepercayaan")
                    ihk_24 = ihk_series_aktif[-24:]

                    fig, ax = plt.subplots(figsize=get_fig_size())
                    ax.plot(ihk_24.index, ihk_24.values, color='#00c9a7', linewidth=2, label='IHK Historis (Aktif)')
                    ax.plot(forecast_mean.index, forecast_mean.values, color='#3b82f6', linewidth=2.5, linestyle='--', marker='o', markersize=4, label='Forecast IHK')
                    ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='#3b82f6', alpha=0.15, label='CI (95%)')

                    ax.axvline(x=ihk_series_aktif.index[-1], color='gray', linestyle=':', linewidth=1.5, label='Akhir Data Aktual')

                    ax.set_facecolor('#1a2235')
                    fig.patch.set_facecolor('#0e1117')
                    ax.tick_params(colors='white')
                    ax.grid(color='white', linestyle='-', linewidth=0.2, alpha=0.3)

                    legend = ax.legend(loc='upper left', facecolor='#1e2848', edgecolor='none')
                    for text in legend.get_texts():
                        text.set_color("white")

                    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, n_forecast // 6)))
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)

                with col_table:
                    st.markdown(f"##### 📋 Tabel Proyeksi ({n_forecast} Bulan)")
                    st.dataframe(df_forecast, hide_index=True, use_container_width=True)

            except Exception as e:
                st.error(f"Terjadi kesalahan sistem: {e}")

# ==========================================
# TAB 3: MODEL SARIMA
# ==========================================
with tab_model:
    st.subheader("Spesifikasi & Diagnostik Model SARIMA")

    info1, info2, info3 = st.columns(3)
    info1.metric("Model Terpilih", nama_model_final)
    info2.metric("MAE (Test)", f"{mae_test:.4f}")
    info3.metric("MAPE (Test)", f"{mape_test:.2f}%")

    st.caption(
        "Model dipilih dari kandidat AIC terendah hasil grid search yang LOLOS uji "
        "Ljung-Box (residual bersifat white noise, p-value > 0.05 di lag 6/12/18/24)."
    )

    if df_grid is not None:
        with st.expander("📑 Lihat Tabel Lengkap Hasil Grid Search"):
            st.dataframe(df_grid, hide_index=True, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_sum1, col_sum2 = st.columns([1, 1])

    with col_sum1:
        st.markdown("**Ringkasan Parameter Model (Summary)**")
        st.text(fit_final.summary().as_text())

    with col_sum2:
        st.markdown("**Grafik Diagnostik Residual**")
        try:
            fig_diag = fit_final.plot_diagnostics(figsize=(10, 8))
            fig_diag.patch.set_facecolor('#0e1117')
            for ax in fig_diag.axes:
                ax.set_facecolor('#1a2235')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
            plt.tight_layout()
            st.pyplot(fig_diag, use_container_width=True)
        except Exception:
            st.warning("Grafik diagnostik tidak dapat ditampilkan.")

# ==========================================
# TAB 4: SIMULASI KEBIJAKAN
# ==========================================
with tab_simulasi:
    st.subheader("🎛️ Simulator Skenario Kebijakan Pengendalian Inflasi Daerah (TPID)")
    st.info("💡 **Informasi Penting:** BPS bertugas *menghitung dan mempublikasikan* data inflasi (IHK). Sementara eksekusi kebijakan penekanan inflasi dilakukan oleh **TPID (Tim Pengendalian Inflasi Daerah)** yang beranggotakan Pemda, Bank Indonesia, Bulog, dan BPS. Di Indonesia, kebijakan ini dirumuskan dalam **Strategi 4K**.")

    col_sim1, col_sim2 = st.columns([1.2, 1])

    with col_sim1:
        st.markdown("**1. Keterjangkauan Harga**")
        subsidi = st.slider("Subsidi Pangan & Operasi Pasar (% Dampak Penurunan IHK)", 0.0, 5.0, 0.0, step=0.1, help="Simulasi seberapa jauh suntikan subsidi pasar murah (Pasar Murah Pemko Medan) dapat menekan indeks harga.")

        st.markdown("**2. Kelancaran Distribusi**")
        ongkir = st.slider("Subsidi Ongkos Angkut / KAD (% Dampak Penurunan IHK)", 0.0, 3.0, 0.0, step=0.1, help="Simulasi efisiensi biaya logistik dan Kerjasama Antar Daerah (KAD).")

        st.markdown("**3. Ketersediaan Pasokan**")
        cadangan = st.number_input("Optimalisasi Cadangan Pangan Daerah (Ribuan Ton)", min_value=0, max_value=1000, value=100, step=10)

    with col_sim2:
        st.markdown("##### 📊 Proyeksi Dampak Intervensi")
        total_dampak = subsidi + ongkir
        st.success(f"**Ekspektasi Penekanan IHK: {-total_dampak:.2f}% dari batas atas baseline.**")

        # Simulasi dampak matematis pada bulan terakhir ramalan overview
        nilai_awal_sim = fc_mean_overview.iloc[-1]
        nilai_akhir_sim = nilai_awal_sim * (1 - (total_dampak / 100))

        st.metric(label="Simulasi IHK (Efek Kebijakan)", value=f"{nilai_akhir_sim:.2f}", delta=f"{-total_dampak:.2f}% Penurunan", delta_color="inverse")

        st.caption("Catatan: **Pilar ke-4 (Komunikasi Efektif)** diasumsikan terkendali dengan baik, ditandai dengan tidak adanya *panic buying* di tengah masyarakat akibat transparansi data oleh BPS dan Pemda.")