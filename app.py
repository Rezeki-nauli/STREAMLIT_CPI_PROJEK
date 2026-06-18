import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Konfigurasi Halaman
st.set_page_config(page_title="Forecasting IHK Medan", layout="wide", page_icon="📈")

st.title("📈 Dashboard Peramalan IHK Kota Medan")
st.markdown("Menggunakan algoritma **Seasonal Autoregressive Integrated Moving Average (SARIMA)**.")
st.divider()

# 1. Memuat Model SARIMA yang sudah dilatih
@st.cache_resource
def load_model():
    # Tambahkan 'src/' di depan 'models/'
    with open('src/models/sarima_ihk_medan.pkl', 'rb') as f:
        return pickle.load(f)
try:
    fit_final = load_model()
except FileNotFoundError:
    st.error("Model belum tersedia! Jalankan perintah 'python src/train_model.py' di terminal terlebih dahulu.")
    st.stop()

# Mengekstrak data historis dari model
ihk_historis = fit_final.data.endog
ihk_index = fit_final.data.row_labels
ihk_series = pd.Series(ihk_historis, index=ihk_index)

# 2. Sidebar untuk interaksi pengguna
st.sidebar.header("⚙️ Pengaturan Parameter")
n_forecast = st.sidebar.slider("Pilih Horizon Waktu (Bulan ke depan):", min_value=1, max_value=36, value=12)

if st.sidebar.button("Jalankan Prediksi 🚀"):
    with st.spinner("Menghitung proyeksi nilai IHK..."):
        # 3. Proses Forecasting
        forecast_obj = fit_final.get_forecast(steps=n_forecast)
        forecast_mean = forecast_obj.predicted_mean
        forecast_ci = forecast_obj.conf_int(alpha=0.05) # CI 95%
        
        last_date = ihk_series.index[-1]
        forecast_index = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=n_forecast, freq='ME')
        forecast_mean.index = forecast_index
        forecast_ci.index = forecast_index
        
        # 4. Membuat Tabel Output
        df_forecast = pd.DataFrame({
            'Bulan': forecast_index.strftime('%B %Y'),
            'Nilai IHK (Proyeksi)': forecast_mean.values.round(2),
            'Batas Bawah (95%)': forecast_ci.iloc[:, 0].values.round(2),
            'Batas Atas (95%)': forecast_ci.iloc[:, 1].values.round(2)
        })

        col1, col2 = st.columns([1, 2])
        
        # Menampilkan Tabel
        with col1:
            st.subheader(f"📋 Tabel Prediksi ({n_forecast} Bulan)")
            st.dataframe(df_forecast, hide_index=True)
            
        # Menampilkan Grafik persis seperti di kodemu
        with col2:
            st.subheader("📊 Grafik Tren dan Interval Kepercayaan")
            
            # Mengambil 24 bulan terakhir untuk visualisasi agar tidak terlalu padat
            ihk_24 = ihk_series[-24:]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ihk_24.index, ihk_24.values, color='steelblue', linewidth=2, label='IHK Historis (24 Bulan Terakhir)')
            ax.plot(forecast_mean.index, forecast_mean.values, color='darkorange', linewidth=2.5, linestyle='--', marker='o', markersize=4, label='Forecast IHK')
            ax.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='darkorange', alpha=0.15, label='Confidence Interval (95%)')
            
            ax.axvline(x=ihk_series.index[-1], color='gray', linestyle=':', linewidth=1.5, label='Akhir Data Aktual')
            ax.set_title(f"Proyeksi IHK Kota Medan {n_forecast} Bulan Kedepan", fontweight='bold')
            ax.set_xlabel("Periode")
            ax.set_ylabel("Nilai IHK")
            ax.legend(loc='upper left')
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, n_forecast//6)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
            plt.tight_layout()
            
            # Menampilkan ke Streamlit
            st.pyplot(fig)
            
    st.success("Peramalan berhasil ditampilkan!")
else:
    st.info("Silakan atur jumlah bulan di sidebar sebelah kiri, lalu klik 'Jalankan Prediksi'.")