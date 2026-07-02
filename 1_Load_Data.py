import streamlit as st
import pandas as pd

st.set_page_config(page_title="Load Data", page_icon="📁")

st.title("📁 Load Data")
st.write(
    "Setiap proyek peramalan (*forecasting*) dimulai dengan data. "
    "Silakan unggah file CSV berisi data historis yang akan dianalisis."
)

# ── Menginisialisasi Session State ──────────────────────────────────────────
# Ini berfungsi sebagai "memori" agar data yang diunggah tidak hilang
if "data_raw" not in st.session_state:
    st.session_state["data_raw"] = None
if "data_name" not in st.session_state:
    st.session_state["data_name"] = ""

# ── Upload CSV ─────────────────────────────────────────────────────────────
st.write("Unggah file CSV dari komputermu. Pastikan baris pertama berisi nama kolom.")
uploaded = st.file_uploader(
    "Pilih file CSV",
    type=["csv"],
    help="Hanya menerima format file CSV.",
)

if uploaded is not None:
    try:
        # Membaca file menggunakan Pandas
        # Gunakan sep=';' jika file CSV komputermu menggunakan format pemisah titik koma
        df = pd.read_csv(uploaded, sep=';') 
        
        st.write("**Pratinjau Data** (10 baris pertama):")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("Gunakan Dataset Ini", type="primary", key="load_csv"):
            # Menyimpan data ke dalam "memori" Streamlit
            st.session_state["data_raw"] = df
            st.session_state["data_name"] = uploaded.name
            
            st.success(
                f"Berhasil memuat **{uploaded.name}** — {df.shape[0]:,} baris, {df.shape[1]} kolom."
            )
    except Exception as e:
        st.error(f"Gagal membaca file CSV: {e}")

# ── Status Dataset Saat Ini ────────────────────────────────────────────────
st.divider()
if st.session_state["data_raw"] is not None:
    st.write(f"**Dataset aktif:** {st.session_state['data_name']}")
    st.dataframe(st.session_state["data_raw"].head(5), use_container_width=True)
    
    st.info("Data siap! Kamu bisa melanjutkan ke tahap Eksplorasi Data atau Pelatihan Model di menu samping.")
else:
    st.write("Belum ada dataset yang dimuat. Silakan unggah file CSV di atas.")