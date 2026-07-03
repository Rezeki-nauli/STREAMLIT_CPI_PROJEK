# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
# import warnings
# import pickle
# import os
# from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from sklearn.metrics import mean_absolute_error
# import itertools

# warnings.filterwarnings('ignore')

# print("⏳ Memulai proses pelacakan data dan pelatihan model SARIMA...")

# # 1. Baca Data dari folder yang benar
# df_raw = pd.read_csv('../data/raw/DATA_IHK_KOTA_MEDAN.csv', sep=';')
# df_raw.columns = ['Tanggal', 'IHK']
# df_raw['Tanggal'] = pd.to_datetime(df_raw['Tanggal'], format='%d/%m/%Y')
# df_raw['IHK'] = pd.to_numeric(df_raw['IHK'], errors='coerce')
# df_raw.set_index('Tanggal', inplace=True)
# df_raw.index = pd.DatetimeIndex(df_raw.index).to_period('M').to_timestamp('M')

# # Filter data aktual mulai 2020
# df_aktual = df_raw.dropna().copy()
# df = df_aktual['2020':].copy()
# ihk = df['IHK']

# # 2. Split Data (Train & Test)
# train = ihk[:'2024-12']
# test  = ihk['2025-01':]

# # 3. Grid Search Sederhana (sesuai kodemu)
# d, D, s = 1, 0, 12 # Asumsi D=0 berdasarkan kodemu, atau sesuaikan jika ADF test-mu bilang 1
# p_range, q_range, P_range, Q_range = range(0, 3), range(0, 3), range(0, 2), range(0, 2)
# kombinasi = list(itertools.product(p_range, q_range, P_range, Q_range))

# print(f"🔍 Memulai Grid Search dengan {len(kombinasi)} kombinasi. Ini mungkin memakan waktu beberapa menit...")
# hasil_grid = []

# for p, q, P, Q in kombinasi:
#     if p == 0 and q == 0 and P == 0 and Q == 0: continue
#     try:
#         model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s),
#                         enforce_stationarity=False, enforce_invertibility=False, trend='c')
#         fit = model.fit(disp=False)
#         hasil_grid.append({'order': (p,d,q), 'seasonal_order': (P,D,Q,s), 'AIC': fit.aic})
#     except:
#         continue

# df_grid = pd.DataFrame(hasil_grid).sort_values('AIC').reset_index(drop=True)
# best = df_grid.iloc[0]
# print(f"🏆 Model Terbaik Ditemukan: SARIMA{best['order']}{best['seasonal_order']} dengan AIC {best['AIC']:.2f}")

# # 4. Latih Model Final dengan Seluruh Data
# model_final = SARIMAX(ihk, order=best['order'], seasonal_order=best['seasonal_order'],
#                       enforce_stationarity=False, enforce_invertibility=False, trend='c')
# fit_final = model_final.fit(disp=False)

# # 5. Simpan Model
# os.makedirs('models', exist_ok=True)
# with open('models/sarima_ihk_medan.pkl', 'wb') as f:
#     pickle.dump(fit_final, f)

# print("✅ Pelatihan selesai! Model berhasil disimpan di 'models/sarima_ihk_medan.pkl'")



"""
train_model.py — Skrip pelatihan model SARIMA untuk IHK Kota Medan.

Diperbarui agar konsisten dengan metodologi hasil perbaikan di
untitled20.py:

1. Data 2014-2019 TIDAK dibuang lagi. Perubahan tahun dasar BPS (2020, 2024)
   diselaraskan lewat BACKWARD SPLICING (rescale proporsional pakai rasio
   Januari/Desember di titik patahan), jadi seluruh histori 2014-sekarang
   tetap dipakai, bukan cuma data >= 2020.
2. Grid search sekarang ikut mencari d dan D (bukan di-fix d=1, D=0),
   rentang p,d,q,P,D,Q semua 0-1, s=12.
3. Model TIDAK dipilih hanya dari AIC terendah. Kandidat top-20 AIC diuji
   Ljung-Box (lag 6,12,18,24); model pertama yang lolos (p-value > 0.05 di
   SEMUA lag) yang dikunci sebagai model final. Ini menghindari memilih
   model yang residualnya masih ada autokorelasi walau AIC-nya kecil.
4. Model final dilatih ulang di SELURUH data historis dengan (p,d,q)(P,D,Q,s)
   YANG SAMA dengan hasil pencarian Ljung-Box di atas.
   Catatan: draft Colab lama (untitled20.py Cell 11) sempat hardcode ke
   SARIMA(1,0,1)(1,0,1)[12] di tahap ini padahal model yang lolos uji
   white-noise adalah SARIMA(1,1,1)(0,1,1)[12] (sesuai catatan skripsi).
   Skrip ini memperbaikinya dengan memakai parameter dinamis dari hasil
   pencarian, BUKAN angka hardcoded, supaya model diagnostik dan model
   yang dipakai forecasting selalu sama.
5. Output disimpan sebagai DICTIONARY (bukan cuma objek model mentah) agar
   dashboard (app.py) bisa menampilkan tabel grid search, evaluasi
   test-set, dan tabel proyeksi tanpa perlu hitung ulang tiap kali dibuka.
   Dictionary ini tetap menyimpan objek model SARIMAXResultsWrapper utuh
   di key 'model' supaya fitur seperti .apply() ke data baru, .summary(),
   dan plot_diagnostics() di app.py tetap berfungsi.

Cara pakai:
    python train_model.py
Membutuhkan file mentah di ../data/raw/DATA_IHK_KOTA_MEDAN.csv
(kolom: Tanggal;IHK, delimiter titik koma, format tanggal dd/mm/YYYY).
"""

import os
import pickle
import warnings
import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.stats.diagnostic as diag
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

RAW_CSV_PATH = "../data/raw/DATA_IHK_KOTA_MEDAN.csv"
OUTPUT_PKL_PATH = "models/sarima_ihk_medan.pkl"
FORECAST_HORIZON = 12
SEASONAL_PERIOD = 12
TRAIN_END = "2024-12"
TEST_START = "2025-01"


def load_and_splice(path: str) -> pd.DataFrame:
    """Baca CSV mentah dan lakukan backward splicing di titik patahan tahun dasar."""
    df = pd.read_csv(path, sep=";")
    df.columns = ["Tanggal", "IHK"]
    if df["IHK"].dtype == "object":
        df["IHK"] = df["IHK"].astype(str).str.replace(",", ".").astype(float)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], format="%d/%m/%Y")
    df = df.sort_values("Tanggal").reset_index(drop=True)

    idx_2020 = df[df["Tanggal"] == "2020-01-01"].index[0]
    idx_2024 = df[df["Tanggal"] == "2024-01-01"].index[0]

    rasio_2020 = df.loc[idx_2020, "IHK"] / df.loc[idx_2020 - 1, "IHK"]
    rasio_2024 = df.loc[idx_2024, "IHK"] / df.loc[idx_2024 - 1, "IHK"]

    df["IHK_Spliced"] = df["IHK"].copy()
    # Selaraskan skala 2020-2023 ke basis 2024
    df.loc[idx_2020:idx_2024 - 1, "IHK_Spliced"] = df.loc[idx_2020:idx_2024 - 1, "IHK"] * rasio_2024
    # Selaraskan skala 2014-2019 ke basis 2024 (lewat basis 2020 dulu)
    df.loc[0:idx_2020 - 1, "IHK_Spliced"] = df.loc[0:idx_2020 - 1, "IHK"] * rasio_2020 * rasio_2024

    df.set_index("Tanggal", inplace=True)
    df.index = pd.DatetimeIndex(df.index).to_period("M").to_timestamp("M")
    return df


def grid_search(train: pd.Series, s: int = SEASONAL_PERIOD) -> pd.DataFrame:
    """Full grid search p,d,q,P,D,Q in {0,1}, urutkan berdasarkan AIC."""
    rng = range(0, 2)
    pdq = list(itertools.product(rng, rng, rng))
    seasonal_pdq = [(x[0], x[1], x[2], s) for x in itertools.product(rng, rng, rng)]

    total = len(pdq) * len(seasonal_pdq)
    print(f"🔍 Grid search: mencoba {total} kombinasi SARIMA...")

    hasil = []
    for order in pdq:
        for seasonal_order in seasonal_pdq:
            if order == (0, 0, 0) and seasonal_order[:3] == (0, 0, 0):
                continue
            try:
                fit = sm.tsa.statespace.SARIMAX(
                    train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
                hasil.append({
                    "Model": f"SARIMA {order} {seasonal_order}",
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "AIC": fit.aic,
                    "BIC": fit.bic,
                })
            except Exception:
                continue

    df_grid = pd.DataFrame(hasil).sort_values("AIC").reset_index(drop=True)
    return df_grid


def pilih_model_lulus_ljungbox(train: pd.Series, df_grid: pd.DataFrame,
                                top_n: int = 20, lags=(6, 12, 18, 24)):
    """Uji top-N kandidat AIC terendah dengan Ljung-Box, kunci yang pertama lolos."""
    for i, row in df_grid.head(top_n).iterrows():
        try:
            fit = sm.tsa.statespace.SARIMAX(
                train, order=row["order"], seasonal_order=row["seasonal_order"],
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            lb = diag.acorr_ljungbox(fit.resid, lags=list(lags), return_df=True)
            if all(lb["lb_pvalue"] > 0.05):
                print(f"✅ Lolos uji Ljung-Box: {row['Model']} (peringkat AIC #{i + 1})")
                return row["order"], row["seasonal_order"], row["Model"], fit
            print(f"❌ Gagal uji Ljung-Box: {row['Model']}")
        except Exception:
            continue
    raise RuntimeError(
        "Tidak ada model dari top-N kandidat yang lolos uji Ljung-Box. "
        "Perlebar pencarian dengan menaikkan nilai top_n."
    )


def main():
    print("⏳ Memulai proses pelatihan model SARIMA (versi diperbaiki)...")

    df = load_and_splice(RAW_CSV_PATH)
    ihk = df["IHK_Spliced"]

    train = ihk[:TRAIN_END]
    test = ihk[TEST_START:]
    print(f"Data Latih (Train): {train.index[0]:%b %Y} - {train.index[-1]:%b %Y} ({len(train)} obs)")
    print(f"Data Uji (Test)   : {test.index[0]:%b %Y} - {test.index[-1]:%b %Y} ({len(test)} obs)")

    df_grid = grid_search(train)
    order, seasonal_order, nama_model, fit_train = pilih_model_lulus_ljungbox(train, df_grid)

    # --- Evaluasi di data test ---
    prediksi = fit_train.get_forecast(steps=len(test))
    prediksi_mean = prediksi.predicted_mean
    prediksi_ci = prediksi.conf_int()

    mae = mean_absolute_error(test, prediksi_mean)
    rmse = np.sqrt(mean_squared_error(test, prediksi_mean))
    mape = float(np.mean(np.abs((test - prediksi_mean) / test)) * 100)
    print(f"📊 Evaluasi {nama_model} di data test: MAE={mae:.4f}  RMSE={rmse:.4f}  MAPE={mape:.2f}%")

    # --- Retrain model final di SELURUH data historis, parameter SAMA (dinamis, bukan hardcoded) ---
    print(f"Melatih ulang '{nama_model}' menggunakan SELURUH data historis...")
    fit_final = sm.tsa.statespace.SARIMAX(
        ihk, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)

    ramalan = fit_final.get_forecast(steps=FORECAST_HORIZON)
    ramalan_mean = ramalan.predicted_mean
    ramalan_ci = ramalan.conf_int()

    tabel_ramalan = pd.DataFrame({
        "Bulan": ramalan_mean.index.strftime("%Y-%m"),
        "Prediksi_IHK": ramalan_mean.values.round(2),
        "Batas_Bawah": ramalan_ci.iloc[:, 0].values.round(2),
        "Batas_Atas": ramalan_ci.iloc[:, 1].values.round(2),
    })

    paket_streamlit = {
        # Objek model utuh -> dipakai app.py untuk .apply(), .summary(), plot_diagnostics()
        "model": fit_final,
        "nama_model": nama_model,
        "order": order,
        "seasonal_order": seasonal_order,

        # Data historis & split
        "data_historis": df[["IHK_Spliced"]],
        "data_train": train,
        "data_test": test,

        # Hasil grid search & evaluasi test-set
        "tabel_grid_search": df_grid,
        "mae_test": mae,
        "rmse_test": rmse,
        "mape_test": mape,
        "prediksi_test_mean": prediksi_mean,
        "prediksi_test_ci": prediksi_ci,

        # Ramalan masa depan (out-of-sample)
        "ramalan_masa_depan_mean": ramalan_mean,
        "ramalan_masa_depan_ci": ramalan_ci,
        "tabel_ramalan": tabel_ramalan,
    }

    os.makedirs(os.path.dirname(OUTPUT_PKL_PATH), exist_ok=True)
    with open(OUTPUT_PKL_PATH, "wb") as f:
        pickle.dump(paket_streamlit, f)

    print(f"✅ Selesai! Model '{nama_model}' berhasil disimpan di '{OUTPUT_PKL_PATH}'")


if __name__ == "__main__":
    main()