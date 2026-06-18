import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import pickle
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import itertools

warnings.filterwarnings('ignore')

print("⏳ Memulai proses pelacakan data dan pelatihan model SARIMA...")

# 1. Baca Data dari folder yang benar
df_raw = pd.read_csv('../data/raw/DATA_IHK_KOTA_MEDAN.csv', sep=';')
df_raw.columns = ['Tanggal', 'IHK']
df_raw['Tanggal'] = pd.to_datetime(df_raw['Tanggal'], format='%d/%m/%Y')
df_raw['IHK'] = pd.to_numeric(df_raw['IHK'], errors='coerce')
df_raw.set_index('Tanggal', inplace=True)
df_raw.index = pd.DatetimeIndex(df_raw.index).to_period('M').to_timestamp('M')

# Filter data aktual mulai 2020
df_aktual = df_raw.dropna().copy()
df = df_aktual['2020':].copy()
ihk = df['IHK']

# 2. Split Data (Train & Test)
train = ihk[:'2024-12']
test  = ihk['2025-01':]

# 3. Grid Search Sederhana (sesuai kodemu)
d, D, s = 1, 0, 12 # Asumsi D=0 berdasarkan kodemu, atau sesuaikan jika ADF test-mu bilang 1
p_range, q_range, P_range, Q_range = range(0, 3), range(0, 3), range(0, 2), range(0, 2)
kombinasi = list(itertools.product(p_range, q_range, P_range, Q_range))

print(f"🔍 Memulai Grid Search dengan {len(kombinasi)} kombinasi. Ini mungkin memakan waktu beberapa menit...")
hasil_grid = []

for p, q, P, Q in kombinasi:
    if p == 0 and q == 0 and P == 0 and Q == 0: continue
    try:
        model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s),
                        enforce_stationarity=False, enforce_invertibility=False, trend='c')
        fit = model.fit(disp=False)
        hasil_grid.append({'order': (p,d,q), 'seasonal_order': (P,D,Q,s), 'AIC': fit.aic})
    except:
        continue

df_grid = pd.DataFrame(hasil_grid).sort_values('AIC').reset_index(drop=True)
best = df_grid.iloc[0]
print(f"🏆 Model Terbaik Ditemukan: SARIMA{best['order']}{best['seasonal_order']} dengan AIC {best['AIC']:.2f}")

# 4. Latih Model Final dengan Seluruh Data
model_final = SARIMAX(ihk, order=best['order'], seasonal_order=best['seasonal_order'],
                      enforce_stationarity=False, enforce_invertibility=False, trend='c')
fit_final = model_final.fit(disp=False)

# 5. Simpan Model
os.makedirs('models', exist_ok=True)
with open('models/sarima_ihk_medan.pkl', 'wb') as f:
    pickle.dump(fit_final, f)

print("✅ Pelatihan selesai! Model berhasil disimpan di 'models/sarima_ihk_medan.pkl'")