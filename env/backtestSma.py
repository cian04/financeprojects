import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Télécharger les données
ticker = 'AAPL'
df = yf.download(ticker, start='2020-01-01', end='2023-12-31')[['Close']]

# 2. Moyennes mobiles
df['SMA20'] = df['Close'].rolling(window=20).mean()
df['SMA50'] = df['Close'].rolling(window=50).mean()

# 3. Signaux de trading
df['Signal'] = 0
df.iloc[20:, df.columns.get_loc('Signal')] = np.where(
    df['SMA20'].iloc[20:] > df['SMA50'].iloc[20:], 1, 0)
df['Position'] = df['Signal'].diff()

# 4. Performance et frais
fee_rate = 0.1 # 0.1% de frais
df['Return'] = df['Close'].pct_change()
df['Strategy_raw'] = df['Return'] * df['Signal'].shift(1)
df['Fee'] = df['Position'].abs() * fee_rate
df['Strategy_net'] = df['Strategy_raw'] - df['Fee']

# === 5. Simulation de portefeuille ===
initial_capital = 100

df['Cumulative Market'] = (1 + df['Return']).cumprod()
df['Cumulative Strategy'] = (1 + df['Strategy_net']).cumprod()

df['Capital_Market'] = df['Cumulative Market'] * initial_capital
df['Capital_Strategy'] = df['Cumulative Strategy'] * initial_capital

# === 6. Résultats finaux ===
final_market = df['Capital_Market'].iloc[-1]
final_strategy = df['Capital_Strategy'].iloc[-1]

print(f"\n💰 Simulation avec 100€ investis entre 2020 et 2023 sur {ticker} :")
print(f"   - Marché (Buy & Hold) sans stratégie : {final_market:.2f}€")
print(f"   - Stratégie SMA20/50 avec frais     : {final_strategy:.2f}€")

# === 7. Points de trade ===
trades = df[df['Position'].notna() & df['Position'] != 0]
print("\n📈 Points de trades détectés :")
print(trades[['Close', 'SMA20', 'SMA50', 'Position']].dropna())

# === 8. Visualisation de la stratégie ===
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Capital_Market'], label='💼 Marché (Buy & Hold)')
plt.plot(df.index, df['Capital_Strategy'], label='⚙️ Stratégie SMA20/50 (avec frais)')
plt.title(f"📊 Évolution du capital (100€ investis) - {ticker}")
plt.xlabel("Date")
plt.ylabel("Capital (€)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()