import yfinance as yf
import pandas as pd

def main():
    tickers = ["AAPL", "MSFT", "TSLA", "NVDA"]

    # On force auto_adjust=True pour obtenir directement des prix ajustés (Close ajusté)
    data = yf.download(
        tickers,
        start="2020-01-01",
        end="2025-01-01",
        auto_adjust=True,     # <- important avec les nouvelles versions
        actions=False,
        progress=True
    )

    # Cas 1 : MultiIndex (OHLCV x tickers) -> on prend Adj Close ou Close si présent
    if isinstance(data.columns, pd.MultiIndex):
        cols0 = data.columns.get_level_values(0)
        if "Adj Close" in cols0:
            data = data["Adj Close"]
        elif "Close" in cols0:
            data = data["Close"]
        else:
            raise ValueError(f"Colonnes inattendues (niveau 0) : {sorted(set(cols0))}")

    # Cas 2 : Index simple -> on a déjà un tableau (dates x tickers) de prix ajustés
    # Rien à faire.

    # Nettoyage / tri
    data = data.sort_index().ffill().dropna(how="all")
    # Optionnel : retirer les lignes avec NaN restants si tu veux un carré parfait
    # data = data.dropna()

    # Sauvegarde
    data.to_csv("data/prices_yf.csv")
    print("✅ Données sauvegardées dans data/prices_yf.csv")
    print("Aperçu colonnes :", list(data.columns))
    print(data.tail())

if __name__ == "__main__":
    main()