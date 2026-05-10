import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# =============================================
# CONFIGURATION
# =============================================
PEA_ELIGIBLE_FILE = "pe_eligible_tickers.txt"
OUTPUT_FILE = "selected_stocks.json"
LOG_FILE = "screener_log.txt"
MAX_ACTIONS = 20
MIN_CAP = 1_000_000_000  # 1 milliard d'euros
MOMENTUM_MIN = 0.05  # 5% de momentum sur 6 mois
RSI_MIN, RSI_MAX = 40, 70
VOLUME_RELATIVE_MIN = 1.1  # Volume 10% supérieur à la moyenne
PER_MAX = 30
DEBT_TO_EBITDA_MAX = 3

# =============================================
# FONCTIONS DE FILTRAGE
# =============================================
def load_pe_eligible_tickers():
    """Charge la liste des tickers éligibles au PEA."""
    try:
        with open(PEA_ELIGIBLE_FILE, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print("❌ Fichier pe_eligible_tickers.txt introuvable!")
        return []

def get_financial_data(ticker):
    """Récupère les données financières pour un ticker."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # Valeurs par défaut si les données sont manquantes
        pe_ratio = info.get('trailingPE', None)
        debt_to_ebitda = info.get('debtToEbitda', None)
        market_cap = info.get('marketCap', 0)
        sector = info.get('sector', 'Unknown')
        country = info.get('country', 'Unknown')
        name = info.get('longName', ticker)

        return {
            'ticker': ticker,
            'name': name,
            'sector': sector,
            'country': country,
            'market_cap': market_cap,
            'pe_ratio': pe_ratio,
            'debt_to_ebitda': debt_to_ebitda
        }
    except Exception as e:
        print(f"⚠️ Impossible de récupérer les données pour {ticker} : {e}")
        return None

def filter_initial_tickers(tickers):
    """Filtre les tickers selon les critères initiaux."""
    financial_data = []
    for ticker in tickers:
        data = get_financial_data(ticker)
        if data:
            financial_data.append(data)

    if not financial_data:
        raise ValueError("Aucune donnée financière disponible!")

    df = pd.DataFrame(financial_data)

    # Filtrer les tickers avec market_cap > MIN_CAP
    df = df[df['market_cap'] >= MIN_CAP]

    # Filtrer les tickers avec pe_ratio <= PER_MAX OU pe_ratio manquant
    pe_mask = df['pe_ratio'].isna() | (df['pe_ratio'] <= PER_MAX)
    df = df[pe_mask]

    # Filtrer les tickers avec debt_to_ebitda <= DEBT_TO_EBITDA_MAX OU manquant
    debt_mask = df['debt_to_ebitda'].isna() | (df['debt_to_ebitda'] <= DEBT_TO_EBITDA_MAX)
    df = df[debt_mask]

    return df

def calculate_momentum(ticker):
    """Calcule le momentum sur 6 mois."""
    try:
        data = yf.download(ticker, period="6mo", interval="1d")
        if len(data) < 30 or 'Close' not in data.columns:
            return None

        initial_price = data['Close'].iloc[0]
        final_price = data['Close'].iloc[-1]
        momentum = (final_price - initial_price) / initial_price
        return momentum
    except Exception as e:
        print(f"⚠️ Erreur pour {ticker} dans calculate_momentum : {e}")
        return None

def calculate_rsi(ticker, window=14):
    """Calcule le RSI sur 14 jours."""
    try:
        data = yf.download(ticker, period="3mo", interval="1d")
        if len(data) < window or 'Close' not in data.columns:
            return None

        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    except Exception as e:
        print(f"⚠️ Erreur pour {ticker} dans calculate_rsi : {e}")
        return None

def calculate_volume_relative(ticker):
    """Calcule le volume relatif (volume actuel / moyenne sur 30 jours)."""
    try:
        data = yf.download(ticker, period="1mo", interval="1d")
        if len(data) < 30 or 'Volume' not in data.columns:
            return None

        avg_volume = data['Volume'].mean()
        last_volume = data['Volume'].iloc[-1]
        return last_volume / avg_volume
    except Exception as e:
        print(f"⚠️ Erreur pour {ticker} dans calculate_volume_relative : {e}")
        return None

def calculate_dynamic_stop_loss(ticker):
    """Calcule un stop-loss dynamique basé sur l'amplitude des 6 derniers mois."""
    try:
        data = yf.download(ticker, period="6mo", interval="1d")
        if len(data) < 30 or 'Close' not in data.columns:
            return None

        high = data['High'].max()
        low = data['Low'].min()
        amplitude = (high - low) / low
        return amplitude * 0.5  # Stop à 50% de l'amplitude
    except Exception as e:
        print(f"⚠️ Erreur pour {ticker} dans calculate_dynamic_stop_loss : {e}")
        return None

def select_top_stocks(df, tickers):
    """Sélectionne les top 20 actions selon les critères."""
    results = []

    for ticker in tickers:
        momentum = calculate_momentum(ticker)
        rsi = calculate_rsi(ticker)
        volume_rel = calculate_volume_relative(ticker)
        stop_loss = calculate_dynamic_stop_loss(ticker)

        if (momentum is None or rsi is None or volume_rel is None or stop_loss is None):
            continue

        if (momentum >= MOMENTUM_MIN and
            RSI_MIN <= rsi <= RSI_MAX and
            volume_rel >= VOLUME_RELATIVE_MIN):

            stock_info = yf.Ticker(ticker).info
            current_price = stock_info.get('currentPrice', None)

            results.append({
                'ticker': ticker,
                'name': stock_info.get('longName', ticker),
                'sector': stock_info.get('sector', 'Unknown'),
                'country': stock_info.get('country', 'Unknown'),
                'price': current_price,
                'momentum_6m': round(momentum, 4),
                'rsi_14j': round(rsi, 2),
                'stop_loss': round(current_price * (1 - stop_loss), 2),
                'amplitude_6m': round(stop_loss * 2, 4),
                'selection_date': datetime.now().strftime("%Y-%m-%d %H:%M")
            })

    # Trier par momentum décroissant et prendre les top 20
    results.sort(key=lambda x: x['momentum_6m'], reverse=True)
    return results[:MAX_ACTIONS]

# =============================================
# FONCTION PRINCIPALE
# =============================================
def main():
    print("🚀 Démarrage du screener PEA...")

    # 1. Charger les tickers éligibles
    tickers = load_pe_eligible_tickers()
    if not tickers:
        print("❌ Aucun ticker éligible trouvé!")
        return

    print(f"✅ {len(tickers)} tickers chargés.")

    # 2. Filtrer les tickers initiaux
    try:
        df_initial = filter_initial_tickers(tickers)
        print(f"✅ {len(df_initial)} tickers après filtre initial.")
    except Exception as e:
        print(f"❌ Erreur lors du filtrage initial : {e}")
        return

    # 3. Sélectionner les top 20 actions
    selected_stocks = select_top_stocks(df_initial, tickers)
    print(f"✅ {len(selected_stocks)} actions sélectionnées.")

    # 4. Sauvegarder les résultats
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(selected_stocks, f, indent=2, ensure_ascii=False)

    print(f"✅ Résultats sauvegardés dans {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
