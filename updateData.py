import yfinance as yf
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import requests  # Pour l'appel API (si vous utilisez une IA externe)

# =============================================
# CONFIGURATION
# =============================================
PEA_ELIGIBLE_FILE = "pe_eligible_tickers.txt"  # Liste des tickers éligibles au PEA
OUTPUT_FILE = "selected_stocks.json"          # Fichier de sortie
LOG_FILE = "screener_log.txt"                 # Log des exécutions
MAX_ACTIONS = 20                              # Nombre d'actions à sélectionner
MIN_CAP = 1_000_000_000                       # Capitalisation minimale (1Md€)
MOMENTUM_MIN = 0.05                           # Momentum 6M minimum (+5%)
RSI_MIN, RSI_MAX = 40, 70                     # Plage RSI acceptable
VOLUME_RELATIVE_MIN = 1.1                     # Volume > 1.1x moyenne 20j
PER_MAX = 30                                  # Exclure les PER > 30 (sauf exceptions)
DEBT_TO_EBITDA_MAX = 3                        # Dette nette < 3x EBITDA

# =============================================
# 1. FILTRAGE INITIAL (GRATUIT - yfinance)
# =============================================
def load_pe_eligible_tickers():
    """Charge la liste des tickers éligibles au PEA (Trade Republic = Europe)"""
    with open(PEA_ELIGIBLE_FILE, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers

def get_sector(ticker):
    """Récupère le secteur d'une action via yfinance"""
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get('sector', 'Unknown')
    except:
        return 'Unknown'

def get_country(ticker):
    """Récupère le pays d'une action via yfinance"""
    try:
        stock = yf.Ticker(ticker)
        country = stock.info.get('country', 'Unknown')
        # Normaliser les pays (ex: "France" -> "FR")
        country_map = {
            "France": "FR", "Germany": "DE", "Netherlands": "NL",
            "Italy": "IT", "Belgium": "BE", "Spain": "ES",
            "Sweden": "SE", "Norway": "NO", "Austria": "AT",
            "Switzerland": "CH", "United Kingdom": "UK"
        }
        return country_map.get(country, country[:2])
    except:
        return 'XX'

def calculate_rsi(series, window=14):
    """Calcule le RSI 14j"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_momentum(data, period="6mo"):
    """Calcule le momentum (perf sur 6M)"""
    if len(data) < 30:
        return 0
    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]
    return (end_price - start_price) / start_price

def filter_initial_tickers(tickers):
    """Filtre les tickers selon vos critères initiaux"""
    filtered = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Vérifier capitalisation
            if info.get('marketCap', 0) < MIN_CAP:
                continue

            # Récupérer 6M de données
            data = yf.download(ticker, period="6mo", interval="1d")
            if len(data) < 30:
                continue

            # Calculer métriques
            momentum = calculate_momentum(data)
            if momentum < MOMENTUM_MIN:
                continue

            rsi = calculate_rsi(data['Close'])
            if rsi.iloc[-1] < RSI_MIN or rsi.iloc[-1] > RSI_MAX:
                continue

            volume_rel = data['Volume'].iloc[-1] / data['Volume'].rolling(20).mean().iloc[-1]
            if volume_rel < VOLUME_RELATIVE_MIN:
                continue

            # Exclure les ETFs
            if info.get('quoteType') == 'ETF':
                continue

            # Stocker les données
            filtered.append({
                "ticker": ticker,
                "name": info.get('shortName', 'Unknown'),
                "sector": get_sector(ticker),
                "country": get_country(ticker),
                "marketCap": info.get('marketCap', 0),
                "momentum_6m": round(momentum * 100, 2),
                "rsi_14j": round(rsi.iloc[-1], 2),
                "volume_relative": round(volume_rel, 2),
                "close": round(data['Close'].iloc[-1], 2),
                "pe_ratio": info.get('trailingPE', np.nan),
                "debt_to_ebitda": info.get('debtToEbitda', np.nan)
            })
        except Exception as e:
            print(f"⚠️ Erreur pour {ticker}: {str(e)}")
            continue

    # Convertir en DataFrame
    df = pd.DataFrame(filtered)

    # Exclure les actions avec PER > 30 ou dette > 3x EBITDA
    df = df[
        (df['pe_ratio'].isna()) | (df['pe_ratio'] <= PER_MAX) |
        (df['debt_to_ebitda'].isna()) | (df['debt_to_ebitda'] <= DEBT_TO_EBITDA_MAX)
    ]

    # Diversification sectorielle : garder 1 action par secteur
    df['sector_rank'] = df.groupby('sector')['momentum_6m'].rank(ascending=False)
    df = df[df['sector_rank'] <= 1].sort_values('momentum_6m', ascending=False)

    # Diversification géographique : limiter à 2 actions par pays
    df['country_rank'] = df.groupby('country')['momentum_6m'].rank(ascending=False)
    df = df[df['country_rank'] <= 2].sort_values('momentum_6m', ascending=False)

    return df.head(60)  # On garde ~60 actions pour la sélection IA

# =============================================
# 2. SÉLECTION IA (1 PROMPT - 60 → 20 actions)
# =============================================
def generate_ai_prompt(df):
    """Génère le prompt pour l'IA (à envoyer via API ou copier-coller)"""
    prompt = f"""
Voici 60 actions européennes éligibles au PEA, filtrées par capitalisation, momentum et volume.
Pour CHAQUE action, analyse :
1. **Actualité récente** (sentiment positif/négatif/neutre) via Yahoo Finance/Investing.com
2. **Catalyseurs à venir** (résultats trimestriels, annonces stratégiques, réglementations)
3. **Ratio risque/rendement estimé** (basé sur volatilité et potentiel à 6-12 mois)
4. **Diversification sectorielle et géographique** (éviter les doublons de secteur/pays)

Sélectionne les 20 actions les plus pertinentes pour un horizon 6-12 mois, en respectant :
- **Diversification maximale** : 1 action par secteur, géographie variée (France, Allemagne, Italie, etc.)
- **Exclure les actions avec PER > 30** (sauf exceptions justifiées)
- **Exclure les actions avec dette nette > 3x EBITDA**
- **Priorité aux actions avec un bon momentum récent** (RSI 14j > 50, perf 1M > +3%)

Données disponibles :
{df[['ticker', 'name', 'sector', 'country', 'momentum_6m', 'rsi_14j', 'close']].to_string(index=False)}

Format de réponse attendu :
TICKERS: [TICKER1, TICKER2, ..., TICKER20]
RAISONS: [2-3 phrases par action expliquant les critères dominants]
SCORE_RAISONNÉ: [Note de 0 à 100 pour chaque action, basée sur momentum + actualité + diversification]
"""
    return prompt

def parse_ai_response(response, df):
    """Parse la réponse de l'IA et retourne les 20 tickers sélectionnés"""
    try:
        # Exemple de réponse attendue (à adapter selon votre IA)
        tickers_selected = response.split("TICKERS:")[1].split("\n")[0].strip().strip("[]").replace(" ", "").split(",")
        return [ticker.strip() for ticker in tickers_selected if ticker.strip() in df['ticker'].values]
    except:
        print("⚠️ Erreur de parsing de la réponse IA. Utilisation des 20 meilleures actions par momentum.")
        return df.sort_values('momentum_6m', ascending=False)['ticker'].head(20).tolist()

# =============================================
# 3. CALCUL DU STOP-LOSS DYNAMIQUE (6 MOIS)
# =============================================
def calculate_dynamic_stop_loss(ticker):
    """Calcule le stop-loss dynamique basé sur l'amplitude et les supports"""
    try:
        data = yf.download(ticker, period="6mo", interval="1d")
        if len(data) < 30:
            return None

        # Calculer les moyennes
        high_mean = data['High'].mean()
        low_mean = data['Low'].mean()
        close_mean = data['Close'].mean()
        close_current = data['Close'].iloc[-1]
        ma20 = data['Close'].rolling(20).mean().iloc[-1]

        # Amplitude moyenne
        amplitude = (high_mean - low_mean) / close_mean

        # Supports
        support_bas = low_mean - (0.5 * amplitude * close_current)
        support_mm20 = ma20 - (0.3 * amplitude * close_current)
        support_pullback = data['Low'].iloc[-20:].min() - (0.2 * amplitude * close_current)

        # Stop-loss final avec marge de sécurité
        stop_loss = min(support_bas, support_mm20, support_pullback) - (0.1 * amplitude * close_current)

        return {
            "ticker": ticker,
            "stop_loss": round(stop_loss, 2),
            "amplitude": round(amplitude * 100, 2),
            "supports": {
                "bas": round(support_bas, 2),
                "mm20": round(support_mm20, 2),
                "pullback": round(support_pullback, 2)
            },
            "close_current": round(close_current, 2),
            "ma20": round(ma20, 2)
        }
    except Exception as e:
        print(f"⚠️ Erreur pour {ticker} (stop-loss): {str(e)}")
        return None

# =============================================
# 4. GÉNÉRATION DU TABLEAU FINAL
# =============================================
def generate_final_table(selected_tickers, df_initial):
    """Génère le tableau final avec toutes les métriques"""
    final_data = []

    for ticker in selected_tickers:
        # Récupérer les données initiales
        stock_data = df_initial[df_initial['ticker'] == ticker].iloc[0]

        # Calculer le stop-loss
        stop_data = calculate_dynamic_stop_loss(ticker)
        if not stop_data:
            continue

        # Calculer le prix d'entrée (moyenne des 5 derniers jours)
        data = yf.download(ticker, period="1mo", interval="1d")
        entry_price = round(data['Close'].iloc[-5:].mean(), 2)

        final_data.append({
            "ticker": ticker,
            "name": stock_data['name'],
            "sector": stock_data['sector'],
            "country": stock_data['country'],
            "momentum_6m": stock_data['momentum_6m'],
            "rsi_14j": stock_data['rsi_14j'],
            "close_current": stop_data['close_current'],
            "entry_price": entry_price,
            "stop_loss": stop_data['stop_loss'],
            "amplitude_6m": stop_data['amplitude'],
            "support_bas": stop_data['supports']['bas'],
            "support_mm20": stop_data['supports']['mm20'],
            "support_pullback": stop_data['supports']['pullback'],
            "ma20": stop_data['ma20'],
            "volume_relative": stock_data['volume_relative'],
            "pe_ratio": stock_data['pe_ratio'],
            "debt_to_ebitda": stock_data['debt_to_ebitda'],
            "selection_date": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

    return pd.DataFrame(final_data)

# =============================================
# 5. FONCTION PRINCIPALE
# =============================================
def main():
    print("🚀 Démarrage du screener PEA...")
    start_time = datetime.now()

    # 1. Charger les tickers éligibles
    print("📂 Chargement des tickers éligibles au PEA...")
    tickers = load_pe_eligible_tickers()
    print(f"✅ {len(tickers)} tickers chargés.")

    # 2. Filtrer les tickers initiaux (60 actions)
    print("🔍 Filtrage initial des actions...")
    df_initial = filter_initial_tickers(tickers)
    print(f"✅ {len(df_initial)} actions filtrées (1 champion par secteur + diversification géographique).")

    # 3. Générer le prompt pour l'IA
    print("🤖 Génération du prompt pour l'IA...")
    ai_prompt = generate_ai_prompt(df_initial)
    print("✅ Prompt généré. Copiez-le dans votre outil IA préféré (ex: ChatGPT, Mistral, etc.).")

    # 4. Récupérer la réponse de l'IA (simulation)
    print("\n📌 **COPIEZ-COLLEZ CE PROMPT DANS VOTRE IA** :")
    print("=" * 80)
    print(ai_prompt)
    print("=" * 80)
    print("\n🔹 Après avoir reçu la réponse de l'IA, collez-la ici (ou modifiez le code pour une API automatique) :")

    # Simulation : on utilise les 20 meilleures par momentum si pas de réponse IA
    ai_response = input("Entrez la réponse de l'IA (ou appuyez sur Entrée pour utiliser les 20 meilleures par momentum) : ").strip()
    selected_tickers = parse_ai_response(ai_response, df_initial) if ai_response else df_initial.sort_values('momentum_6m', ascending=False)['ticker'].head(20).tolist()

    print(f"\n✅ {len(selected_tickers)} actions sélectionnées par l'IA.")

    # 5. Générer le tableau final
    print("📊 Génération du tableau final...")
    final_df = generate_final_table(selected_tickers, df_initial)

    # 6. Sauvegarder les résultats
    final_df.to_json(OUTPUT_FILE, orient="records", indent=2)
    print(f"\n💾 Résultats sauvegardés dans {OUTPUT_FILE}")

    # 7. Afficher le tableau final
    print("\n📋 **TABLEAU FINAL DES 20 ACTIONS SÉLECTIONNÉES**")
    print("=" * 120)
    print(final_df[['ticker', 'name', 'sector', 'country', 'momentum_6m', 'rsi_14j', 'entry_price', 'stop_loss', 'amplitude_6m']].to_string(index=False))
    print("=" * 120)

    # 8. Log l'exécution
    with open(LOG_FILE, "a") as f:
        f.write(f"\n{datetime.now()}: {len(selected_tickers)} actions sélectionnées. Fichier: {OUTPUT_FILE}\n")

    print(f"\n⏱️ Exécution terminée en {(datetime.now() - start_time).total_seconds():.2f} secondes.")
    print("🔹 Prochaine étape : Programmez les achats/ventes dans Trade Republic avec les seuils indiqués.")

# =============================================
# EXÉCUTION
# =============================================
if __name__ == "__main__":
    main()
