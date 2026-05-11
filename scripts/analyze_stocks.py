#!/usr/bin/env python3
"""
PEA Tracker - Analyse quotidienne optimisée
- Présélection 100% quantitative (yfinance, aucun coût)
- 1 seul appel MammouthIA pour les 20 finalistes
"""

import json, os, time, warnings
from datetime import datetime, date
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from openai import OpenAI

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG MAMMOUTHIA
# On utilise le meilleur modèle finance disponible via
# l'API compatible OpenAI de MammouthIA
# ============================================================
MAMMOUTH_API_KEY = os.environ.get("MAMMOUTH_API_KEY", "")

client = OpenAI(
    api_key=MAMMOUTH_API_KEY,
    base_url="https://api.mammouth.ai/v1",  # endpoint MammouthIA
)

# Modèle le plus pertinent pour l'analyse financière
# MammouthIA donne accès à plusieurs modèles - on prend le meilleur dispo
AI_MODEL = "gpt-4o"   # fallback: "claude-3-5-sonnet", "mistral-large"

# Frais Trade Republic
TR_FEE = 1.0          # € par ordre
TR_FEE_TOTAL = 2.0    # € aller + retour
MIN_GAIN_PCT = 2.5    # gain net minimum pertinent après frais

# ============================================================
# UNIVERS PEA - Actions européennes éligibles PEA français
# Source : grandes capitalisations européennes, disponibles
# sur Trade Republic, vérifiées éligibles PEA (siège UE/EEE)
# Mise à jour manuelle trimestrielle suffisante
# ============================================================
PEA_UNIVERSE = {
    # --- TECHNOLOGIE ---
    "Technologie": [
        ("SAP.DE",    "SAP"),
        ("CAP.PA",    "Capgemini"),
        ("DSY.PA",    "Dassault Systèmes"),
        ("STM.PA",    "STMicroelectronics"),
        ("ASML.AS",   "ASML"),
        ("NOKIA.HE",  "Nokia"),
        ("DASSAV.PA", "Dassault Aviation"),
        ("ATOS.PA",   "Atos"),
        ("SQLI.PA",   "SQLI"),
        ("LDL.PA",    "Lectra"),
    ],
    # --- FINANCE & BANQUE ---
    "Finance": [
        ("BNP.PA",   "BNP Paribas"),
        ("ACA.PA",   "Crédit Agricole"),
        ("GLE.PA",   "Société Générale"),
        ("CS.PA",    "AXA"),
        ("DBK.DE",   "Deutsche Bank"),
        ("BBVA.MC",  "BBVA"),
        ("SAN.MC",   "Santander"),
        ("ISP.MI",   "Intesa Sanpaolo"),
        ("ING.AS",   "ING"),
        ("KBC.BR",   "KBC Groupe"),
    ],
    # --- SANTÉ & PHARMA ---
    "Santé": [
        ("SAN.PA",   "Sanofi"),
        ("EL.PA",    "EssilorLuxottica"),
        ("BIM.PA",   "bioMérieux"),
        ("IPSEN.PA", "Ipsen"),
        ("ORPEA.PA", "Orpea"),
        ("EUROFINS.PA","Eurofins"),
        ("NOVN.SW",  "Novartis"),
        ("ROG.SW",   "Roche"),
        ("UCB.BR",   "UCB"),
        ("GEVE.PA",  "Genfit"),
    ],
    # --- ÉNERGIE ---
    "Énergie": [
        ("TTE.PA",   "TotalEnergies"),
        ("ENGI.PA",  "Engie"),
        ("IBE.MC",   "Iberdrola"),
        ("ENEL.MI",  "Enel"),
        ("RWE.DE",   "RWE"),
        ("ORSTED.CO","Ørsted"),
        ("VOPAK.AS", "Vopak"),
        ("SQY.PA",   "Sqy"),
        ("NEOEN.PA", "Neoen"),
        ("ENPH.PA",  "Energisme"),
    ],
    # --- INDUSTRIE ---
    "Industrie": [
        ("AI.PA",    "Air Liquide"),
        ("SU.PA",    "Schneider Electric"),
        ("LR.PA",    "Legrand"),
        ("DG.PA",    "Vinci"),
        ("SIE.DE",   "Siemens"),
        ("ABB.ST",   "ABB"),
        ("VOLV-B.ST","Volvo"),
        ("ALO.PA",   "Alstom"),
        ("SPIE.PA",  "SPIE"),
        ("TKTT.PA",  "Tikehau Capital"),
    ],
    # --- LUXE & CONSO ---
    "Luxe & Conso": [
        ("MC.PA",    "LVMH"),
        ("RMS.PA",   "Hermès"),
        ("OR.PA",    "L'Oréal"),
        ("CDI.PA",   "Christian Dior"),
        ("RCO.PA",   "Rémy Cointreau"),
        ("RI.PA",    "Pernod Ricard"),
        ("ADS.DE",   "Adidas"),
        ("PUM.DE",   "Puma"),
        ("MELE.PA",  "Mélèze"),
        ("LANC.PA",  "Lancaster"),
    ],
    # --- AUTOMOBILE ---
    "Automobile": [
        ("STLA.MI",  "Stellantis"),
        ("RNO.PA",   "Renault"),
        ("BMW.DE",   "BMW"),
        ("VOW3.DE",  "Volkswagen"),
        ("MBG.DE",   "Mercedes-Benz"),
        ("RACE.MI",  "Ferrari"),
        ("MICH.PA",  "Michelin"),
        ("FG.PA",    "Faurecia"),
        ("SAF.PA",   "Safran"),
        ("GT.PA",    "Getinge"),
    ],
    # --- TELECOM ---
    "Télécom": [
        ("ORA.PA",   "Orange"),
        ("VIV.PA",   "Vivendi"),
        ("TEF.MC",   "Telefónica"),
        ("DTE.DE",   "Deutsche Telekom"),
        ("TELIA.ST", "Telia"),
        ("PROX.BR",  "Proximus"),
        ("ALTICE.NV","Altice Europe"),
        ("ILI.PA",   "Iliad"),
        ("ATEME.PA", "Ateme"),
        ("IQSTL.PA", "iQSTEL"),
    ],
    # --- IMMOBILIER ---
    "Immobilier": [
        ("URW.AS",   "Unibail-Rodamco"),
        ("COV.PA",   "Covivio"),
        ("KLE.PA",   "Klépierre"),
        ("ICAD.PA",  "iCAD"),
        ("ARGAN.PA", "Argan"),
        ("MRM.PA",   "MRM"),
        ("SFL.PA",   "SFL"),
        ("CBRE.PA",  "CBRE"),
        ("TFII.PA",  "TF1"),
        ("FDR.PA",   "Foncière des Régions"),
    ],
    # --- MATÉRIAUX & CHIMIE ---
    "Matériaux": [
        ("AIR.PA",   "Airbus"),
        ("ARK.PA",   "Arkema"),
        ("SOLV.BR",  "Solvay"),
        ("BASF.DE",  "BASF"),
        ("LINDE.DE", "Linde"),
        ("SWT.PA",   "Saint-Gobain"),
        ("BL.PA",    "Bollore"),
        ("VIE.PA",   "Vicat"),
        ("CIM.PA",   "Ciments Français"),
        ("ERA.PA",   "Eramet"),
    ],
    # --- AÉRO & DÉFENSE ---
    "Aéro & Défense": [
        ("HO.PA",    "Thales"),
        ("AM.PA",    "Dassault Aviation"),
        ("LDO.MI",   "Leonardo"),
        ("BA.L",     "BAE Systems"),
        ("RHIM.L",   "RHI Magnesita"),
        ("HEICO.PA", "Heico"),
        ("MTU.DE",   "MTU Aero"),
        ("ROLLS.L",  "Rolls-Royce"),
        ("AIR.PA",   "Airbus"),
        ("MBDA.PA",  "MBDA"),
    ],
    # --- DISTRIBUTION ---
    "Distribution": [
        ("CA.PA",    "Carrefour"),
        ("MRK.DE",   "Merck KGaA"),
        ("ITX.MC",   "Inditex"),
        ("FNAC.PA",  "Fnac Darty"),
        ("AMS.MC",   "Amadeus"),
        ("EXO.PA",   "Exosens"),
        ("BON.PA",   "Bonduelle"),
        ("LI.PA",    "Leclerc Invest"),
        ("HMB.ST",   "H&M"),
        ("ZURN.SW",  "Zurich Insurance"),
    ],
}

SECTORS = list(PEA_UNIVERSE.keys())   # 12 secteurs → 12 tickets minimum
TOP_PER_SECTOR = 2                    # On garde 2 meilleurs par secteur avant IA
FINAL_COUNT = 20                      # Sélection finale


# ============================================================
# ÉTAPE 1 : COLLECTE DONNÉES (yfinance, 100% gratuit)
# ============================================================

def get_stock_data(ticker: str) -> dict | None:
    """Récupère prix + historique pour un ticker. Retourne None si échec."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="6mo", auto_adjust=True)
        if hist is None or len(hist) < 30:
            return None
        info = {}
        try:
            info = t.info or {}
        except Exception:
            pass
        return {"hist": hist, "info": info}
    except Exception:
        return None


# ============================================================
# ÉTAPE 2 : SCORING QUANTITATIF PUR (aucun coût IA)
# ============================================================

def compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd  = ema12 - ema26
    sig   = macd.ewm(span=9).mean()
    return float(macd.iloc[-1]), float(sig.iloc[-1]), float((macd - sig).iloc[-1])


def compute_bollinger(series: pd.Series, period: int = 20):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = (mid + 2 * std).iloc[-1]
    lower = (mid - 2 * std).iloc[-1]
    price = series.iloc[-1]
    pos   = (price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    return float(upper), float(lower), float(pos)


def compute_atr(hist: pd.DataFrame, period: int = 14) -> float:
    high, low, close = hist["High"], hist["Low"], hist["Close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return float(tr.rolling(period).mean().iloc[-1])


def compute_trend_slope(series: pd.Series, days: int = 30) -> float:
    """Pente de régression linéaire sur `days` jours, en % du prix."""
    s = series.tail(days).values
    x = np.arange(len(s))
    c = np.polyfit(x, s, 1)
    return float(c[0] * days / s[-1] * 100)


def score_stock(ticker: str, name: str, sector: str) -> dict | None:
    """
    Calcule un score quantitatif complet (0-100) sans aucun appel IA.
    Retourne None si données insuffisantes.
    """
    data = get_stock_data(ticker)
    if data is None:
        print(f"    ⚠ {ticker} : données insuffisantes")
        return None

    hist = data["hist"]
    info = data["info"]
    close = hist["Close"]
    price = float(close.iloc[-1])

    # --- Indicateurs techniques ---
    ma20  = float(close.rolling(20).mean().iloc[-1])
    ma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    rsi   = compute_rsi(close)
    macd_val, macd_sig, macd_hist = compute_macd(close)
    bb_up, bb_low, bb_pos = compute_bollinger(close)
    atr   = compute_atr(hist)
    atr_pct = atr / price * 100
    trend = compute_trend_slope(close, 30)

    # Supports & résistances (3 mois)
    hist3m   = hist.tail(63)
    support  = float(hist3m["Low"].min())
    resist   = float(hist3m["High"].max())

    # Variations
    chg1d  = float((price / close.iloc[-2] - 1) * 100) if len(close) >= 2  else 0.0
    chg1m  = float((price / close.iloc[-22] - 1) * 100) if len(close) >= 22 else 0.0
    chg3m  = float((price / close.iloc[-63] - 1) * 100) if len(close) >= 63 else 0.0

    vol20  = float(hist["Volume"].rolling(20).mean().iloc[-1])
    vol_rel= float(hist["Volume"].iloc[-1] / vol20) if vol20 > 0 else 1.0

    # --- Score technique (0-45) ---
    ts = 0
    if ma50  and price > ma50:  ts += 10
    if ma200 and price > ma200: ts += 10
    elif not ma200 and price > ma20: ts += 5

    if 40 <= rsi <= 60:   ts += 12
    elif 30 <= rsi < 40:  ts += 10   # survente légère = opportunité
    elif rsi < 30:        ts += 8    # survente forte
    elif 60 < rsi <= 70:  ts += 6
    else:                 ts += 2    # surachat > 70

    if macd_hist > 0 and macd_val > macd_sig: ts += 8
    elif macd_hist > 0:                       ts += 4

    if bb_pos < 0.25:    ts += 5   # proche bande basse
    elif bb_pos < 0.5:   ts += 3

    if trend > 3:  ts += 5
    elif trend > 0: ts += 2

    ts = min(45, ts)

    # --- Score fondamental (0-40) ---
    pe     = info.get("trailingPE") or info.get("forwardPE")
    roe    = info.get("returnOnEquity")
    rev_g  = info.get("revenueGrowth")
    earn_g = info.get("earningsGrowth")
    de     = info.get("debtToEquity")
    div    = info.get("dividendYield")
    beta   = info.get("beta")
    target = info.get("targetMeanPrice")
    mktcap = info.get("marketCap")

    upside = ((target / price) - 1) * 100 if target and price > 0 else None

    fs = 20  # base
    if pe:
        if   10 <= pe <= 18: fs += 10
        elif 18 < pe <= 28:  fs += 5
        elif pe > 28:        fs -= 5
        elif pe < 10 and pe > 0: fs += 7

    if roe  and roe > 0.15:  fs += 6
    if rev_g and rev_g > 0.05: fs += 6
    if earn_g and earn_g > 0.05: fs += 5
    if de   and de < 80:     fs += 3
    if upside and upside > 15: fs += 5
    elif upside and upside > 8: fs += 3

    fs = min(40, max(0, fs))

    # --- Score momentum (0-15) ---
    ms = 0
    if chg1m > 3:   ms += 5
    elif chg1m > 0: ms += 2
    if chg3m > 5:   ms += 5
    elif chg3m > 0: ms += 2
    if vol_rel > 1.3: ms += 5
    elif vol_rel > 1.0: ms += 2
    ms = min(15, ms)

    # --- Score global ---
    total = ts + fs + ms   # max = 100

    # --- Prix d'entrée & stop-loss ---
    atr_stop     = price - 2.5 * atr
    support_stop = support * 0.975
    stop_loss    = round(max(atr_stop, support_stop), 2)

    if rsi > 65:
        entry = round(price * 0.98, 2)
        entry_tip = "Attendre un pull-back de ~2 % (RSI élevé)"
    elif rsi < 35:
        entry = round(price * 1.005, 2)
        entry_tip = "RSI en survente : entrée par tranche possible"
    else:
        entry = round(price, 2)
        entry_tip = "Zone d'entrée actuelle correcte"

    # Objectif 1 mois
    obj_trend = price * (1 + max(0.04, trend / 100 * 1.2))
    target_1m = round(min(resist * 0.97, obj_trend), 2)

    risk   = entry - stop_loss
    reward = target_1m - entry
    rr     = round(reward / risk, 2) if risk > 0 else 0.0

    fees_pct    = TR_FEE_TOTAL / entry * 100
    net_gain    = round((reward / entry * 100) - fees_pct, 2)
    pertinent   = net_gain >= MIN_GAIN_PCT and rr >= 1.5

    # Malus si le trade ne couvre pas les frais
    if not pertinent:
        total = int(total * 0.75)

    return {
        # Identité
        "ticker":   ticker,
        "name":     name,
        "sector":   sector,
        # Prix
        "price":    round(price,  2),
        "chg1d":   round(chg1d,  2),
        "chg1m":   round(chg1m,  2),
        "chg3m":   round(chg3m,  2),
        # Technique
        "ma20":    round(ma20,   2),
        "ma50":    round(ma50,   2) if ma50  else None,
        "ma200":   round(ma200,  2) if ma200 else None,
        "rsi":     round(rsi,    1),
        "macd":    round(macd_val, 3),
        "macd_sig":round(macd_sig, 3),
        "macd_hist":round(macd_hist,3),
        "bb_up":   round(bb_up,  2),
        "bb_low":  round(bb_low, 2),
        "bb_pos":  round(bb_pos, 2),
        "atr":     round(atr,    2),
        "atr_pct": round(atr_...⏹
