#!/usr/bin/env python3
"""
PEA Tracker - Analyse quotidienne optimisee
- Presélection 100% quantitative (yfinance, aucun cout)
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

MAMMOUTH_API_KEY = os.environ.get("MAMMOUTH_API_KEY", "")

client = OpenAI(
    api_key=MAMMOUTH_API_KEY,
    base_url="https://api.mammouth.ai/v1",
)

AI_MODEL = "gpt-4o"

TR_FEE = 1.0
TR_FEE_TOTAL = 2.0
MIN_GAIN_PCT = 1.0
MAX_PRICE = 250.0
TOP_N = 20

PEA_UNIVERSE = {
    "Technologie": [
        ("SAP.DE",    "SAP"),
        ("CAP.PA",    "Capgemini"),
        ("DSY.PA",    "Dassault Systemes"),
        ("STM.PA",    "STMicroelectronics"),
        ("ASML.AS",   "ASML"),
        ("NOKIA.HE",  "Nokia"),
        ("ATOS.PA",   "Atos"),
        ("LDL.PA",    "Lectra"),
    ],
    "Finance": [
        ("BNP.PA",   "BNP Paribas"),
        ("ACA.PA",   "Credit Agricole"),
        ("GLE.PA",   "Societe Generale"),
        ("CS.PA",    "AXA"),
        ("DBK.DE",   "Deutsche Bank"),
        ("BBVA.MC",  "BBVA"),
        ("SAN.MC",   "Santander"),
        ("ISP.MI",   "Intesa Sanpaolo"),
        ("ING.AS",   "ING"),
        ("KBC.BR",   "KBC Groupe"),
    ],
    "Sante": [
        ("SAN.PA",   "Sanofi"),
        ("EL.PA",    "EssilorLuxottica"),
        ("BIM.PA",   "bioMerieux"),
        ("IPSEN.PA", "Ipsen"),
        ("UCB.BR",   "UCB"),
        ("NOVN.SW",  "Novartis"),
        ("ROG.SW",   "Roche"),
    ],
    "Energie": [
        ("TTE.PA",   "TotalEnergies"),
        ("ENGI.PA",  "Engie"),
        ("IBE.MC",   "Iberdrola"),
        ("ENEL.MI",  "Enel"),
        ("RWE.DE",   "RWE"),
    ],
    "Industrie": [
        ("AI.PA",    "Air Liquide"),
        ("SU.PA",    "Schneider Electric"),
        ("LR.PA",    "Legrand"),
        ("DG.PA",    "Vinci"),
        ("SIE.DE",   "Siemens"),
        ("ABB.ST",   "ABB"),
        ("ALO.PA",   "Alstom"),
        ("SPIE.PA",  "SPIE"),
    ],
    "Luxe & Conso": [
        ("MC.PA",    "LVMH"),
        ("RMS.PA",   "Hermes"),
        ("OR.PA",    "LOreal"),
        ("RI.PA",    "Pernod Ricard"),
        ("ADS.DE",   "Adidas"),
        ("PUM.DE",   "Puma"),
    ],
    "Automobile": [
        ("STLA.MI",  "Stellantis"),
        ("RNO.PA",   "Renault"),
        ("BMW.DE",   "BMW"),
        ("VOW3.DE",  "Volkswagen"),
        ("MBG.DE",   "Mercedes-Benz"),
        ("RACE.MI",  "Ferrari"),
    ],
}


# ============================================================
# ETAPE 1 : SCORING QUANTITATIF
# ============================================================

def score_stock(ticker: str, name: str, sector: str) -> dict | None:
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="6mo")
        if hist.empty or len(hist) < 20:
            return None

        close = hist["Close"].dropna()
        volume = hist["Volume"].dropna()
        price = float(close.iloc[-1])

        if price <= 0:
            return None

        # Moyennes mobiles
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else ma20

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = float(100 - (100 / (1 + rs)).iloc[-1]) if not rs.isna().iloc[-1] else 50.0

        # Volatilite et volume
        returns = close.pct_change().dropna()
        volatility = float(returns.std() * np.sqrt(252) * 100)
        vol_avg = float(volume.rolling(20).mean().iloc[-1])
        vol_ratio = float(volume.iloc[-1] / vol_avg) if vol_avg > 0 else 1.0

        # Variation 1 mois
        perf_1m = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0.0

        # Objectif de prix (resistance simple)
        high_3m = float(close.tail(63).max())
        target = round(high_3m * 1.03, 2)
        stop = round(price * 0.95, 2)
        net_gain = ((target - price) / price * 100) - TR_FEE_TOTAL
        rr = (target - price) / (price - stop) if (price - stop) > 0 else 0

        # Score composite (sans filtrage dur)
        score = 0.0
        if price > ma20:
            score += 2
        if ma20 > ma50:
            score += 2
        if 35 <= rsi <= 65:
            score += 2
        elif rsi < 35:
            score += 3  # survente = opportunite
        if vol_ratio > 1.2:
            score += 1
        if net_gain >= MIN_GAIN_PCT:
            score += 2
        if rr >= 1.0:
            score += 2
        if perf_1m > 0:
            score += 1

        entry_tip = (
            f"Entree sous {price:.2f}, objectif {target:.2f}, stop {stop:.2f} "
            f"(RR {rr:.1f}:1, gain net estimé {net_gain:.1f}%)"
        )

        return {
            "ticker":     ticker,
            "name":       name,
            "sector":     sector,
            "price":      round(price, 2),
            "ma20":       round(ma20, 2),
            "ma50":       round(ma50, 2),
            "rsi":        round(rsi, 1),
            "volatility": round(volatility, 1),
            "vol_ratio":  round(vol_ratio, 2),
            "perf_1m":    round(perf_1m, 2),
            "target":     target,
            "stop":       stop,
            "net_gain":   round(net_gain, 2),
            "rr":         round(rr, 2),
            "score":      round(score, 2),
            "entry_tip":  entry_tip,
        }

    except Exception as e:
        print(f"  Erreur {ticker} : {e}")
        return None


# ============================================================
# ETAPE 2 : SELECTION DES CANDIDATS
# ============================================================

def select_candidates() -> list[dict]:
    all_stocks = []
    for sector, tickers in PEA_UNIVERSE.items():
        for ticker, name in tickers:
            print(f"  Scoring {ticker}...", end="
")
            result = score_stock(ticker, name, sector)
            if result:
                all_stocks.append(result)
            time.sleep(0.3)

    all_stocks.sort(key=lambda x: x["score"], reverse=True)
    return all_stocks[:TOP_N]


# ============================================================
# ETAPE 3 : ANALYSE IA
# ============================================================

def get_ai_analysis(stocks: list[dict]) -> dict:
    if not stocks:
        return {}

    summary = []
    for s in stocks:
        summary.append(
            f"{s['ticker']} ({s['name']}) | Secteur: {s['sector']} | "
            f"Prix: {s['price']} | MA20: {s['ma20']} | MA50: {s['ma50']} | "
            f"RSI: {s['rsi']} | Volatilite: {s['volatility']}% | "
            f"Perf 1M: {s['perf_1m']}% | Objectif: {s['target']} | "
            f"Stop: {s['stop']} | Gain net: {s['net_gain']}% | RR: {s['rr']} | Score: {s['score']}"
        )

    prompt = f"""Tu es un analyste financier expert en bourse europeenne et en PEA.

Voici les {len(stocks)} meilleures actions selectionnees par scoring quantitatif :

{chr(10).join(summary)}

Pour CHAQUE action, fournis une analyse JSON avec ces champs :
- ticker (string)
- signal : "ACHETER", "SURVEILLER" ou "EVITER"
- conviction : entier de 1 (faible) a 5 (forte)
- resume : 2 phrases max sur la situation actuelle
- bull_case : argument principal haussier
- bear_case : risque principal
- chartiste : lecture technique courte (support, resistance, tendance)
- conseil : conseil operationnel concret pour un investisseur PEA

Reponds UNIQUEMENT avec un JSON valide de la forme :
{{"analyses": [ ... ]}}
"""

    try:
        resp = client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000,
        )
        raw = resp.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)
        analyses = data.get("analyses", data) if isinstance(data, dict) else data
        return {a["ticker"]: a for a in analyses if "ticker" in a}

    except Exception as e:
        print(f"  Erreur IA : {e}")
        return {}


# ============================================================
# ETAPE 4 : SAUVEGARDE JSON
# ============================================================

def save_results(stocks: list[dict], ai_map: dict):
    output = []
    for s in stocks:
        ai = ai_map.get(s["ticker"], {})
        signal = ai.get("signal", "SURVEILLER")
        # On garde TOUTES les actions, meme EVITER (le front-end fait le tri)
        output.append({
            **s,
            "signal":     signal,
            "conviction": ai.get("conviction", 3),
            "resume":     ai.get("resume", "Analyse en cours de chargement."),
            "bull_case":  ai.get("bull_case", ""),
            "bear_case":  ai.get("bear_case", ""),
            "chartiste":  ai.get("chartiste", ""),
            "conseil":    ai.get("conseil", s["entry_tip"]),
        })

    result = {
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date":       date.today().isoformat(),
        "count":      len(output),
        "stocks":     output,
    }

    os.makedirs("data", exist_ok=True)
    path = "data/selections.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  Sauvegarde : {path} ({len(output)} actions)")
    return path


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PEA Tracker - Analyse du", datetime.utcnow().strftime("%d/%m/%Y %H:%M UTC"))
    print("=" * 60)

    print("\nETAPE 1-2 : Collecte et scoring quantitatif...")
    candidates = select_candidates()
    print(f"\n  {len(candidates)} actions selectionnees")

    print("\nETAPE 3 : Analyse IA...")
    ai_map = get_ai_analysis(candidates)
    print(f"  {len(ai_map)} analyses IA recues")

    print("\nETAPE 4 : Sauvegarde...")
    save_results(candidates, ai_map)

    print("\nTermine !")
    print("=" * 60)


if __name__ == "__main__":
    main()
