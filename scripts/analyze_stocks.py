#!/usr/bin/env python3
"""
PEA Tracker - Analyse quotidienne optimisee
- Preselection 100% quantitative (yfinance, aucun cout)
- 2 appels MammouthIA : actions 1-10 puis 11-20
- Remplacement des EVITER par les suivants du secteur
- Logique quantitative professionnelle (ATR, R/R garanti 1:2)
- [V3] Zero EVITER garanti dans le Top 20 (reserve ultime)
- [V3] Airbus classe dans Industrie
- [V3] Pastille de style MOMENTUM / REBOND / NEUTRE
- [V3] Formule gain validee (aucune correction necessaire)
"""

import json, os, time, warnings, math
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

AI_MODEL        = "gpt-4o"
TR_FEE          = 1.0
TR_FEE_TOTAL    = 2.0
MIN_GAIN_PCT    = 3.0
MAX_PRICE       = 250.0
MIN_SCORE_ACHAT = 55

# ============================================================
# UNIVERS PEA
# [V3-B] Airbus deplace dans Industrie (correction classification)
# ============================================================

PEA_UNIVERSE = {
    "Technologie": [
        ("CAP.F",       "Capgemini"),
        ("STM.F",       "STMicroelectronics"),
        ("SAP.DE",      "SAP"),
        ("IFX.DE",      "Infineon"),
        ("BSI.F",       "Be Semiconductor"),
        ("EKE.F",       "Esker"),
        ("SOP.F",       "Sopra Steria"),
        ("DSY.F",       "Dassault Systemes"),
        ("NOA3.F",      "Nokia"),
        ("LDL.PA",      "Lectra"),
        ("SOI.F",       "Soitec"),
        ("ALT.F",       "Alten"),
        ("SWP.F",       "Sword Group"),
        ("INPST.AS",    "Inpost"),
        ("PHI1.F",      "Philips"),
        ("NEX.PA",      "Nexans"),
        ("TEP.F",       "Teleperformance"),
        ("OVH.PA",      "OVHcloud"),
        ("AUB.PA",      "Aubay"),
        ("WAVE.PA",     "Wavestone"),
        ("ATO.PA",      "Atos"), # Attention : Situation financière tendue
    ],
    "Industrie & Defense": [
        ("SND.F",       "Schneider Electric"),
        ("AIR.F",       "Airbus"),
        ("THA.F",       "Thales"),
        ("DAA.F",       "Dassault Aviation"),
        ("LDO.F",       "Leonardo"),
        ("SGO.F",       "Saint-Gobain"),
        ("RXL.F",       "Rexel"),
        ("PRY.F",       "Prysmian"),
        ("V7A.F",       "Verallia"),
        ("IPH.F",       "Interpump Group"),
        ("GTT.PA",      "GTT"),
        ("ALU.F",       "Alstom"),
        ("ELIS.PA",     "Elis"),
    ],
    "Finance": [
        ("BNP.F",       "BNP Paribas"),
        ("CBK.DE",      "Commerzbank"),
        ("DBK.DE",      "Deutsche Bank"),
        ("ACA.F",       "Credit Agricole"),
        ("GLE.F",       "Societe Generale"),
        ("AXAF.F",      "AXA"),
        ("BBVA.MC",     "BBVA"),
        ("SAN.MC",      "Santander"),
        ("ISP.MI",      "Intesa Sanpaolo"),
        ("INGA.AS",     "ING Groep"),
        ("AMUN.PA",     "Amundi"),
        ("COFA.PA",     "Coface"),
        ("SCOR.PA",     "SCOR"),
        ("FDJ.PA",      "Francaise des Jeux"),
        ("TIKR.PA",     "Tikehau Capital"),
        ("ABC.PA",      "ABC Arbitrage"),
    ],
    "Sante": [
        ("SAN.F",       "Sanofi"),
        ("EL.F",        "EssilorLuxottica"),
        ("BIM.F",       "bioMerieux"),
        ("IPN.F",       "Ipsen"),
        ("UCB.BR",      "UCB"),
        ("VLA.PA",      "Valneva"),
        ("SRT3.DE",     "Sartorius Stedim"),
        ("EMEIS.PA",    "Emeis"), # Ex-Orpea
        ("ALCLS.PA",    "Clariane"), # Ex-Korian
        ("ERPI.PA",     "Euroapi"),
    ],
    "Luxe & Conso": [
        ("RI.F",        "Pernod Ricard"),
        ("ADS.DE",      "Adidas"),
        ("PUM.DE",      "Puma"),
        ("BN.F",        "Danone"),
        ("UNA.AS",      "Unilever"),
        ("CA.F",        "Carrefour"),
        ("ACR.F",       "Accor"),
        ("SK.F",        "SEB"),
        ("RCO.PA",      "Remy Cointreau"),
        ("CARLB.CO",    "Carlsberg"),
        ("FNAC.PA",     "Fnac Darty"),
        ("BEN.PA",      "Beneteau"),
    ],
    "Automobile": [
        ("STLAM.MI",    "Stellantis"),
        ("RNO.PA",      "Renault"),
        ("MBG.DE",      "Mercedes-Benz"),
        ("BMW.DE",      "BMW"),
        ("VOW3.DE",     "Volkswagen"),
        ("OPM.PA",      "OpMobility"), # Ex-Plastic Omnium
        ("VLEO.PA",     "Valeo"),
        ("AYV.PA",      "Ayvens"), # Ex-ALD
    ],
    "Immobilier": [
        ("URW.AS",      "Unibail-Rodamco"),
        ("GFC.PA",      "Gecina"),
        ("COV.PA",      "Covivio"),
        ("LIH.F",       "Vonovia"),
        ("ARGAN.PA",    "Argan"),
        ("NSE.PA",      "Nexity"),
        ("MERY.PA",     "Mercialys"),
        ("WDP.BR",      "Warehouses De Pauw"),
    ],
    "Energie & Materiaux": [
        ("AIQU.F",      "Air Liquide"),
        ("TTE.F",       "TotalEnergies"),
        ("ENI.MI",      "ENI"),
        ("AKE.PA",      "Arkema"),
        ("BAS.DE",      "BASF"),
        ("HEI.DE",      "Heidelberg Materials"),
        ("VK.PA",       "Vallourec"),
        ("SOLB.BR",     "Solvay"),
        ("UMI.BR",      "Umicore"),
        ("DB6.F",       "Derichebourg"),
    ],
    "Telecom & Media": [
        ("ORA.F",       "Orange"),
        ("DTE.DE",      "Deutsche Telekom"),
        ("TEF.MC",      "Telefonica"),
        ("PUB.PA",      "Publicis"),
        ("VIV.PA",      "Vivendi"),
        ("MMB.PA",      "Lagardere"),
        ("TFI.PA",      "TF1"),
        ("M6.PA",       "M6 Metropole Television"),
    ],
}

SECTORS        = list(PEA_UNIVERSE.keys())
TOP_PER_SECTOR = 3
FINAL_COUNT    = 20


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def to_float(val):
    try:
        if val is None:
            return None
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, (int, str)) or obj is None:
        return obj
    else:
        try:
            f = float(obj)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except (ValueError, TypeError):
            return str(obj)


# ============================================================
# COLLECTE DONNEES
# ============================================================

def get_stock_data(ticker: str) -> dict | None:
    try:
        t    = yf.Ticker(ticker)
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
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
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
    s = series.tail(days).values
    x = np.arange(len(s))
    c = np.polyfit(x, s, 1)
    return float(c[0] * days / s[-1] * 100)


# ============================================================
# SCORING
# ============================================================

def score_stock(ticker: str, name: str, sector: str) -> dict | None:
    data = get_stock_data(ticker)
    if data is None:
        print(f"    Pas de donnees : {ticker}")
        return None

    hist  = data["hist"]
    info  = data["info"]
    close = hist["Close"]
    price = float(close.iloc[-1])

    # ── Filtre prix ──────────────────────────────────────────────────
    if price > MAX_PRICE:
        print(f"    Elimine (prix {price:.2f}EUR > {MAX_PRICE}EUR) : {ticker}")
        return None

    # ── ATR (14 séances) ─────────────────────────────────────────────
    atr = compute_atr(hist, period=14)
    if atr <= 0:
        print(f"    Elimine (ATR nul) : {ticker}")
        return None
    atr_pct = round(atr / price * 100, 2)

    # ── Indicateurs techniques ───────────────────────────────────────
    ma20  = float(close.rolling(20).mean().iloc[-1])
    ma50  = float(close.rolling(50).mean().iloc[-1])  if len(close) >= 50  else None
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    rsi   = compute_rsi(close)
    macd_val, macd_sig, macd_hist_val = compute_macd(close)
    bb_up, bb_low, bb_pos = compute_bollinger(close)
    trend = compute_trend_slope(close, 30)

    chg1d = float((price / close.iloc[-2]  - 1) * 100) if len(close) >= 2  else 0.0
    chg1m = float((price / close.iloc[-22] - 1) * 100) if len(close) >= 22 else 0.0
    chg3m = float((price / close.iloc[-63] - 1) * 100) if len(close) >= 63 else 0.0

    vol20   = float(hist["Volume"].rolling(20).mean().iloc[-1])
    vol_rel = float(hist["Volume"].iloc[-1] / vol20) if vol20 > 0 else 1.0

    hist3m  = hist.tail(63)
    support = float(hist3m["Low"].min())
    resist  = float(hist3m["High"].max())

    # ── Niveaux de trading basés sur l'ATR ───────────────────────────
    entry       = round(price - (0.2 * atr), 2)
    support_20j = float(hist["Low"].rolling(20).min().iloc[-1])
    stop_loss   = round(min(entry - (2 * atr), support_20j * 0.99), 2)
    risque      = entry - stop_loss
    target_1m   = round(entry + (risque * 2), 2)

    frais_entree = 0.005 # 0.5%
    frais_sortie = 0.005 # 0.5%
    prix_revient = entry * (1 + frais_entree)
    net_sortie   = target_1m * (1 - frais_sortie)
    net_gain     = round(((net_sortie / prix_revient) - 1) * 100, 2)

    rr        = round((target_1m - entry) / risque, 2) if risque > 0 else 0.0
    rr_label  = f"1:{rr}"

    if net_gain < MIN_GAIN_PCT:
        print(f"    Elimine (gain net {net_gain:.2f}% < {MIN_GAIN_PCT}%) : {ticker}")
        return None

    # ── [V3-C] Pastille de style ─────────────────────────────────────
    # REBOND  : RSI survendu  → opportunite de retour a la moyenne
    # MOMENTUM: RSI suracheté → train en marche, tendance forte
    # NEUTRE  : zone intermediaire
    if rsi < 40:
        style = "REBOND"
    elif rsi > 60:
        style = "MOMENTUM"
    else:
        style = "NEUTRE"

    # ── Score technique (0–45) ───────────────────────────────────────
    ts = 0
    if ma50  and price > ma50:  ts += 10
    if ma200 and price > ma200: ts += 10
    elif not ma200 and price > ma20: ts += 5

    if 40 <= rsi <= 60:   ts += 12
    elif 30 <= rsi < 40:  ts += 10
    elif rsi < 30:        ts += 8
    elif 60 < rsi <= 70:  ts += 6
    else:                 ts += 2

    if macd_hist_val > 0 and macd_val > macd_sig: ts += 8
    elif macd_hist_val > 0:                       ts += 4

    if bb_pos < 0.25:  ts += 5
    elif bb_pos < 0.5: ts += 3

    if trend > 3:   ts += 5
    elif trend > 0: ts += 2

    ts = min(45, ts)

    # ── Score fondamental (0–40) ─────────────────────────────────────
    pe     = to_float(info.get("trailingPE") or info.get("forwardPE"))
    roe    = to_float(info.get("returnOnEquity"))
    rev_g  = to_float(info.get("revenueGrowth"))
    earn_g = to_float(info.get("earningsGrowth"))
    de     = to_float(info.get("debtToEquity"))
    div    = to_float(info.get("dividendYield"))
    beta   = to_float(info.get("beta"))
    target = to_float(info.get("targetMeanPrice"))
    mktcap = info.get("marketCap")

    upside = ((target / price) - 1) * 100 if target and price > 0 else None

    fs = 20
    if pe is not None:
        if   10 <= pe <= 18: fs += 10
        elif 18 < pe <= 28:  fs += 5
        elif pe > 28:        fs -= 5
        elif 0 < pe < 10:    fs += 7

    if roe    is not None and roe > 0.15:    fs += 6
    if rev_g  is not None and rev_g > 0.05:  fs += 6
    if earn_g is not None and earn_g > 0.05: fs += 5
    if de     is not None and de < 80:       fs += 3
    if upside is not None and upside > 15:   fs += 5
    elif upside is not None and upside > 8:  fs += 3

    fs = min(40, max(0, fs))

    # ── Score momentum (0–15) ────────────────────────────────────────
    ms = 0
    if chg1m > 3:   ms += 5
    elif chg1m > 0: ms += 2
    if chg3m > 5:   ms += 5
    elif chg3m > 0: ms += 2
    if vol_rel > 1.3:   ms += 5
    elif vol_rel > 1.0: ms += 2
    ms = min(15, ms)

    total = ts + fs + ms

    # ── Conseil d'entrée contextuel ──────────────────────────────────
    if rsi < 35:
        entry_tip = "Zone de survente : entree progressive recommandee."
    elif price > ma20 and macd_hist_val > 0:
        entry_tip = "Tendance confirmee : entree au prix du marche."
    else:
        entry_tip = "Attendre confirmation : entree sur repli ou cassure."

    prices_raw = close.tail(120).tolist()
    prices_6m  = [round(float(p), 2) for p in prices_raw
                  if p is not None and not math.isnan(float(p))]

    return {
        "ticker":     ticker,
        "name":       name,
        "sector":     sector,
        "price":      round(price, 2),
        "atr":        round(atr, 2),
        "atr_pct":    atr_pct,
        "rsi":        round(rsi, 1),
        "macd":       round(macd_val, 4),
        "macd_sig":   round(macd_sig, 4),
        "macd_hist":  round(macd_hist_val, 4),
        "bb_pos":     round(bb_pos, 3),
        "trend":      round(trend, 2),
        "chg1d":      round(chg1d, 2),
        "chg1m":      round(chg1m, 2),
        "chg3m":      round(chg3m, 2),
        "vol_rel":    round(vol_rel, 2),
        "support":    round(support, 2),
        "resist":     round(resist, 2),
        "ma20":       round(ma20, 2),
        "ma50":       round(ma50, 2) if ma50 else None,
        "ma200":      round(ma200, 2) if ma200 else None,
        "style":      style,          # [V3-C] Pastille de style
        "entry":      entry,
        "entry_tip":  entry_tip,
        "stop_loss":  stop_loss,
        "target_1m":  target_1m,
        "rr":         rr,
        "rr_label":   rr_label,
        "net_gain":   net_gain,
        "score":      total,
        "score_tech": ts,
        "score_fond": fs,
        "score_mom":  ms,
        "prices_6m":  prices_6m,
    }


# ============================================================
# SELECTION PAR SECTEUR
# ============================================================

def select_candidates() -> tuple[list[dict], dict, list[dict]]:
    """
    Retourne :
    - la liste des 20 finalistes triés par score décroissant
    - un dict sector -> liste des actions scorées hors top N (réserve sectorielle)
    - [V3-A] all_scored : toutes les actions scorées, triées par score desc
              (sert de réserve ultime dans save_results)
    """
    all_results     = []
    sector_reserves = {}

    for sector, stocks in PEA_UNIVERSE.items():
        print(f"\n  Secteur : {sector} ({len(stocks)} actions)")
        sector_results = []
        for ticker, name in stocks:
            print(f"    Analyse {ticker}...")
            result = score_stock(ticker, name, sector)
            if result:
                sector_results.append(result)
            time.sleep(0.3)

        sector_results.sort(key=lambda x: x["score"], reverse=True)
        top = sector_results[:TOP_PER_SECTOR]
        print(f"    Top {TOP_PER_SECTOR} retenus : {[x['ticker'] for x in top]}")
        all_results.extend(top)
        sector_reserves[sector] = sector_results[TOP_PER_SECTOR:]

    all_results.sort(key=lambda x: x["score"], reverse=True)

    # [V3-A] Construire all_scored = top secteurs + toutes les réserves
    all_scored = list(all_results)
    for reserve in sector_reserves.values():
        all_scored.extend(reserve)
    all_scored.sort(key=lambda x: x["score"], reverse=True)

    return all_results[:FINAL_COUNT], sector_reserves, all_scored


# ============================================================
# APPEL IA
# ============================================================

def call_ai_batch(stocks: list[dict], batch_label: str) -> dict:
    if not stocks:
        return {}

    stocks_info = []
    for s in stocks:
        stocks_info.append({
            "ticker":    s["ticker"],
            "name":      s["name"],
            "sector":    s["sector"],
            "price":     s["price"],
            "rsi":       s["rsi"],
            "macd_hist": s["macd_hist"],
            "trend":     s["trend"],
            "chg1m":     s["chg1m"],
            "chg3m":     s["chg3m"],
            "score":     s["score"],
            "style":     s["style"],    # [V3-C] transmis a l'IA pour contexte
            "net_gain":  s["net_gain"],
            "rr_label":  s["rr_label"],
        })

    prompt = f"""Tu es un analyste financier senior specialise sur les actions europeennes cotees sur PEA.

Analyse les {len(stocks)} actions suivantes et fournis pour chacune une analyse fondamentale concise et differentee.

Donnees quantitatives :
{json.dumps(stocks_info, ensure_ascii=False, indent=2)}

REGLES STRICTES :
1. Signal ACHETER uniquement si score >= {MIN_SCORE_ACHAT} ET net_gain > 0. Sinon SURVEILLER.
2. Signal EVITER uniquement si risque fondamental serieux et avere (dette critique, fraude, faillite imminente).
3. Chaque "resume" doit decrire l'avantage competitif UNIQUE de l'entreprise. Ne JAMAIS ecrire le meme resume pour deux entreprises differentes.
4. bull_case et bear_case bases sur l'actualite recente du secteur, pas des generalites.
5. Le champ "style" de chaque action est "{{"REBOND" si RSI < 40, "MOMENTUM" si RSI > 60, "NEUTRE" sinon}}". Adapter le conseil en consequence : pour REBOND insister sur la patience et le point d'entree bas ; pour MOMENTUM insister sur la dynamique et la gestion du stop.

Format JSON STRICT :
{{
  "analyses": [
    {{
      "ticker": "XXX.XX",
      "signal": "ACHETER|SURVEILLER|EVITER",
      "conviction": 1-5,
      "resume": "Avantage compétitif distinctif de l'entreprise : moat, brevets, position dominante, marque, part de marché. Max 20 mots. Ex: 'Marque iconique avec un pricing power fort et une distribution mondiale difficile à répliquer.'",
      "bull_case": "Catalyseur concret issu de l'actualité ou du contexte macro qui pourrait faire monter le titre. Max 15 mots. Ex: 'Le retour en grâce du luxe en Chine et la réouverture des marchés asiatiques dopent les perspectives.'",
      "bear_case": "Risque réel et actuel lié à l'actualité ou au contexte sectoriel. Max 15 mots. Ex: 'Un ralentissement de la consommation américaine et la guerre des prix avec Nike pèsent sur les marges.'",
      "chartiste": "Niveau technique clé à surveiller : support à défendre, résistance à franchir, rebond attendu ou consolidation en cours. Ne pas répéter l'entrée/stop/cible. Max 20 mots. Ex: 'Support solide à 127€ à défendre. Résistance majeure à 161€ à franchir pour confirmer la tendance haussière.'"
    }}
  ]
}}

Reponds avec le JSON complet pour les {len(stocks)} actions sans aucun texte avant ou apres."""

    print(f"  Appel MammouthIA ({AI_MODEL}) - {batch_label}...")

    try:
        resp = client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000,
        )
        raw           = resp.choices[0].message.content.strip()
        finish_reason = resp.choices[0].finish_reason

        if finish_reason == "length":
            print(f"  AVERTISSEMENT : reponse tronquee ({batch_label})")

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data     = json.loads(raw)
            analyses = data.get("analyses", data) if isinstance(data, dict) else data
            result   = {a["ticker"]: a for a in analyses if "ticker" in a}
            print(f"  {len(result)} analyses parsees avec succes ({batch_label})")
            return result

        except json.JSONDecodeError:
            print(f"  JSON incomplet, tentative de recuperation partielle ({batch_label})...")
            import re
            objects  = re.findall(r'\{[^{}]{50,}\}', raw)
            analyses = []
            for obj in objects:
                try:
                    parsed = json.loads(obj)
                    if "ticker" in parsed and "signal" in parsed:
                        analyses.append(parsed)
                except Exception:
                    continue
            print(f"  {len(analyses)} analyses recuperees par parsing partiel ({batch_label})")
            return {a["ticker"]: a for a in analyses}

    except Exception as e:
        print(f"  Erreur IA ({batch_label}) : {e}")
        return {}


# ============================================================
# ANALYSE IA + REMPLACEMENT DES EVITER
# ============================================================

def get_ai_analysis(
    candidates: list[dict],
    sector_reserves: dict,
) -> tuple[list[dict], dict]:
    """
    1. Analyse les 20 candidats en 2 lots (1-10, 11-20)
    2. Remplace les EVITER par les suivants du secteur
    3. Retourne la liste finale et le dict ai_map complet
    """
    MAX_BATCH              = 10
    MAX_REPLACEMENT_ROUNDS = 5

    lot1 = candidates[:10]
    lot2 = candidates[10:]

    print(f"\n  Lot 1 : actions 1-10 ({len(lot1)} actions)")
    ai_map = call_ai_batch(lot1, "actions 1-10")

    print(f"\n  Lot 2 : actions 11-20 ({len(lot2)} actions)")
    ai_map.update(call_ai_batch(lot2, "actions 11-20"))

    stock_data_map = {s["ticker"]: s for s in candidates}
    reserve_index  = {sector: 0 for sector in sector_reserves}

    sector_first = {}
    for s in candidates:
        sec = s["sector"]
        if sec not in sector_first:
            sector_first[sec] = s
    for sec, reserve in sector_reserves.items():
        if sec not in sector_first and reserve:
            sector_first[sec] = reserve[0]

    final_list = list(candidates)

    for round_num in range(1, MAX_REPLACEMENT_ROUNDS + 1):
        to_replace = [
            (i, s) for i, s in enumerate(final_list)
            if ai_map.get(s["ticker"], {}).get("signal", "SURVEILLER") == "EVITER"
        ]

        if not to_replace:
            print(f"\n  Aucun EVITER restant. Remplacement termine apres {round_num - 1} tour(s).")
            break

        print(f"\n  Tour {round_num} : {len(to_replace)} action(s) a remplacer : "
              f"{[x[1]['ticker'] for x in to_replace]}")

        replacements             = []
        tickers_already_in_final = {s["ticker"] for s in final_list}

        for idx, evited_stock in to_replace:
            sector  = evited_stock["sector"]
            reserve = sector_reserves.get(sector, [])
            ri      = reserve_index[sector]
            found   = None

            while ri < len(reserve):
                candidate = reserve[ri]
                ri += 1
                if candidate["ticker"] not in tickers_already_in_final:
                    found = candidate
                    stock_data_map[candidate["ticker"]] = candidate
                    tickers_already_in_final.add(candidate["ticker"])
                    break

            reserve_index[sector] = ri

            if found:
                replacements.append((idx, found))
                print(f"    Remplacement : {evited_stock['ticker']} -> {found['ticker']} ({found['name']})")
            else:
                fallback = sector_first.get(sector)
                if fallback and fallback["ticker"] != evited_stock["ticker"]:
                    print(f"    Reserve epuisee pour {sector}. "
                          f"Fallback sur {fallback['ticker']} ({fallback['name']})")
                    replacements.append((idx, fallback))
                else:
                    print(f"    Aucun remplacant possible pour {evited_stock['ticker']}, conservation forcee.")
                    replacements.append((idx, evited_stock))

        new_stocks_to_analyze = []
        for idx, new_stock in replacements:
            if new_stock["ticker"] != final_list[idx]["ticker"]:
                final_list[idx] = new_stock
                if new_stock["ticker"] not in ai_map:
                    new_stocks_to_analyze.append(new_stock)

        if not new_stocks_to_analyze:
            print(f"  Aucun nouveau stock a analyser au tour {round_num}.")
            break

        print(f"\n  Analyse IA des {len(new_stocks_to_analyze)} remplacants...")
        for batch_start in range(0, len(new_stocks_to_analyze), MAX_BATCH):
            batch = new_stocks_to_analyze[batch_start:batch_start + MAX_BATCH]
            label = f"remplacants tour {round_num} ({batch_start + 1}-{batch_start + len(batch)})"
            ai_map.update(call_ai_batch(batch, label))

    else:
        print(f"\n  Limite de {MAX_REPLACEMENT_ROUNDS} tours atteinte.")

    print(f"\n  Selection finale : {[s['ticker'] for s in final_list]}")
    return final_list, ai_map


# ============================================================
# SAUVEGARDE
# [V3-A] Réserve ultime : garantit zéro EVITER dans le JSON final
# ============================================================

def save_results(stocks: list[dict], ai_map: dict, all_scored: list[dict]):
    """
    all_scored : toutes les actions scorées (select_candidates),
                 triées par score desc — sert de réserve ultime
                 pour remplacer tout EVITER résiduel.
    """
    tickers_in_final = {s["ticker"] for s in stocks}

    # Réserve ultime : actions scorées non présentes dans le Top 20
    ultimate_reserve = [
        s for s in all_scored
        if s["ticker"] not in tickers_in_final
    ]
    reserve_idx = 0

    output = []
    for s in stocks:
        ai     = ai_map.get(s["ticker"], {})
        signal = ai.get("signal", "SURVEILLER")

        # CORRECTION : Sécurité RSI (Si RSI > 75, on passe en SURVEILLER)
        if s["rsi"] > 75:
            signal = "SURVEILLER"
        
        # [V3-A] Si EVITER résiduel → remplacer par le suivant de la réserve ultime
        if signal == "EVITER":
            replaced = False
            while reserve_idx < len(ultimate_reserve):
                candidate        = ultimate_reserve[reserve_idx]
                reserve_idx     += 1
                candidate_ai     = ai_map.get(candidate["ticker"], {})
                candidate_signal = candidate_ai.get("signal", "SURVEILLER")
                if candidate_signal != "EVITER":
                    print(f"  [V3-A] Remplacement ultime EVITER : "
                          f"{s['ticker']} -> {candidate['ticker']} ({candidate['name']})")
                    s      = candidate
                    ai     = candidate_ai
                    signal = candidate_signal
                    # Mettre a jour le set pour eviter les doublons si plusieurs EVITER
                    tickers_in_final.add(candidate["ticker"])
                    replaced = True
                    break
            if not replaced:
                # Reserve ultime epuisee : on degrade en SURVEILLER plutot que d'afficher EVITER
                print(f"  [V3-A] Reserve ultime epuisee pour {s['ticker']} : degrade en SURVEILLER")
                signal = "SURVEILLER"

        output.append({
            **s,
            "signal":     signal,
            "conviction": ai.get("conviction", 3),
            "resume":     ai.get("resume",    "Donnees fondamentales en cours de chargement."),
            "bull_case":  ai.get("bull_case", ""),
            "bear_case":  ai.get("bear_case", ""),
            "chartiste":  ai.get("chartiste", ""),
            "conseil":    ai.get("conseil",   s.get("entry_tip", "")),
            "style":      s.get("style", "NEUTRE"),   # [V3-C] toujours present dans l'output
        })

    # Tri final par score décroissant
    output.sort(key=lambda x: x["score"], reverse=True)

    result = {
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date":       date.today().isoformat(),
        "count":      len(output),
        "stocks":     output,
    }

    result_clean = clean_for_json(result)

    os.makedirs("data", exist_ok=True)
    path = "data/selections.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result_clean, f, ensure_ascii=False, indent=2)

    print(f"\n  Sauvegarde : {path} ({len(output)} actions)")
    return path


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PEA Tracker - Analyse du", datetime.utcnow().strftime("%d/%m/%Y %H:%M UTC"))
    print("=" * 60)

    print("\nETAPE 1-2 : Collecte et scoring quantitatif...")
    # [V3-A] select_candidates retourne maintenant 3 valeurs
    candidates, sector_reserves, all_scored = select_candidates()
    print(f"\n  {len(candidates)} actions selectionnees")

    print("\nETAPE 3 : Analyse IA avec remplacement des EVITER...")
    final_list, ai_map = get_ai_analysis(candidates, sector_reserves)
    print(f"  {len(ai_map)} analyses IA recues au total")

    print("\nETAPE 4 : Sauvegarde...")
    # [V3-A] all_scored transmis a save_results pour la reserve ultime
    save_results(final_list, ai_map, all_scored)

    print("\nTermine !")
    print("=" * 60)


if __name__ == "__main__":
    main()
