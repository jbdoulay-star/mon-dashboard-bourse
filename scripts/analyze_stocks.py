#!/usr/bin/env python3
"""
PEA Tracker - Analyse quotidienne optimisee
- Preselection 100% quantitative (yfinance, aucun cout)
- 2 appels MammouthIA : actions 1-10 puis 11-20
- Remplacement des EVITER par les suivants du secteur
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

AI_MODEL = "gpt-4o"

TR_FEE = 1.0
TR_FEE_TOTAL = 2.0
MIN_GAIN_PCT = 1.0
MAX_PRICE = 250.0

PEA_UNIVERSE = {
    "Technologie": [
        ("CAP.PA",      "Capgemini"),
        ("STM.PA",      "STMicroelectronics"),
        ("NOKIA.HE",    "Nokia"),
        ("ATOS.PA",     "Atos"),
        ("LDL.PA",      "Lectra"),
        ("SOITEC.PA",   "Soitec"),
        ("ALTEN.PA",    "Alten"),
        ("SWORD.PA",    "Sword Group"),
        ("SAP.DE",      "SAP"),
        ("IFX.DE",      "Infineon"),
        ("SOP.PA",      "Sopra Steria"),
        ("INPST.AS",    "Inpost"),
        ("PHIA.AS",     "Philips"),
        ("NEX.PA",      "Nexans"),
        ("MGI.PA",      "MGI Digital"),
        ("TIT.MI",      "Telecom Italia"),
        ("TEP.PA",      "Teleperformance"),
        ("OVH.PA",      "OVHcloud"),
        ("BIGBEN.PA",   "BigBen Interactive"),
        ("AUBAY.PA",    "Aubay"),
        ("TXCOM.PA",    "Txcom"),
        ("INFE.PA",     "Infotel"),
        ("WGRD.PA",     "Wavestone"),
        ("DSY.PA",      "Dassault Systemes"),
        ("HEX1V.HE",    "Hexagon"),
    ],
    "Finance": [
        ("BNP.PA",      "BNP Paribas"),
        ("ACA.PA",      "Credit Agricole"),
        ("GLE.PA",      "Societe Generale"),
        ("CS.PA",       "AXA"),
        ("DBK.DE",      "Deutsche Bank"),
        ("BBVA.MC",     "BBVA"),
        ("SAN.MC",      "Santander"),
        ("ISP.MI",      "Intesa Sanpaolo"),
        ("ING.AS",      "ING"),
        ("KBC.BR",      "KBC Groupe"),
        ("ABCA.PA",     "ABC Arbitrage"),
        ("AMUN.PA",     "Amundi"),
        ("CNP.PA",      "CNP Assurances"),
        ("COV.PA",      "Coface"),
        ("CBK.DE",      "Commerzbank"),
        ("MUV2.DE",     "Munich Re"),
        ("ALV.DE",      "Allianz"),
        ("INGA.AS",     "ING Groep"),
        ("EXO.MI",      "Exor"),
        ("SCOR.PA",     "SCOR"),
        ("FDJ.PA",      "Francaise des Jeux"),
        ("CRBP2.PA",    "Credit Agricole Brie Picardie"),
        ("MRM.PA",      "MRM"),
        ("TIKR.PA",     "Tikehau Capital"),
        ("CIC.PA",      "CIC"),
    ],
    "Sante": [
        ("SAN.PA",      "Sanofi"),
        ("EL.PA",       "EssilorLuxottica"),
        ("BIM.PA",      "bioMerieux"),
        ("IPSEN.PA",    "Ipsen"),
        ("UCB.BR",      "UCB"),
        ("OSE.PA",      "OSE Immunotherapeutics"),
        ("LNA.PA",      "LNA Sante"),
        ("DBV.PA",      "DBV Technologies"),
        ("ELIS.PA",     "Elis"),
        ("ORPEA.PA",    "Orpea"),
        ("VALNEVA.PA",  "Valneva"),
        ("GMED.PA",     "Guerbet"),
        ("IPHA.PA",     "Innate Pharma"),
        ("GENFIT.PA",   "Genfit"),
        ("NANOB.PA",    "Nanobiotix"),
        ("ABIVAX.PA",   "Abivax"),
        ("ONXEO.PA",    "Onxeo"),
        ("TXPA.PA",     "Transgene"),
    ],
    "Energie": [
        ("TTE.PA",      "TotalEnergies"),
        ("ENGI.PA",     "Engie"),
        ("IBE.MC",      "Iberdrola"),
        ("ENEL.MI",     "Enel"),
        ("RWE.DE",      "RWE"),
        ("VIE.PA",      "Veolia"),
        ("EDP.LS",      "EDP"),
        ("EDPR.LS",     "EDP Renovaveis"),
        ("GALP.LS",     "Galp Energia"),
        ("ENI.MI",      "ENI"),
        ("NESTE.HE",    "Neste"),
        ("FORTUM.HE",   "Fortum"),
        ("VLTSA.PA",    "Voltalia"),
        ("NEOEN.PA",    "Neoen"),
        ("MCPHY.PA",    "McPhy Energy"),
        ("OMV.VI",      "OMV"),
        ("ALD.PA",      "ALD Automotive"),
        ("ALBIOMA.PA",  "Albioma"),
    ],
    "Industrie": [
        ("AI.PA",       "Air Liquide"),
        ("SU.PA",       "Schneider Electric"),
        ("LR.PA",       "Legrand"),
        ("DG.PA",       "Vinci"),
        ("ALO.PA",      "Alstom"),
        ("SPIE.PA",     "SPIE"),
        ("GTT.PA",      "GTT"),
        ("SAF.PA",      "Safran"),
        ("ADP.PA",      "Aeroports de Paris"),
        ("ERA.PA",      "Eramet"),
        ("AF.PA",       "Air France-KLM"),
        ("ABB.ST",      "ABB"),
        ("MBG.DE",      "Mercedes-Benz"),
        ("BMW.DE",      "BMW"),
        ("VOW3.DE",     "Volkswagen"),
        ("HAG.DE",      "Henkel"),
        ("KNEBV.HE",    "Kone"),
        ("WRT1V.HE",    "Wartsila"),
        ("STERV.HE",    "Stora Enso"),
        ("FGR.PA",      "Figeac Aero"),
        ("HAULOTTE.PA", "Haulotte"),
        ("DEME.BR",     "DEME Group"),
        ("EIFF.PA",     "Eiffage"),
        ("BVI.PA",      "Bureau Veritas"),
        ("GET.PA",      "Getlink"),
        ("MANU.PA",     "Manitou"),
        ("LISI.PA",     "Lisi"),
        ("GL.PA",       "GL Events"),
        ("FLEURY.PA",   "Fleury Michon"),
    ],
    "Luxe & Conso": [
        ("OR.PA",       "LOreal"),
        ("RI.PA",       "Pernod Ricard"),
        ("ADS.DE",      "Adidas"),
        ("PUM.DE",      "Puma"),
        ("SEB.PA",      "SEB"),
        ("BN.PA",       "Danone"),
        ("UNA.AS",      "Unilever"),
        ("CARLB.CO",    "Carlsberg"),
        ("SMCP.PA",     "SMCP"),
        ("FNAC.PA",     "Fnac Darty"),
        ("CA.PA",       "Carrefour"),
        ("RCO.PA",      "Remy Cointreau"),
        ("BEN.PA",      "Beneteau"),
        ("CDA.PA",      "Compagnie des Alpes"),
        ("AURES.PA",    "Aures Technologies"),
    ],
    "Automobile": [
        ("STLA.MI",     "Stellantis"),
        ("RNO.PA",      "Renault"),
        ("LI.PA",       "Plastic Omnium"),
        ("VLEO.PA",     "Valeo"),
        ("GTX.PA",      "Garrett Motion"),
        ("MBG.DE",      "Mercedes-Benz"),
        ("BMW.DE",      "BMW"),
        ("VOW3.DE",     "Volkswagen"),
        ("ELCO.PA",     "Electra"),
    ],
    "Immobilier": [
        ("URW.AS",      "Unibail-Rodamco"),
        ("CLT.PA",      "Carmila"),
        ("COV.PA",      "Covivio"),
        ("ARGAN.PA",    "Argan"),
        ("MRM.PA",      "MRM"),
        ("ALTAG.PA",    "Altarea"),
        ("NSE.PA",      "Nexity"),
        ("MONTEA.BR",   "Montea"),
        ("COFB.BR",     "Cofinimmo"),
        ("WDP.BR",      "Warehouses De Pauw"),
        ("GFCM.PA",     "Gecina"),
        ("MERY.PA",     "Mercialys"),
    ],
    "Telecom & Media": [
        ("ORA.PA",      "Orange"),
        ("TEF.MC",      "Telefonica"),
        ("DTE.DE",      "Deutsche Telekom"),
        ("PRX.AS",      "Prosus"),
        ("PUBP.PA",     "Publicis"),
        ("MMB.PA",      "Lagardere"),
        ("TEP.PA",      "Teleperformance"),
        ("PROX.BR",     "Proximus"),
        ("ILD.PA",      "Iliad"),
        ("NRJ.PA",      "NRJ Group"),
        ("TFI.PA",      "TF1"),
        ("M6.PA",       "M6 Metropole Television"),
        ("VIV.PA",      "Vivendi"),
    ],
    "Materiaux & Chimie": [
        ("AIR.PA",      "Airbus"),
        ("ARKEMA.PA",   "Arkema"),
        ("VK.PA",       "Vallourec"),
        ("SOLB.BR",     "Solvay"),
        ("UMI.BR",      "Umicore"),
        ("TITAN.AT",    "Titan Cement"),
        ("HEI.DE",      "Heidelberg Materials"),
        ("SDF.DE",      "K+S"),
        ("WACKER.DE",   "Wacker Chemie"),
        ("BASF.DE",     "BASF"),
        ("LANXE.DE",    "Lanxess"),
        ("DEINF.PA",    "Derichebourg"),
        ("LIN.DE",      "Linde"),
    ],
}

SECTORS = list(PEA_UNIVERSE.keys())
TOP_PER_SECTOR = 3
FINAL_COUNT = 20


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

    if price > MAX_PRICE:
        print(f"    Elimine (prix {price:.2f}EUR > {MAX_PRICE}EUR) : {ticker}")
        return None

    ma20  = float(close.rolling(20).mean().iloc[-1])
    ma50  = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    rsi   = compute_rsi(close)
    macd_val, macd_sig, macd_hist = compute_macd(close)
    bb_up, bb_low, bb_pos = compute_bollinger(close)
    atr   = compute_atr(hist)
    atr_pct = round(atr / price * 100, 2)
    trend = compute_trend_slope(close, 30)

    hist3m  = hist.tail(63)
    support = float(hist3m["Low"].min())
    resist  = float(hist3m["High"].max())

    chg1d = float((price / close.iloc[-2]  - 1) * 100) if len(close) >= 2  else 0.0
    chg1m = float((price / close.iloc[-22] - 1) * 100) if len(close) >= 22 else 0.0
    chg3m = float((price / close.iloc[-63] - 1) * 100) if len(close) >= 63 else 0.0

    vol20   = float(hist["Volume"].rolling(20).mean().iloc[-1])
    vol_rel = float(hist["Volume"].iloc[-1] / vol20) if vol20 > 0 else 1.0

    # Score technique (0-45)
    ts = 0
    if ma50  and price > ma50:  ts += 10
    if ma200 and price > ma200: ts += 10
    elif not ma200 and price > ma20: ts += 5

    if 40 <= rsi <= 60:   ts += 12
    elif 30 <= rsi < 40:  ts += 10
    elif rsi < 30:        ts += 8
    elif 60 < rsi <= 70:  ts += 6
    else:                 ts += 2

    if macd_hist > 0 and macd_val > macd_sig: ts += 8
    elif macd_hist > 0:                       ts += 4

    if bb_pos < 0.25:  ts += 5
    elif bb_pos < 0.5: ts += 3

    if trend > 3:   ts += 5
    elif trend > 0: ts += 2

    ts = min(45, ts)

    # Score fondamental (0-40)
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

    # Score momentum (0-15)
    ms = 0
    if chg1m > 3:    ms += 5
    elif chg1m > 0:  ms += 2
    if chg3m > 5:    ms += 5
    elif chg3m > 0:  ms += 2
    if vol_rel > 1.3:    ms += 5
    elif vol_rel > 1.0:  ms += 2
    ms = min(15, ms)

    total = ts + fs + ms

    # Niveaux de trading
    atr_stop     = price - 2.5 * atr
    support_stop = support * 0.975
    stop_loss    = round(max(atr_stop, support_stop), 2)

    if rsi > 65:
        entry     = round(price * 0.98, 2)
        entry_tip = "Attendre pull-back ~2% (RSI eleve)"
    elif rsi < 35:
        entry     = round(price * 1.005, 2)
        entry_tip = "RSI en survente : entree par tranche possible"
    else:
        entry     = round(price, 2)
        entry_tip = "Zone d'entree actuelle correcte"

    obj_trend = price * (1 + max(0.04, trend / 100 * 1.2))
    target_1m = round(min(resist * 0.97, obj_trend), 2)

    risk    = entry - stop_loss
    reward  = target_1m - entry
    rr      = round(reward / risk, 2) if risk > 0 else 0.0

    fees_pct = TR_FEE_TOTAL / entry * 100
    net_gain = round((reward / entry * 100) - fees_pct, 2)

    pertinent = bool(net_gain >= MIN_GAIN_PCT and rr >= 1.0)
    if not pertinent:
        total = max(0, total - 15)

    prices_raw = close.tail(120).tolist()
    prices_6m  = [round(float(p), 2) for p in prices_raw
                  if p is not None and not math.isnan(float(p))]

    return {
        "ticker":     ticker,
        "name":       name,
        "sector":     sector,
        "price":      round(price, 2),
        "chg1d":      round(chg1d, 2),
        "chg1m":      round(chg1m, 2),
        "chg3m":      round(chg3m, 2),
        "ma20":       round(ma20, 2),
        "ma50":       round(ma50, 2) if ma50  else None,
        "ma200":      round(ma200, 2) if ma200 else None,
        "rsi":        round(rsi, 1),
        "macd":       round(macd_val, 3),
        "macd_sig":   round(macd_sig, 3),
        "macd_hist":  round(macd_hist, 3),
        "bb_up":      round(bb_up, 2),
        "bb_low":     round(bb_low, 2),
        "bb_pos":     round(bb_pos, 2),
        "atr":        round(atr, 2),
        "atr_pct":    atr_pct,
        "trend":      round(trend, 2),
        "support":    round(support, 2),
        "resist":     round(resist, 2),
        "vol_rel":    round(vol_rel, 2),
        "pe":         round(pe, 1)       if pe     is not None else None,
        "roe":        round(roe * 100, 1) if roe   is not None else None,
        "upside":     round(upside, 1)   if upside is not None else None,
        "div":        round(div * 100, 2) if div   is not None else None,
        "beta":       round(beta, 2)     if beta   is not None else None,
        "mktcap":     int(mktcap)        if mktcap is not None else None,
        "entry":      entry,
        "entry_tip":  entry_tip,
        "stop_loss":  stop_loss,
        "target_1m":  target_1m,
        "rr":         rr,
        "net_gain":   net_gain,
        "pertinent":  pertinent,
        "score":      total,
        "score_tech": ts,
        "score_fond": fs,
        "score_mom":  ms,
        "prices_6m":  prices_6m,
    }


# ============================================================
# SELECTION PAR SECTEUR
# ============================================================

def select_candidates() -> tuple[list[dict], dict]:
    """
    Retourne :
    - la liste des 20 finalistes (triés par score)
    - un dict sector -> liste des actions scorées du secteur (réserve pour remplacements)
    """
    all_results = []
    sector_reserves = {}  # ticker déjà dans les top N exclus

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

        # La réserve = actions scorées au-delà du top N, dans l'ordre du score
        reserve = sector_results[TOP_PER_SECTOR:]
        sector_reserves[sector] = reserve

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:FINAL_COUNT], sector_reserves


# ============================================================
# APPEL IA
# ============================================================

def call_ai_batch(stocks: list[dict], batch_label: str) -> dict:
    """Appelle l'IA pour un lot d'actions (max 10) et retourne dict ticker -> analyse."""
    if not stocks:
        return {}

    prompt = f"""Tu es un analyste financier expert. Analyse ces {len(stocks)} actions et réponds UNIQUEMENT en JSON valide.

Actions:
{chr(10).join([f"- {s['ticker']} ({s['name']}): PE={s.get('pe','N/A')}, ROE={s.get('roe','N/A')}%, upside={s.get('upside','N/A')}%, trend={s.get('trend','N/A')}%, RSI={s.get('rsi','N/A')}, support={s.get('support','N/A')}, resist={s.get('resist','N/A')}, bb_pos={s.get('bb_pos','N/A')}" for s in stocks])}

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

Réponds avec le JSON complet pour les {len(stocks)} actions sans aucun texte avant ou après."""

    print(f"  Appel MammouthIA ({AI_MODEL}) - {batch_label}...")

    try:
        resp = client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000,
        )
        raw = resp.choices[0].message.content.strip()
        finish_reason = resp.choices[0].finish_reason

        if finish_reason == "length":
            print(f"  AVERTISSEMENT : reponse tronquee ({batch_label})")

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            data = json.loads(raw)
            analyses = data.get("analyses", data) if isinstance(data, dict) else data
            result = {a["ticker"]: a for a in analyses if "ticker" in a}
            print(f"  {len(result)} analyses parsees avec succes ({batch_label})")
            return result

        except json.JSONDecodeError:
            print(f"  JSON incomplet, tentative de recuperation partielle ({batch_label})...")
            import re
            objects = re.findall(r'\{[^{}]{50,}\}', raw)
            analyses = []
            for obj in objects:
                try:
                    parsed = json.loads(obj)
                    if "ticker" in parsed and "signal" in parsed:
                        analyses.append(parsed)
                except:
                    continue
            print(f"  {len(analyses)} analyses recuperees par parsing partiel ({batch_label})")
            return {a["ticker"]: a for a in analyses}

    except Exception as e:
        print(f"  Erreur IA ({batch_label}) : {e}")
        return {}


def get_ai_analysis(candidates: list[dict], sector_reserves: dict) -> tuple[list[dict], dict]:
    """
    1. Analyse les 20 candidats en 2 lots (1-10, 11-20)
    2. Remplace les EVITER par les suivants du secteur, par vagues de max 10
    3. Si réserve épuisée sans trouver ACHETER/SURVEILLER, revient sur la 1ère action du secteur
    4. Retourne la liste finale et le dict ai_map complet
    """
    MAX_BATCH = 10
    MAX_REPLACEMENT_ROUNDS = 5

    # --- Etape 1 : analyse initiale ---
    lot1 = candidates[:10]
    lot2 = candidates[10:]

    print(f"\n  Lot 1 : actions 1-10 ({len(lot1)} actions)")
    ai_map = call_ai_batch(lot1, "actions 1-10")

    print(f"\n  Lot 2 : actions 11-20 ({len(lot2)} actions)")
    ai_map.update(call_ai_batch(lot2, "actions 11-20"))

    # Index des données quantitatives par ticker
    stock_data_map = {s["ticker"]: s for s in candidates}

    # Suivi de l'index courant dans la réserve par secteur
    reserve_index = {sector: 0 for sector in sector_reserves}

    # Première action de chaque secteur dans les candidats initiaux (fallback)
    # On la récupère depuis PEA_UNIVERSE via sector_reserves + candidates
    # Plus simple : on garde la tête de liste scorée par secteur
    sector_first = {}  # secteur -> stock (le mieux scoré du secteur, toutes réserves confondues)
    for s in candidates:
        sec = s["sector"]
        if sec not in sector_first:
            sector_first[sec] = s
    for sec, reserve in sector_reserves.items():
        if sec not in sector_first and reserve:
            sector_first[sec] = reserve[0]

    # Liste finale courante
    final_list = list(candidates)

    # --- Etapes de remplacement ---
    for round_num in range(1, MAX_REPLACEMENT_ROUNDS + 1):
        # Identifier les EVITER dans la liste finale
        to_replace = []
        for i, s in enumerate(final_list):
            ticker = s["ticker"]
            signal = ai_map.get(ticker, {}).get("signal", "SURVEILLER")
            if signal == "EVITER":
                to_replace.append((i, s))

        if not to_replace:
            print(f"\n  Aucun EVITER restant. Remplacement termine apres {round_num - 1} tour(s).")
            break

        print(f"\n  Tour {round_num} : {len(to_replace)} action(s) a remplacer : "
              f"{[x[1]['ticker'] for x in to_replace]}")

        replacements = []
        tickers_already_in_final = {s["ticker"] for s in final_list}

        for idx, evited_stock in to_replace:
            sector = evited_stock["sector"]
            reserve = sector_reserves.get(sector, [])
            ri = reserve_index[sector]

            # Chercher le prochain candidat non déjà présent dans la réserve
            found = None
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
                # Réserve épuisée : fallback sur la 1ère action du secteur
                fallback = sector_first.get(sector)
                if fallback and fallback["ticker"] != evited_stock["ticker"]:
                    print(f"    Reserve epuisee pour {sector}. "
                          f"Fallback sur {fallback['ticker']} ({fallback['name']})")
                    replacements.append((idx, fallback))
                else:
                    # Même action ou pas de fallback : on garde l'EVITER faute de mieux
                    print(f"    Aucun remplacant possible pour {evited_stock['ticker']}, "
                          f"conservation forcee.")
                    replacements.append((idx, evited_stock))

        # Appliquer les remplacements dans final_list
        new_stocks_to_analyze = []
        for idx, new_stock in replacements:
            old_ticker = final_list[idx]["ticker"]
            if new_stock["ticker"] != old_ticker:
                final_list[idx] = new_stock
                # N'analyser que si pas déjà dans ai_map
                if new_stock["ticker"] not in ai_map:
                    new_stocks_to_analyze.append(new_stock)

        if not new_stocks_to_analyze:
            print(f"  Aucun nouveau stock a analyser au tour {round_num}.")
            break

        # Analyser les nouveaux stocks en lots de max 10
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
# ============================================================

def save_results(stocks: list[dict], ai_map: dict):
    output = []
    for s in stocks:
        ai = ai_map.get(s["ticker"], {})
        signal = ai.get("signal", "SURVEILLER")
        output.append({
            **s,
            "signal":     signal,
            "conviction": ai.get("conviction", 3),
            "resume":     ai.get("resume", "Donnees fondamentales en cours de chargement."),
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
    candidates, sector_reserves = select_candidates()
    print(f"\n  {len(candidates)} actions selectionnees")

    print("\nETAPE 3 : Analyse IA avec remplacement des EVITER...")
    final_list, ai_map = get_ai_analysis(candidates, sector_reserves)
    print(f"  {len(ai_map)} analyses IA recues au total")

    print("\nETAPE 4 : Sauvegarde...")
    save_results(final_list, ai_map)

    print("\nTermine !")
    print("=" * 60)


if __name__ == "__main__":
    main()
