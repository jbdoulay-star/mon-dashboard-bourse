#!/usr/bin/env python3
"""
PEA Tracker - Analyse quotidienne optimisee
- Preselection 100% quantitative (yfinance, aucun cout)
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
        ("TEMN.SW",     "Temenos"),
        ("ALSO.SW",     "Also Holding"),
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
        ("NOVN.SW",     "Novartis"),
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
        ("BALN.SW",     "Baloise"),
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
        ("LHN.SW",      "Holcim"),
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
        ("NESN.SW",     "Nestle"),
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
# FONCTION UTILITAIRE - CORRECTION PRINCIPALE
# ============================================================

def to_float(val):
    """Convertit une valeur en float de manière sécurisée."""
    try:
        if val is None:
            return None
        return float(val)
    except (ValueError, TypeError):
        return None


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

    vol20  = float(hist["Volume"].rolling(20).mean().iloc[-1])
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

    # Score fondamental (0-40) - CORRECTION : to_float() sur toutes les métriques
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

    if roe   is not None and roe > 0.15:     fs += 6
    if rev_g is not None and rev_g > 0.05:   fs += 6
    if earn_g is not None and earn_g > 0.05: fs += 5
    if de    is not None and de < 80:        fs += 3
    if upside is not None and upside > 15:   fs += 5
    elif upside is not None and upside > 8:  fs += 3

    fs = min(40, max(0, fs))

    # Score momentum (0-15)
    ms = 0
    if chg1m > 3:    ms += 5
    elif chg1m > 0:  ms += 2
    if chg3m > 5:    ms += 5
    elif chg3m > 0:  ms += 2
    if vol_rel > 1.3:   ms += 5
    elif vol_rel > 1.0: ms += 2
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

    pertinent = net_gain >= MIN_GAIN_PCT and rr >= 1.0
    if not pertinent:
        total = max(0, total - 15)

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
        "pe":         round(pe, 1) if pe is not None else None,
        "roe":        round(roe * 100, 1) if roe is not None else None,
        "upside":     round(upside, 1) if upside is not None else None,
        "div":        round(div * 100, 2) if div is not None else None,
        "beta":       round(beta, 2) if beta is not None else None,
        "mktcap":     mktcap,
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
        "prices_6m": [round(float(p), 2) for p in close.tail(120).tolist()],
    }


# ============================================================
# SELECTION PAR SECTEUR
# ============================================================

def select_candidates() -> list[dict]:
    all_results = []
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

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:FINAL_COUNT]


# ============================================================
# ANALYSE IA (1 seul appel)
# ============================================================

def build_prompt(stocks: list[dict]) -> str:
    lines = []
    for i, s in enumerate(stocks, 1):
        lines.append(
            f"{i}. {s['name']} ({s['ticker']}) | Score={s['score']}/100 | "
            f"Prix={s['price']}EUR | RSI={s['rsi']} | Tendance={s['trend']:+.1f}% | "
            f"R/R={s['rr']} | Gain net={s['net_gain']}% | "
            f"Entree={s['entry']} | Stop={s['stop_loss']} | Obj={s['target_1m']} | "
            f"PE={s.get('pe','N/A')} | ROE={s.get('roe','N/A')}% | "
            f"Upside analyst={s.get('upside','N/A')}% | Div={s.get('div','N/A')}% | "
            f"MA50={'AU-DESSUS' if s.get('ma50') and s['price'] > s['ma50'] else 'EN-DESSOUS'} | "
            f"BB_pos={s.get('bb_pos','N/A')} | ATR%={s.get('atr_pct','N/A')}"
        )
    stocks_text = "\n".join(lines)

    return f"""Tu es un analyste financier expert en actions europeennes eligibles PEA.

Voici les {len(stocks)} meilleures actions selectionnees aujourd'hui par scoring quantitatif :

{stocks_text}

Pour chacune, fournis en JSON un tableau "analyses" avec ces champs :
- ticker (string)
- signal (string): "ACHETER", "SURVEILLER" ou "EVITER"
- conviction (int): 1 a 5
- resume (string): 1-2 phrases resumant les fondamentaux cles (valorisation, croissance, sante financiere)
- bull_case (string): 1 phrase - raison principale scenario optimiste
- bear_case (string): 1 phrase - raison principale scenario pessimiste
- chartiste (string): 3-4 phrases max : tendance actuelle, conseil precis sur timing entree, zone surveillance, vigilance stop-loss
- conseil (string): conseil operationnel court (1 phrase)

Reponds UNIQUEMENT avec le JSON valide, sans markdown, sans explication."""


def get_ai_analysis(stocks: list[dict]) -> dict:
    if not MAMMOUTH_API_KEY:
        print("  Pas de cle API - analyse IA ignoree")
        return {}

    # Prompt raccourci pour eviter la troncature
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

    print(f"  Appel MammouthIA ({AI_MODEL})...")

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
            print("  AVERTISSEMENT : reponse tronquee par limite de tokens")

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        # Tentative de parsing direct
        try:
            data = json.loads(raw)
            analyses = data.get("analyses", data) if isinstance(data, dict) else data
            result = {a["ticker"]: a for a in analyses if "ticker" in a}
            print(f"  {len(result)} analyses parsees avec succes")
            return result

        except json.JSONDecodeError:
            print("  JSON incomplet, tentative de recuperation partielle...")
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
            print(f"  {len(analyses)} analyses recuperees par parsing partiel")
            return {a["ticker"]: a for a in analyses}

    except Exception as e:
        print(f"  Erreur IA : {e}")
        return {}

# ============================================================
# SAUVEGARDE JSON
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

    os.makedirs("data", exist_ok=True)
    path = "data/selections.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

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
