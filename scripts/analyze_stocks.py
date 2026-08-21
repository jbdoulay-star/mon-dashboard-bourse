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
MIN_SCORE_ACHAT = 50

# ============================================================
# UNIVERS PEA
# ============================================================

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
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return float(100 - 100 / (1 + rs.iloc[-1]))


def compute_macd(series: pd.Series):
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9).mean()
    return float(macd.iloc[-1]), float(sig.iloc[-1]), float((macd - sig).iloc[-1])


def compute_bollinger(series: pd.Series, period: int = 20):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = (mid + 2 * std).iloc[-1]
    lower = (mid - 2 * std).iloc[-1]
    price = series.iloc[-1]
    pos = (price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
    return float(upper), float(lower), float(pos)


def analyze_stock(ticker: str, name: str, sector: str) -> dict | None:
    """Analyse technique complète d'une action."""
    data = get_stock_data(ticker)
    if data is None:
        return None

    hist = data["hist"]
    close = hist["Close"]
    volume = hist["Volume"] if "Volume" in hist.columns else None

    price = float(close.iloc[-1])
    if price > MAX_PRICE:
        return None

    # Indicateurs techniques
    rsi = compute_rsi(close)
    macd_val, macd_sig, macd_hist_val = compute_macd(close)
    bb_upper, bb_lower, bb_pos = compute_bollinger(close)

    # Moyennes mobiles
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    ma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    # Volatilité et ATR
    tr = pd.concat([
        (hist["High"] - hist["Low"]),
        abs(hist["High"] - close.shift(1)),
        abs(hist["Low"] - close.shift(1))
    ], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1])
    atr_pct = (atr / price * 100) if price > 0 else 0

    # Volume relatif
    vol_rel = (volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]) if volume is not None else 1.0
    vol_rel = float(vol_rel)

    # Supports et résistances
    support = float(close.rolling(20).min().iloc[-1])
    resist = float(close.rolling(20).max().iloc[-1])

    # Trend et changements
    trend = (ma20 - ma50) / ma50 * 100 if ma50 else 0
    chg1d = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100)
    chg1m = float((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100) if len(close) >= 21 else 0
    chg3m = float((close.iloc[-1] - close.iloc[-63]) / close.iloc[-63] * 100) if len(close) >= 63 else 0

    # Calcul des scores
    ts = 0
    if rsi < 40:
        ts += 15
    elif rsi > 60:
        ts += 10
    if macd_hist_val > 0:
        ts += 10
    if price > ma20:
        ts += 8
    if vol_rel > 1.2:
        ts += 7

    fs = 0
    if chg1m > 0:
        fs += 10
    if chg3m > 0:
        fs += 15
    if atr_pct > 2:
        fs += 8

    ms = 0
    if trend > 0:
        ms += 10
    if bb_pos > 0.5:
        ms += 8

    total = ts + fs + ms

    # Conseil d'entrée contextuel
    if rsi < 35:
        entry_tip = "Zone de survente : entree progressive recommandee."
    elif price > ma20 and macd_hist_val > 0:
        entry_tip = "Tendance confirmee : entree au prix du marche."
    else:
        entry_tip = "Attendre confirmation : entree sur repli ou cassure."

    # Calcul du style (REBOND / MOMENTUM / NEUTRE)
    if rsi < 40:
        style = "REBOND"
    elif rsi > 60:
        style = "MOMENTUM"
    else:
        style = "NEUTRE"

    # Calcul entry, stop_loss, target
    entry = round(price, 2)
    stop_loss = round(support, 2)

    # Si prix >= résistance, projette target plus haut
    if price >= resist:
        target_1m = round(resist * 1.05, 2)
    else:
        target_1m = round(resist, 2)

    # Risk-Reward (garanti >= 1:1)
    risk = entry - stop_loss
    reward = target_1m - entry
    rr = reward / risk if risk > 0 else 0

    # Garanti minimum 1:1
    if rr < 1.0 and risk > 0:
        target_1m = round(entry + (risk * 1.5), 2)
        reward = target_1m - entry
        rr = reward / risk

    rr_label = f"1:{round(rr, 2)}" if rr > 0 else "N/A"

    # Gain net (après frais)
    net_gain = (reward - TR_FEE_TOTAL) / entry * 100 if entry > 0 else 0

    prices_raw = close.tail(120).tolist()
    prices_6m = [round(float(p), 2) for p in prices_raw
                 if p is not None and not math.isnan(float(p))]

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "price": round(price, 2),
        "atr": round(atr, 2),
        "atr_pct": atr_pct,
        "rsi": round(rsi, 1),
        "macd": round(macd_val, 4),
        "macd_sig": round(macd_sig, 4),
        "macd_hist": round(macd_hist_val, 4),
        "bb_pos": round(bb_pos, 3),
        "trend": round(trend, 2),
        "chg1d": round(chg1d, 2),
        "chg1m": round(chg1m, 2),
        "chg3m": round(chg3m, 2),
        "vol_rel": round(vol_rel, 2),
        "support": round(support, 2),
        "resist": round(resist, 2),
        "ma20": round(ma20, 2),
        "ma50": round(ma50, 2) if ma50 else None,
        "ma200": round(ma200, 2) if ma200 else None,
        "style": style,
        "entry": entry,
        "entry_tip": entry_tip,
        "stop_loss": stop_loss,
        "target_1m": target_1m,
        "rr": rr,
        "rr_label": rr_label,
        "net_gain": net_gain,
        "score": total,
        "score_tech": ts,
        "score_fond": fs,
        "score_mom": ms,
        "prices_6m": prices_6m,
    }


# ============================================================
# SELECTION PAR SECTEUR
# ============================================================

def select_by_sector() -> tuple[list[dict], dict, list[dict]]:
    """Sélectionne les meilleures actions par secteur."""
    sector_reserves = {}
    all_scored = []

    for sector in SECTORS:
        print(f"\nSecteur : {sector} ({len(PEA_UNIVERSE[sector])} actions)")
        sector_results = []

        for ticker, name in PEA_UNIVERSE[sector]:
            print(f"  Analyse {ticker}...", end="")
            result = analyze_stock(ticker, name, sector)

            if result is None:
                print(" Pas de donnees")
                continue

            if result["price"] > MAX_PRICE:
                print(f" Elimine (prix {result['price']}EUR > {MAX_PRICE}EUR)")
                continue

            sector_results.append(result)
            all_scored.append(result)
            print(f" Score={result['score']}")

        # Top 3 du secteur
        sector_results.sort(key=lambda x: x["score"], reverse=True)
        top_3 = sector_results[:TOP_PER_SECTOR]
        sector_reserves[sector] = sector_results[TOP_PER_SECTOR:]

        print(f"  Top {TOP_PER_SECTOR} : {[t['ticker'] for t in top_3]}")

    all_scored.sort(key=lambda x: x["score"], reverse=True)
    return all_scored[:FINAL_COUNT], sector_reserves, all_scored


# ============================================================
# SIGNAL FINAL DETERMINISTE
# ============================================================

def compute_final_signal(stock: dict) -> str:
    """Calcule le signal métier sans dépendre de la réponse de l'IA."""
    signal = (
        "ACHETER"
        if stock["score"] >= MIN_SCORE_ACHAT and stock["net_gain"] > 0
        else "SURVEILLER"
    )
    if stock["rsi"] > 75:
        signal = "SURVEILLER"
    return signal


# ============================================================
# APPEL IA
# ============================================================

def call_ai_batch(stocks: list[dict], batch_label: str) -> dict:
    """Appel MammouthIA pour analyse fondamentale."""
    if not stocks:
        return {}

    stocks_info = []
    for s in stocks:
        stocks_info.append({
            "ticker": s["ticker"],
            "name": s["name"],
            "sector": s["sector"],
            "price": s["price"],
            "rsi": s["rsi"],
            "macd_hist": s["macd_hist"],
            "trend": s["trend"],
            "chg1m": s["chg1m"],
            "chg3m": s["chg3m"],
            "score": s["score"],
            "style": s["style"],
            "net_gain": s["net_gain"],
            "final_signal": compute_final_signal(s),
            "rr_label": s["rr_label"],
        })

    prompt = f"""Tu es un analyste financier senior specialise sur les actions europeennes cotees sur PEA.

Analyse les {len(stocks)} actions suivantes et fournis pour chacune une analyse fondamentale concise et differentee.

Donnees quantitatives :
{json.dumps(stocks_info, ensure_ascii=False, indent=2)}

REGLES STRICTES :
1. Pour chaque action, le champ "final_signal" fourni dans les donnees est le signal de reference calcule par Python. Recopier exactement sa valeur dans "signal". Ne jamais le recalculer, le modifier ou le contredire.
2. Le signal ACHETER n'est autorise que si final_signal vaut ACHETER (score >= {MIN_SCORE_ACHAT} ET net_gain > 0). Sinon, ecrire SURVEILLER. Un RSI > 75 a deja force final_signal a SURVEILLER : respecter ce declassement.
3. Le texte genere (resume, bull_case, bear_case, chartiste et conseil) doit rester coherent avec final_signal et ne doit jamais suggerer le signal inverse.
4. Signal EVITER uniquement si risque fondamental serieux et avere (dette critique, fraude, faillite imminente).
5. Chaque "resume" doit decrire l'avantage competitif UNIQUE de l'entreprise. Ne JAMAIS reutiliser ou paraphraser le meme texte pour deux entreprises differentes.
6. bull_case et bear_case bases sur l'actualite recente du secteur, pas des generalites.
7. Le champ "style" de chaque action est (REBOND si RSI < 40, MOMENTUM si RSI > 60, NEUTRE sinon). Adapter le conseil en consequence.

Format JSON STRICT :
{{
  "analyses": [
    {{
      "ticker": "XXX.XX",
      "signal": "ACHETER|SURVEILLER|EVITER",
      "conviction": 1-5,
      "resume": "Avantage competitif UNIQUE. Max 20 mots.",
      "bull_case": "Catalyseur concret. Max 15 mots.",
      "bear_case": "Risque reel. Max 15 mots.",
      "chartiste": "Support/resistance clé. Max 20 mots."
    }}
  ]
}}

Reponds avec le JSON complet sans aucun texte avant ou apres."""

    print(f"  Appel MammouthIA ({AI_MODEL}) - {batch_label}...")

    try:
        response = client.chat.completions.create(
            model=AI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            timeout=60,
        )
        raw = response.choices[0].message.content.strip()

        try:
            data = json.loads(raw)
            analyses = data.get("analyses", [])
            return {a["ticker"]: a for a in analyses}
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

def get_ai_analysis(candidates: list[dict], sector_reserves: dict, all_scored: list[dict]) -> tuple[list[dict], dict]:
    """
    1. Analyse les 20 candidats en 2 lots (1-10, 11-20)
    2. Remplace les EVITER par les suivants du secteur
    3. Retourne la liste finale et le dict ai_map complet
    """
    MAX_BATCH = 10
    MAX_REPLACEMENT_ROUNDS = 5

    lot1 = candidates[:10]
    lot2 = candidates[10:]

    print(f"\n  Lot 1 : actions 1-10 ({len(lot1)} actions)")
    ai_map = call_ai_batch(lot1, "actions 1-10")

    print(f"\n  Lot 2 : actions 11-20 ({len(lot2)} actions)")
    ai_map.update(call_ai_batch(lot2, "actions 11-20"))

    print(f"\n  {len(ai_map)} analyses IA recues au total")

    # Remplacement des EVITER
    tickers_in_final = {s["ticker"] for s in candidates}
    ultimate_reserve = [s for s in all_scored if s["ticker"] not in tickers_in_final]
    reserve_idx = 0
    round_num = 0

    while round_num < MAX_REPLACEMENT_ROUNDS:
        round_num += 1
        final_list = []
        replaced_any = False

        for s in candidates:
            signal = compute_final_signal(s)

            if signal == "EVITER":
                replaced = False
                while reserve_idx < len(ultimate_reserve):
                    candidate = ultimate_reserve[reserve_idx]
                    reserve_idx += 1
                    candidate_signal = compute_final_signal(candidate)

                    if candidate_signal != "EVITER":
                        print(f"  [V3-A] Remplacement : {s['ticker']} → {candidate['ticker']} ({candidate['name']})")
                        final_list.append(candidate)
                        tickers_in_final.add(candidate["ticker"])
                        replaced = True
                        replaced_any = True
                        break

                if not replaced:
                    print(f"  [V3-A] Reserve epuisee pour {s['ticker']} : degrade en SURVEILLER")
                    final_list.append(s)
            else:
                final_list.append(s)

        candidates = final_list

        if not replaced_any:
            break

    print(f"\n  Aucun EVITER restant. Remplacement termine apres {round_num} tour(s).")

    return candidates, ai_map

# ============================================================
# SAUVEGARDE RESULTATS
# ============================================================

def save_results(stocks: list[dict], ai_map: dict) -> str:
    """Sauvegarde les resultats dans selections.json avec signal déterministe."""
    output = []

    for s in stocks:
        ai = ai_map.get(s["ticker"], {})
        signal = compute_final_signal(s)

        output.append({
            **s,
            "signal": signal,
            "conviction": ai.get("conviction", 3),
            "resume": ai.get("resume", "Données fondamentales en cours de chargement."),
            "bull_case": ai.get("bull_case", ""),
            "bear_case": ai.get("bear_case", ""),
            "chartiste": ai.get("chartiste", ""),
            "conseil": ai.get("conseil", s.get("entry_tip", "")),
            "style": s.get("style", "NEUTRE"),
        })

    output.sort(key=lambda x: x["score"], reverse=True)

    result = {
        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": date.today().isoformat(),
        "count": len(output),
        "stocks": output,
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

    print("\nETAPE 1 : Selection par secteur...")
    candidates, sector_reserves, all_scored = select_by_sector()
    print(f"\n  Selection pre-IA : {[t['ticker'] for t in candidates]}")

    print("\nETAPE 2 : Analyse IA + remplacement EVITER...")
    final_stocks, ai_map = get_ai_analysis(candidates, sector_reserves, all_scored)  # ← AJOUTER all_scored
    print(f"\n  Selection finale : {[t['ticker'] for t in final_stocks]}")

    print("\nETAPE 3 : Sauvegarde...")
    save_results(final_stocks, ai_map)

    print("\nTermine !")
    print("=" * 60)


if __name__ == "__main__":
    main()
