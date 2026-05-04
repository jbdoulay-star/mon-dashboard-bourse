# ============================================================
#  Screener PEA Pro — v2.4
#  - Santé fondamentale enrichie :
#    + Scénario défavorable (risque principal)
#    + Scénario favorable (catalyseur principal)
# ============================================================

import os
import json
import logging
import re
import io
import base64
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pytz
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from openai import OpenAI

# ─────────────────────────────────────────
#  1. LOGS
# ─────────────────────────────────────────
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("logs/screener.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("screener")

# ─────────────────────────────────────────
#  2. CONFIGURATION
# ─────────────────────────────────────────
client = OpenAI(
    api_key=os.getenv("MAMMOUTH_API_KEY"),
    base_url="https://api.mammouth.ai/v1"
)

CACHE_FILE         = "cache_data.json"
CACHE_EXPIRY_HOURS = 6
MAX_WORKERS        = 6
MODEL              = "gpt-4o"
BATCH_SIZE         = 10

# ─────────────────────────────────────────
#  3. TICKERS
# ─────────────────────────────────────────
BASE_TICKERS = [
    # ── CAC 40 / Grandes caps françaises ──────────────────
    "AI.PA",      # Air Liquide
    "AIR.PA",     # Airbus
    "ALO.PA",     # Alstom
    "CS.PA",      # AXA
    "BNP.PA",     # BNP Paribas
    "EN.PA",      # Bouygues
    "CA.PA",      # Carrefour
    "OR.PA",      # L'Oréal
    "MC.PA",      # LVMH
    "ML.PA",      # Michelin
    "RI.PA",      # Pernod Ricard
    "RNO.PA",     # Renault
    "SAF.PA",     # Safran
    "SAN.PA",     # Sanofi
    "SGO.PA",     # Saint-Gobain
    "SU.PA",      # Schneider Electric
    "GLE.PA",     # Société Générale
    "STM.PA",     # STMicroelectronics
    "TTE.PA",     # TotalEnergies
    "DG.PA",      # Vinci
    "HO.PA",      # Thales
    "CAP.PA",     # Capgemini
    "DSY.PA",     # Dassault Systèmes
    "ERF.PA",     # Eurofins Scientific
    "ENGI.PA",    # Engie
    "ELIS.PA",    # Elis

    # ── Mid-caps françaises ───────────────────────────────
    "ERA.PA",     # Eramet
    "EL.PA",      # EssilorLuxottica
    "ENX.PA",     # Euronext
    "ETL.PA",     # Eutelsat
    "NEX.PA",     # Nexans
    "GTT.PA",     # GTT
    "RMS.PA",     # Hermès International
    "IDL.PA",     # ID Logistics
    "NK.PA",      # Imerys
    "SOI.PA",     # Soitec
    "SPIE.PA",    # SPIE
    "TE.PA",      # Technip Energies
    "THEP.PA",    # Thermador Groupe
    "VK.PA",      # Vallourec
    "VU.PA",      # Vusion Group
    "VIR.PA",     # Viridien
    "BLC.PA",     # Bastide Le Confort
    "BDU.PA",     # Bonduelle
    "BVI.PA",     # Bureau Veritas
    "RCF.PA",     # Teleperformance
    "FDE.PA",     # Française Energie
    "MEMS.PA",    # Memscap
    "ALMDG.PA",   # MGI Digital Graphic
    "ALMDT.PA",   # Median Technologies
    "EXENS.PA",   # Exosens
    "EXA.PA",     # Exail Technologies
    "NOA3.PA",    # Nokia (Paris)
    "NAE.PA",     # North Atlantic En.
    "ALAGP.PA",   # Agripower
    "ALRIB.PA",   # Riber

    # ── Valeurs européennes ───────────────────────────────
    "ASML.AS",    # ASML
    "BESI.AS",    # BE Semiconductor
    "PHI1.AS",    # Koninklijke Philips
    "SHL.DE",     # Siemens Healthineers
    "ADS.DE",     # Adidas
    "BMW.DE",     # BMW
    "VOS.DE",     # Vossloh

    # ── ETFs PEA éligibles ────────────────────────────────
    "EWLD.PA",    # Lyxor MSCI World PEA
    "RS2K.PA",    # Amundi Russell 2000 PEA
    "PANX.PA",    # Amundi Nasdaq-100 PEA
    "PUST.PA",    # Lyxor S&P 500 PEA
    "PAEEM.PA",   # Amundi MSCI Emerging PEA
    "PASI.PA",    # Amundi MSCI Asia PEA
    "PINE.PA",    # Amundi India PEA
    "EDEF.PA",    # BNP Easy MSCI Def. EU
]

# ─────────────────────────────────────────
#  4. CACHE
# ─────────────────────────────────────────
def load_cache() -> Optional[list]:
    if not Path(CACHE_FILE).exists():
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        ts = datetime.fromisoformat(cached.get("timestamp", "2000-01-01"))
        if datetime.now() - ts < timedelta(hours=CACHE_EXPIRY_HOURS):
            log.info(f"✅ Cache valide ({CACHE_FILE}), skip API calls.")
            return cached["data"]
        log.info("⏰ Cache expiré, recalcul...")
    except Exception as e:
        log.warning(f"Cache illisible : {e}")
    return None

def save_cache(data: list):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f, ensure_ascii=False)
    log.info(f"💾 Cache sauvegardé ({CACHE_FILE})")

# ─────────────────────────────────────────
#  5. FETCH DONNÉES MARCHÉ
# ─────────────────────────────────────────
def compute_rsi(prices: list, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains  = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)

def pre_score(rsi: float, perf_6m: float, volume_ratio: float) -> int:
    score = 50
    if rsi < 35:
        score += 20
    elif rsi < 45:
        score += 10
    elif rsi > 70:
        score -= 15
    if perf_6m > 15:
        score += 20
    elif perf_6m > 5:
        score += 10
    elif perf_6m < -10:
        score -= 10
    if volume_ratio > 1.5:
        score += 10
    return max(0, min(100, score))

def fetch_ticker(symbol: str) -> Optional[dict]:
    try:
        tk   = yf.Ticker(symbol)
        info = tk.info
        hist = tk.history(period="6mo", interval="1d")

        if hist.empty or len(hist) < 20:
            log.warning(f"⚠️  {symbol} : historique insuffisant")
            return None

        closes       = hist["Close"].tolist()
        volumes      = hist["Volume"].tolist()
        price        = round(closes[-1], 2)
        price_6m_ago = closes[0]
        perf_6m      = round((price - price_6m_ago) / price_6m_ago * 100, 1)
        rsi          = compute_rsi(closes)
        avg_vol      = np.mean(volumes[:-1]) if len(volumes) > 1 else 1
        vol_ratio    = round(volumes[-1] / avg_vol, 2) if avg_vol > 0 else 1.0
        ma20         = round(np.mean(closes[-20:]), 2)
        ma50         = round(np.mean(closes[-50:]), 2) if len(closes) >= 50 else ma20

        isin = info.get("isin", "N/A")
        name = info.get("longName") or info.get("shortName") or symbol

        score = pre_score(rsi, perf_6m, vol_ratio)

        log.info(
            f"  {'🟢' if perf_6m >= 0 else '🔴'} {symbol} | "
            f"Prix: {price} | RSI: {rsi} | ISIN: {isin}"
        )

        return {
            "symbol":    symbol,
            "name":      name,
            "isin":      isin,
            "price":     price,
            "perf_6m":   perf_6m,
            "rsi":       rsi,
            "vol_ratio": vol_ratio,
            "ma20":      ma20,
            "ma50":      ma50,
            "pre_score": score,
            "closes":    closes[-130:],
        }
    except Exception as e:
        log.error(f"❌ {symbol} : {e}")
        return None

def fetch_all_tickers(tickers: list) -> list:
    results = []
    log.info(f"📡 Récupération de {len(tickers)} tickers ({MAX_WORKERS} workers)...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_ticker, s): s for s in tickers}
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)
    log.info(f"✅ {len(results)}/{len(tickers)} tickers récupérés.")
    return results

# ─────────────────────────────────────────
#  6. GRAPHIQUE SPARKLINE
# ─────────────────────────────────────────
def make_sparkline(closes: list, perf: float) -> str:
    fig, ax = plt.subplots(figsize=(3, 0.9))
    color   = "#00c896" if perf >= 0 else "#fc5c7d"
    ax.plot(closes, color=color, linewidth=1.5)
    ax.fill_between(range(len(closes)), closes, min(closes), alpha=0.15, color=color)
    ax.axis("off")
    fig.patch.set_alpha(0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=80, bbox_inches="tight", transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ─────────────────────────────────────────
#  7. ANALYSES IA
# ─────────────────────────────────────────
def _default_result(d: dict) -> dict:
    return {
        "sante":       "Analyse momentanément indisponible.",
        "risque":      "Indisponible.",
        "catalyseur":  "Indisponible.",
        "tendance":    "Analyse momentanément indisponible.",
        "conseil":     "Veuillez réessayer ultérieurement.",
        "prix_entree": str(d.get("price", "N/A")) + "€",
        "score":       d.get("pre_score", 50),
    }

def _extract_field(text: str, field: str) -> str:
    pattern = rf"{field}\s*:\s*(.+?)(?=\n[A-Z_]+\s*:|$)"
    match   = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def build_prompt_batch(batch: list) -> str:
    lines = []
    for d in batch:
        lines.append(
            "- " + d["symbol"] + " (" + d["name"] + ")"
            + " | Prix: " + str(d["price"]) + "€"
            + " | Perf6M: " + str(d["perf_6m"]) + "%"
            + " | RSI: " + str(d["rsi"])
            + " | MA20: " + str(d["ma20"])
            + " | MA50: " + str(d["ma50"])
            + " | VolRatio: " + str(d["vol_ratio"])
        )

    prompt = (
        "Tu es un analyste financier expert en bourse européenne et PEA français.\n"
        "Analyse les actions suivantes et réponds UNIQUEMENT dans ce format exact pour chaque action :\n\n"
        "SYMBOL: <ticker>\n"
        "SANTE: <1 phrase synthétique sur la santé fondamentale : rentabilité, valorisation, bilan>\n"
        "RISQUE: <1 phrase sur le principal risque ou scénario défavorable>\n"
        "CATALYSEUR: <1 phrase sur le principal catalyseur ou scénario très favorable>\n"
        "TENDANCE: <1 phrase sur la tendance technique/chartiste : momentum, supports, résistances>\n"
        "CONSEIL: <1 phrase de conseil opérationnel précis>\n"
        "PRIX_ENTREE: <prix d'entrée recommandé en euros, ex: 145.50€>\n"
        "SCORE: <score global de 0 à 100>\n\n"
        "Actions à analyser :\n"
        + "\n".join(lines)
        + "\n\nRéponds en français. Sois précis et concis. "
        + "Respecte scrupuleusement le format ci-dessus pour chaque action."
    )
    return prompt

def parse_batch_response(text: str, batch: list) -> dict:
    results = {}
    blocks  = re.split(r"(?=SYMBOL\s*:)", text, flags=re.IGNORECASE)
    for block in blocks:
        sym_match = re.search(r"SYMBOL\s*:\s*(\S+)", block, re.IGNORECASE)
        if not sym_match:
            continue
        symbol = sym_match.group(1).strip().upper()

        sante       = _extract_field(block, "SANTE")
        risque      = _extract_field(block, "RISQUE")
        catalyseur  = _extract_field(block, "CATALYSEUR")
        tendance    = _extract_field(block, "TENDANCE")
        conseil     = _extract_field(block, "CONSEIL")
        prix_entree = _extract_field(block, "PRIX_ENTREE")
        score_str   = _extract_field(block, "SCORE")

        try:
            score = int(re.search(r"\d+", score_str).group())
        except Exception:
            score = 50

        results[symbol] = {
            "sante":       sante       or "N/A",
            "risque":      risque      or "N/A",
            "catalyseur":  catalyseur  or "N/A",
            "tendance":    tendance    or "N/A",
            "conseil":     conseil     or "N/A",
            "prix_entree": prix_entree or "N/A",
            "score":       score,
        }
    return results

def call_ai_batch(batch: list) -> list:
    prompt = build_prompt_batch(batch)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000,
        )
        text    = response.choices[0].message.content
        parsed  = parse_batch_response(text, batch)
        results = []
        for d in batch:
            sym = d["symbol"].upper()
            ai  = parsed.get(sym) or _default_result(d)
            results.append({**d, **ai})
        return results
    except Exception as e:
        log.error(f"❌ Erreur API IA : {e}")
        return [{**d, **_default_result(d)} for d in batch]

def run_ai_analysis(market_data: list) -> list:
    batches = [
        market_data[i:i + BATCH_SIZE]
        for i in range(0, len(market_data), BATCH_SIZE)
    ]
    log.info(f"🤖 {len(batches)} batch(s) IA à traiter...")
    all_results = []
    for idx, batch in enumerate(batches):
        syms = [d["symbol"] for d in batch]
        log.info(f"  Batch {idx+1}/{len(batches)} — {syms}")
        results = call_ai_batch(batch)
        all_results.extend(results)
    all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return all_results

# ─────────────────────────────────────────
#  8. GÉNÉRATION HTML
# ─────────────────────────────────────────
def score_color(score: int) -> str:
    if score >= 75:
        return "#00c896"
    elif score >= 50:
        return "#f6c90e"
    else:
        return "#fc5c7d"

def perf_badge(perf: float) -> str:
    color = "#00c896" if perf >= 0 else "#fc5c7d"
    sign  = "+" if perf >= 0 else ""
    return (
        '<span style="background:' + color + '22; color:' + color + ';'
        'padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:600;">'
        + sign + str(perf) + '%</span>'
    )

def build_row(d: dict) -> str:
    spark = make_sparkline(d["closes"], d["perf_6m"])
    sc    = d.get("score", 50)
    clr   = score_color(sc)

    return """
    <tr>
      <td>
        <div class="stock-name">""" + d["name"] + """</div>
        <div class="stock-sub">
          """ + d["symbol"] + """ &nbsp;·&nbsp;
          <span class="isin">""" + str(d.get("isin", "N/A")) + """</span>
        </div>
      </td>
      <td>
        <img src="data:image/png;base64,""" + spark + """"
             alt="spark" style="height:36px;"/>
      </td>
      <td style="text-align:right;">
        <div style="font-size:1.05rem;font-weight:700;color:#e2e8f0;">
          """ + str(d["price"]) + """€
        </div>
        <div style="margin-top:4px;">""" + perf_badge(d["perf_6m"]) + """</div>
      </td>
      <td style="text-align:center;">
        <div class="score-circle"
             style="border-color:""" + clr + """;color:""" + clr + """;">
          """ + str(sc) + """
        </div>
      </td>
      <td>
        <div class="pill">RSI <b>""" + str(d["rsi"]) + """</b></div>
        <div class="pill">MA20 <b>""" + str(d["ma20"]) + """€</b></div>
        <div class="pill">MA50 <b>""" + str(d["ma50"]) + """€</b></div>
        <div class="pill">Vol× <b>""" + str(d["vol_ratio"]) + """</b></div>
      </td>
      <td>
        <div class="sante-box">
          <div class="sante-text">""" + d.get("sante", "N/A") + """</div>
          <div class="scenario-row">
            <span class="scenario-bad">
              ⚠️ <b>Risque :</b> """ + d.get("risque", "N/A") + """
            </span>
            <span class="scenario-good">
              🚀 <b>Catalyseur :</b> """ + d.get("catalyseur", "N/A") + """
            </span>
          </div>
        </div>
      </td>
      <td>
        <div class="analysis-text">""" + d.get("tendance", "N/A") + """</div>
      </td>
      <td>
        <div class="conseil-box">
          <div class="prix-entree">🎯 Entrée : """ + d.get("prix_entree", "N/A") + """</div>
          <div class="conseil-text">""" + d.get("conseil", "N/A") + """</div>
        </div>
      </td>
    </tr>"""

def build_html(data_list: list) -> str:
    update_ts = datetime.now(pytz.timezone("Europe/Paris")).strftime("%d/%m/%Y %H:%M")
    rows_html = "\n".join(build_row(d) for d in data_list)

    return """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Screener PEA Pro</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #0f1117;
      color: #e2e8f0;
      font-family: 'Segoe UI', system-ui, sans-serif;
      min-height: 100vh;
    }

    /* ── Header ── */
    .header {
      background: linear-gradient(135deg, #1a1d2e 0%, #0f1117 100%);
      border-bottom: 1px solid rgba(255,255,255,0.06);
      padding: 28px 40px 20px;
    }
    .header h1 {
      font-size: 1.8rem;
      font-weight: 800;
      letter-spacing: -0.5px;
      color: #f0f4f8;
    }
    .header h1 span { color: #00c896; }
    .header-meta {
      margin-top: 6px;
      font-size: 0.78rem;
      color: #718096;
    }

    /* ── Table wrapper ── */
    .table-wrapper {
      padding: 24px 20px;
      overflow-x: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 1100px;
    }
    thead th {
      background: #1a1d2e;
      color: #718096;
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      white-space: nowrap;
    }
    tbody tr {
      border-bottom: 1px solid rgba(255,255,255,0.04);
      transition: background 0.15s;
    }
    tbody tr:hover { background: rgba(255,255,255,0.03); }
    tbody td {
      padding: 14px;
      vertical-align: top;
    }

    /* ── Stock name ── */
    .stock-name {
      font-weight: 700;
      font-size: 0.92rem;
      color: #f0f4f8;
    }
    .stock-sub {
      font-size: 0.72rem;
      color: #4a5568;
      margin-top: 2px;
    }
    .isin {
      font-family: monospace;
      color: #718096;
    }

    /* ── Score circle ── */
    .score-circle {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 44px;
      height: 44px;
      border-radius: 50%;
      border: 2px solid;
      font-size: 0.85rem;
      font-weight: 800;
    }

    /* ── Pills indicateurs ── */
    .pill {
      display: inline-block;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.07);
      border-radius: 6px;
      padding: 2px 7px;
      font-size: 0.72rem;
      color: #a0aec0;
      margin: 2px 2px 2px 0;
      white-space: nowrap;
    }

    /* ── Santé fondamentale box ── */
    .sante-box {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .sante-text {
      font-size: 0.82rem;
      color: #a0aec0;
      line-height: 1.5;
    }
    .scenario-row {
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .scenario-bad {
      font-size: 0.76rem;
      color: #fc5c7d;
      background: rgba(252, 92, 125, 0.08);
      border: 1px solid rgba(252, 92, 125, 0.2);
      border-radius: 6px;
      padding: 4px 8px;
      line-height: 1.45;
      display: block;
    }
    .scenario-good {
      font-size: 0.76rem;
      color: #00c896;
      background: rgba(0, 200, 150, 0.08);
      border: 1px solid rgba(0, 200, 150, 0.2);
      border-radius: 6px;
      padding: 4px 8px;
      line-height: 1.45;
      display: block;
    }

    /* ── Tendance chartiste ── */
    .analysis-text {
      font-size: 0.82rem;
      color: #a0aec0;
      line-height: 1.55;
    }

    /* ── Conseil box ── */
    .conseil-box {
      background: rgba(99, 179, 237, 0.06);
      border: 1px solid rgba(99, 179, 237, 0.15);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .prix-entree {
      font-size: 1.0rem;
      font-weight: 700;
      color: #00c896;
      margin-bottom: 6px;
      letter-spacing: -0.3px;
    }
    .conseil-text {
      font-size: 0.8rem;
      color: #a0aec0;
      line-height: 1.5;
    }

    /* ── Footer ── */
    .footer {
      text-align: center;
      padding: 20px;
      font-size: 0.72rem;
      color: #4a5568;
      border-top: 1px solid rgba(255,255,255,0.04);
    }
  </style>
</head>
<body>

  <div class="header">
    <h1>📈 Screener <span>PEA Pro</span></h1>
    <div class="header-meta">
      Dernière mise à jour : """ + update_ts + """ &nbsp;·&nbsp;
      """ + str(len(data_list)) + """ valeurs analysées &nbsp;·&nbsp;
      Trié par score IA décroissant
    </div>
  </div>

  <div class="table-wrapper">
    <table>
      <thead>
        <tr>
          <th>Action</th>
          <th>Tendance 6M</th>
          <th style="text-align:right;">Prix</th>
          <th style="text-align:center;">Score</th>
          <th>Indicateurs</th>
          <th>Santé Fondamentale</th>
          <th>Tendance Chartiste</th>
          <th>Conseil &amp; Entrée</th>
        </tr>
      </thead>
      <tbody>
        """ + rows_html + """
      </tbody>
    </table>
  </div>

  <div class="footer">
    ⚠️ Les analyses sont générées par IA et ne constituent pas un conseil financier.
    &nbsp;|&nbsp; Screener PEA Pro v2.4
  </div>

</body>
</html>"""

# ─────────────────────────────────────────
#  9. POINT D'ENTRÉE
# ─────────────────────────────────────────
if __name__ == "__main__":
    log.info("═══════════════════════════════════════════")
    log.info("   Screener PEA Pro v2.4 — Démarrage       ")
    log.info("═══════════════════════════════════════════")

    data_list = load_cache()

    if data_list is None:
        market_data = fetch_all_tickers(BASE_TICKERS)
        data_list   = run_ai_analysis(market_data)
        save_cache(data_list)

    log.info(f"📊 {len(data_list)} actions prêtes.")

    html = build_html(data_list)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    log.info("🌐 index.html généré avec succès !")
    log.info("═══════════════════════════════════════════")
