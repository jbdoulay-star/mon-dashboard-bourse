# ============================================================
#  Screener PEA Pro — v2.2
#  - ISIN récupéré via yfinance
#  - 5 tickers pour les tests
#  - Analyses IA enrichies (3 lignes + prix d'entrée obligatoire)
#  - Gestion robuste des erreurs IA
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
MAX_WORKERS        = 4
MODEL              = "gpt-4o-mini"   # Modèle léger pour les tests
BATCH_SIZE         = 5

# ─────────────────────────────────────────
#  3. TICKERS — Limité à 5 pour les tests
# ─────────────────────────────────────────
BASE_TICKERS = [
    "CAP.PA",   # Capgemini
    "OR.PA",    # L'Oréal
    "TTE.PA",   # TotalEnergies
    "BNP.PA",   # BNP Paribas
    "AI.PA",    # Air Liquide
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
        json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f, ensure_ascii=False, indent=2)
    log.info(f"💾 Cache sauvegardé ({len(data)} actions).")

# ─────────────────────────────────────────
#  5. RÉCUPÉRATION DONNÉES MARCHÉ + ISIN
# ─────────────────────────────────────────
def get_isin(ticker_obj, symbol: str) -> str:
    """
    Tente de récupérer l'ISIN via plusieurs méthodes yfinance.
    """
    try:
        info = ticker_obj.info

        # Méthode 1 : champ direct 'isin'
        if info.get("isin"):
            return info["isin"]

        # Méthode 2 : yfinance .isin (propriété dédiée)
        isin_val = ticker_obj.isin
        if isin_val and isin_val != "-":
            return isin_val

    except Exception as e:
        log.warning(f"ISIN non trouvé pour {symbol} : {e}")

    return "N/A"

def fetch_ticker_data(symbol: str) -> Optional[dict]:
    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info

        # Prix
        price = (
            info.get("currentPrice") or
            info.get("regularMarketPrice") or
            info.get("previousClose", 0)
        )

        # Historique 6 mois pour sparkline + variation
        hist = ticker.history(period="6mo")
        if hist.empty:
            log.warning(f"⚠️ Pas d'historique pour {symbol}")
            return None

        prices_6m  = hist["Close"].tolist()
        variation  = ((prices_6m[-1] - prices_6m[0]) / prices_6m[0] * 100) if prices_6m[0] else 0

        # RSI (14 périodes)
        closes = hist["Close"]
        delta  = closes.diff()
        gain   = delta.clip(lower=0).rolling(14).mean()
        loss   = (-delta.clip(upper=0)).rolling(14).mean()
        rs     = gain / loss
        rsi    = float((100 - (100 / (1 + rs))).iloc[-1]) if not rs.empty else 50.0

        # MACD
        ema12  = closes.ewm(span=12).mean()
        ema26  = closes.ewm(span=26).mean()
        macd   = float((ema12 - ema26).iloc[-1])

        # Données fondamentales
        per    = info.get("trailingPE", None)
        high52 = info.get("fiftyTwoWeekHigh", price)
        low52  = info.get("fiftyTwoWeekLow",  price)
        name   = info.get("longName") or info.get("shortName", symbol)
        currency = info.get("currency", "EUR")

        # ── ISIN ──────────────────────────────────────────────
        isin = get_isin(ticker, symbol)

        # Pré-score local (heuristique rapide, sans IA)
        pre_score = 50
        if rsi < 35:   pre_score += 15
        elif rsi > 70: pre_score -= 15
        if variation > 5:  pre_score += 10
        elif variation < -10: pre_score -= 10
        if per and 8 < per < 20: pre_score += 10
        pre_score = max(0, min(100, pre_score))

        log.info(f"✅ {symbol} | Prix: {price:.2f} | RSI: {rsi:.1f} | ISIN: {isin}")

        return {
            "symbol":     symbol,
            "name":       name,
            "isin":       isin,
            "price":      round(price, 2),
            "currency":   currency,
            "variation":  round(variation, 2),
            "rsi":        round(rsi, 2),
            "macd":       round(macd, 4),
            "per":        round(per, 1) if per else None,
            "high52":     round(high52, 2),
            "low52":      round(low52, 2),
            "prices_6m":  [round(p, 2) for p in prices_6m],
            "pre_score":  pre_score,
            # Champs IA (remplis après)
            "sante":      "N/A",
            "tendance":   "N/A",
            "conseil":    "N/A",
            "prix_entree": "N/A",
            "score":      pre_score,
        }

    except Exception as e:
        log.error(f"❌ Erreur fetch {symbol} : {e}")
        return None

def fetch_all_tickers(symbols: list) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_ticker_data, s): s for s in symbols}
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    results.sort(key=lambda x: x["pre_score"], reverse=True)
    log.info(f"📊 {len(results)}/{len(symbols)} tickers récupérés.")
    return results

# ─────────────────────────────────────────
#  6. PROMPT IA — Enrichi
# ─────────────────────────────────────────
def build_prompt_batch(batch: list) -> str:
    """
    Prompt compact mais riche :
    - Santé fondamentale : 2-3 phrases
    - Tendance chartiste : 2-3 phrases
    - Conseil : recommandation qualitative + prix d'entrée obligatoire
    - Score : 0-100
    """
    lines = []
    for d in batch:
        per_str = f"PER={d['per']}" if d['per'] else "PER=N/D"
        lines.append(
            f"#{d['symbol']} | {d['name']} | Prix={d['price']}{d['currency']} | "
            f"{per_str} | RSI={d['rsi']} | MACD={'↑' if d['macd']>0 else '↓'} | "
            f"52s=[{d['low52']}-{d['high52']}] | Var6M={d['variation']:+.1f}%"
        )

    actions_str = "\n".join(lines)

    return f"""Tu es un analyste financier expert en actions européennes éligibles PEA.
Analyse les {len(batch)} actions suivantes et réponds UNIQUEMENT avec le format demandé.

DONNÉES :
{actions_str}

FORMAT DE RÉPONSE OBLIGATOIRE (répète pour chaque action) :
===SYMBOL===
[SANTE]: 2 à 3 phrases sur la santé fondamentale : rentabilité, valorisation (PER), forces/faiblesses.
[TENDANCE]: 2 à 3 phrases sur la tendance chartiste : RSI, MACD, supports/résistances, momentum.
[CONSEIL]: Une recommandation claire (Acheter / Renforcer / Attendre / Éviter) + la tactique précise (ex: attendre consolidation sur support, valider cassure des X€, renforcer sur repli vers X€). OBLIGATOIRE : terminer par "Prix d'entrée conseillé : X.XX€"
[SCORE]: Note de 0 à 100 (entier uniquement)

RÈGLES :
- Chaque section = 2-3 phrases maximum, claires et précises
- Le prix d'entrée DOIT toujours être fourni, même si c'est le prix actuel
- Pas de markdown, pas de bullet points, texte simple
- Répondre pour CHAQUE action sans exception
"""

def parse_batch_response(text: str, batch: list) -> dict:
    results = {}
    for d in batch:
        symbol = d["symbol"]
        try:
            # Cherche le bloc de chaque action
            pattern = rf"===\s*{re.escape(symbol)}\s*===(.*?)(?====|\Z)"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

            if not match:
                log.warning(f"⚠️ Bloc introuvable pour {symbol}")
                results[symbol] = _default_result(d)
                continue

            bloc = match.group(1)

            sante    = _extract(bloc, "SANTE")
            tendance = _extract(bloc, "TENDANCE")
            conseil_full = _extract(bloc, "CONSEIL")
            score_str    = _extract(bloc, "SCORE")

            # Extraire le prix d'entrée du conseil
            prix_match = re.search(
                r"[Pp]rix\s+d['']entr[ée]e?\s+conseill[ée]?\s*[:=]?\s*([0-9]+[.,][0-9]+)\s*€?",
                conseil_full
            )
            prix_entree = prix_match.group(1).replace(",", ".") + "€" if prix_match else f"{d['price']}€"

            # Nettoyer le conseil (retirer la ligne prix d'entrée pour éviter doublon)
            conseil_clean = re.sub(
                r"[Pp]rix\s+d['']entr[ée]e?\s+conseill[ée]?\s*[:=]?\s*[0-9]+[.,][0-9]+\s*€?\.?",
                "", conseil_full
            ).strip()

            try:
                score = int(re.search(r"\d+", score_str).group())
                score = max(0, min(100, score))
            except:
                score = d["pre_score"]

            results[symbol] = {
                "sante":       sante    or "Analyse indisponible.",
                "tendance":    tendance or "Analyse indisponible.",
                "conseil":     conseil_clean or "Consulter un conseiller.",
                "prix_entree": prix_entree,
                "score":       score,
            }

        except Exception as e:
            log.error(f"❌ Parse erreur {symbol} : {e}")
            results[symbol] = _default_result(d)

    return results

def _extract(text: str, key: str) -> str:
    match = re.search(rf'\[{key}\]\s*:\s*(.*?)(?=\[|$)', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

def _default_result(d: dict) -> dict:
    return {
        "sante":       "Données insuffisantes pour l'analyse.",
        "tendance":    "Données insuffisantes pour l'analyse.",
        "conseil":     "Analyse momentanément indisponible.",
        "prix_entree": f"{d['price']}€",
        "score":       d["pre_score"],
    }

# ─────────────────────────────────────────
#  7. APPELS IA PAR BATCH
# ─────────────────────────────────────────
def call_ai_batch(batch: list) -> dict:
    prompt = build_prompt_batch(batch)
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        text = response.choices[0].message.content
        log.info(f"🤖 Batch IA reçu ({len(batch)} actions, {len(text)} chars)")
        return parse_batch_response(text, batch)
    except Exception as e:
        log.error(f"❌ Erreur API IA : {e}")
        return {d["symbol"]: _default_result(d) for d in batch}

def run_ai_analysis(market_data: list) -> list:
    # Découpage en batches
    batches = [market_data[i:i+BATCH_SIZE] for i in range(0, len(market_data), BATCH_SIZE)]
    log.info(f"🤖 {len(batches)} batch(es) IA à traiter...")

    all_results = {}
    for i, batch in enumerate(batches):
        log.info(f"  → Batch {i+1}/{len(batches)} : {[d['symbol'] for d in batch]}")
        results = call_ai_batch(batch)
        all_results.update(results)

    # Injection des résultats IA dans les données
    for d in market_data:
        ai = all_results.get(d["symbol"], _default_result(d))
        d.update(ai)

    # Tri final par score IA
    market_data.sort(key=lambda x: x["score"], reverse=True)
    return market_data

# ─────────────────────────────────────────
#  8. SPARKLINE
# ─────────────────────────────────────────
def generate_sparkline(prices: list, variation: float) -> str:
    fig, ax = plt.subplots(figsize=(2.5, 0.8))
    color = "#00c896" if variation >= 0 else "#ff4d6d"
    ax.plot(prices, color=color, linewidth=1.5)
    ax.fill_between(range(len(prices)), prices, min(prices), alpha=0.15, color=color)
    ax.axis("off")
    fig.patch.set_alpha(0)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=80)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

# ─────────────────────────────────────────
#  9. GÉNÉRATION HTML
# ─────────────────────────────────────────
def build_html(data_list: list) -> str:
    paris_tz  = pytz.timezone("Europe/Paris")
    update_ts = datetime.now(paris_tz).strftime("%d/%m/%Y à %H:%M")

    rows_html = ""
    for d in data_list:
        sparkline = generate_sparkline(d["prices_6m"], d["variation"])

        var_color = "#00c896" if d["variation"] >= 0 else "#ff4d6d"
        var_arrow = "▲" if d["variation"] >= 0 else "▼"

        rsi_color = "#ff4d6d" if d["rsi"] > 70 else ("#00c896" if d["rsi"] < 30 else "#a0aec0")
        macd_color = "#00c896" if d["macd"] > 0 else "#ff4d6d"
        macd_label = "▲" if d["macd"] > 0 else "▼"

        score     = d["score"]
        score_color = "#00c896" if score >= 70 else ("#f6c90e" if score >= 50 else "#ff4d6d")

        per_str = f"PER {d['per']}" if d['per'] else "PER N/D"

        # ISIN badge
        isin_html = f'<span class="isin-badge">{d["isin"]}</span>' if d["isin"] != "N/A" else '<span class="isin-badge isin-na">ISIN N/D</span>'

        rows_html += f"""
        <tr>
          <!-- Colonne Action -->
          <td class="col-action">
            <div class="ticker-symbol">{d['symbol']}</div>
            <div class="ticker-name">{d['name']}</div>
            {isin_html}
            <div class="ticker-per">{per_str}</div>
          </td>

          <!-- Sparkline + variation -->
          <td class="col-spark">
            <img src="{sparkline}" alt="trend" class="sparkline"/>
            <div style="color:{var_color}; font-size:0.78rem; margin-top:2px; font-weight:600;">
              {var_arrow} {abs(d['variation']):.2f}%
            </div>
          </td>

          <!-- Prix -->
          <td class="col-price">
            <span class="price-value">{d['price']:.2f}€</span>
            <div class="price-range">
              <span style="color:#ff4d6d;">↓{d['low52']:.1f}</span>
              <span style="color:#718096;"> / </span>
              <span style="color:#00c896;">↑{d['high52']:.1f}</span>
            </div>
          </td>

          <!-- Score -->
          <td class="col-score">
            <span class="score-value" style="color:{score_color};">{score}</span>
            <div class="score-label">/ 100</div>
          </td>

          <!-- Indicateurs RSI / MACD -->
          <td class="col-indicators">
            <span class="badge" style="background:rgba(255,255,255,0.06); color:{rsi_color};">
              RSI {d['rsi']:.1f}
            </span>
            <span class="badge" style="background:rgba(255,255,255,0.06); color:{macd_color};">
              MACD {macd_label}
            </span>
          </td>

          <!-- Santé fondamentale -->
          <td class="col-sante">
            <p class="analysis-text">{d['sante']}</p>
          </td>

          <!-- Tendance chartiste -->
          <td class="col-tendance">
            <p class="analysis-text">{d['tendance']}</p>
          </td>

          <!-- Conseil & Prix d'entrée -->
          <td class="col-conseil">
            <div class="conseil-box">
              <div class="prix-entree">🎯 {d['prix_entree']}</div>
              <p class="conseil-text">{d['conseil']}</p>
            </div>
          </td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Screener PEA Pro</title>
  <style>
    /* ── Reset & base ── */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
      background: #0d1117;
      color: #e2e8f0;
      min-height: 100vh;
    }}

    /* ── Header ── */
    .header {{
      background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
      padding: 24px 32px;
      border-bottom: 1px solid rgba(255,255,255,0.08);
    }}
    .header h1 {{
      font-size: 1.6rem;
      font-weight: 700;
      color: #fff;
      letter-spacing: -0.5px;
    }}
    .header h1 span {{ color: #00c896; }}
    .header-meta {{
      font-size: 0.8rem;
      color: #718096;
      margin-top: 4px;
    }}

    /* ── Table ── */
    .table-wrapper {{
      overflow-x: auto;
      padding: 16px 24px 40px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1200px;
    }}
    thead th {{
      background: #161b22;
      color: #718096;
      font-size: 0.72rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      padding: 12px 14px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      text-align: left;
    }}
    tbody tr {{
      border-bottom: 1px solid rgba(255,255,255,0.04);
      transition: background 0.15s;
    }}
    tbody tr:hover {{ background: rgba(255,255,255,0.03); }}
    td {{
      padding: 16px 14px;
      vertical-align: top;
    }}

    /* ── Colonnes ── */
    .col-action    {{ min-width: 160px; }}
    .col-spark     {{ min-width: 120px; text-align: center; }}
    .col-price     {{ min-width: 100px; text-align: right; }}
    .col-score     {{ min-width: 80px;  text-align: center; }}
    .col-indicators{{ min-width: 110px; }}
    .col-sante     {{ min-width: 200px; max-width: 240px; }}
    .col-tendance  {{ min-width: 200px; max-width: 240px; }}
    .col-conseil   {{ min-width: 220px; max-width: 270px; }}

    /* ── Action cell ── */
    .ticker-symbol {{
      font-size: 0.72rem;
      color: #63b3ed;
      font-weight: 700;
      letter-spacing: 0.05em;
      margin-bottom: 2px;
    }}
    .ticker-name {{
      font-size: 0.95rem;
      font-weight: 600;
      color: #e2e8f0;
      margin-bottom: 5px;
    }}
    .isin-badge {{
      display: inline-block;
      font-size: 0.68rem;
      font-family: 'Courier New', monospace;
      color: #a0aec0;
      background: rgba(255,255,255,0.06);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 4px;
      padding: 2px 6px;
      margin-bottom: 4px;
      letter-spacing: 0.04em;
    }}
    .isin-na {{
      color: #4a5568;
      border-color: rgba(255,255,255,0.04);
    }}
    .ticker-per {{
      font-size: 0.72rem;
      color: #718096;
    }}

    /* ── Sparkline ── */
    .sparkline {{ max-width: 110px; }}

    /* ── Prix ── */
    .price-value {{
      font-size: 1.1rem;
      font-weight: 700;
      color: #e2e8f0;
    }}
    .price-range {{
      font-size: 0.72rem;
      margin-top: 4px;
      white-space: nowrap;
    }}

    /* ── Score ── */
    .score-value {{
      font-size: 2rem;
      font-weight: 800;
      line-height: 1;
    }}
    .score-label {{
      font-size: 0.7rem;
      color: #4a5568;
      margin-top: 2px;
    }}

    /* ── Badges RSI/MACD ── */
    .badge {{
      display: inline-block;
      font-size: 0.72rem;
      font-weight: 600;
      border-radius: 4px;
      padding: 3px 7px;
      margin: 2px 0;
      white-space: nowrap;
    }}
    .col-indicators {{ display: flex; flex-direction: column; gap: 4px; }}

    /* ── Textes d'analyse ── */
    .analysis-text {{
      font-size: 0.82rem;
      color: #a0aec0;
      line-height: 1.55;
    }}

    /* ── Conseil box ── */
    .conseil-box {{
      background: rgba(99, 179, 237, 0.06);
      border: 1px solid rgba(99, 179, 237, 0.15);
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .prix-entree {{
      font-size: 1.0rem;
      font-weight: 700;
      color: #00c896;
      margin-bottom: 6px;
      letter-spacing: -0.3px;
    }}
    .conseil-text {{
      font-size: 0.8rem;
      color: #a0aec0;
      line-height: 1.5;
    }}

    /* ── Footer ── */
    .footer {{
      text-align: center;
      padding: 20px;
      font-size: 0.72rem;
      color: #4a5568;
      border-top: 1px solid rgba(255,255,255,0.04);
    }}
  </style>
</head>
<body>

  <div class="header">
    <h1>📈 Screener <span>PEA Pro</span></h1>
    <div class="header-meta">
      Dernière mise à jour : {update_ts} &nbsp;·&nbsp;
      {len(data_list)} valeurs analysées &nbsp;·&nbsp;
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
        {rows_html}
      </tbody>
    </table>
  </div>

  <div class="footer">
    ⚠️ Les analyses sont générées par IA et ne constituent pas un conseil financier. &nbsp;|&nbsp;
    Screener PEA Pro v2.2
  </div>

</body>
</html>"""

# ─────────────────────────────────────────
#  10. POINT D'ENTRÉE
# ─────────────────────────────────────────
if __name__ == "__main__":
    log.info("═══════════════════════════════════════════")
    log.info("   Screener PEA Pro v2.2 — Démarrage       ")
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
