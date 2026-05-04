# ============================================================
#  Screener PEA Pro — v2.8
#  - Indicateurs : + Volume relatif + ATR (volatilité)
#  - Santé fondamentale : + scénario optimiste & pessimiste
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
    "AI.PA",   "AIR.PA",  "ALO.PA",  "CS.PA",   "BNP.PA",
    "EN.PA",   "CA.PA",   "OR.PA",   "MC.PA",   "ML.PA",
    "RI.PA",   "RNO.PA",  "SAF.PA",  "SAN.PA",  "SGO.PA",
    "SU.PA",   "GLE.PA",  "STM.PA",  "TTE.PA",  "DG.PA",
    "HO.PA",   "CAP.PA",  "DSY.PA",  "ERF.PA",  "ENGI.PA",
    "ELIS.PA",
    # ── Mid-caps françaises ───────────────────────────────
    "ACA.PA",  "LR.PA",   "PUB.PA",  "VIE.PA",  "ORA.PA",
    "VIV.PA",  "KER.PA",  "RMS.PA",  "EL.PA",   "TEP.PA",
    "WLN.PA",  "GTT.PA",
    # ── ETFs PEA éligibles ────────────────────────────────
    "WPEA.PA", "EWLD.PA", "PCEU.PA", "C6E.PA",  "PAEEM.PA",
    "PANX.PA", "PUST.PA", "BNKE.PA", "MWRD.PA",
]

# ─────────────────────────────────────────
#  4. CACHE
# ─────────────────────────────────────────
def load_cache() -> Optional[list]:
    p = Path(CACHE_FILE)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            cached = json.load(f)
        ts = datetime.fromisoformat(cached["timestamp"])
        if datetime.now() - ts < timedelta(hours=CACHE_EXPIRY_HOURS):
            log.info(f"✅ Cache valide ({ts:%H:%M:%S}) — {len(cached['data'])} entrées")
            return cached["data"]
        log.info("♻️  Cache expiré — recalcul...")
    except Exception as e:
        log.warning(f"Cache illisible : {e}")
    return None

def save_cache(data: list):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "data": data},
                  f, ensure_ascii=False, indent=2)
    log.info("💾 Cache sauvegardé.")

# ─────────────────────────────────────────
#  5. DONNÉES MARCHÉ
# ─────────────────────────────────────────
def fetch_ticker(symbol: str) -> Optional[dict]:
    try:
        tk   = yf.Ticker(symbol)
        info = tk.info
        hist = tk.history(period="6mo", interval="1d", auto_adjust=True)
        if hist.empty or len(hist) < 20:
            log.warning(f"⚠️  {symbol} — historique insuffisant")
            return None

        closes  = hist["Close"].dropna().values
        volumes = hist["Volume"].dropna().values
        highs   = hist["High"].dropna().values
        lows    = hist["Low"].dropna().values
        price   = float(closes[-1])

        # ── RSI 14 ──────────────────────────────────────
        def rsi(src, p=14):
            d  = np.diff(src)
            g  = np.where(d > 0, d, 0.0)
            l  = np.where(d < 0, -d, 0.0)
            ag = np.convolve(g, np.ones(p)/p, 'valid')
            al = np.convolve(l, np.ones(p)/p, 'valid')
            rs = np.where(al == 0, 100, ag / (al + 1e-10))
            return float(100 - 100 / (1 + rs[-1]))

        # ── MACD ────────────────────────────────────────
        def ema(src, n):
            k, e = 2/(n+1), src[0]
            for v in src[1:]: e = v*k + e*(1-k)
            return e

        macd_val   = ema(closes, 12) - ema(closes, 26)
        signal_val = ema(closes[-9:], 9) if len(closes) >= 9 else 0.0

        # ── Moyennes mobiles ────────────────────────────
        mm20  = float(np.mean(closes[-20:]))
        mm50  = float(np.mean(closes[-50:]))  if len(closes) >= 50  else mm20
        mm200 = float(np.mean(closes[-200:])) if len(closes) >= 200 else mm50

        # ── Performance 6M ──────────────────────────────
        perf6m = (closes[-1] / closes[0] - 1) * 100

        # ── Volume relatif (5j vs moyenne 20j) ─────────
        vol_5j   = float(np.mean(volumes[-5:]))  if len(volumes) >= 5  else 0.0
        vol_20j  = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else 1.0
        vol_rel  = vol_5j / vol_20j if vol_20j > 0 else 1.0   # >1 = volumes en hausse

        # ── ATR 14 (Average True Range) ─────────────────
        def atr(hi, lo, cl, p=14):
            tr_list = []
            for i in range(1, len(cl)):
                tr_list.append(max(
                    hi[i] - lo[i],
                    abs(hi[i] - cl[i-1]),
                    abs(lo[i] - cl[i-1])
                ))
            return float(np.mean(tr_list[-p:])) if len(tr_list) >= p else 0.0

        atr_val     = atr(highs, lows, closes)
        atr_pct     = (atr_val / price * 100) if price > 0 else 0.0  # ATR en % du prix

        # ── Miniature graphique ─────────────────────────
        fig, ax = plt.subplots(figsize=(3.2, 1.1))
        color   = "#00c896" if closes[-1] >= closes[0] else "#fc5c7d"
        ax.plot(closes, color=color, linewidth=1.4)
        ax.fill_between(range(len(closes)), closes, closes[0],
                        alpha=0.15, color=color)
        ax.axis("off")
        fig.patch.set_alpha(0)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    transparent=True, dpi=72)
        plt.close(fig)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        return {
            "symbol"   : symbol,
            "name"     : info.get("longName", symbol),
            "isin"     : info.get("isin", "N/A"),
            "price"    : price,
            "currency" : info.get("currency", "EUR"),
            "per"      : info.get("trailingPE"),
            "eps"      : info.get("trailingEps"),
            "revenue"  : info.get("totalRevenue"),
            "mktcap"   : info.get("marketCap"),
            "rsi"      : rsi(closes),
            "macd"     : float(macd_val),
            "signal"   : float(signal_val),
            "mm20"     : mm20,
            "mm50"     : mm50,
            "mm200"    : mm200,
            "perf6m"   : float(perf6m),
            "vol_rel"  : vol_rel,
            "atr_pct"  : atr_pct,
            "img_b64"  : img_b64,
            "sector"   : info.get("sector", "N/A"),
        }
    except Exception as e:
        log.error(f"❌ {symbol} — {e}")
        return None

def fetch_all_tickers(symbols: list) -> list:
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_ticker, s): s for s in symbols}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)
                log.info(
                    f"✔  {r['symbol']:12s} | {r['price']:.2f} {r['currency']} "
                    f"| RSI {r['rsi']:.1f} | VolRel {r['vol_rel']:.2f}x "
                    f"| ATR {r['atr_pct']:.2f}%"
                )
    return results

# ─────────────────────────────────────────
#  6. ANALYSE IA — PROMPT v2.8
# ─────────────────────────────────────────
SYSTEM_PROMPT = """Tu es un analyste financier expert en analyse technique et fondamentale.
Tu analyses des actions éligibles au PEA français.
Réponds UNIQUEMENT avec le format demandé, sans texte autour."""

def build_user_prompt(batch: list) -> str:
    lines = []
    for d in batch:
        lines.append(
            f"===SYMBOL={d['symbol']}===\n"
            f"Nom: {d['name']} | Prix: {d['price']:.2f} {d['currency']}\n"
            f"RSI: {d['rsi']:.1f} | MACD: {d['macd']:.3f} | Signal: {d['signal']:.3f}\n"
            f"MM20: {d['mm20']:.2f} | MM50: {d['mm50']:.2f} | MM200: {d['mm200']:.2f}\n"
            f"Perf 6M: {d['perf6m']:.1f}% | PER: {d['per']} | Secteur: {d['sector']}\n"
            f"Volume relatif 5j/20j: {d['vol_rel']:.2f}x | ATR%: {d['atr_pct']:.2f}%\n"
        )
    prompt = "\n".join(lines)
    prompt += """

Pour CHAQUE action, réponds exactement dans ce format (remplace les crochets) :

===SYMBOL===
[SANTE]: 2-3 phrases : rentabilité, PER, forces/faiblesses fondamentales.
[OPTIMISTE]: 1 phrase : principal catalyseur ou scénario haussier fondamental.
[PESSIMISTE]: 1 phrase : principal risque ou scénario baissier fondamental.
[TENDANCE]: 2-3 phrases : RSI, MACD, supports/résistances, momentum.
[GAIN_1AN]: XX.X (gain potentiel estimé sur 1 an en %, chiffre seul)
[RATIO_RR]: X.X (ratio Gain/Risque, chiffre seul)
[CONSEIL]: Recommandation claire (Acheter/Renforcer/Attendre/Éviter) + tactique précise.
[PRIX_ENTREE]: XX.XX (prix d'entrée conseillé, chiffre seul)
[STOP_LOSS]: XX.XX (niveau d'invalidation haussière, chiffre seul, strictement inférieur au prix d'entrée)
[CONTEXTE_STOP]: 1-2 phrases : quel niveau technique est cassé et conséquences.
[SCORE]: 0-100 (entier seul)
"""
    return prompt

def _default_result(d: dict) -> dict:
    return {**d,
        "sante"         : "Données insuffisantes.",
        "optimiste"     : "N/A",
        "pessimiste"    : "N/A",
        "tendance"      : "Analyse indisponible.",
        "gain_1an"      : "N/A",
        "ratio_rr"      : "N/A",
        "conseil"       : "Analyse indisponible.",
        "prix_entree"   : "N/A",
        "stop_loss"     : "N/A",
        "contexte_stop" : "N/A",
        "score"         : 0,
    }

def parse_ai_response(response_text: str, batch: list) -> list:
    results    = []
    symbol_map = {d["symbol"]: d for d in batch}

    blocks = re.split(r'===([A-Z0-9.\-]+)===', response_text)
    i = 1
    while i < len(blocks) - 1:
        sym  = blocks[i].strip()
        body = blocks[i+1]
        i   += 2
        if sym not in symbol_map:
            continue

        d = symbol_map[sym].copy()

        def extract(tag):
            m = re.search(rf'\[{tag}\]:\s*(.+?)(?=\n\[|\Z)', body, re.S)
            return m.group(1).strip() if m else "N/A"

        d["sante"]          = extract("SANTE")
        d["optimiste"]      = extract("OPTIMISTE")
        d["pessimiste"]     = extract("PESSIMISTE")
        d["tendance"]       = extract("TENDANCE")
        d["gain_1an"]       = extract("GAIN_1AN")
        d["ratio_rr"]       = extract("RATIO_RR")
        d["conseil"]        = extract("CONSEIL")
        d["prix_entree"]    = extract("PRIX_ENTREE")
        d["stop_loss"]      = extract("STOP_LOSS")
        d["contexte_stop"]  = extract("CONTEXTE_STOP")

        score_raw = extract("SCORE")
        try:
            d["score"] = int(re.search(r'\d+', score_raw).group())
        except:
            d["score"] = 0

        results.append(d)

    found = {r["symbol"] for r in results}
    for d in batch:
        if d["symbol"] not in found:
            results.append(_default_result(d))

    return results

def run_ai_analysis(market_data: list) -> list:
    all_results = []
    batches = [market_data[i:i+BATCH_SIZE]
               for i in range(0, len(market_data), BATCH_SIZE)]
    log.info(f"🤖 {len(batches)} batch(es) IA — {len(market_data)} actions")

    for idx, batch in enumerate(batches, 1):
        log.info(f"  Batch {idx}/{len(batches)} — {[d['symbol'] for d in batch]}")
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(batch)},
                ],
                temperature=0.3,
                max_tokens=6000,
            )
            text   = resp.choices[0].message.content
            parsed = parse_ai_response(text, batch)
            all_results.extend(parsed)
            log.info(f"  ✔ Batch {idx} — {len(parsed)} résultats")
        except Exception as e:
            log.error(f"  ❌ Batch {idx} erreur IA : {e}")
            for d in batch:
                all_results.append(_default_result(d))

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results

# ─────────────────────────────────────────
#  7. CONSTRUCTION HTML v2.8
# ─────────────────────────────────────────
def score_color(s: int) -> str:
    if s >= 75: return "#00c896"
    if s >= 50: return "#f6ad55"
    return "#fc5c7d"

def rsi_color(r: float) -> str:
    if r >= 70: return "#fc5c7d"
    if r <= 30: return "#00c896"
    return "#a0aec0"

def build_html(data_list: list) -> str:
    tz        = pytz.timezone("Europe/Paris")
    update_ts = datetime.now(tz).strftime("%d/%m/%Y à %H:%M")

    rows_html = ""
    for d in data_list:
        sc    = d.get("score", 0)
        rsi_v = d.get("rsi", 50)
        price = d.get("price", 0)
        curr  = d.get("currency", "EUR")
        sym   = d.get("symbol", "")
        name  = d.get("name", sym)
        isin  = d.get("isin", "N/A")

        perf     = d.get("perf6m", 0)
        perf_col = "#00c896" if perf >= 0 else "#fc5c7d"
        perf_str = f"+{perf:.1f}%" if perf >= 0 else f"{perf:.1f}%"

        macd_v  = d.get("macd", 0)
        sig_v   = d.get("signal", 0)
        mm20    = d.get("mm20", 0)
        mm50    = d.get("mm50", 0)

        # ── Volume relatif ───────────────────────────────
        vol_rel     = d.get("vol_rel", 1.0)
        vol_rel_str = f"{vol_rel:.2f}x"
        # Vert si volumes en hausse (>1.2), rouge si en baisse (<0.8)
        vol_color   = "#00c896" if vol_rel >= 1.2 else "#fc5c7d" if vol_rel <= 0.8 else "#a0aec0"

        # ── ATR % ────────────────────────────────────────
        atr_pct     = d.get("atr_pct", 0.0)
        atr_str     = f"{atr_pct:.2f}%"
        # ATR élevé = forte volatilité = orange/ambre (neutre, informatif)
        atr_color   = "#f6ad55" if atr_pct >= 2.0 else "#a0aec0"

        # ── Gain 1an / Ratio R/R ─────────────────────────
        gain_1an = d.get("gain_1an", "N/A")
        ratio_rr = d.get("ratio_rr", "N/A")

        try:
            g         = float(str(gain_1an).replace("%", "").strip())
            gain_str  = f"+{g:.1f}%" if g >= 0 else f"{g:.1f}%"
            gain_color= "#00c896" if g >= 0 else "#fc5c7d"
        except:
            gain_str  = str(gain_1an)
            gain_color= "#a0aec0"

        try:
            rr       = float(str(ratio_rr).replace("x", "").strip())
            rr_str   = f"{rr:.1f}x"
            rr_color = "#00c896" if rr >= 2 else "#f6ad55" if rr >= 1 else "#fc5c7d"
        except:
            rr_str   = str(ratio_rr)
            rr_color = "#a0aec0"

        img_tag = ""
        if d.get("img_b64"):
            img_tag = (f'<img src="data:image/png;base64,{d["img_b64"]}" '
                       f'style="width:100%;max-width:220px;display:block;margin:4px 0;">')

        # ── Scénarios optimiste / pessimiste ─────────────
        optimiste  = d.get("optimiste",  "N/A")
        pessimiste = d.get("pessimiste", "N/A")

        rows_html += f"""
        <tr>
          <!-- ACTION -->
          <td>
            <div class="stock-name">{name}</div>
            <div class="stock-isin">ISIN : {isin}</div>
            <div class="stock-sym">{sym}</div>
          </td>

          <!-- TENDANCE 6M -->
          <td>
            {img_tag}
            <div style="color:{perf_col};font-weight:600;font-size:0.85rem;
                        text-align:center;">{perf_str}</div>
          </td>

          <!-- PRIX -->
          <td style="text-align:right;">
            <div class="price-val">{price:.2f} {curr}</div>
          </td>

          <!-- SCORE -->
          <td style="text-align:center;">
            <div class="score-badge" style="background:{score_color(sc)};">{sc}</div>
          </td>

          <!-- INDICATEURS -->
          <td>
            <div class="indic-row">
              <span class="indic-label">RSI</span>
              <span class="indic-val" style="color:{rsi_color(rsi_v)};">{rsi_v:.1f}</span>
            </div>
            <div class="indic-row">
              <span class="indic-label">MACD</span>
              <span class="indic-val"
                style="color:{'#00c896' if macd_v > sig_v else '#fc5c7d'};">
                {macd_v:.3f}
              </span>
            </div>
            <div class="indic-row">
              <span class="indic-label">MM20</span>
              <span class="indic-val"
                style="color:{'#00c896' if price > mm20 else '#fc5c7d'};">
                {mm20:.2f}
              </span>
            </div>
            <div class="indic-row">
              <span class="indic-label">MM50</span>
              <span class="indic-val"
                style="color:{'#00c896' if price > mm50 else '#fc5c7d'};">
                {mm50:.2f}
              </span>
            </div>
            <div class="indic-row">
              <span class="indic-label">Vol.Rel</span>
              <span class="indic-val" style="color:{vol_color};"
                title="Volume moyen 5j vs moyenne 20j — >1 = hausse des volumes">
                {vol_rel_str}
              </span>
            </div>
            <div class="indic-row">
              <span class="indic-label">ATR%</span>
              <span class="indic-val" style="color:{atr_color};"
                title="Average True Range en % du prix — mesure la volatilité quotidienne">
                {atr_str}
              </span>
            </div>
          </td>

          <!-- SANTÉ FONDAMENTALE -->
          <td>
            <div class="text-cell">{d.get('sante', 'N/A')}</div>
            <div class="scenario-optimiste">
              🟢 <strong>Optimiste :</strong> {optimiste}
            </div>
            <div class="scenario-pessimiste">
              🔴 <strong>Pessimiste :</strong> {pessimiste}
            </div>
          </td>

          <!-- TENDANCE CHARTISTE -->
          <td>
            <div class="text-cell">{d.get('tendance', 'N/A')}</div>
            <div class="gain-rr-row">
              <span class="gain-badge" style="color:{gain_color};">
                📈 Gain 1an : <strong>{gain_str}</strong>
              </span>
              <span class="rr-badge" style="color:{rr_color};">
                ⚖️ R/R : <strong>{rr_str}</strong>
              </span>
            </div>
          </td>

          <!-- CONSEIL -->
          <td>
            <div class="conseil-box">
              <div class="conseil-text">{d.get('conseil', 'N/A')}</div>
              <div class="entry-val">
                🟢 Entrée : <strong>{d.get('prix_entree', 'N/A')} {curr}</strong>
              </div>
              <div class="stop-val">
                🔴 Stop-loss : <strong>{d.get('stop_loss', 'N/A')} {curr}</strong>
              </div>
              <div class="stop-ctx">{d.get('contexte_stop', 'N/A')}</div>
            </div>
          </td>
        </tr>"""

    return """<!DOCTYPE html>
<html lang="fr">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Screener PEA Pro v2.8</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d1117;
      color: #e6edf3;
      font-family: 'Segoe UI', system-ui, sans-serif;
      font-size: 0.88rem;
    }
    .header {
      background: linear-gradient(135deg, #161b22 0%, #1f2937 100%);
      padding: 22px 32px;
      border-bottom: 1px solid #30363d;
    }
    .header h1 { font-size: 1.6rem; font-weight: 700; color: #e6edf3; }
    .header h1 span { color: #00c896; }
    .header-meta { color: #8b949e; font-size: 0.8rem; margin-top: 4px; }
    .table-wrapper { overflow-x: auto; padding: 16px; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #161b22;
      border-radius: 10px;
      overflow: hidden;
    }
    thead tr {
      background: #1f2937;
      color: #8b949e;
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: .05em;
    }
    th, td {
      padding: 12px 14px;
      border-bottom: 1px solid #21262d;
      vertical-align: top;
    }
    th { font-weight: 600; }
    tr:hover { background: #1c2128; }

    .stock-name { font-weight: 700; color: #e6edf3; font-size: 0.92rem; }
    .stock-isin { color: #58a6ff; font-size: 0.75rem; margin-top: 2px; font-family: monospace; }
    .stock-sym  { color: #8b949e; font-size: 0.75rem; margin-top: 2px; }

    .price-val  { font-weight: 700; font-size: 1.0rem; color: #e6edf3; }

    .score-badge {
      display: inline-block;
      padding: 5px 12px;
      border-radius: 20px;
      font-weight: 700;
      font-size: 1.0rem;
      color: #0d1117;
      min-width: 48px;
      text-align: center;
    }

    .indic-row   { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 3px; }
    .indic-label { color: #8b949e; font-size: 0.78rem; }
    .indic-val   { font-weight: 600; font-size: 0.82rem; cursor: default; }

    .text-cell { color: #c9d1d9; font-size: 0.82rem; line-height: 1.5; }

    /* Scénarios */
    .scenario-optimiste {
      margin-top: 8px;
      font-size: 0.80rem;
      color: #00c896;
      line-height: 1.4;
      border-left: 2px solid #00c89655;
      padding-left: 6px;
    }
    .scenario-pessimiste {
      margin-top: 5px;
      font-size: 0.80rem;
      color: #fc5c7d;
      line-height: 1.4;
      border-left: 2px solid #fc5c7d55;
      padding-left: 6px;
    }

    .gain-rr-row {
      display: flex;
      gap: 10px;
      margin-top: 8px;
      flex-wrap: wrap;
    }
    .gain-badge, .rr-badge {
      font-size: 0.80rem;
      background: #1f2937;
      border-radius: 6px;
      padding: 3px 8px;
    }

    .conseil-box  { display: flex; flex-direction: column; gap: 6px; }
    .conseil-text { color: #c9d1d9; font-size: 0.82rem; line-height: 1.5; }
    .entry-val    { color: #00c896; font-size: 0.82rem; }
    .stop-val     { color: #fc5c7d; font-size: 0.82rem; }
    .stop-ctx {
      color: #8b949e;
      font-size: 0.78rem;
      font-style: italic;
      line-height: 1.4;
      border-left: 2px solid #fc5c7d33;
      padding-left: 6px;
      margin-top: 2px;
    }

    .footer {
      text-align: center;
      color: #484f58;
      font-size: 0.75rem;
      padding: 18px;
      border-top: 1px solid #21262d;
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
          <th>Conseil · Prix d'entrée &amp; Stop-loss</th>
        </tr>
      </thead>
      <tbody>
        """ + rows_html + """
      </tbody>
    </table>
  </div>

  <div class="footer">
    ⚠️ Les analyses sont générées par IA et ne constituent pas un conseil financier.
    &nbsp;|&nbsp; Screener PEA Pro v2.8
  </div>

</body>
</html>"""

# ─────────────────────────────────────────
#  8. POINT D'ENTRÉE
# ─────────────────────────────────────────
if __name__ == "__main__":
    log.info("═══════════════════════════════════════════")
    log.info("   Screener PEA Pro v2.8 — Démarrage       ")
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
