# ============================================================
#  Screener PEA Pro — v2.1
#  Optimisation coût IA : batch, prompt court, modèle léger
# ============================================================

import os
import json
import logging
import re
import io
import base64
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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
MAX_WORKERS        = 8

# ── Modèle utilisé selon le score pré-filtré ──────────────
# Les actions peu prometteuses → modèle léger (gpt-4o-mini)
# Les meilleures opportunités  → modèle complet (gpt-4o)
MODEL_LIGHT  = "gpt-4o-mini"   # ~20x moins cher
MODEL_FULL   = "gpt-4o"
SCORE_THRESHOLD = 60            # Seuil pour passer au modèle full

# ── Taille des batches pour l'appel IA groupé ─────────────
BATCH_SIZE = 6                  # 6 actions par appel = ~6x moins d'appels

BASE_TICKERS = [
    "MC.PA",   "OR.PA",   "RMS.PA",  "TTE.PA",  "SAN.PA",
    "AIR.PA",  "AI.PA",   "BNP.PA",  "DG.PA",   "KER.PA",
    "ASML.AS", "SAP.DE",  "SIE.DE",  "SU.PA",   "CS.PA",
    "ALV.DE",  "BMW.DE",  "VOW3.DE", "BAS.DE",  "BAYN.DE",
    "SAF.PA",  "ENGI.PA", "RNO.PA",  "GLE.PA",  "ACA.PA",
    "ML.PA",   "VIE.PA",  "STM.PA",  "ORA.PA",  "CAP.PA",
    "DSY.PA",  "PUB.PA",  "BN.PA",   "URW.PA",  "VIV.PA",
    "EDEN.PA"
]

# ─────────────────────────────────────────
#  3. CACHE
# ─────────────────────────────────────────
def load_cache() -> list | None:
    try:
        if not Path(CACHE_FILE).exists():
            return None
        with open(CACHE_FILE, encoding="utf-8") as f:
            cache = json.load(f)
        age_h = (datetime.now() - datetime.fromisoformat(cache["timestamp"])).seconds / 3600
        if age_h < CACHE_EXPIRY_HOURS:
            log.info(f"✅ Cache valide ({age_h:.1f}h) — chargement sans appel IA.")
            return cache["data"]
        log.info(f"⏰ Cache expiré ({age_h:.1f}h).")
    except Exception as e:
        log.warning(f"Cache illisible : {e}")
    return None


def save_cache(data: list) -> None:
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"timestamp": datetime.now().isoformat(), "data": data}, f, ensure_ascii=False, indent=2)
        log.info(f"💾 Cache sauvegardé ({len(data)} entrées).")
    except Exception as e:
        log.warning(f"Sauvegarde cache impossible : {e}")

# ─────────────────────────────────────────
#  4. INDICATEURS TECHNIQUES
#     (calculés localement, sans IA)
# ─────────────────────────────────────────
def compute_rsi(series, period: int = 14) -> float:
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("inf"))
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2) if not rsi.empty else 50.0


def compute_macd(series) -> dict:
    ema12  = series.ewm(span=12, adjust=False).mean()
    ema26  = series.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    histo  = macd - signal
    return {
        "macd":   round(float(macd.iloc[-1]),   4),
        "signal": round(float(signal.iloc[-1]), 4),
        "histo":  round(float(histo.iloc[-1]),  4),
    }


def compute_sma(series, period: int) -> float:
    sma = series.rolling(period).mean()
    return round(float(sma.iloc[-1]), 2) if len(sma) >= period else 0.0


def pre_score(rsi: float, macd_histo: float, var_6m: float, per) -> int:
    """
    Score pré-IA basé uniquement sur les indicateurs techniques.
    Permet de décider quel modèle IA utiliser (ou même de passer l'IA).
    """
    score = 50

    # RSI
    if rsi < 30:   score += 15   # Survendu = opportunité
    elif rsi > 70: score -= 15   # Suracheté = risque
    elif rsi < 45: score += 5

    # MACD
    if macd_histo > 0: score += 10
    else:              score -= 10

    # Momentum 6 mois
    if var_6m > 10:    score += 10
    elif var_6m > 0:   score += 5
    elif var_6m < -15: score -= 10
    else:              score -= 5

    # PER
    try:
        p = float(per)
        if 10 < p < 20:  score += 10
        elif p > 40:     score -= 10
        elif p < 0:      score -= 5
    except (TypeError, ValueError):
        pass

    return max(0, min(100, score))

# ─────────────────────────────────────────
#  5. COLLECTE DES DONNÉES MARCHÉ
#     (parallélisée, sans appel IA)
# ─────────────────────────────────────────
def fetch_ticker_data(symbol: str) -> dict | None:
    """Récupère toutes les données marché pour un ticker."""
    try:
        tk   = yf.Ticker(symbol)
        info = tk.info
        hist = tk.history(period="6mo")

        if not info or hist.empty:
            log.warning(f"⚠️  {symbol} — données insuffisantes.")
            return None

        curr_price = info.get("regularMarketPrice") or info.get("previousClose", 0)
        if curr_price <= 0:
            log.warning(f"⚠️  {symbol} — prix invalide.")
            return None

        closes    = hist["Close"]
        rsi       = compute_rsi(closes)
        macd      = compute_macd(closes)
        sma20     = compute_sma(closes, 20)
        sma50     = compute_sma(closes, 50)
        var_6m    = round(((closes.iloc[-1] / closes.iloc[0]) - 1) * 100, 2)
        per       = info.get("trailingPE", "N/A")
        p_score   = pre_score(rsi, macd["histo"], var_6m, per)

        log.info(f"📥 {symbol} récupéré — pré-score: {p_score} | RSI: {rsi}")

        return {
            "ticker":    symbol,
            "nom":       info.get("longName", symbol),
            "isin":      info.get("isin", "N/A"),
            "prix":      curr_price,
            "high52":    info.get("fiftyTwoWeekHigh"),
            "low52":     info.get("fiftyTwoWeekLow"),
            "per":       per,
            "var_6m":    var_6m,
            "rsi":       rsi,
            "macd":      macd,
            "sma20":     sma20,
            "sma50":     sma50,
            "pre_score": p_score,
            "hist":      closes.tolist(),   # Sérialisable pour le cache
            "chart":     generate_sparkline(hist, rsi),
        }
    except Exception as e:
        log.error(f"❌ Erreur fetch {symbol}: {e}", exc_info=True)
        return None


def fetch_all_tickers(tickers: list) -> list:
    """Collecte parallèle des données marché (sans IA)."""
    results = []
    log.info(f"📡 Collecte parallèle de {len(tickers)} tickers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_ticker_data, s): s for s in tickers}
        for future in as_completed(futures):
            r = future.result()
            if r:
                results.append(r)
    log.info(f"✅ {len(results)}/{len(tickers)} tickers collectés.")
    return results

# ─────────────────────────────────────────
#  6. SPARKLINE
# ─────────────────────────────────────────
def generate_sparkline(hist, rsi: float = 50) -> str:
    try:
        if hist.empty:
            return ""
        color = "#ef4444" if rsi > 70 else "#10b981" if rsi < 30 else "#3b82f6"
        plt.figure(figsize=(3, 1), dpi=80)
        plt.plot(hist["Close"].values, color=color, linewidth=2)
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception as e:
        log.debug(f"Sparkline error: {e}")
        return ""

# ─────────────────────────────────────────
#  7. APPELS IA OPTIMISÉS
# ─────────────────────────────────────────

# ── 7a. Prompt ultra-compact ──────────────────────────────
def build_prompt_batch(batch: list[dict]) -> str:
    """
    Construit un prompt unique pour un batch de N actions.
    Un seul appel API pour N actions = division du coût par N.
    """
    lines = []
    for d in batch:
        rsi_lbl  = "survendu" if d["rsi"] < 30 else "suracheté" if d["rsi"] > 70 else "neutre"
        macd_lbl = "haussier" if d["macd"]["histo"] > 0 else "baissier"
        lines.append(
            f"- {d['ticker']} | prix:{d['prix']}€ | 52s:[{d['low52']},{d['high52']}]"
            f" | PER:{d['per']} | var6m:{d['var_6m']}%"
            f" | RSI:{d['rsi']}({rsi_lbl}) | MACD:{macd_lbl}"
            f" | SMA20:{d['sma20']} SMA50:{d['sma50']}"
        )

    tickers_str = ", ".join(d["ticker"] for d in batch)
    data_block  = "\n".join(lines)

    # Prompt minimaliste = moins de tokens
    return f"""Expert financier. Analyse ces {len(batch)} actions PEA.
Pour CHAQUE action, réponds avec ce format EXACT (pas d'intro, pas de commentaire) :

TICKER|SANTE|TENDANCE|CONSEIL|SCORE

Règles :
- Une ligne par action
- SANTE, TENDANCE, CONSEIL : max 20 mots chacun
- SCORE : entier 0-100
- Séparateur : | (pipe)

Actions :
{data_block}

Ordre de réponse : {tickers_str}"""


def parse_batch_response(text: str, batch: list[dict]) -> dict:
    """
    Parse la réponse batch et retourne un dict {ticker: {sante, tendance, conseil, score}}.
    Robuste aux lignes malformées.
    """
    results = {}
    ticker_set = {d["ticker"] for d in batch}

    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 5:
            continue
        ticker = parts[0].upper()
        if ticker not in ticker_set:
            continue
        try:
            results[ticker] = {
                "sante":    parts[1],
                "tendance": parts[2],
                "conseil":  parts[3],
                "score":    max(0, min(100, int(re.search(r"\d+", parts[4]).group()))),
            }
        except Exception:
            log.warning(f"Ligne malformée pour {ticker}: {line}")

    # Fallback pour les tickers manquants
    for d in batch:
        if d["ticker"] not in results:
            log.warning(f"⚠️  Pas de réponse IA pour {d['ticker']} — fallback pré-score.")
            results[d["ticker"]] = {
                "sante":    "Données insuffisantes.",
                "tendance": "Analyse indisponible.",
                "conseil":  "Surveiller avant d'entrer.",
                "score":    d["pre_score"],
            }
    return results


def call_ai_batch(batch: list[dict], model: str) -> dict:
    """Appelle l'IA pour un batch d'actions. Retourne le dict parsé."""
    prompt = build_prompt_batch(batch)
    log.info(f"🤖 Appel IA ({model}) — batch de {len(batch)} : {[d['ticker'] for d in batch]}")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,        # Moins de variabilité = réponses plus courtes
            max_tokens=150 * len(batch),  # Budget tokens proportionnel au batch
        )
        raw = response.choices[0].message.content
        log.debug(f"Réponse brute :\n{raw}")
        return parse_batch_response(raw, batch)
    except Exception as e:
        log.error(f"❌ Erreur appel IA batch: {e}", exc_info=True)
        return {d["ticker"]: {
            "sante": "Erreur IA.", "tendance": "N/A",
            "conseil": "N/A", "score": d["pre_score"]
        } for d in batch}


def run_ai_analysis(market_data: list) -> list:
    """
    Stratégie double-modèle :
      1. Toutes les actions → gpt-4o-mini en batch (pré-tri)
      2. Top actions (score > SCORE_THRESHOLD) → gpt-4o pour affiner
    """
    log.info("═══ Phase 1 : Analyse rapide (modèle léger) ═══")

    # ── Phase 1 : Batch sur modèle léger ──────────────
    ai_results = {}
    batches = [market_data[i:i+BATCH_SIZE] for i in range(0, len(market_data), BATCH_SIZE)]
    for i, batch in enumerate(batches):
        log.info(f"Batch {i+1}/{len(batches)}...")
        ai_results.update(call_ai_batch(batch, MODEL_LIGHT))

    # Fusion des données marché + résultats IA phase 1
    for d in market_data:
        ai = ai_results.get(d["ticker"], {})
        d.update({
            "sante":    ai.get("sante",    "N/A"),
            "tendance": ai.get("tendance", "N/A"),
            "conseil":  ai.get("conseil",  "N/A"),
            "score":    ai.get("score",    d["pre_score"]),
        })

    # ── Phase 2 : Affinage sur le top (modèle complet) ──
    top = [d for d in market_data if d["score"] >= SCORE_THRESHOLD]
    log.info(f"═══ Phase 2 : Affinage top {len(top)} actions (modèle complet) ═══")

    if top:
        top_batches = [top[i:i+BATCH_SIZE] for i in range(0, len(top), BATCH_SIZE)]
        top_results = {}
        for i, batch in enumerate(top_batches):
            log.info(f"Top-batch {i+1}/{len(top_batches)}...")
            top_results.update(call_ai_batch(batch, MODEL_FULL))

        # Mise à jour uniquement du top
        for d in market_data:
            if d["ticker"] in top_results:
                ai = top_results[d["ticker"]]
                d.update({
                    "sante":    ai["sante"],
                    "tendance": ai["tendance"],
                    "conseil":  ai["conseil"],
                    "score":    ai["score"],
                })
                log.info(f"✅ {d['ticker']} affiné — score final: {ai['score']}")

    return sorted(market_data, key=lambda x: x["score"], reverse=True)

# ─────────────────────────────────────────
#  8. ESTIMATION DU COÛT
# ─────────────────────────────────────────
def estimate_cost(n_total: int, n_top: int) -> None:
    """Affiche une estimation du coût en tokens."""
    # Approximations : ~200 tokens input + ~150 tokens output par action
    tokens_light = (n_total * 200 + (n_total // BATCH_SIZE + 1) * 100)
    tokens_full  = (n_top   * 200 + (n_top  // BATCH_SIZE + 1) * 100)
    # Prix : gpt-4o-mini $0.15/1M input, $0.60/1M output
    #        gpt-4o      $5.00/1M input, $15.0/1M output
    cost_light = (tokens_light / 1_000_000) * 0.15
    cost_full  = (tokens_full  / 1_000_000) * 5.00
    log.info(f"💰 Estimation coût — Phase 1 (mini): ~${cost_light:.4f} | Phase 2 (full): ~${cost_full:.4f} | Total: ~${cost_light+cost_full:.4f}")

# ─────────────────────────────────────────
#  9. GÉNÉRATION HTML (inchangée)
# ─────────────────────────────────────────
def rsi_badge(rsi: float) -> str:
    if rsi > 70:
        return f'<span class="px-2 py-0.5 rounded-full bg-red-500/20 text-red-400 text-[10px] font-bold">RSI {rsi} ⚠️</span>'
    if rsi < 30:
        return f'<span class="px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 text-[10px] font-bold">RSI {rsi} ✅</span>'
    return f'<span class="px-2 py-0.5 rounded-full bg-slate-600/40 text-slate-400 text-[10px] font-bold">RSI {rsi}</span>'


def macd_badge(histo: float) -> str:
    if histo > 0:
        return '<span class="px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-400 text-[10px] font-bold">MACD 📈</span>'
    return '<span class="px-2 py-0.5 rounded-full bg-red-500/20 text-red-400 text-[10px] font-bold">MACD 📉</span>'


def build_html(data_list: list) -> str:
    date_now = datetime.now(pytz.timezone("Europe/Paris")).strftime("%d/%m/%Y %H:%M")

    header = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Screener PEA Pro v2.1</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ background:#020617; color:white; font-family:'Inter',sans-serif; }}
        .tbl {{ table-layout:fixed; width:100%; }}
        .col-action {{ width:180px; }}
        .col-chart  {{ width:140px; }}
        .col-price  {{ width:100px; }}
        .col-score  {{ width:100px; }}
        .col-indic  {{ width:160px; }}
        .col-flex   {{ width:calc((100% - 680px) / 3); }}
    </style>
</head>
<body class="p-8">
<div class="max-w-[1900px] mx-auto">
    <header class="mb-10 flex justify-between items-end">
        <div>
            <h1 class="text-4xl font-black tracking-tighter text-transparent bg-clip-text
                        bg-gradient-to-r from-blue-400 to-emerald-400 uppercase">
                Screener PEA : Analyse & Potentiel
            </h1>
            <p class="text-slate-500 font-bold text-xs mt-2 uppercase tracking-widest">
                Mise à jour : {date_now} •
                <span id="live-status">Synchro live...</span>
            </p>
        </div>
    </header>
    <div class="bg-slate-900/50 border border-white/10 rounded-3xl overflow-hidden shadow-2xl">
    <table class="tbl">
        <thead>
            <tr class="bg-slate-800/40 text-slate-400 text-[10px] uppercase
                       tracking-[0.2em] border-b border-white/5">
                <th class="p-5 col-action">Action</th>
                <th class="p-5 col-chart text-center">Tendance 6M</th>
                <th class="p-5 col-price text-center">Prix</th>
                <th class="p-5 col-score text-center">Score</th>
                <th class="p-5 col-indic text-center">Indicateurs</th>
                <th class="p-5 col-flex">Santé Fondamentale</th>
                <th class="p-5 col-flex">Tendance Chartiste</th>
                <th class="p-5 col-flex">Conseil & Entrée</th>
            </tr>
        </thead>
        <tbody class="divide-y divide-white/5">
"""

    rows = ""
    for d in data_list:
        var_color = "text-emerald-400" if d["var_6m"] >= 0 else "text-red-400"
        macd_histo = d["macd"]["histo"] if isinstance(d.get("macd"), dict) else d.get("macd_histo", 0)
        rows += f"""
        <tr class="hover:bg-white/[0.02] transition-colors">
            <td class="p-5">
                <div class="flex flex-col">
                    <span class="text-blue-400 font-black text-[10px]">{d['ticker']}</span>
                    <span class="text-[15px] font-bold truncate">{d['nom']}</span>
                    <span class="text-[11px] text-slate-500 font-mono mt-1">{d['isin']}</span>
                </div>
            </td>
            <td class="p-5 text-center">
                <img src="data:image/png;base64,{d['chart']}"
                     class="w-full h-auto opacity-80" alt="Chart">
                <span class="text-[10px] {var_color} font-bold mt-1 block">
                    {'▲' if d['var_6m'] >= 0 else '▼'} {abs(d['var_6m'])}%
                </span>
            </td>
            <td class="p-5 text-center font-black text-[16px] price-tag"
                data-symbol="{d['ticker']}">{d['prix']}€</td>
            <td class="p-5 text-center font-black text-3xl italic text-emerald-400">
                {d['score']}
            </td>
            <td class="p-5 text-center">
                <div class="flex flex-col gap-1 items-center">
                    {rsi_badge(d['rsi'])}
                    {macd_badge(macd_histo)}
                    <span class="text-slate-500 text-[10px] mt-1">{d.get('sma_label','')}</span>
                </div>
            </td>
            <td class="p-5 text-slate-300 text-[13px] leading-relaxed italic">
                {d['sante']}
            </td>
            <td class="p-5 text-slate-300 text-[13px] leading-relaxed italic
                       border-l border-white/5">
                {d['tendance']}
            </td>
            <td class="p-5 border-l border-white/5">
                <div class="bg-blue-500/10 p-4 rounded-xl border border-blue-500/20
                            text-blue-200 text-[13px]">
                    {d['conseil']}
                </div>
            </td>
        </tr>"""

    footer = """
        </tbody>
    </table>
    </div>
</div>
<script>
async function updatePrices() {
    const tags = document.querySelectorAll('.price-tag');
    for (let tag of tags) {
        const sym = tag.getAttribute('data-symbol');
        try {
            const r = await fetch(
                `https://query1.finance.yahoo.com/v8/finance/chart/${sym}?interval=1m&range=1d`
            );
            const j = await r.json();
            const p = j.chart.result[0].meta.regularMarketPrice;
            if (p) tag.innerText = p.toFixed(2) + '€';
        } catch(e) {}
    }
    document.getElementById('live-status').innerHTML =
        '<span class="text-emerald-500">● PRIX SYNCHRONISÉS</span>';
}
window.onload = updatePrices;
</script>
</body>
</html>"""

    return header + rows + footer

# ─────────────────────────────────────────
#  10. POINT D'ENTRÉE
# ─────────────────────────────────────────
if __name__ == "__main__":
    log.info("═══════════════════════════════════════════")
    log.info("   Screener PEA Pro v2.1 — Démarrage       ")
    log.info("═══════════════════════════════════════════")

    data_list = load_cache()

    if data_list is None:
        # Étape 1 : collecte des données marché (parallèle, sans IA)
        market_data = fetch_all_tickers(BASE_TICKERS)

        # Estimation du coût avant d'appeler l'IA
        n_top = len([d for d in market_data if d["pre_score"] >= SCORE_THRESHOLD])
        estimate_cost(len(market_data), n_top)

        # Étape 2 : analyses IA optimisées
        data_list = run_ai_analysis(market_data)

        save_cache(data_list)

    log.info(f"📊 {len(data_list)} actions prêtes.")

    html = build_html(data_list)
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html)

    log.info("🌐 index.html généré avec succès !")
    log.info("═══════════════════════════════════════════")
