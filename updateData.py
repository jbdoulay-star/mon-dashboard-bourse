import os
import yfinance as yf
from openai import OpenAI
import re
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
import pytz

# Configuration API
client = OpenAI(
    api_key=os.getenv("MAMMOUTH_API_KEY"),
    base_url="https://api.mammouth.ai/v1"
)

base_tickers = [
    "MC", "OR", "RMS", "TTE", "SAN", "AIR", "AI", "BNP", "DG", "KER",
    "CDI", "EL", "ASML", "SAP", "SIE", "SU", "CS", "ALV", "BMW", "VOW3",
    "BAS", "BAYN", "SAF", "ENGI", "RNO", "GLE", "ACA", "ML", "VIE", "STM",
    "ORA", "LR", "CAP", "DSY", "ATO", "PUB", "BN", "URW", "VIV", "EDEN",
    "GET", "HO", "DIM", "WLN", "GFC", "BOL", "DEC", "COFA", "NEX", "COV",
    "EN", "FDJ", "TEP", "SGO", "FR", "SOP", "MERY", "ICAD", "ERF", "KORI",
    "KPN", "AD.AS"
]

def generate_sparkline(ticker_obj):
    try:
        hist = ticker_obj.history(period="6mo")
        if hist.empty: return ""
        prices = hist['Close']
        median_3m = prices.tail(90).median()
        high, low = prices.max(), prices.min()
        plt.figure(figsize=(3, 1), dpi=80)
        plt.plot(prices.values, color='#3b82f6', linewidth=1.5)
        plt.axhline(y=high, color='#10b981', linestyle='-', linewidth=1, alpha=0.8)
        plt.axhline(y=low, color='#ef4444', linestyle='-', linewidth=1, alpha=0.8)
        plt.axhline(y=median_3m, color='#94a3b8', linestyle='--', linewidth=0.8)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except: return ""

def get_isin(ticker_obj):
    try:
        isin = ticker_obj.isin
        return isin if isin and len(isin) > 5 else ticker_obj.info.get('isin', 'N/A')
    except: return 'N/A'

def get_best_ticker(base):
    for suffix in [".F", ".DE", ".PA"]:
        t = yf.Ticker(base + suffix)
        try:
            if t.fast_info['last_price'] > 0: return t, base + suffix
        except: continue
    return yf.Ticker(base + ".PA"), base + ".PA"

def analyze_stock(symbol, price, info):
    prompt = f"Analyse {symbol} à {price}€. Réponds strictement : SCORE: [0-100], SANTE: [Détail], TENDANCE: [Détail], CONSEIL: [Prix]."
    try:
        response = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content
        score = re.search(r"SCORE:\s*(\d+)", response)
        sante = re.search(r"SANTE:(.*?)(?=TENDANCE:|$)", response, re.DOTALL)
        tendance = re.search(r"TENDANCE:(.*?)(?=CONSEIL:|$)", response, re.DOTALL)
        conseil = re.search(r"CONSEIL:(.*)", response, re.DOTALL)
        return {
            "score": int(score.group(1)) if score else 0,
            "sante": sante.group(1).strip() if sante else "N/A",
            "tendance": tendance.group(1).strip() if tendance else "N/A",
            "conseil": conseil.group(1).strip() if conseil else "N/A"
        }
    except: return {"score": 0, "sante": "N/A", "tendance": "N/A", "conseil": "N/A"}

all_data = []
now = datetime.now(pytz.timezone('Europe/Paris')).strftime("%d/%m/%Y %H:%M")

for base in base_tickers:
    ticker_obj, full_ticker = get_best_ticker(base)
    try:
        price = round(ticker_obj.fast_info['last_price'], 2)
        all_data.append({
            "ticker": full_ticker,
            "nom": ticker_obj.info.get('longName', base),
            "isin": get_isin(ticker_obj),
            "prix": price,
            "chart": generate_sparkline(ticker_obj),
            **analyze_stock(full_ticker, price, ticker_obj.info)
        })
    except: continue

all_data.sort(key=lambda x: x['score'], reverse=True)

# --- GENERATION HTML ---
html_header = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Screener PEA Live</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ background-color: #020617; color: #f8fafc; font-family: 'Inter', sans-serif; }}
        .glass {{ background: rgba(15, 23, 42, 0.9); backdrop-filter: blur(12px); }}
    </style>
</head>
<body class="p-4">
    <div class="max-w-[1700px] mx-auto">
        <div class="mb-10 text-center md:text-left">
            <h1 class="text-5xl font-black bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent italic uppercase tracking-tighter">
                Screener PEA : Analyse & Potentiel
            </h1>
            <p class="text-slate-400 font-bold mt-2">Dernière analyse complète : <span class="text-blue-400">{now}</span></p>
            <div id="live-indicator" class="inline-flex items-center mt-2 px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-500 text-[10px] font-bold uppercase tracking-widest">
                <span class="w-2 h-2 bg-emerald-500 rounded-full animate-pulse mr-2"></span>
                Mise à jour des prix live au chargement...
            </div>
        </div>
        
        <div class="overflow-x-auto rounded-3xl border border-slate-800 shadow-2xl">
            <table class="w-full text-left">
                <thead class="glass sticky top-0 border-b border-slate-800">
                    <tr class="text-[11px] uppercase tracking-widest text-slate-500">
                        <th class="p-6">Ticker & ISIN</th>
                        <th class="p-6">Canal (6 mois)</th>
                        <th class="p-6 text-center">Prix Live</th>
                        <th class="p-6 text-center">Potentiel IA</th>
                        <th class="p-6">Santé Fondamentale</th>
                        <th class="p-6">Tendance Chartiste</th>
                        <th class="p-6">Conseil & Entrée</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-800/40">
"""

html_rows = ""
for d in all_data:
    html_rows += f"""
    <tr class="hover:bg-blue-900/10 transition-colors">
        <td class="p-6">
            <div class="flex flex-col">
                <span class="text-blue-400 font-black text-sm">{d['ticker']}</span>
                <span class="text-white text-sm font-semibold truncate max-w-[180px]">{d['nom']}</span>
                <span class="text-[13px] text-slate-400 font-bold mt-1 tracking-wider">{d['isin']}</span>
            </div>
        </td>
        <td class="p-6">
            <img src="data:image/png;base64,{d['chart']}" class="w-44 h-auto" alt="Chart">
        </td>
        <td class="p-6 text-center">
            <span class="price-val text-xl font-black text-emerald-400" data-symbol="{d['ticker']}">{d['prix']}€</span>
        </td>
        <td class="p-6 text-center">
            <div class="text-2xl font-black text-blue-400">{d['score']}%</div>
        </td>
        <td class="p-6 text-[14px] text-slate-300 leading-relaxed italic min-w-[280px]">
            {d['sante']}
        </td>
        <td class="p-6 text-[14px] text-slate-300 leading-relaxed italic min-w-[280px]">
            {d['tendance']}
        </td>
        <td class="p-6">
            <div class="bg-slate-900/80 p-4 rounded-xl border border-blue-500/20 text-blue-100 text-[13px] font-medium leading-snug">
                {d['conseil']}
            </div>
        </td>
    </tr>
    """

html_footer = """
                </tbody>
            </table>
        </div>
    </div>

    <script>
    // SCRIPT DE MISE À JOUR DES PRIX AU CHARGEMENT (Temps Réel)
    async function updatePrices() {
        const priceCells = document.querySelectorAll('.price-val');
        const status = document.getElementById('live-indicator');
        
        for (let cell of priceCells) {
            const symbol = cell.getAttribute('data-symbol');
            try {
                // Utilisation d'un service proxy pour récupérer le prix live sans quitter la page
                const response = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1m&range=1d`);
                const data = await response.json();
                const livePrice = data.chart.result[0].meta.regularMarketPrice;
                if(livePrice) {
                    cell.innerText = livePrice.toFixed(2) + '€';
                    cell.classList.add('animate-pulse');
                    setTimeout(() => cell.classList.remove('animate-pulse'), 1000);
                }
            } catch (e) { console.error("Erreur prix live pour " + symbol); }
        }
        status.innerHTML = '<span class="w-2 h-2 bg-blue-500 rounded-full mr-2"></span> Prix mis à jour à l\\'instant';
    }
    
    // On lance la mise à jour dès que la page est ouverte
    window.onload = updatePrices;
    </script>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_header + html_rows + html_footer)
