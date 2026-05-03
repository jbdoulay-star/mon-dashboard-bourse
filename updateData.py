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

# Liste des actions
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
    """Génère le graphique 6 mois avec canal et médiane"""
    try:
        hist = ticker_obj.history(period="6mo")
        if hist.empty: return ""
        
        prices = hist['Close']
        median_3m = prices.tail(90).median()
        high = prices.max()
        low = prices.min()

        # Création du graphique ultra-compact
        plt.figure(figsize=(3, 1), dpi=100)
        plt.plot(prices.values, color='#3b82f6', linewidth=2)
        plt.axhline(y=high, color='#10b981', linestyle='-', linewidth=1, alpha=0.8) # Haut canal
        plt.axhline(y=low, color='#ef4444', linestyle='-', linewidth=1, alpha=0.8)  # Bas canal
        plt.axhline(y=median_3m, color='#94a3b8', linestyle='--', linewidth=1)     # Médiane 3m
        
        plt.axis('off')
        plt.gca().set_facecolor('none')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Erreur graphique : {e}")
        return ""

def get_isin(ticker_obj):
    try:
        return ticker_obj.isin if ticker_obj.isin else "N/A"
    except:
        return "N/A"

def get_best_ticker(base):
    for suffix in [".F", ".DE", ".PA"]:
        t = yf.Ticker(base + suffix)
        try:
            # Vérification si le ticker existe
            p = t.fast_info['last_price']
            if p > 0: return t, base + suffix
        except: continue
    return yf.Ticker(base + ".PA"), base + ".PA"

def analyze_stock(symbol, price):
    prompt = f"Analyse {symbol} à {price}€. Réponds avec : SCORE: [0-100], SANTE: [Détail], TENDANCE: [Détail], CONSEIL: [Prix entrée]."
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
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
    except:
        return {"score": 0, "sante": "N/A", "tendance": "N/A", "conseil": "N/A"}

# Collecte des données
all_data = []
now = datetime.now(pytz.timezone('Europe/Paris')).strftime("%d/%m/%Y %H:%M")

for base in base_tickers:
    ticker_obj, full_ticker = get_best_ticker(base)
    try:
        current_price = round(ticker_obj.fast_info['last_price'], 2)
        analysis = analyze_stock(full_ticker, current_price)
        
        all_data.append({
            "ticker": full_ticker,
            "nom": ticker_obj.info.get('longName', base),
            "isin": get_isin(ticker_obj),
            "prix": current_price,
            "chart": generate_sparkline(ticker_obj),
            **analysis
        })
    except:
        continue

# Tri par score décroissant
all_data.sort(key=lambda x: x['score'], reverse=True)

# Génération HTML
html_header = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Screener PEA</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        body {{ background-color: #020617; color: #f8fafc; font-family: 'Inter', sans-serif; }}
        .glass {{ background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(10px); }}
    </style>
</head>
<body class="p-4 md:p-8">
    <div class="max-w-[1700px] mx-auto">
        <header class="mb-10">
            <h1 class="text-5xl font-black bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent italic uppercase tracking-tighter">
                Screener PEA : Analyse & Potentiel
            </h1>
            <div class="flex items-center gap-4 mt-2">
                <p class="text-slate-400 font-bold">Mise à jour analyse : <span class="text-blue-400">{now}</span></p>
                <div id="live-status" class="flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest text-emerald-500 bg-emerald-500/10 px-3 py-1 rounded-full">
                    <span class="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></span>
                    Prix actualisés au chargement
                </div>
            </div>
        </header>

        <div class="overflow-x-auto rounded-3xl border border-slate-800 shadow-2xl">
            <table class="w-full text-left border-collapse">
                <thead>
                    <tr class="glass text-[11px] uppercase tracking-widest text-slate-500 border-b border-slate-800">
                        <th class="p-6">Action & ISIN</th>
                        <th class="p-6 text-center">Graphique (6m)</th>
                        <th class="p-6 text-center">Prix</th>
                        <th class="p-6 text-center">Potentiel</th>
                        <th class="p-6">Santé Fondamentale</th>
                        <th class="p-6">Tendance Chartiste</th>
                        <th class="p-6">Conseil & Entrée</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-800/50">
"""

rows_html = ""
for d in all_data:
    rows_html += f"""
    <tr class="hover:bg-slate-800/30 transition-colors">
        <td class="p-6">
            <div class="flex flex-col">
                <span class="text-blue-400 font-black text-sm">{d['ticker']}</span>
                <span class="text-white font-bold text-sm truncate max-w-[180px]">{d['nom']}</span>
                <span class="text-[13px] text-slate-400 font-bold mt-1">{d['isin']}</span>
            </div>
        </td>
        <td class="p-6 text-center">
            <img src="data:image/png;base64,{d['chart']}" class="w-40 h-auto mx-auto" alt="Chart">
        </td>
        <td class="p-6 text-center">
            <div class="text-xl font-black text-emerald-400 price-tag" data-symbol="{d['ticker']}">{d['prix']}€</div>
        </td>
        <td class="p-6 text-center">
            <div class="text-2xl font-black text-blue-400">{d['score']}%</div>
        </td>
        <td class="p-6 text-[14px] text-slate-300 leading-relaxed italic min-w-[300px]">
            {d['sante']}
        </td>
        <td class="p-6 text-[14px] text-slate-300 leading-relaxed italic border-l border-slate-800/30 min-w-[300px]">
            {d['tendance']}
        </td>
        <td class="p-6 border-l border-slate-800/30">
            <div class="bg-slate-900/90 p-4 rounded-xl border border-blue-500/20 text-blue-100 text-[13px] font-semibold">
                {d['conseil']}
            </div>
        </td>
    </tr>
    """

html_footer = """
                </tbody>
            </table>
        </div>
        <p class="mt-6 text-center text-[10px] text-slate-600 uppercase tracking-widest font-bold">
            Graphique : Vert = Haut 6m | Rouge = Bas 6m | Pointillés = Médiane 3m
        </p>
    </div>

    <script>
    // Tentative de mise à jour des prix en direct au chargement
    async function refreshPrices() {
        const tags = document.querySelectorAll('.price-tag');
        for (let tag of tags) {
            const symbol = tag.getAttribute('data-symbol');
            try {
                const res = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${symbol}?interval=1m&range=1d`);
                const data = await res.json();
                const price = data.chart.result[0].meta.regularMarketPrice;
                if (price) tag.innerText = price.toFixed(2) + '€';
            } catch (e) {
                console.log("Impossible de rafraîchir le prix live pour " + symbol);
            }
        }
        document.getElementById('live-status').innerHTML = '<span class="w-2 h-2 bg-blue-500 rounded-full mr-1"></span> Prix synchronisés';
    }
    window.onload = refreshPrices;
    </script>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_header + rows_html + html_footer)
