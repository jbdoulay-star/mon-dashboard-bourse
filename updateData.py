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

# Liste des actions (vous pouvez en ajouter jusqu'à 60+)
base_tickers = [
    "MC", "OR", "RMS", "TTE", "SAN", "AIR", "AI", "BNP", "DG", "KER",
    "CDI", "EL", "ASML", "SAP", "SIE", "SU", "CS", "ALV", "BMW", "VOW3",
    "BAS", "BAYN", "SAF", "ENGI", "RNO", "GLE", "ACA", "ML", "VIE", "STM",
    "ORA", "LR", "CAP", "DSY", "PUB", "BN", "URW", "VIV", "EDEN"
]

def generate_sparkline(ticker_obj):
    try:
        hist = ticker_obj.history(period="6mo")
        if hist.empty: return ""
        prices = hist['Close']
        plt.figure(figsize=(3, 1), dpi=80)
        plt.plot(prices.values, color='#3b82f6', linewidth=2)
        plt.axhline(y=prices.max(), color='#10b981', linestyle='--', alpha=0.3)
        plt.axhline(y=prices.min(), color='#ef4444', linestyle='--', alpha=0.3)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except: return ""

data_list = []
for t in base_tickers:
    symbol = t if "." in t else f"{t}.PA"
    print(f"Analyse approfondie de {symbol}...")
    try:
        tk = yf.Ticker(symbol)
        info = tk.info
        name = info.get('longName', t)
        isin = info.get('isin', 'N/A')
        price = info.get('regularMarketPrice') or info.get('previousClose', 0)
        
        # PROMPT AMÉLIORÉ : On force l'utilisation de balises pour ne plus se tromper
        prompt = f"""Analyse l'action {name} ({symbol}). 
        Réponds strictement avec ce format :
        [SANTE]: (ton analyse fondamentale ici)
        [TENDANCE]: (ton analyse graphique ici)
        [CONSEIL]: (ton conseil d'achat et prix d'entrée ici)
        [SCORE]: (donne un chiffre entre 0 et 100 uniquement)"""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        full_text = response.choices[0].message.content
        
        # Extraction intelligente par balises
        sante = re.search(r"\[SANTE\]:(.*?)\[", full_text, re.S)
        tendance = re.search(r"\[TENDANCE\]:(.*?)\[", full_text, re.S)
        conseil = re.search(r"\[CONSEIL\]:(.*?)\[", full_text, re.S)
        score_find = re.search(r"\[SCORE\]:\s*(\d+)", full_text)

        # Nettoyage et valeurs par défaut si l'IA oublie une balise
        sante_txt = sante.group(1).strip() if sante else "Analyse indisponible."
        tendance_txt = tendance.group(1).strip() if tendance else "Tendance neutre."
        # Si [CONSEIL] est la dernière balise, le regex au dessus peut échouer, on gère :
        if not conseil:
             conseil_txt = full_text.split("[CONSEIL]:")[-1].split("[SCORE]")[0].strip()
        else:
             conseil_txt = conseil.group(1).strip()
        
        score_val = int(score_find.group(1)) if score_find else 50

        data_list.append({
            'ticker': symbol, 'nom': name, 'isin': isin, 'prix': price,
            'score': score_val, 'sante': sante_txt, 'tendance': tendance_txt,
            'conseil': conseil_txt, 'chart': generate_sparkline(tk)
        })
    except Exception as e:
        print(f"Erreur sur {symbol}: {e}")

# --- TRI DES ACTIONS PAR POTENTIEL (Du plus haut au plus bas) ---
data_list.sort(key=lambda x: x['score'], reverse=True)

# Génération HTML
date_now = datetime.now(pytz.timezone('Europe/Paris')).strftime("%d/%m/%Y %H:%M")
html_header = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Screener PEA Live - Expert</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap" rel="stylesheet">
    <style>
        body {{ background: #020617; color: white; font-family: 'Inter', sans-serif; }}
        .glass {{ background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }}
        .score-badge {{ background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%); }}
    </style>
</head>
<body class="p-4 md:p-8 text-[13px]">
    <div class="max-w-[1750px] mx-auto">
        <header class="mb-8 flex justify-between items-center">
            <div>
                <h1 class="text-3xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">SCREENER PEA : TOP OPPORTUNITÉS</h1>
                <p class="text-slate-500 font-bold uppercase tracking-widest text-[10px] mt-1">Mise à jour : {date_now} • <span id="live-status" class="text-emerald-500">Synchronisation Live...</span></p>
            </div>
            <div class="text-right hidden md:block">
                <span class="text-slate-500 text-[10px] block">TRIÉ PAR</span>
                <span class="text-blue-400 font-bold">MEILLEUR POTENTIEL</span>
            </div>
        </header>

        <div class="glass rounded-3xl overflow-hidden shadow-2xl">
            <table class="w-full text-left">
                <thead>
                    <tr class="bg-slate-900/50 text-slate-400 text-[10px] uppercase tracking-widest border-b border-white/5">
                        <th class="p-5">Action</th>
                        <th class="p-5 text-center">Tendance 6M</th>
                        <th class="p-5 text-center">Prix Live</th>
                        <th class="p-5 text-center">Potentiel</th>
                        <th class="p-5">Analyse Fondamentale</th>
                        <th class="p-5">Analyse Chartiste</th>
                        <th class="p-5">Conseil d'Entrée</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-white/5">
"""

rows_html = ""
for d in data_list:
    # Couleur du score
    color = "text-emerald-400" if d['score'] > 70 else "text-blue-400" if d['score'] > 50 else "text-slate-400"
    
    rows_html += f"""
    <tr class="hover:bg-white/[0.02] transition-colors">
        <td class="p-5">
            <div class="flex flex-col">
                <span class="text-blue-400 font-black text-[11px]">{d['ticker']}</span>
                <span class="text-[16px] font-bold tracking-tight">{d['nom']}</span>
                <span class="text-[10px] text-slate-500 font-mono mt-1">{d['isin']}</span>
            </div>
        </td>
        <td class="p-5 text-center">
            <img src="data:image/png;base64,{d['chart']}" class="w-32 h-auto mx-auto brightness-110" alt="Chart">
        </td>
        <td class="p-5 text-center">
            <div class="text-lg font-black price-tag" data-symbol="{d['ticker']}">{d['prix']}€</div>
        </td>
        <td class="p-5 text-center">
            <div class="text-3xl font-black {color} italic">{d['score']}%</div>
        </td>
        <td class="p-5 text-slate-300 leading-relaxed italic max-w-xs">{d['sante']}</td>
        <td class="p-5 text-slate-300 leading-relaxed italic border-l border-white/5 max-w-xs">{d['tendance']}</td>
        <td class="p-5 border-l border-white/5">
            <div class="bg-blue-500/5 p-4 rounded-xl border border-blue-500/10 text-blue-100 font-medium">
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
    async function updatePrices() {
        const tags = document.querySelectorAll('.price-tag');
        for (let tag of tags) {
            const sym = tag.getAttribute('data-symbol');
            try {
                const r = await fetch(`https://query1.finance.yahoo.com/v8/finance/chart/${sym}?interval=1m&range=1d`);
                const j = await r.json();
                const p = j.chart.result[0].meta.regularMarketPrice;
                if (p) tag.innerText = p.toFixed(2) + '€';
            } catch (e) {}
        }
        document.getElementById('live-status').innerText = 'Prix en Direct Synchronisés';
    }
    window.onload = updatePrices;
    </script>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_header + rows_html + html_footer)
