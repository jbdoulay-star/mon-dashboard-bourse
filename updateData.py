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

# Liste corrigée avec les bons marchés (Frankfurt .DE, Amsterdam .AS)
base_tickers = [
    "MC.PA", "OR.PA", "RMS.PA", "TTE.PA", "SAN.PA", "AIR.PA", "AI.PA", "BNP.PA", "DG.PA", "KER.PA",
    "ASML.AS", "SAP.DE", "SIE.DE", "SU.PA", "CS.PA", "ALV.DE", "BMW.DE", "VOW3.DE",
    "BAS.DE", "BAYN.DE", "SAF.PA", "ENGI.PA", "RNO.PA", "GLE.PA", "ACA.PA", "ML.PA", "VIE.PA", "STM.PA",
    "ORA.PA", "CAP.PA", "DSY.PA", "PUB.PA", "BN.PA", "URW.PA", "VIV.PA", "EDEN.PA"
]

def generate_sparkline(hist):
    try:
        if hist.empty: return ""
        prices = hist['Close']
        plt.figure(figsize=(3, 1), dpi=80)
        plt.plot(prices.values, color='#3b82f6', linewidth=2)
        plt.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    except: return ""

data_list = []
for symbol in base_tickers:
    print(f"Analyse de {symbol}...")
    try:
        tk = yf.Ticker(symbol)
        info = tk.info
        hist = tk.history(period="6mo")
        
        name = info.get('longName', symbol)
        isin = info.get('isin', 'N/A')
        curr_price = info.get('regularMarketPrice') or info.get('previousClose', 0)
        
        # On prépare des données réelles pour "forcer" l'IA à analyser
        stats_contexte = f"""
        Données actuelles pour {name}:
        - Prix: {curr_price}€
        - Plus haut 52 sem: {info.get('fiftyTwoWeekHigh')}€
        - Plus bas 52 sem: {info.get('fiftyTwoWeekLow')}€
        - PER: {info.get('trailingPE', 'N/A')}
        - Variation 6 mois: {round(((hist['Close'][-1]/hist['Close'][0])-1)*100, 2) if not hist.empty else 0}%
        """
        
        prompt = f"""Tu es un expert financier. Analyse l'action {name} ({symbol}) avec ces données : {stats_contexte}.
        Réponds STRICTEMENT avec ce format (pas d'introduction) :
        [SANTE]: Ton analyse fondamentale courte.
        [TENDANCE]: Ton analyse technique courte.
        [CONSEIL]: Ton conseil d'achat et prix d'entrée précis.
        [SCORE]: Un chiffre entre 0 et 100 uniquement."""
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        full_text = response.choices[0].message.content
        
        # Extraction robuste
        sante = re.search(r"\[SANTE\]:(.*?)\[", full_text, re.S)
        tendance = re.search(r"\[TENDANCE\]:(.*?)\[", full_text, re.S)
        conseil = re.search(r"\[CONSEIL\]:(.*?)\[", full_text, re.S)
        score_find = re.search(r"\[SCORE\]:\s*(\d+)", full_text)

        data_list.append({
            'ticker': symbol, 'nom': name, 'isin': isin, 'prix': curr_price,
            'score': int(score_find.group(1)) if score_find else 50,
            'sante': sante.group(1).strip() if sante else "Analyse en cours...",
            'tendance': tendance.group(1).strip() if tendance else "Tendance neutre.",
            'conseil': conseil.group(1).strip() if conseil else full_text.split("[CONSEIL]:")[-1].strip(),
            'chart': generate_sparkline(hist)
        })
    except Exception as e: print(f"Erreur {symbol}: {e}")

# Tri par potentiel
data_list.sort(key=lambda x: x['score'], reverse=True)

# Génération HTML avec colonnes fixes et égales
date_now = datetime.now(pytz.timezone('Europe/Paris')).strftime("%d/%m/%Y %H:%M")
html_header = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Screener PEA Pro</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ background: #020617; color: white; font-family: 'Inter', sans-serif; }}
        .table-fixed-layout {{ table-layout: fixed; width: 100%; }}
        .col-action {{ width: 180px; }}
        .col-chart {{ width: 140px; }}
        .col-price {{ width: 100px; }}
        .col-score {{ width: 100px; }}
        .col-flexible {{ width: calc((100% - 520px) / 3); }}
    </style>
</head>
<body class="p-8">
    <div class="max-w-[1800px] mx-auto">
        <header class="mb-10 flex justify-between items-end">
            <div>
                <h1 class="text-4xl font-black tracking-tighter text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400 uppercase">Screener PEA : Analyse & Potentiel</h1>
                <p class="text-slate-500 font-bold text-xs mt-2 uppercase tracking-widest">Mise à jour : {date_now} • <span id="live-status">Synchro live...</span></p>
            </div>
        </header>

        <div class="bg-slate-900/50 border border-white/10 rounded-3xl overflow-hidden shadow-2xl">
            <table class="table-fixed-layout">
                <thead>
                    <tr class="bg-slate-800/40 text-slate-400 text-[10px] uppercase tracking-[0.2em] border-b border-white/5">
                        <th class="p-5 col-action">Action</th>
                        <th class="p-5 col-chart text-center">Tendance 6M</th>
                        <th class="p-5 col-price text-center">Prix</th>
                        <th class="p-5 col-score text-center">Potentiel</th>
                        <th class="p-5 col-flexible">Santé Fondamentale</th>
                        <th class="p-5 col-flexible">Tendance Chartiste</th>
                        <th class="p-5 col-flexible">Conseil & Entrée</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-white/5">
"""

rows_html = ""
for d in data_list:
    rows_html += f"""
    <tr class="hover:bg-white/[0.02] transition-colors">
        <td class="p-5">
            <div class="flex flex-col">
                <span class="text-blue-400 font-black text-[10px]">{d['ticker']}</span>
                <span class="text-[15px] font-bold truncate">{d['nom']}</span>
                <span class="text-[11px] text-slate-500 font-mono mt-1">{d['isin']}</span>
            </div>
        </td>
        <td class="p-5 text-center">
            <img src="data:image/png;base64,{d['chart']}" class="w-full h-auto opacity-80" alt="Chart">
        </td>
        <td class="p-5 text-center font-black text-[16px] price-tag" data-symbol="{d['ticker']}">{d['prix']}€</td>
        <td class="p-5 text-center font-black text-3xl italic text-emerald-400">{d['score']}%</td>
        <td class="p-5 text-slate-300 text-[13px] leading-relaxed italic">{d['sante']}</td>
        <td class="p-5 text-slate-300 text-[13px] leading-relaxed italic border-l border-white/5">{d['tendance']}</td>
        <td class="p-5 border-l border-white/5">
            <div class="bg-blue-500/10 p-4 rounded-xl border border-blue-500/20 text-blue-200 text-[13px]">
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
        document.getElementById('live-status').innerHTML = '<span class="text-emerald-500">● PRIX SYNCHRONISÉS</span>';
    }
    window.onload = updatePrices;
    </script>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_header + rows_html + html_footer)
