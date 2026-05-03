import os
import yfinance as yf
from openai import OpenAI
import re

# Configuration API
client = OpenAI(
    api_key=os.getenv("MAMMOUTH_API_KEY"),
    base_url="https://api.mammouth.ai/v1"
)

# Liste exhaustive des 62 actions
base_tickers = [
    "MC", "OR", "RMS", "TTE", "SAN", "AIR", "AI", "BNP", "DG", "KER",
    "CDI", "EL", "ASML", "SAP", "SIE", "SU", "CS", "ALV", "BMW", "VOW3",
    "BAS", "BAYN", "SAF", "ENGI", "RNO", "GLE", "ACA", "ML", "VIE", "STM",
    "ORA", "LR", "CAP", "DSY", "ATO", "PUB", "BN", "URW", "VIV", "EDEN",
    "GET", "HO", "DIM", "WLN", "GFC", "BOL", "DEC", "COFA", "NEX", "COV",
    "EN", "FDJ", "TEP", "SGO", "FR", "SOP", "MERY", "ICAD", "ERF", "KORI",
    "KPN", "AD.AS"
]

def get_isin(ticker_obj):
    """Récupère l'ISIN de manière robuste"""
    try:
        isin = ticker_obj.isin
        if isin and isin != "-" and len(isin) > 5:
            return isin
        # Fallback si l'attribut .isin échoue
        return ticker_obj.info.get('isin', 'N/A')
    except:
        return 'N/A'

def get_best_ticker(base):
    for suffix in [".F", ".DE", ".PA"]:
        t = yf.Ticker(base + suffix)
        try:
            if t.fast_info['last_price'] > 0:
                return t, base + suffix
        except:
            continue
    return yf.Ticker(base + ".PA"), base + ".PA"

def analyze_stock(symbol, price, info):
    prompt = f"""Analyse financière pour {symbol} au prix de {price}€.
    Secteur: {info.get('sector', 'N/A')}.
    Tu dois impérativement répondre en suivant ce schéma précis :
    SCORE: [Note de 0 à 100 uniquement le chiffre]
    SANTE: [Analyse santé fondamentale]
    TENDANCE: [Analyse tendance chartiste]
    CONSEIL: [Conseil achat et prix d'entrée]"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content

        # Extraction robuste par Regex
        score_match = re.search(r"SCORE:\s*(\d+)", response)
        sante_match = re.search(r"SANTE:(.*?)(?=TENDANCE:|$)", response, re.DOTALL)
        tendance_match = re.search(r"TENDANCE:(.*?)(?=CONSEIL:|$)", response, re.DOTALL)
        conseil_match = re.search(r"CONSEIL:(.*)", response, re.DOTALL)

        return {
            "score": int(score_match.group(1)) if score_match else 0,
            "sante": sante_match.group(1).strip() if sante_match else "Analyse indisponible",
            "tendance": tendance_match.group(1).strip() if tendance_match else "Analyse indisponible",
            "conseil": conseil_match.group(1).strip() if conseil_match else "Attendre signal"
        }
    except:
        return {"score": 0, "sante": "Erreur", "tendance": "Erreur", "conseil": "Erreur"}

all_data = []
print(f"Lancement de l'analyse sur {len(base_tickers)} actions...")

for base in base_tickers:
    ticker_obj, full_ticker = get_best_ticker(base)
    try:
        price = round(ticker_obj.fast_info['last_price'], 2)
        info = ticker_obj.info
        nom = info.get('longName', base)
        isin = get_isin(ticker_obj)
        
        analysis = analyze_stock(full_ticker, price, info)
        
        all_data.append({
            "ticker": full_ticker,
            "nom": nom,
            "isin": isin,
            "prix": price,
            "score": analysis['score'],
            "sante": analysis['sante'],
            "tendance": analysis['tendance'],
            "conseil": analysis['conseil']
        })
        print(f"Analysé : {full_ticker} - Score : {analysis['score']}")
    except:
        continue

# TRI PAR POTENTIEL (DÉCROISSANT)
all_data.sort(key=lambda x: x['score'], reverse=True)

# Génération HTML
html_header = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Screener PEA - Top 62</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f172a; color: white; }
        .col-analyse { min-width: 250px; }
    </style>
</head>
<body class="p-8">
    <div class="max-w-7xl mx-auto">
        <h1 class="text-3xl font-black mb-2 bg-gradient-to-r from-blue-400 to-emerald-400 bg-clip-text text-transparent italic">
            SCREENER PEA : ANALYSE & POTENTIEL
        </h1>
        <p class="text-slate-400 text-sm mb-8 italic">Classement en temps réel des 62 actions majeures par potentiel de gain immédiat.</p>
        
        <div class="overflow-hidden rounded-xl border border-slate-800 shadow-2xl">
            <table class="w-full text-left border-collapse bg-slate-900/50">
                <thead>
                    <tr class="bg-slate-800/50 text-[10px] uppercase tracking-widest text-slate-400">
                        <th class="p-4">Ticker & ISIN</th>
                        <th class="p-4 text-center">Prix</th>
                        <th class="p-4 text-center">Potentiel IA</th>
                        <th class="p-4">Santé Fondamentale</th>
                        <th class="p-4 border-l border-slate-800">Tendance Chartiste</th>
                        <th class="p-4 border-l border-slate-800">Conseil & Entrée</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-800">
"""

html_rows = ""
for i, data in enumerate(all_data):
    # Gestion des couleurs de score
    score_color = "text-emerald-400" if data['score'] > 70 else "text-yellow-400" if data['score'] > 40 else "text-rose-400"
    rank_style = "bg-blue-500/10" if i < 3 else ""
    
    html_rows += f"""
    <tr class="hover:bg-slate-800/40 transition-colors {rank_style}">
        <td class="p-4">
            <div class="flex flex-col">
                <span class="text-blue-400 font-bold text-sm">{data['ticker']}</span>
                <span class="text-white text-xs font-medium truncate max-w-[180px]">{data['nom']}</span>
                <span class="text-[10px] text-slate-500 font-mono mt-1 uppercase">{data['isin']}</span>
            </div>
        </td>
        <td class="p-4 text-center font-bold text-emerald-500">{data['prix']}€</td>
        <td class="p-4 text-center">
            <div class="text-xl font-black {score_color}">{data['score']}%</div>
        </td>
        <td class="p-4 text-[11px] text-slate-300 leading-relaxed col-analyse italic">
            {data['sante']}
        </td>
        <td class="p-4 text-[11px] text-slate-300 leading-relaxed col-analyse border-l border-slate-800/50 italic">
            {data['tendance']}
        </td>
        <td class="p-4 text-xs border-l border-slate-800/50">
            <div class="bg-slate-900/80 p-2 rounded text-blue-200 font-medium">
                {data['conseil']}
            </div>
        </td>
    </tr>
    """

html_footer = """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_header + html_rows + html_footer)
