import os
import yfinance as yf
from openai import OpenAI

# Connexion à MammouthAI
client = OpenAI(
    api_key=os.getenv("MAMMOUTH_API_KEY"),
    base_url="https://api.mammouth.ai/v1"
)

# Liste exhaustive des 62 actions PEA (Tickers racines)
base_tickers = [
    "MC", "OR", "RMS", "TTE", "SAN", "AIR", "AI", "BNP", "DG", "KER",
    "CDI", "EL", "ASML", "SAP", "SIE", "SU", "CS", "ALV", "BMW", "VOW3",
    "BAS", "BAYN", "SAF", "ENGI", "RNO", "GLE", "ACA", "ML", "VIE", "STM",
    "ORA", "LR", "CAP", "DSY", "ATO", "PUB", "BN", "URW", "VIV", "EDEN",
    "GET", "HO", "DIM", "WLN", "GFC", "BOL", "DEC", "COFA", "NEX", "COV",
    "EN", "FDJ", "TEP", "SGO", "FR", "SOP", "MERY", "ICAD", "ERF", "KORI",
    "KPN", "AD.AS"
]

def get_best_ticker(base):
    # Logique de priorité : Francfort (.F) > Xetra (.DE) > Paris (.PA)
    for suffix in [".F", ".DE", ".PA"]:
        t = yf.Ticker(base + suffix)
        try:
            price = t.fast_info['last_price']
            if price and price > 0:
                return t, base + suffix
        except:
            continue
    return yf.Ticker(base + ".PA"), base + ".PA"

def analyze_stock(symbol, price, info):
    prompt = f"""Analyse professionnelle pour {symbol} ({price}€).
    Secteur: {info.get('sector', 'Inconnu')}.
    Réponds strictement sous ce format :
    SCORE: [Note de 0 à 100 du potentiel de gain à court terme]
    SANTE: [Bilan financier court]
    TENDANCE: [Analyse graphique/momentum]
    CONSEIL: [Achat/Vente/Attendre + prix d'entrée]"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        res = response.choices[0].message.content
        
        # Initialisation
        score, sante, tendance, conseil = 0, "N/A", "N/A", "N/A"
        
        for line in res.split('\n'):
            if "SCORE:" in line: 
                try: score = int(''.join(filter(str.isdigit, line)))
                except: score = 0
            elif "SANTE:" in line: sante = line.replace("SANTE:", "").strip()
            elif "TENDANCE:" in line: tendance = line.replace("TENDANCE:", "").strip()
            elif "CONSEIL:" in line: conseil = line.replace("CONSEIL:", "").strip()
            
        return score, sante, tendance, conseil
    except:
        return 0, "Analyse indisponible", "Analyse indisponible", "Analyse indisponible"

# Collecte des données pour les 62 actions
all_data = []
print(f"Début de l'analyse de {len(base_tickers)} actions...")

for base in base_tickers:
    try:
        stock_obj, full_ticker = get_best_ticker(base)
        price = stock_obj.fast_info['last_price']
        info = stock_obj.info
        
        score, sante, tendance, conseil = analyze_stock(full_ticker, round(price, 2), info)
        
        all_data.append({
            "ticker": full_ticker,
            "nom": info.get('longName', base),
            "isin": info.get('isin', 'N/A'),
            "prix": round(price, 2),
            "score": score,
            "sante": sante,
            "tendance": tendance,
            "conseil": conseil
        })
        print(f"Analysé : {full_ticker} (Score: {score})")
    except Exception as e:
        continue

# CLASSEMENT PAR POTENTIEL DÉCROISSANT
all_data.sort(key=lambda x: x['score'], reverse=True)

# Génération du HTML
html_header = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Screener PEA Top 62 - MammouthAI</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .col-analyse { max-width: 300px; min-width: 200px; }
    </style>
</head>
<body class="bg-[#0f172a] text-slate-200 p-4 md:p-8">
    <div class="max-w-[1700px] mx-auto">
        <header class="mb-8 border-b border-slate-800 pb-6">
            <h1 class="text-4xl font-black text-white bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
                Screener PEA : Analyse & Potentiel
            </h1>
            <p class="text-slate-400 mt-2 italic">Classement en temps réel des 62 actions majeures par potentiel de gain immédiat.</p>
        </header>
        
        <div class="overflow-x-auto rounded-2xl border border-slate-800 bg-[#1e293b]/50 shadow-2xl backdrop-blur-sm">
            <table class="w-full text-left border-collapse">
                <thead>
                    <tr class="bg-slate-900/80 text-[11px] uppercase tracking-widest font-bold">
                        <th class="p-4 border-b border-slate-800">Ticker</th>
                        <th class="p-4 border-b border-slate-800 text-center">Prix</th>
                        <th class="p-4 border-b border-slate-800 text-center bg-blue-600/20">Potentiel IA</th>
                        <th class="p-4 border-b border-slate-800">Santé Fondamentale</th>
                        <th class="p-4 border-b border-slate-800">Tendance Chartiste</th>
                        <th class="p-4 border-b border-slate-800">Conseil & Entrée</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-800/50">
"""

html_rows = ""
for i, data in enumerate(all_data):
    score_color = "text-emerald-400" if data['score'] >= 75 else "text-amber-400" if data['score'] >= 50 else "text-rose-400"
    rank_style = "bg-emerald-500/10" if i < 3 else "" # Mise en avant du Top 3
    
    html_rows += f"""
    <tr class="hover:bg-slate-700/30 transition-all {rank_style}">
        <td class="p-4">
            <div class="flex items-center">
                <span class="text-[10px] text-slate-500 mr-2 font-mono">#{i+1}</span>
                <div>
                    <div class="font-mono font-bold text-blue-400 text-sm">{data['ticker']}</div>
                    <div class="text-[11px] text-white font-semibold truncate max-w-[150px]">{data['nom']}</div>
                    <div class="text-[9px] text-slate-500 font-mono italic">{data['isin']}</div>
                </div>
            </div>
        </td>
        <td class="p-4 text-center">
            <span class="px-2 py-1 bg-slate-900 rounded border border-slate-700 font-bold text-emerald-500 text-sm">
                {data['prix']}€
            </span>
        </td>
        <td class="p-4 text-center bg-blue-600/5">
            <div class="text-2xl font-black {score_color}">{data['score']}%</div>
        </td>
        <td class="p-4 text-xs text-slate-300 italic col-analyse leading-relaxed">
            {data['sante']}
        </td>
        <td class="p-4 text-xs text-slate-300 italic col-analyse border-l border-slate-800/50 leading-relaxed">
            {data['tendance']}
        </td>
        <td class="p-4 text-xs font-bold border-l border-slate-800/50">
            <div class="bg-slate-900/50 p-2 rounded border border-slate-800/50 text-blue-300">
                {data['conseil']}
            </div>
        </td>
    </tr>
    """

html_footer = """
                </tbody>
            </table>
        </div>
        <footer class="mt-8 text-center text-slate-600 text-[10px]">
            Données calculées via Yahoo Finance & MammouthAI (GPT-4o). 
            Le score de potentiel est une estimation algorithmique et ne constitue pas un conseil financier.
        </footer>
    </div>
</body>
</html>
"""

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_header + html_rows + html_footer)
