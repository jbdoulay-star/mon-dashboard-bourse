import os
import yfinance as yf
from openai import OpenAI

# Connexion à MammouthAI
client = OpenAI(
    api_key=os.getenv("MAMMOUTH_API_KEY"),
    base_url="https://api.mammouth.ai/v1"
)

# Liste des bases (sans suffixe) pour appliquer la priorité des marchés
base_tickers = [
    "MC", "OR", "RMS", "TTE", "SAN", "AIR", "AI", "BNP",
    "DG", "KER", "CDI", "EL", "ASML", "SAP", "SIE"
]

def get_best_ticker(base):
    # Priorité : Francfort (.F), puis Xetra (.DE), puis Paris (.PA)
    for suffix in [".F", ".DE", ".PA"]:
        t = yf.Ticker(base + suffix)
        try:
            # On vérifie si on arrive à obtenir un prix
            if t.fast_info['last_price'] is not None:
                return t, base + suffix
        except:
            continue
    return yf.Ticker(base + ".PA"), base + ".PA" # Fallback Paris

def analyze_stock(symbol, price, info):
    prompt = f"""Analyse professionnelle pour {symbol} ({price}€).
    Secteur: {info.get('sector', 'N/A')}. PER: {info.get('trailingPE', 'N/A')}.
    Répond obligatoirement sous ce format précis sans rien d'autre :
    SANTE: (max 15 mots) | TENDANCE: (max 15 mots) | CONSEIL: (Achat/Vente/Attendre + prix d'entrée)"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.choices[0].message.content.replace('*', '')
        # Découpage de la réponse par le séparateur "|"
        parts = content.split('|')
        sante = parts[0].replace('SANTE:', '').strip() if len(parts) > 0 else "N/A"
        tendance = parts[1].replace('TENDANCE:', '').strip() if len(parts) > 1 else "N/A"
        conseil = parts[2].replace('CONSEIL:', '').strip() if len(parts) > 2 else "N/A"
        return sante, tendance, conseil
    except:
        return "Analyse indisponible", "Analyse indisponible", "Analyse indisponible"

# Génération du HTML avec colonnes séparées
html_header = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Screener PEA - IA</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 p-2 md:p-6">
    <div class="max-w-full mx-auto">
        <h1 class="text-3xl font-bold text-blue-400 mb-2">Screener PEA - Analyse IA Multi-Marchés</h1>
        <p class="text-gray-400 mb-6 text-sm">Priorité Marchés : Francfort > Xetra > Paris</p>
        
        <div class="overflow-x-auto shadow-2xl rounded-xl border border-gray-700">
            <table class="w-full text-left border-collapse bg-gray-800 table-auto">
                <thead>
                    <tr class="bg-gray-700 text-blue-200 text-xs uppercase tracking-wider">
                        <th class="p-3 border-r border-gray-600">Ticker</th>
                        <th class="p-3 border-r border-gray-600">Nom</th>
                        <th class="p-3 border-r border-gray-600">ISIN</th>
                        <th class="p-3 border-r border-gray-600 text-center">Prix</th>
                        <th colspan="3" class="p-3 text-center bg-blue-900/30 text-blue-300">Analyse Expert IA</th>
                    </tr>
                    <tr class="bg-gray-750 text-[10px] text-gray-400 uppercase">
                        <th colspan="4" class="border-r border-gray-600"></th>
                        <th class="p-2 border-r border-gray-700 w-1/4">Santé Fondamentale</th>
                        <th class="p-2 border-r border-gray-700 w-1/4">Tendance Chartiste</th>
                        <th class="p-2 w-1/4">Conseil & Entrée</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-gray-700">
"""

html_footer = """
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""

html_rows = ""
for base in base_tickers:
    try:
        stock_obj, full_ticker = get_best_ticker(base)
        price = stock_obj.fast_info['last_price']
        info = stock_obj.info
        isin = info.get('isin', 'N/A')
        nom = info.get('longName', base)
        
        sante, tendance, conseil = analyze_stock(full_ticker, round(price, 2), info)
        
        # Couleur dynamique pour le conseil
        color_conseil = "text-green-400" if "Achat" in conseil else "text-yellow-400" if "Attendre" in conseil else "text-red-400"

        html_rows += f"""
        <tr class="hover:bg-gray-750 border-b border-gray-700">
            <td class="p-3 font-mono text-blue-300 text-sm font-bold">{full_ticker}</td>
            <td class="p-3 text-white text-sm font-medium">{nom}</td>
            <td class="p-3 text-gray-500 text-[10px]">{isin}</td>
            <td class="p-3 text-green-400 font-bold text-center border-r border-gray-700">{round(price, 2)}€</td>
            <td class="p-3 text-xs text-gray-300 italic border-r border-gray-700">{sante}</td>
            <td class="p-3 text-xs text-gray-300 italic border-r border-gray-700">{tendance}</td>
            <td class="p-3 text-xs font-bold {color_conseil}">{conseil}</td>
        </tr>
        """
    except Exception as e:
        print(f"Erreur sur {base}: {e}")
        continue

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_header + html_rows + html_footer)
