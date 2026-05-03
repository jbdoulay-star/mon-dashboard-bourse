import os
import yfinance as yf
from openai import OpenAI

# Connexion à MammouthAI
client = OpenAI(
    api_key=os.getenv("MAMMOUTH_API_KEY"),
    base_url="https://api.mammouth.ai/v1"
)

# Liste des actions (vous pouvez en ajouter d'autres ici)
tickers = [
    "MC.PA", "OR.PA", "RMS.PA", "TTE.PA", "SAN.PA", "AIR.PA", "AI.PA", "BNP.PA",
    "DG.PA", "KER.PA", "OR.PA", "CDI.PA", "EL.PA", "ASML.AS", "SAP.DE", "SIE.DE"
]

def analyze_stock(symbol, price, info):
    prompt = f"""Analyse courte pour {symbol} (Prix: {price}€). 
    Secteur: {info.get('sector', 'N/A')}. PER: {info.get('trailingPE', 'N/A')}.
    Donne 3 points brefs : 1. Santé 2. Tendance 3. Conseil (Achat/Vente/Attendre) + Prix d'entrée.
    Pas de gras (**), pas de tirets, reste très concis (max 200 caractères)."""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        # On nettoie le texte pour enlever les étoiles et caractères spéciaux
        return response.choices[0].message.content.replace('*', '').replace('#', '').strip()
    except Exception as e:
        return "Analyse temporairement indisponible."

# Génération du HTML
html_header = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Screener IA Live</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 p-4 md:p-8">
    <div class="max-w-6xl mx-auto">
        <h1 class="text-3xl font-bold text-blue-400 mb-2">Mon Dashboard PEA - Analyse IA</h1>
        <p class="text-gray-400 mb-6">Mise à jour automatique via MammouthAI</p>
        <div class="overflow-x-auto shadow-2xl rounded-lg">
            <table class="w-full text-left border-collapse bg-gray-800">
                <thead>
                    <tr class="bg-gray-700 text-blue-300 uppercase text-sm">
                        <th class="p-4">Action</th>
                        <th class="p-4">Prix</th>
                        <th class="p-4">Analyse Expert IA</th>
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
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        price = stock.fast_info['last_price']
        info = stock.info
        analysis = analyze_stock(ticker, round(price, 2), info)
        
        html_rows += f"""
        <tr class="hover:bg-gray-750 transition-colors">
            <td class="p-4 font-bold text-white">{ticker}</td>
            <td class="p-4 text-green-400">{round(price, 2)}€</td>
            <td class="p-4 text-sm text-gray-300 leading-relaxed">{analysis}</td>
        </tr>
        """
    except:
        continue

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_header + html_rows + html_footer)
