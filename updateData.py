import os
import yfinance as yf
from openai import OpenAI

# Connexion à MammouthAI (compatible OpenAI)
client = OpenAI(
    api_key=os.getenv("MAMMOUTH_API_KEY"),
    base_url="https://api.mammouth.ai/v1"
)

# Liste des 62 actions PEA Trade Republic (Echantillon des principales)
tickers = [
    "MC.PA", "OR.PA", "RMS.PA", "TTE.PA", "SAN.PA", "AIR.PA", "AI.PA", "BNP.PA",
    "DG.PA", "KER.PA", "OR.PA", "CDI.PA", "EL.PA", "ASML.AS", "SAP.DE", "SIE.DE"
    # Ajoutez ici les autres codes .PA (Paris) ou .DE (Allemagne) pour compléter les 62
]

def analyze_stock(symbol, price, info):
    prompt = f"""
    Analyse l'action {symbol} qui vaut actuellement {price}€.
    Secteur: {info.get('sector', 'Inconnu')}. 
    PER: {info.get('trailingPE', 'N/A')}.
    Donne une réponse courte en 3 colonnes :
    1. Analyse Fondamentale (Santé)
    2. Analyse Chartiste (Tendance)
    3. Conseil : "Achat", "Attendre" ou "Vente" + Prix d'entrée optimal.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o", # Ou un autre modèle disponible sur Mammouth
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erreur IA : {str(e)}"

# Génération de la page HTML
html_header = """
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Screener PEA - IA Live</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 p-8">
    <h1 class="text-3xl font-bold mb-6 text-blue-400">Mon Dashboard PEA - Mise à jour IA</h1>
    <p class="mb-4 text-gray-400">Actualisé automatiquement à 7h, 12h et 17h via MammouthAI.</p>
    <table class="w-full text-left border-collapse bg-gray-800 rounded-lg overflow-hidden">
        <thead>
            <tr class="bg-gray-700 text-blue-300">
                <th class="p-4">Action</th>
                <th class="p-4">Prix</th>
                <th class="p-4">Analyse Expert (IA)</th>
            </tr>
        </thead>
        <tbody>
"""

html_rows = ""
for symbol in tickers:
    stock = yf.Ticker(symbol)
    price = stock.fast_info['last_price']
    info = stock.info
    analysis = analyze_stock(symbol, round(price, 2), info)
    
    html_rows += f"""
    <tr class="border-b border-gray-700 hover:bg-gray-750">
        <td class="p-4 font-bold">{symbol}</td>
        <td class="p-4">{round(price, 2)}€</td>
        <td class="p-4 text-sm text-gray-300">{analysis}</td>
    </tr>
    """

html_footer = "</tbody></table></body></html>"

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html_header + html_rows + html_footer)
