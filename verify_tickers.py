# verify_tickers.py
# Lance ce script AVANT le screener pour valider tous les tickers

import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import json

TICKER_MATRIX = {
    "AI.PA"   : {"F": "AIL.F",   "DE": "AIL.DE",   "PA": "AI.PA"},
    "AIR.PA"  : {"F": "AIR.F",   "DE": "AIR.DE",   "PA": "AIR.PA"},
    "ALO.PA"  : {"F": "AOB.F",   "DE": "AOB.DE",   "PA": "ALO.PA"},
    "CS.PA"   : {"F": "CS.F",    "DE": "CS.DE",    "PA": "CS.PA"},
    "BNP.PA"  : {"F": "BNP.F",   "DE": "BNP.DE",   "PA": "BNP.PA"},
    "EN.PA"   : {"F": "EN.F",    "DE": "EN.DE",    "PA": "EN.PA"},
    "CA.PA"   : {"F": "CARR.F",  "DE": "CARR.DE",  "PA": "CA.PA"},
    "OR.PA"   : {"F": "LOI.F",   "DE": "LOI.DE",   "PA": "OR.PA"},
    "MC.PA"   : {"F": "MOH.F",   "DE": "MOH.DE",   "PA": "MC.PA"},
    "ML.PA"   : {"F": "ML.F",    "DE": "ML.DE",    "PA": "ML.PA"},
    "RNO.PA"  : {"F": "RNO.F",   "DE": "RNO.DE",   "PA": "RNO.PA"},
    "SAF.PA"  : {"F": "SAF.F",   "DE": "SAF.DE",   "PA": "SAF.PA"},
    "SAN.PA"  : {"F": "SAN.F",   "DE": "SAN.DE",   "PA": "SAN.PA"},
    "SGO.PA"  : {"F": "SGO.F",   "DE": "SGO.DE",   "PA": "SGO.PA"},
    "SU.PA"   : {"F": "SU.F",    "DE": "SU.DE",    "PA": "SU.PA"},
    "GLE.PA"  : {"F": "GZF.F",   "DE": "GZF.DE",   "PA": "GLE.PA"},
    "STM.PA"  : {"F": "SGM.F",   "DE": "SGM.DE",   "PA": "STM.PA"},
    "TTE.PA"  : {"F": "TTE.F",   "DE": "TTE.DE",   "PA": "TTE.PA"},
    "DG.PA"   : {"F": "DGR.F",   "DE": "DGR.DE",   "PA": "DG.PA"},
    "HO.PA"   : {"F": "HAL.F",   "DE": "HAL.DE",   "PA": "HO.PA"},
    "CAP.PA"  : {"F": "CPG.F",   "DE": "CPG.DE",   "PA": "CAP.PA"},
    "DSY.PA"  : {"F": "DSY.F",   "DE": "DSY.DE",   "PA": "DSY.PA"},
    "ENGI.PA" : {"F": "ENGI.F",  "DE": "ENGI.DE",  "PA": "ENGI.PA"},
    "ACA.PA"  : {"F": "ACA.F",   "DE": "ACA.DE",   "PA": "ACA.PA"},
    "PUB.PA"  : {"F": "PUB.F",   "DE": "PUB.DE",   "PA": "PUB.PA"},
    "VIE.PA"  : {"F": "VIE.F",   "DE": "VIE.DE",   "PA": "VIE.PA"},
    "ORA.PA"  : {"F": "ORA.F",   "DE": "ORA.DE",   "PA": "ORA.PA"},
    "VIV.PA"  : {"F": "VIV.F",   "DE": "VIV.DE",   "PA": "VIV.PA"},
    "KER.PA"  : {"F": "KER.F",   "DE": "KER.DE",   "PA": "KER.PA"},
    "RMS.PA"  : {"F": "RMS.F",   "DE": "RMS.DE",   "PA": "RMS.PA"},
    "EL.PA"   : {"F": "ESX.F",   "DE": "ESX.DE",   "PA": "EL.PA"},
    "WLN.PA"  : {"F": "WLN.F",   "DE": "WLN.DE",   "PA": "WLN.PA"},
    "GTT.PA"  : {"F": "GTT.F",   "DE": "GTT.DE",   "PA": "GTT.PA"},
    "ASML.AS" : {"F": "ASML.F",  "DE": "ASML.DE",  "PA": "ASML.AS"},
    "SAP.DE"  : {"F": "SAP.F",   "DE": "SAP.DE",   "PA": "SAP.DE"},
    "SIE.DE"  : {"F": "SIE.F",   "DE": "SIE.DE",   "PA": "SIE.DE"},
    "ALV.DE"  : {"F": "ALV.F",   "DE": "ALV.DE",   "PA": "ALV.DE"},
    "BMW.DE"  : {"F": "BMW.F",   "DE": "BMW.DE",   "PA": "BMW.DE"},
    "VOW3.DE" : {"F": "VOW3.F",  "DE": "VOW3.DE",  "PA": "VOW3.DE"},
    "BAS.DE"  : {"F": "BAS.F",   "DE": "BAS.DE",   "PA": "BAS.DE"},
    "BAYN.DE" : {"F": "BAYN.F",  "DE": "BAYN.DE",  "PA": "BAYN.DE"},
}

def check_ticker(symbol):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period="5d", interval="1d")
        info = t.info
        
        if hist.empty:
            return symbol, False, None, None
        
        last_price = round(hist["Close"].iloc[-1], 2)
        currency   = info.get("currency", "?")
        name       = info.get("shortName", "?")
        
        return symbol, True, last_price, currency, name
    except Exception as e:
        return symbol, False, None, None, str(e)

def verify_all():
    print("\n" + "═"*80)
    print("  VÉRIFICATION DES TICKERS — Francfort .F | XETRA .DE | Paris .PA")
    print("═"*80 + "\n")

    results = {}
    all_symbols = []

    for base, markets in TICKER_MATRIX.items():
        for mkt, sym in markets.items():
            if sym:
                all_symbols.append((base, mkt, sym))

    # Test en parallèle
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(check_ticker, sym): (base, mkt, sym)
            for base, mkt, sym in all_symbols
        }

        for future in futures:
            base, mkt, sym = futures[future]
            result = future.result()
            symbol, ok, price, currency, name = result if len(result) == 5 else (*result, "N/A")

            if base not in results:
                results[base] = {}
            results[base][mkt] = {
                "ticker"  : sym,
                "ok"      : ok,
                "price"   : price,
                "currency": currency,
                "name"    : name
            }

    # ── Affichage rapport ──────────────────────────────────────────
    issues = []

    for base, markets in sorted(results.items()):
        print(f"  📌 {base}")
        for mkt in ["F", "DE", "PA"]:
            if mkt not in markets:
                continue
            d = markets[mkt]
            status = "✅" if d["ok"] else "❌"
            price  = f"{d['price']} {d['currency']}" if d["ok"] else "— ÉCHEC"
            label  = {"F": "Francfort .F", "DE": "XETRA .DE  ", "PA": "Paris .PA  "}[mkt]
            name   = d.get("name", "") or ""
            print(f"    {status} [{label}] {d['ticker']:<14} {price:<18} {name[:40]}")

            if not d["ok"] and mkt in ["F", "DE"]:
                issues.append(f"{base} → {d['ticker']} ({label}) INVALIDE")

        print()

    # ── Résumé ────────────────────────────────────────────────────
    print("═"*80)
    print(f"  ⚠️  {len(issues)} problème(s) détecté(s) :\n")
    for i in issues:
        print(f"    • {i}")

    # ── Sauvegarde JSON ───────────────────────────────────────────
    with open("ticker_verification.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n  💾 Résultats sauvegardés dans ticker_verification.json")
    print("═"*80 + "\n")

if __name__ == "__main__":
    verify_all()
