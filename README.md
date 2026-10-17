# Finanz-Multitool – Installation

Alle Python-Dateien und `requirements.txt` müssen im **selben Hauptverzeichnis** deines GitHub-Repositories liegen. Beim Hochladen die Dateinamen exakt beibehalten. Keine Versionszusätze anhängen.

## Dateien

- `Finanz_Multitool.py` – Streamlit-Anwendung mit acht Reitern
- `momentum_engine.py` – Momentum-Strategien, Depot- und SPY-Berechnungen
- `momentum_risk_guard.py` – Momentum-Risikomodul
- `finanz_assets.py` – Assetnamen und S&P-500-Zuordnung
- `requirements.txt` – benötigte Python-Pakete

## Einrichtung in Streamlit Cloud

1. Dateien direkt im Repository-Hauptverzeichnis hochladen; gleichnamige alte Dateien ersetzen.
2. Die Startdatei **`Finanz_Multitool.py`** in Streamlit Cloud beibehalten.
3. Commit speichern und die App neu laden; bei hartnäckigem alten Fehler `Manage app → Reboot app` wählen.
4. `requirements.txt` wird für die Paketinstallation verwendet.

Lokaler Start (optional):

```bash
pip install -r requirements.txt
streamlit run Finanz_Multitool.py
```

Für die optionale Nachrichtenanalyse `NEWS_API_KEY` in Streamlit Secrets hinterlegen. **Keine echten API-Schlüssel in GitHub hochladen.**

Optionale eigene historische Indexdaten (`sp500_membership_history.csv`) oder Sektordaten (`sp500_sectors.csv`) sind nicht enthalten. Ohne vollständige historische Indexmitgliedschaftsdaten bleiben langfristige Momentum-Backtests anfällig für Survivorship Bias. Auch ein getesteter Import ersetzt keinen Live-Test gegen Yahoo Finance oder andere externe Anbieter.
