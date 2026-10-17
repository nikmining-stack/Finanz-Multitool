# Finanz-Multitool – direkte Bereitstellung

1. Lade `Finanz_Multitool.py` in das Hauptverzeichnis deines GitHub-Repositories hoch und überschreibe die bisherige Datei.
2. Lass die bestehende `requirements.txt` unverändert, sofern sie die beiliegenden Abhängigkeiten bereits enthält. Andernfalls ersetze sie durch die Datei aus diesem Paket.
3. Streamlit-Startdatei bleibt `Finanz_Multitool.py`. Nach Commit unter `Manage app` → `Reboot app` neu starten.

**Wichtig:** Die Startdatei enthält jetzt alle drei zuvor ausgelagerten Hilfsmodule (`momentum_engine.py`, `momentum_risk_guard.py`, `finanz_assets.py`). Alte Kopien dürfen im Repository bleiben, werden aber nicht mehr importiert und können später archiviert werden. Es müssen keine weiteren Python-Dateien ersetzt werden.

**Geschwindigkeit:** Trotz Zusammenführung werden die acht Reiter weiterhin bei Auswahl geladen. Der aufwendige Momentum-Kursdatenabruf wird nicht beim allgemeinen Start ausgeführt. Das physische Auslagern in mehrere Dateien hatte den ersten Import etwas übersichtlicher gemacht, war jedoch nicht der Hauptgrund für die bessere Laufzeit.

**Stand der Prüfung:** Syntax und das Vorhandensein der importierten Funktionen wurden geprüft. Eine vollständige Live-Integration auf deiner Streamlit-Cloud-Instanz wurde nicht durchgeführt.
