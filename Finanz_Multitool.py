"""Integriertes Finanz-Terminal V10.2 – lazy Tabs, Momentum-Modul und Periodenübersicht.

Start: streamlit run Finanz_Multitool.py
WICHTIG: Tab 7 ist eine Forschungs-/Signaloberfläche, keine Orderausführung.
Momentum-Originalbacktest bleibt brutto; getrennte Euro-Depot-Simulation mit modellierten Kosten und Steuern.
"""
import os
import gc
import logging
from io import StringIO
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
import pandas_market_calendars as mcal
from momentum_risk_guard import (build_risk_guard, simulate_risk_overlay,
                                 diagnostic_forward_losses)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logging.basicConfig(level=logging.WARNING)
LOG = logging.getLogger("finanz_terminal")
st.set_page_config(page_title="Finanz-Terminal", layout="wide", page_icon="🧭")

# API Keys: über Umgebungsvariable oder Streamlit-Secrets (niemals ins Git-Repo kopieren)
def app_secret(name, default=""):
    try:
        secret = st.secrets.get(name, None)
        if secret not in (None, ""):
            return str(secret)
    except Exception:
        pass
    return os.getenv(name, default)

NEWS_API_KEY = app_secret("NEWS_API_KEY")
# Keine angeblich aktuellen Leitzinsen ohne Datenquelle ausgeben.
EZB_EINLAGEZINS = app_secret("EZB_EINLAGEZINS")
FED_ZINSBAND = app_secret("FED_ZINSBAND")

# ---------------------------------------------
# GLOBALE HILFSFUNKTIONEN / UNTERNEHMENSNAMEN
# ---------------------------------------------
def format_de(number):
    if number is None or not np.isfinite(float(number)):
        return "–"
    return f"{float(number):,.0f} €".replace(",", ".")


def format_de_num(number, dec=2):
    if number is None or not np.isfinite(float(number)):
        return "–"
    return f"{float(number):,.{dec}f}".replace(",", "X").replace(".", ",").replace("X", ".")


from finanz_assets import (
    MOMENTUM_ETFS, S_AND_P_CSV, load_sp500_constituents,
    get_sp500_name_mapping, get_sp500_sector_mapping, get_asset_name,
)

def extract_yf_field(raw, field="Close"):
    """Normalisiert yfinance Einzel-/MultiIndex-Strukturen."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        if field in raw.columns.get_level_values(0):
            part = raw[field]
        elif field in raw.columns.get_level_values(1):
            part = raw.xs(field, level=1, axis=1)
        else:
            return pd.DataFrame()
    else:
        if field not in raw.columns:
            return pd.DataFrame()
        part = raw[[field]]
    if isinstance(part, pd.Series):
        part = part.to_frame()
    part = part.copy().apply(pd.to_numeric, errors="coerce")
    part.index = pd.DatetimeIndex(part.index).tz_localize(None).normalize()
    return part[~part.index.duplicated(keep="last")].sort_index()


def plot_handelsblatt_chart(series, title, color, val_format=2, suffix="", y_range=None, d_tick=None):
    if series is None:
        series = pd.Series(dtype=float)
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) < 2:
        st.markdown(f"**{title}**")
        st.caption("Keine aktuellen Kursdaten verfügbar")
        return
    cur_val, prev_val, start_val = float(series.iloc[-1]), float(series.iloc[-2]), float(series.iloc[0])
    day_pct = 100 * (cur_val / prev_val - 1) if prev_val != 0 else np.nan
    col = "#27ae60" if day_pct >= 0 else "#e74c3c"
    arrow = "▲" if day_pct >= 0 else "▼"
    st.markdown(
        f"<p style='color:{color};font-weight:700;font-size:1.02rem;margin-bottom:0;min-height:48px'>"
        f"{title}: {format_de_num(cur_val,val_format)}{suffix} "
        f"<span style='color:{col};font-size:0.82rem'>({arrow} {format_de_num(day_pct)} %)</span></p>",
        unsafe_allow_html=True,
    )
    fig = px.line(x=series.index, y=series.to_numpy())
    fig.update_traces(line=dict(color=color, width=2.2))
    fig.add_scatter(x=[series.index[0], series.index[-1]], y=[start_val, cur_val],
                    mode="markers", marker=dict(color=color,size=7), showlegend=False)
    yd = dict(tickfont=dict(size=10, color="gray"), showgrid=True,
              gridcolor="rgba(180,180,180,0.25)", zeroline=False)
    if y_range is not None:
        yd["range"] = y_range
    if d_tick is not None:
        yd["dtick"] = d_tick
    fig.update_layout(height=240, margin=dict(l=35, r=10, t=10, b=30),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      showlegend=False,
                      xaxis=dict(showgrid=False,tickmode="array",tickvals=[series.index[0],series.index[-1]],
                                 ticktext=[series.index[0].strftime('%b %Y'),series.index[-1].strftime('%b %Y')],
                                 tickfont=dict(size=11,color="gray")),yaxis=yd)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})


@st.cache_data(ttl=3600, max_entries=60, show_spinner=False)
def fetch_historical(ticker, max_history=False, years=10):
    t = str(ticker).strip().upper()
    if not t:
        return pd.Series(dtype=float)
    try:
        kwargs = dict(progress=False, auto_adjust=True, group_by="column", threads=False, timeout=20)
        if max_history:
            raw = yf.download(t, period="max", **kwargs)
        else:
            start = pd.Timestamp.now() - pd.DateOffset(years=int(years))
            raw = yf.download(t, start=start.strftime("%Y-%m-%d"), **kwargs)
        closes = extract_yf_field(raw, "Close")
        return closes.iloc[:, 0].dropna().astype(float) if not closes.empty else pd.Series(dtype=float)
    except Exception as exc:
        LOG.warning("Kursabruf %s: %s", t, exc)
        return pd.Series(dtype=float)


@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def fetch_macro_data(ticker_dict, days=365):
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(days))
    result = {}
    for name, ticker in ticker_dict.items():
        series = fetch_historical(ticker, years=2)
        if not series.empty:
            result[name] = series.loc[series.index >= cutoff]
    return result


@st.cache_data(ttl=4*3600, show_spinner=False)
def fetch_policy_rates():
    """Liefert Live-Zinssätze, falls amtliche öffentliche Endpunkte verfügbar sind.

    ECB: Deposit Facility (Änderungszeitreihe).
    Fed: Federal Reserve Bank of St. Louis, Zielband-Unter-/Obergrenze.
    Es werden KEINE erfundenen Zinssätze bei Fehlern zurückgegeben.
    """
    ecb_rate, fed_range = None, None
    try:
        url = "https://data-api.ecb.europa.eu/service/data/FM/B.U2.EUR.4F.KR.DFR.LEV"
        r = requests.get(url, params={"lastNObservations": 1, "format": "csvdata"}, timeout=12)
        r.raise_for_status()
        frame = pd.read_csv(StringIO(r.text))
        values = pd.to_numeric(frame.get("OBS_VALUE",pd.Series(dtype=float)),errors="coerce").dropna()
        if not values.empty and -5 < values.iloc[-1] < 30:
            ecb_rate = float(values.iloc[-1])
    except Exception as exc:
        LOG.warning("EZB-API: %s",exc)
    try:
        r=requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                       params={"id":"DFEDTARL,DFEDTARU"},timeout=12)
        r.raise_for_status()
        frame=pd.read_csv(StringIO(r.text))
        if {"DFEDTARL","DFEDTARU"}.issubset(frame.columns):
            rates=frame[["DFEDTARL","DFEDTARU"]].apply(pd.to_numeric,errors="coerce").dropna()
            if not rates.empty:
                lo,hi=map(float,rates.iloc[-1])
                if -5 < lo <= hi < 30:
                    fed_range=f"{lo:.2f}–{hi:.2f}".replace(".",",")
    except Exception as exc:
        LOG.warning("FRED-Zinsband: %s",exc)
    return ecb_rate,fed_range


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_eurusd(days=365):
    series = fetch_historical("EURUSD=X", years=2)
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    return series.loc[series.index >= cutoff] if not series.empty else series


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_bonds_robust(days=365):
    tickers = {"USA (3 Monate)":"^IRX", "USA (5 Jahre)":"^FVX",
               "USA (10 Jahre)":"^TNX", "USA (30 Jahre)":"^TYX"}
    return fetch_macro_data(tickers, days=days)


@st.cache_data(ttl=1800, max_entries=1, show_spinner=False)
def fetch_fear_and_greed():
    """Aktueller CNN Fear & Greed; bei CNN-Sperre mit datiertem Drittanbieter-Spiegel.

    Der Feed von CNN ist nicht öffentlich dokumentiert und kann aus Cloud-Umgebungen
    HTTP 403/418 liefern. Keine erfundenen Werte; veraltete Daten werden verworfen.
    Rückgabe: dict(score, label, source, asof, issue).
    """
    labels = {
        "extreme fear": "Extreme Angst", "fear": "Angst",
        "neutral": "Neutral", "greed": "Gier",
        "extreme greed": "Extreme Gier",
    }
    browser_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://www.cnn.com",
        "Referer": "https://www.cnn.com/markets/fear-and-greed",
    }
    # Auch bei Wochenenden/Feiertagen werden ältere Stände bis max. 7 Tage
    # toleriert; bei längerer Nichtaktualisierung erscheint kein Kennwert.
    today = pd.Timestamp.now(tz="UTC").date()
    failures = []

    def validated(score, rating, date_value, source_name):
        score = float(score)
        if not np.isfinite(score) or not 0 <= score <= 100:
            raise ValueError("Ungültiger Score (außerhalb 0–100)")
        date = pd.to_datetime(date_value, utc=True, errors="coerce")
        if pd.isna(date):
            raise ValueError("Quellendatum fehlt oder ist ungültig")
        asof = date.date()
        age = (today - asof).days
        if age < -1 or age > 7:
            raise ValueError(f"Daten sind {age} Tage alt bzw. liegen in der Zukunft")
        rating = str(rating or "").strip().lower()
        if rating not in labels:
            rating = ("extreme fear" if score <= 25 else "fear" if score <= 45
                      else "neutral" if score <= 55 else "greed" if score <= 75
                      else "extreme greed")
        return {"score": score, "label": labels[rating], "source": source_name,
                "asof": asof.isoformat(), "issue": ""}

    def download(url, headers, timeout=7):
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response

    # 1. Primärdaten direkt von CNN
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        payload = download(url, browser_headers).json()
        current = payload.get("fear_and_greed", {})
        stamp = current.get("timestamp")
        if not stamp:
            history = payload.get("fear_and_greed_historical", {}).get("data", [])
            if history:
                stamp = pd.to_datetime(history[-1].get("x"), unit="ms", utc=True)
        return validated(current.get("score"), current.get("rating"), stamp, "CNN (Direktabruf)")
    except Exception as exc:
        failures.append(f"CNN: {type(exc).__name__}: {str(exc)[:90]}")
        LOG.warning("CNN Fear & Greed Direktabruf fehlgeschlagen: %s", exc)

    # 2. Öffentlicher GitHub-Spiegel mit Zeitstempel (Darthh)
    try:
        url = ("https://raw.githubusercontent.com/Darthh/"
               "fear-and-greed-index-data/main/data/latest.json")
        mirror = download(url, {"User-Agent": browser_headers["User-Agent"],
                                "Accept": "application/json"}, timeout=7).json()
        result = validated(mirror.get("value"), mirror.get("rating"),
                           mirror.get("date"), "CNN-Daten: Drittanbieter-Spiegel (GitHub)")
        return result
    except Exception as exc:
        failures.append(f"Spiegel 1: {type(exc).__name__}: {str(exc)[:90]}")
        LOG.warning("Fear & Greed GitHub-Spiegel 1 fehlgeschlagen: %s", exc)

    # 3. Zweiter täglich aktualisierter CSV-Spiegel (whit3rabbit)
    try:
        url = "https://raw.githubusercontent.com/whit3rabbit/fear-greed-data/main/fear-greed.csv"
        resp = download(url, {"User-Agent": browser_headers["User-Agent"]}, timeout=7)
        data = pd.read_csv(StringIO(resp.text))
        if not {"Date", "Fear Greed", "Rating"}.issubset(data.columns):
            raise ValueError("Fehlende CSV-Spalten")
        dates = pd.to_datetime(data["Date"], errors="coerce")
        if dates.notna().sum() == 0:
            raise ValueError("CSV enthält keine gültigen Datumswerte")
        row = data.loc[dates.idxmax()]
        return validated(row["Fear Greed"], row["Rating"], row["Date"],
                         "CNN-Daten: alternativer GitHub-Spiegel")
    except Exception as exc:
        failures.append(f"Spiegel 2: {type(exc).__name__}: {str(exc)[:90]}")
        LOG.warning("Fear & Greed GitHub-Spiegel 2 fehlgeschlagen: %s", exc)

    # Echte Ausfälle nicht als neutralen Score (50) darstellen.
    return {"score": None, "label": "Nicht verfügbar", "source": "",
            "asof": None, "issue": " | ".join(failures)}

# Momentum-Berechnungen sind in einem eigenen Modul ausgelagert.
# Die UI wird erst beim Öffnen des entsprechenden Tabs ausgeführt.
from momentum_engine import (
    _mt_symbol, _mt_memberships, _mt_sectors, _mt_next_session,
    fetch_sp500_universe, fetch_sp500_data, calculate_momentum_strategy,
    german_stock_tax_rate, simulate_momentum_eur_depot, simulate_spy_buy_hold_eur,
    momentum_period_overview,
)

# ---------------------------------------------
# STEUER-BERECHNUNG: 2026 (vereinfachtes Modell)
# ---------------------------------------------
def german_income_tax_2026(zve):
    """§ 32a EStG 2026 – Grundtarif, nicht Splittingtarif."""
    x = float(np.floor(max(0, zve)))
    if x <= 12348:
        return 0.0
    if x <= 17799:
        y = (x - 12348) / 10000.0
        tax = (914.51 * y + 1400) * y
    elif x <= 69878:
        z = (x - 17799) / 10000.0
        tax = (173.10 * z + 2397) * z + 1034.87
    elif x <= 277825:
        tax = 0.42 * x - 11135.63
    else:
        tax = 0.45 * x - 19470.38
    return float(np.floor(max(0, tax)))


def calculate_german_tax(brutto, werbungskosten=1230, is_pkv=False, pkv_beitrag=4800,
                         has_grv=True, childless=True, additional_health_rate=2.9):
    """Orientierung für Arbeitnehmer, unverheiratet, Steuerjahr 2026.
    Vereinfachungen: PV-Kinderfreibeträge/Sachsen, Altersentlastung, Vorsorgepauschalen,
    Sonderausgaben, Kirchensteuer, Familien- und Grenzfälle nicht vollständig modelliert.
    PKV-Beitrag = selbst getragener Jahresbeitrag, ohne AG-Zuschussmodell.
    """
    brutto = max(0.,float(brutto))
    bbg_rv, bbg_kv = 101400., 69750.
    sv_rv = min(brutto, bbg_rv) * 0.093 if has_grv else 0.
    sv_av = min(brutto, bbg_rv) * 0.013 if has_grv else 0.
    if is_pkv:
        sv_kv_pv = float(pkv_beitrag)
    else:
        # Arbeitnehmer: 7,3% KV + 0,5 * Zusatzbeitrag
        sv_kv = min(brutto, bbg_kv) * (0.073 + additional_health_rate / 200.)
        # vereinfachter bundesweiter PV-Satz, kinderlos / 1 Kind, ohne Sachsen
        sv_pv = min(brutto, bbg_kv) * (0.024 if childless else 0.018)
        sv_kv_pv = sv_kv + sv_pv
    sv_total = sv_rv + sv_av + sv_kv_pv
    # Nicht alle PKV-Beiträge sind als Basisabsicherung in voller Höhe abzugsfähig;
    # hier ausdrücklich vereinfachte Obergrenzen-Näherung.
    zve = max(0., brutto - max(0.,float(werbungskosten)) - sv_rv - sv_kv_pv)
    est = german_income_tax_2026(zve)
    soli = min(0.055 * est, 0.119 * max(0., est - 20350.))
    taxes = est + soli
    return {"netto":brutto - sv_total - taxes, "steuer":taxes, "sv":sv_total,
            "zve":zve, "est":est, "soli":soli}


# ---------------------------------------------
# THEMEN-ETF (TAB 2)
# ---------------------------------------------
@st.cache_resource
def load_sentiment_analyzer():
    return SentimentIntensityAnalyzer()


THEMES = {
    "Kupfer": {"keywords": '("copper mining" OR "copper deficit")', "ticker":"COPX"},
    "Seltene Erden": {"keywords": '("rare earth elements" OR "lithium extraction")', "ticker":"REMX"},
    "Wasser": {"keywords": '("water scarcity" OR "water infrastructure")', "ticker":"CGW"},
    "Nukleartechnologie": {"keywords": '("nuclear power" OR "uranium market")', "ticker":"URNM"},
    "Pharma & Biotech": {"keywords": '("pharmaceutical" OR "drug approval")', "ticker":"PJP"},
    "Finanzen & Banken": {"keywords": '("banking sector" OR "interest rates")', "ticker":"IXG"},
    "Infrastruktur": {"keywords": '("infrastructure spending" OR "construction")', "ticker":"IGF"},
    "Globale Dividenden": {"keywords": '("dividend yield" OR "value investing")', "ticker":"VHYL"},
    "Physisches Gold": {"keywords": '("physical gold" OR "safe haven asset")', "ticker":"GLD"},
}


@st.cache_data(ttl=3600, max_entries=50, show_spinner=False)
def fetch_articles(query: str, days: int) -> list[dict]:
    """NewsAPI: 'everything' kann bei kostenfreiem Tarif auf ~1 Monat begrenzt sein."""
    from_date = (datetime.now() - timedelta(days=min(int(days),28))).strftime("%Y-%m-%d")
    params = {"q":query, "from":from_date, "sortBy":"publishedAt",
              "language":"en", "pageSize":100, "apiKey":NEWS_API_KEY}
    try:
        res = requests.get("https://newsapi.org/v2/everything", params=params, timeout=15)
        res.raise_for_status()
        payload = res.json()
        if payload.get("status") != "ok":
            raise RuntimeError(payload.get("message", "NewsAPI gab keine gültige Antwort."))
        return payload.get("articles", [])
    except Exception as exc:
        LOG.warning("NewsAPI Fehler (%s): %s", query[:30], exc)
        raise RuntimeError(f"NewsAPI-Fehler: {exc}") from exc


def deduplicate_texts(texts: list[str], threshold: float=0.8) -> list[str]:
    texts = [x for x in texts[:100] if isinstance(x,str) and x.strip()]
    if len(texts) < 2:
        return texts
    try:
        matrix = TfidfVectorizer(stop_words="english",max_features=1000).fit_transform(texts)
        similarity = cosine_similarity(matrix)
    except ValueError:
        return texts
    keep, seen = [], set()
    for i in range(len(texts)):
        if i in seen:
            continue
        keep.append(texts[i])
        for j in range(i+1,len(texts)):
            if similarity[i,j] >= threshold:
                seen.add(j)
    return keep


def calculate_sentiment(texts: list[str], analyzer) -> float:
    return float(np.mean([analyzer.polarity_scores(s)["compound"] for s in texts])) if texts else 0.


def get_price_momentum(ticker: str) -> str:
    hist = fetch_historical(ticker,years=1)
    if len(hist) < 50:
        return "Daten fehlen"
    return "🟢 Positiv" if hist.iloc[-1] > hist.rolling(50).mean().iloc[-1] else "🔴 Negativ"


@st.cache_data(ttl=4*3600, show_spinner=False)
def load_all_thematic_data():
    if not NEWS_API_KEY:
        raise ValueError("Bitte NEWS_API_KEY hinterlegen.")
    analyzer = load_sentiment_analyzer()
    result = []
    # Das 28-Tage-Fenster ersetzt die oft nicht verfügbare 30-Tage-Suche.
    for name,cfg in THEMES.items():
        news28 = fetch_articles(cfg["keywords"],28)
        now = pd.Timestamp.now(tz="UTC")
        def valid_articles(days):
            selected=[]
            for article in news28:
                published = pd.to_datetime(article.get("publishedAt"), utc=True, errors="coerce")
                if pd.notna(published) and published >= now - pd.Timedelta(days=days):
                    selected.append(article)
            return selected
        articles7 = valid_articles(7)
        articles28 = valid_articles(28)
        def build_text(items):
            return deduplicate_texts([f"{a.get('title') or ''}. {a.get('description') or ''}" for a in items if a.get("title")])
        texts7, texts28 = build_text(articles7), build_text(articles28)
        # Vergleich: Anteil 7 Tage / erwarteten Anteil von 28 Tagen.
        buzz = len(texts7) / max(1., len(texts28) * 7/28)
        s7, s28 = calculate_sentiment(texts7,analyzer), calculate_sentiment(texts28,analyzer)
        i7, i28 = int(round((s7+1)*50)), int(round((s28+1)*50))
        change=i7-i28
        direction = ("🚀 Stark steigend" if change>10 else "↗️ Steigend" if change>3 else
                     "⚠️ Stark fallend" if change<-10 else "↘️ Fallend" if change<-3 else "➡️ Stabil")
        result.append({"Thema":name,"Stimmungs-Index (0-100)":i7,"Trend vs Vormonat":direction,
                       "News Buzz (Lautstärke)":f"{buzz:.2f}x", "Markt-Momentum":get_price_momentum(cfg['ticker']),
                       "News (7 / 28 Tage)":f"{len(texts7)} / {len(texts28)}",
                       "_sort_score":buzz*(1+s7)})
    return pd.DataFrame(result).sort_values("_sort_score",ascending=False).drop(columns="_sort_score").reset_index(drop=True)


# ---------------------------------------------
# HELFER MONTE-CARLO
# ---------------------------------------------
def gaussian_mc(log_returns_mean, log_returns_std, capital, monthly_save, years, simulations=1000):
    """Historisch kalibrierte i.i.d.-Lognormal-Monte-Carlo-Simulation.

    Einzahlung EXAKT alle 21 simulierten Börsentage (12-mal je Modelljahr),
    jeweils nach der Wertentwicklung des Einzahlungsmonats.
    Rückgabe-Signatur wie V6, damit beide bestehenden Tabs unverändert aufrufbar sind.
    """
    years = int(years)
    n_days = 252 * years
    rng = np.random.default_rng(42)
    daily = rng.normal(float(log_returns_mean), max(0., float(log_returns_std)),
                       size=(n_days, int(simulations)))
    paths = np.empty((n_days + 1, int(simulations)), dtype=float)
    paths[0] = float(capital)
    for day in range(1, n_days + 1):
        deposit = float(monthly_save) if day % 21 == 0 else 0.0
        paths[day] = paths[day - 1] * np.exp(daily[day - 1]) + deposit
    annual = paths[np.arange(0, n_days + 1, 252)]
    unlevered_cagr = np.expm1(np.sum(daily, axis=0) / years) * 100
    return annual, paths[-1], unlevered_cagr


def mc_money_weighted_return(start_capital, monthly_save, horizon_years, end_value):
    """Monatlicher IRR mit tatsächlichem monatlichem Zahlungszeitpunkt im MC-Modell.

    Als annualisierte Rendite angegeben. Kann bei unplausiblen Parametern NaN liefern.
    """
    months = 12 * int(horizon_years)
    c, m, target = float(start_capital), float(monthly_save), float(end_value)
    if months <= 0 or (c <= 0 and m <= 0) or target < 0:
        return np.nan
    def residual(monthly_rate):
        factor = 1 + monthly_rate
        if factor <= 0:
            return -target
        try:
            fv = c * (factor ** months)
            if abs(monthly_rate) < 1e-12:
                fv += m * months
            else:
                fv += m * ((factor ** months - 1) / monthly_rate)
            return fv - target
        except OverflowError:
            return float('inf')
    low, high = -0.5, 1.0
    for _ in range(75):
        mid = (low + high) / 2
        if residual(mid) > 0:
            high = mid
        else:
            low = mid
    monthly_rate = (low + high) / 2
    return ((1 + monthly_rate) ** 12 - 1) * 100


def mc_window(series_or_frame, choice):
    if choice == 'Gesamte Historie':
        return series_or_frame
    years = int(str(choice).split()[0])
    cutoff = series_or_frame.index.max() - pd.DateOffset(years=years)
    return series_or_frame.loc[series_or_frame.index >= cutoff]


def render_mc_plot(yearly, years, title, capital=0, monthly_save=0):
    yrs = np.arange(int(years) + 1)
    fig = go.Figure()
    for i in range(min(30, yearly.shape[1])):
        fig.add_trace(go.Scatter(x=yrs, y=yearly[:, i], mode='lines', showlegend=False,
                                 line=dict(color='rgba(130,130,130,0.12)', width=1), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=yrs, y=float(capital) + 12 * float(monthly_save) * yrs,
                             mode='lines', name='Eingezahltes Kapital',
                             line=dict(color='#327ac2', width=3, dash='dot')))
    for q, label, color, dash in [(5, '5%-Perzentil', '#cb403b', 'dash'),
                                  (50, 'Median', '#d7a300', 'solid'),
                                  (95, '95%-Perzentil', '#288e4b', 'dash')]:
        fig.add_trace(go.Scatter(x=yrs, y=np.percentile(yearly, q, axis=1), name=label,
                                 line=dict(color=color, width=3, dash=dash)))
    fig.update_layout(title=title, height=450, xaxis_title='Jahre', yaxis_title='Depotwert (€)',
                      margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig, use_container_width=True)
    paid = float(capital) + 12 * float(monthly_save) * int(years)
    median = float(np.percentile(yearly[-1], 50))
    cols = st.columns(3)
    cols[0].metric('Einzahlungen gesamt', format_de(paid))
    cols[1].metric('Depotwert im Median', format_de(median))
    cols[2].metric('Vermögenszuwachs im Median', format_de(median - paid))
    st.caption(f'Modellierte geldgewichtete Median-Rendite (IRR): '
               f'{format_de_num(mc_money_weighted_return(capital, monthly_save, years, median), 2)} % p. a. '
               'Die Einzahlungslinie zeigt nur eigenes Geld, keine Kursgewinne.')


# ==========================================
# TAB 8: KURSCHART MIT SMA 50 UND SMA 200
# ==========================================
CHART_PERIODS = (
    "1 Monat", "3 Monate", "6 Monate", "1 Jahr", "2 Jahre",
    "5 Jahre", "10 Jahre", "Gesamte Historie"
)


def prepare_sma_chart(prices):
    """Tageskurse aufbereiten; SMA anhand der letzten 50/200 Datenpunkte.

    Die Durchschnitte entstehen VOR dem zeitlichen Zuschneiden des Charts,
    damit die Linien auch am Beginn kurzer Ansichtsfenster korrekt sind.
    """
    empty = pd.DataFrame(columns=["Kurs", "SMA 50", "SMA 200"])
    if prices is None or len(prices) == 0:
        return empty
    series = pd.to_numeric(prices, errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    series = series[series > 0]
    if series.empty:
        return empty
    series = series.loc[~series.index.duplicated(keep="last")].sort_index()
    result = pd.DataFrame({"Kurs": series.astype(float)})
    result["SMA 50"] = result["Kurs"].rolling(50, min_periods=50).mean()
    result["SMA 200"] = result["Kurs"].rolling(200, min_periods=200).mean()
    return result


def restrict_sma_chart_period(data, period):
    """Auf sichtbare Zeitspanne begrenzen (ohne SMA-Aufwärmdaten zu löschen)."""
    if data.empty or period == "Gesamte Historie":
        return data.copy()
    units = {
        "1 Monat": pd.DateOffset(months=1),
        "3 Monate": pd.DateOffset(months=3),
        "6 Monate": pd.DateOffset(months=6),
        "1 Jahr": pd.DateOffset(years=1),
        "2 Jahre": pd.DateOffset(years=2),
        "5 Jahre": pd.DateOffset(years=5),
        "10 Jahre": pd.DateOffset(years=10),
    }
    if period not in units:
        raise ValueError(f"Unbekannter Chart-Zeitraum: {period}")
    cutoff = data.index.max() - units[period]
    return data.loc[data.index >= cutoff].copy()


@st.cache_data(ttl=900, max_entries=12, show_spinner=False)
def fetch_asset_chart_prices(ticker, period):
    """Tagesdaten für ein Yahoo-Finance-Symbol; separat vom Terminal-Cache.

    Auch für kurze Ansichten werden wenigstens zwei Jahre heruntergeladen,
    damit der SMA 200 früh im sichtbaren Ausschnitt gezeichnet werden kann.
    """
    history_years = {
        "1 Monat": 2, "3 Monate": 2, "6 Monate": 2,
        "1 Jahr": 2, "2 Jahre": 3, "5 Jahre": 6, "10 Jahre": 11,
    }
    if period not in CHART_PERIODS:
        return pd.Series(dtype=float)
    try:
        kwargs = dict(progress=False, auto_adjust=True, group_by="column",
                      threads=False, timeout=20, interval="1d")
        if period == "Gesamte Historie":
            raw = yf.download(ticker, period="max", **kwargs)
        else:
            start = pd.Timestamp.now() - pd.DateOffset(years=history_years[period])
            raw = yf.download(ticker, start=start.strftime("%Y-%m-%d"), **kwargs)
        close = extract_yf_field(raw, "Close")
        if close.empty:
            return pd.Series(dtype=float)
        return close.iloc[:, 0].dropna().astype(float)
    except Exception as exc:
        LOG.warning("Chartdaten für %s nicht abrufbar: %s", ticker, exc)
        return pd.Series(dtype=float)


# ---------------------------------------------
# APP LAYOUT: ALLE SIEBEN BESTEHENDEN TABS + NEUER CHART-TAB
# ---------------------------------------------
st.title("📊 Integriertes Finanz-Terminal")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8=st.tabs([
    "🌐 Makro & Markt", "🧭 Thematic Trends", "📈 Backtesting", "🎲 MC Asset",
    "🥧 MC Portfolio", "💶 Steuern", "🚀 S&P 500 Dual-Tracker",
    "📉 Chart & SMA",
], on_change="rerun", key="finanz_v10_active_tab")

# ==========================================
# TAB 1: MAKRO & MARKT DASHBOARD
# ==========================================
if tab1.open:
    with tab1:
        idx_config={"DAX":{"ticker":"^GDAXI","color":"#00407a"},
                    "S&P 500":{"ticker":"^GSPC","color":"#007a33"},
                    "NASDAQ":{"ticker":"^IXIC","color":"#d9531e"},
                    "STOXX 600":{"ticker":"^STOXX","color":"#6c2b6d"}}
        com_config={"Gold (Unze)":{"ticker":"GC=F","color":"#c99b24","mult":1,"unit":" €/Unze"},
                    "Silber (Unze)":{"ticker":"SI=F","color":"#757d85","mult":1,"unit":" €/Unze"},
                    "Öl Brent (Barrel)":{"ticker":"BZ=F","color":"#333333","mult":1,"unit":" €/Barrel"},
                    "Kaffee (Pfund)":{"ticker":"KC=F","color":"#6f4e37","mult":.01,"unit":" €/lb"},
                    "Kupfer (kg)":{"ticker":"HG=F","color":"#b66b42","mult":2.2046226218,"unit":" €/kg"}}
        bond_colors={"USA (3 Monate)":"#389e9f","USA (5 Jahre)":"#e0782a",
                     "USA (10 Jahre)":"#235789","USA (30 Jahre)":"#7a1c3d"}
        eurusd_series=fetch_eurusd(365)
        official_ecb,official_fed=fetch_policy_rates()
        ecb_display=(f"{official_ecb:.2f}".replace(".",",") if official_ecb is not None else EZB_EINLAGEZINS)
        fed_display=official_fed or FED_ZINSBAND
        eurusd_ok=len(eurusd_series)>1 and np.isfinite(eurusd_series.iloc[-1]) and eurusd_series.iloc[-1]>0
        cur_fx=float(eurusd_series.iloc[-1]) if eurusd_ok else None
        day_fx=((cur_fx/eurusd_series.iloc[-2]-1)*100) if eurusd_ok and eurusd_series.iloc[-2]>0 else np.nan
        macro, fngcol=st.columns([2.2,1])
        with macro:
            st.subheader("💱 Währung & Zentralbanken")
            c1,c2,c3=st.columns(3)
            c1.metric("🇪🇺 EZB-Einlagezinssatz", f"{ecb_display} %" if ecb_display else "–")
            c2.metric("🇺🇸 Fed-Zinsband", f"{fed_display} %" if fed_display else "–")
            c3.metric("EUR/USD", f"${cur_fx:.4f}" if eurusd_ok else "Nicht verfügbar",
                      f"{day_fx:+.2f} %" if np.isfinite(day_fx) else None)
            if not ecb_display or not fed_display:
                st.caption("Falls amtliche Zinsschnittstellen ausfallen, lassen sich `EZB_EINLAGEZINS` "
                           "und `FED_ZINSBAND` in den Secrets als manuelle Werte pflegen. "
                           "Fehlende Werte werden nicht geschätzt.")
            if not eurusd_ok:
                st.warning('EUR/USD konnte nicht geladen werden; Rohstoffumrechnung wird übersprungen.')
        with fngcol:
            st.subheader("🧭 CNN Fear & Greed")
            # Manueller erneuter Abruf ohne globale Löschung anderer Marktdaten-Caches.
            if st.button("↻ Indikator aktualisieren", key="fng_refresh", type="tertiary"):
                fetch_fear_and_greed.clear()
            with st.spinner("Lade Fear-&-Greed-Index ..."):
                fng = fetch_fear_and_greed()
            score = fng["score"]
            if score is None:
                st.metric("Marktstimmung", "–")
                st.warning("CNN und die Ersatzdatenquellen liefern derzeit keinen "
                           "ausreichend aktuellen Messwert. Es wird kein Wert geschätzt.")
                with st.expander("Abrufdiagnose"):
                    st.code(fng.get("issue", "Keine technischen Details"))
                st.markdown("[Fear & Greed direkt bei CNN ansehen]"
                            "(https://www.cnn.com/markets/fear-and-greed)")
            else:
                score_color = ("#e74c3c" if score < 25 else "#f39c12" if score < 45
                               else "#f1c40f" if score <= 55 else "#2ecc71" if score <= 75
                               else "#27ae60")
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number", value=score,
                    number={"valueformat": ".0f", "font": {"size": 43}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": score_color, "thickness": .28},
                        "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                        "steps": [
                            {"range": [0, 25], "color": "#f5d1d1"},
                            {"range": [25, 45], "color": "#f8dfbd"},
                            {"range": [45, 55], "color": "#f9f0bf"},
                            {"range": [55, 75], "color": "#cdefd9"},
                            {"range": [75, 100], "color": "#bde6cc"},
                        ],
                    },
                ))
                gauge.update_layout(
                    height=220, margin=dict(l=12, r=12, t=20, b=3),
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(gauge, use_container_width=True,
                                config={"displayModeBar": False})
                st.markdown(f"**{fng['label']}**")
                st.caption(f"Stand: {pd.Timestamp(fng['asof']).strftime('%d.%m.%Y')} · "
                           f"Quelle: {fng['source']}")
                if "Spiegel" in fng["source"]:
                    st.caption("Ersatzdatenquelle: von CNN abgeleitete Drittanbieterdaten; "
                               "können zeitverzögert sein.")
        st.divider()
        st.markdown("### 📈 Indizes (letzte 365 Tage)")
        indices=fetch_macro_data({k:v["ticker"] for k,v in idx_config.items()})
        for column,(name,cfg) in zip(st.columns(4),idx_config.items()):
            with column:
                plot_handelsblatt_chart(indices.get(name),name,cfg["color"],val_format=0,suffix=" Pkt.")
        st.divider()
        st.markdown("### 🛢️ Rohstoffe in Euro (letzte 365 Tage)")
        commodities=fetch_macro_data({k:v["ticker"] for k,v in com_config.items()})
        for column,(name,cfg) in zip(st.columns(5),com_config.items()):
            with column:
                series=commodities.get(name)
                if series is None or not eurusd_ok:
                    st.markdown(f"**{name}**")
                    st.caption("Keine Daten oder kein EUR/USD-Kurs")
                else:
                    paired=pd.concat([series.rename("usd"),eurusd_series.rename("fx")],axis=1,join="inner").dropna()
                    plot_handelsblatt_chart(paired["usd"]*cfg["mult"]/paired["fx"],name,cfg["color"],
                                           val_format=2,suffix=cfg["unit"])
        st.caption("Rohstoff-Futures statt physischer Kassapreise. Kaffee-Futures: US-Cent/lb; "
                   "Kupfer (HG=F): USD/lb, umgerechnet in €/kg (1 kg ≈ 2,20462 lb). "
                   "Future-Roll-Effekte und Wechselkurse beeinflussen die Zeitreihe.")
        st.divider()
        st.markdown("### 🏛️ US-Zinsstrukturkurve (letzte 365 Tage)")
        bonds=fetch_bonds_robust()
        for column,(name,color) in zip(st.columns(4),bond_colors.items()):
            with column:
                plot_handelsblatt_chart(bonds.get(name),name,color,val_format=2,suffix=" %")
        st.caption("Quelle: Yahoo-Finance-Renditeindikatoren (^IRX, ^FVX, ^TNX, ^TYX), indikative Marktdaten.")

    # ==========================================
    # TAB 2: THEMATIC ETF TREND & STIMMUNG
# ==========================================
if tab2.open:
    with tab2:
        st.markdown("### 🧭 Thematic Trends – Nachrichtenstimmung & ETF-Momentum")
        st.caption("Englische Nachrichten über NewsAPI, VADER-Sentiment und ETF-Kurstrend; "
                   "Stichproben- und Medienbias möglich. Historischer Vergleich 7 vs. 28 Tage.")
        if st.button("🔄 Themen neu laden",key="thematic_refresh"):
            load_all_thematic_data.clear()
            fetch_articles.clear()
        if not NEWS_API_KEY:
            st.info("Bitte `NEWS_API_KEY` als Umgebungsvariable oder in `.streamlit/secrets.toml` eintragen.")
        else:
            try:
                with st.spinner("Analysiere Nachrichten und Kursentwicklung..."):
                    thematic=load_all_thematic_data()
                st.subheader("Aktuelle Stimmungslage")
                fig=px.bar(thematic,x="Stimmungs-Index (0-100)",y="Thema",orientation="h",
                           color="Stimmungs-Index (0-100)",color_continuous_scale="RdYlGn",
                           range_color=[30,70],text="Stimmungs-Index (0-100)")
                fig.add_vline(x=50,line_dash="dot",line_color="gray")
                fig.update_layout(xaxis_range=[0,100],yaxis={"categoryorder":"total ascending"})
                st.plotly_chart(fig,use_container_width=True)
                st.subheader("Detailauswertung")
                st.dataframe(thematic,hide_index=True,use_container_width=True)
            except Exception as exc:
                st.error(f"News-Auswertung gescheitert: {exc}")

    # ==========================================
    # TAB 3: BACKTESTING EINZELTITEL
# ==========================================
if tab3.open:
    with tab3:
        st.markdown("### 📈 Strategie-Backtest")
        c1,c2,c3=st.columns(3)
        with c1:
            bt_ticker=st.text_input("Zu testender Ticker (z.B. AAPL)","AAPL",key="bt_ticker_input").strip().upper()
            if bt_ticker:
                st.caption(f"Asset: **{get_asset_name(bt_ticker)}**")
        with c2:
            bt_start=st.date_input("Startdatum",datetime.now()-timedelta(days=365*5),key="bt_start_input")
        with c3:
            bt_end=st.date_input("Enddatum",datetime.now(),key="bt_end_input")
        choice=st.radio("Handelsstrategie",["Aktives Trading (SMA 200 Trendfolge)","Buy & Hold"],
                        horizontal=True,key="bt_strategy_radio")
        if st.button("🚀 Backtest ausführen",key="btn_bt"):
            if not bt_ticker or bt_start>=bt_end:
                st.error("Bitte einen gültigen Ticker und Startdatum vor Enddatum angeben.")
            else:
                with st.spinner("Berechne Backtest..."):
                    try:
                        stock=fetch_historical(bt_ticker,max_history=True)
                        if len(stock)<201:
                            raise ValueError("Zu wenige historische Daten für SMA200.")
                        data=pd.DataFrame({"Price":stock})
                        for t,label in [("SPY","SPY"),("GLD","GLD")]:
                            s=fetch_historical(t,max_history=True)
                            if not s.empty:
                                data[label]=s.reindex(data.index)
                        data["sma200"]=data["Price"].rolling(200,min_periods=200).mean()
                        data["signal"]=(data["Price"]>data["sma200"]).astype(float).shift(1).fillna(0.)
                        data["return_stock"]=data["Price"].pct_change(fill_method=None)
                        active="Aktives Trading" in choice
                        data["strategy"]=data["return_stock"]*(data["signal"] if active else 1.)
                        for t in ["SPY","GLD"]:
                            if t in data:
                                data[f"ret_{t}"]=data[t].pct_change(fill_method=None)
                        selected=data.loc[(data.index>=pd.Timestamp(bt_start)) & (data.index<=pd.Timestamp(bt_end))].copy()
                        if len(selected)<3:
                            raise ValueError("Im gewünschten Zeitraum sind nicht genug Kursdaten verfügbar.")
                        # Erstes Datum ist der gemeinsame Nullpunkt, keine Rendite aus der Vorperiode.
                        columns={f"{bt_ticker} ({'SMA200' if active else 'Buy & Hold'})":"strategy"}
                        if "ret_SPY" in selected:
                            columns["S&P 500 (SPY)"]="ret_SPY"
                        if "ret_GLD" in selected:
                            columns["Gold (GLD)"]="ret_GLD"
                        paths=pd.DataFrame(index=selected.index)
                        for label,col in columns.items():
                            returns=selected[col].copy().fillna(0.)
                            returns.iloc[0]=0.
                            paths[label]=((1.+returns).cumprod()-1)*100
                        fig=px.line(paths,title=f"Vergleich ab {selected.index[0]:%d.%m.%Y}",height=500)
                        fig.update_layout(yaxis_title="Gesamtrendite (%)",xaxis_title="Datum")
                        st.plotly_chart(fig,use_container_width=True)
                        st.dataframe(pd.DataFrame([{name:f"{format_de_num(paths[name].iloc[-1])} %" for name in paths}]),
                                     hide_index=True,use_container_width=True)
                        st.caption("Das SMA200-Signal wird um einen Handelstag verzögert verwendet. "
                                   "Dividenden sind näherungsweise über bereinigte Kurse berücksichtigt; "
                                   "Kosten, Steuern und FX bleiben unberücksichtigt.")
                    except Exception as exc:
                        st.error(f"Backtest fehlgeschlagen: {exc}")

    # ==========================================
    # TAB 4: EINZEL-ASSET MONTE-CARLO
# ==========================================
if tab4.open:
    with tab4:
        st.markdown("### 🎲 Einzel-Asset Monte-Carlo")
        c1,c2,c3,c4=st.columns(4)
        with c1:
            mc_ticker=st.text_input("Ticker für historische Daten","QQQ",key="mc1").strip().upper()
            if mc_ticker:
                st.caption(f"Asset: **{get_asset_name(mc_ticker)}**")
        with c2:
            start_cap=st.number_input("Startkapital (€)",min_value=0,value=10000,step=1000,key="mc2")
        with c3:
            monthly_save=st.number_input("Monatl. Sparrate (€)",min_value=0,value=500,step=50,key="mc3")
        with c4:
            years=st.slider("Anlagehorizont (Jahre)",min_value=1,max_value=50,value=30,key="mc4")
        mc_history=st.selectbox('Historischer Zeitraum für die Kalibrierung',
                                ['Gesamte Historie','5 Jahre','10 Jahre','20 Jahre'],
                                key='mc_history')
        if st.button("🎲 Simulation starten",key="btn_mc"):
            with st.spinner("Simuliere Einzelwert..."):
                try:
                    closes=mc_window(fetch_historical(mc_ticker,max_history=True),mc_history)
                    if len(closes)<253:
                        raise ValueError("Mindestens ein Jahr historische Kurse erforderlich.")
                    logr=np.log(closes/closes.shift(1)).replace([np.inf,-np.inf],np.nan).dropna()
                    yearly,final_values,cagrs=gaussian_mc(float(logr.mean()),float(logr.std()),
                                                           start_cap,monthly_save,years)
                    st.caption(f'Historische Stichprobe: {closes.index.min():%d.%m.%Y} – '
                               f'{closes.index.max():%d.%m.%Y} ({len(logr)} Tagesrenditen).')
                    render_mc_plot(yearly,years,f"Monte-Carlo Simulation ({years} Jahre, {mc_ticker})",
                                   start_cap,monthly_save)
                    c1,c2,c3=st.columns(3)
                    for col,p,label in zip([c1,c2,c3],[5,50,95],["5%-Perzentil","Median","95%-Perzentil"]):
                        col.metric(label,format_de(np.percentile(final_values,p)),
                                   f"{format_de_num(np.percentile(cagrs,p),1)} % simulierte CAGR",delta_color="off")
                    st.caption("Illustrative i.i.d.-Lognormal-Simulation ohne Krisen-Cluster, historische "
                               "Regimewechsel oder Steuern. Die 'CAGR' bezieht sich auf den Kursprozess "
                               "ohne Sparraten, der Endwert enthält Sparraten. Keine Prognose.")
                except Exception as exc:
                    st.error(f"Monte-Carlo-Berechnung fehlgeschlagen: {exc}")

    # ==========================================
    # TAB 5: PORTFOLIO MONTE-CARLO
# ==========================================
if tab5.open:
    with tab5:
        st.markdown("### 🥧 Portfolio Monte-Carlo Simulation")
        left,right=st.columns([1.1,1.9])
        with left:
            st.markdown("##### ⚙️ Allokation & Parameter")
            edited=st.data_editor(pd.DataFrame({"Ticker":["QQQ","SPY","GLD"],
                                                 "Gewichtung":[60.,20.,20.]}),
                                  num_rows="dynamic",hide_index=True,use_container_width=True,key="portfolio_editor")
            names=[f"• **{str(row['Ticker']).strip().upper()}**: {get_asset_name(row['Ticker'])}"
                   for _,row in edited.iterrows() if pd.notna(row.get("Ticker")) and str(row["Ticker"]).strip()]
            if names:
                st.caption("**Erkannte Assets:**\n"+"\n".join(names))
            cap=st.number_input("Startkapital (€)",min_value=0,value=25000,step=1000,key="p_cap")
            save=st.number_input("Monatl. Sparrate (€)",min_value=0,value=750,step=50,key="p_save")
            horizon=st.slider("Anlagehorizont (Jahre)",1,50,30,key="p_years")
            port_history=st.selectbox('Historischer Zeitraum für die Kalibrierung',
                                      ['Gesamte Historie','5 Jahre','10 Jahre','20 Jahre'],
                                      key='portfolio_history')
            clicked=st.button("🎲 Portfolio simulieren",use_container_width=True,type="primary",key="p_run")
        with right:
            if clicked:
                try:
                    rows=edited.copy()
                    rows["Ticker"]=rows["Ticker"].fillna("").astype(str).str.strip().str.upper()
                    rows["Gewichtung"]=pd.to_numeric(rows["Gewichtung"],errors="coerce")
                    rows=rows.loc[rows["Ticker"]!=""]
                    if rows.empty or rows["Gewichtung"].isna().any() or (rows["Gewichtung"]<0).any():
                        raise ValueError("Bitte gültige, nichtnegative Gewichtungen eingeben.")
                    rows=rows.groupby("Ticker",as_index=False)["Gewichtung"].sum()
                    if rows["Gewichtung"].sum()<=0:
                        raise ValueError("Gesamtgewichtung muss größer als null sein.")
                    frames={t:fetch_historical(t,max_history=True) for t in rows["Ticker"]}
                    missing=[t for t,s in frames.items() if s.empty]
                    if missing:
                        raise ValueError("Fehlende Kursdaten für: "+", ".join(missing))
                    close=mc_window(pd.concat(frames,axis=1,join="inner").dropna(),port_history)
                    if len(close)<253:
                        raise ValueError("Zu wenig gemeinsame Kursdaten (mindestens 253 Tage).")
                    weights=rows.set_index("Ticker").loc[close.columns,"Gewichtung"].to_numpy(dtype=float)
                    weights=weights/weights.sum()
                    st.dataframe(pd.DataFrame({"Ticker":close.columns,
                                               "Asset / Name":[get_asset_name(t) for t in close.columns],
                                               "Effektive Gewichtung": [f"{w*100:.1f} %" for w in weights]}),
                                 hide_index=True,use_container_width=True)
                    stock_returns=close.pct_change(fill_method=None).dropna()
                    # Gleichgewichtet entsprechend Vorgabe jeden Tag rebalanciertes Portfolio:
                    port_simple=stock_returns.to_numpy() @ weights
                    if np.any(port_simple<=-1):
                        raise ValueError("Ungültige historische Portfoliorenditen.")
                    port_log=np.log1p(port_simple)
                    yearly,final_values,cagrs=gaussian_mc(float(port_log.mean()),float(port_log.std(ddof=1)),
                                                           cap,save,horizon)
                    st.caption(f'Gemeinsame historische Stichprobe: {close.index.min():%d.%m.%Y} – '
                               f'{close.index.max():%d.%m.%Y} ({len(port_log)} Tagesrenditen).')
                    render_mc_plot(yearly,horizon,f"Depot-Simulation ({horizon} Jahre)",cap,save)
                    c1,c2,c3=st.columns(3)
                    for col,p,label in zip([c1,c2,c3],[5,50,95],["5%-Perzentil","Median","95%-Perzentil"]):
                        col.metric(label,format_de(np.percentile(final_values,p)),
                                   f"{format_de_num(np.percentile(cagrs,p),1)} % simulierte CAGR",delta_color="off")
                    st.caption("Historische Korrelation ist in den aggregierten Portfoliorenditen enthalten. "
                               "Die Simulation nimmt konstante Gewichte/tägliches Rebalancing, "
                               "lognormalverteilte i.i.d.-Erträge und keine Steuern/Kosten an. "
                               "Renditen/Endwerte sind keine Vorhersagen.")
                except Exception as exc:
                    st.error(f"Portfoliosimulation fehlgeschlagen: {exc}")

    # ==========================================
    # TAB 6: EINKOMMEN & STEUERN
# ==========================================
if tab6.open:
    with tab6:
        st.markdown("### 💶 Steuer- und Netto-Rechner")
        st.caption("Vereinfachte Einzelveranlagungs-Rechnung für 2026 – keine Steuerberatung und keine "
                   "vollständige Lohnsteuer-/Sozialversicherungsabrechnung.")
        left,right=st.columns([1,2])
        with left:
            brutto=st.number_input("Bruttojahresgehalt (€)",min_value=0,value=75000,step=1000)
            kv=st.radio("Krankenversicherung",["Gesetzlich (GKV)","Privat (PKV)"],horizontal=True)
            pkv=st.number_input("Selbst getragener PKV-Jahresbeitrag (€)",min_value=0,value=4800,step=100) if kv=="Privat (PKV)" else 0
            rv=st.radio("Renten-/Arbeitslosenversicherung",["Pflichtig","Befreit"],horizontal=True)
            childless=st.checkbox("Kinderlos (PV-Zuschlag, ab 23 J.)",value=True)
            extras=st.number_input("Werbungskosten (€)",min_value=0,value=10000,step=500)
            gkv_add=st.number_input("GKV-Zusatzbeitrag gesamt (%)",min_value=0.,max_value=8.,value=2.9,step=.1) if kv=="Gesetzlich (GKV)" else 2.9
        with right:
            kwargs=dict(is_pkv=kv=="Privat (PKV)",pkv_beitrag=pkv,
                        has_grv=rv=="Pflichtig",childless=childless,additional_health_rate=gkv_add)
            baseline=calculate_german_tax(brutto,werbungskosten=1230,**kwargs)
            optimized=calculate_german_tax(brutto,werbungskosten=max(1230,extras),**kwargs)
            difference=baseline["steuer"]-optimized["steuer"]
            st.success(f"**Modellierte Steuerersparnis ggü. 1.230 € Pauschbetrag: {format_de(max(0.,difference))}**")
            c1,c2,c3=st.columns(3)
            c1.metric("Jahresbrutto",format_de(brutto))
            c2.metric("Modellierte Steuern",format_de(optimized["steuer"]))
            c3.metric("Arbeitnehmer-SV",format_de(optimized["sv"]))
            st.divider()
            c4,c5=st.columns(2)
            c4.metric("Modelliertes Jahresnetto",format_de(optimized["netto"]))
            c5.metric("Ø Monatsnetto",format_de(optimized["netto"]/12))
            st.info("2026: Grundtarif §32a EStG, BBG RV 101.400 €, BBG KV/PV 69.750 €. "
                    "Vereinfachungen bei PKV, Eltern mit mehreren Kindern, Sachsen, Splitting, "
                    "sonstigen Sonderausgaben, Vorsorgeaufwendungen und Arbeitgeberzuschüssen. "
                    "Die Differenz ist eine rechnerische Steuerersparnis, **keine zugesagte Erstattung**.")


# Depot- und Journalfunktionen müssen außerhalb jeder tabX.open-Bedingung stehen,
# damit Tab 7 auch bei direktem Aufruf ohne vorherigen Besuch von Tab 6 funktioniert.
# ================================================================
# V10.1: PERSOENLICHER DEPOTMODUS (global, unabhängig von lazy Tabs)
# Speicherung waehrend der Session, dauerhafte JSON-Datei via Export/Import.
# ================================================================
import json
import re
import math
from copy import deepcopy

_MT_DEPOT_SCHEMA = 1
_MT_SYMBOL_PATTERN = re.compile(r'^[A-Z0-9][A-Z0-9.\-]{0,14}$')


def mt_depot_now():
    return datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds')


def mt_depot_empty():
    return {'schema_version': _MT_DEPOT_SCHEMA, 'strategy': 'A',
            'cash_eur': 0.0, 'positions': {}, 'journal': [],
            'acknowledged_signal': None, 'updated_at': mt_depot_now()}


def mt_depot_number(value, name, allow_zero=True):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name}: ungültiger Zahlenwert.') from exc
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0) or value > 1e9:
        raise ValueError(f'{name}: Wert muss endlich und zwischen 0 und 1 Mrd. liegen.')
    return value


def mt_depot_ticker(value):
    ticker = str(value).strip().upper().replace('.', '-')
    if not _MT_SYMBOL_PATTERN.fullmatch(ticker):
        raise ValueError(f'Ungültiges Aktienkürzel: {str(value)[:30]}')
    return ticker


def mt_depot_validate(state):
    """Validiert JSON-Backup strikt; liest weder Code noch fremde Dateipfade."""
    if not isinstance(state, dict) or state.get('schema_version') != _MT_DEPOT_SCHEMA:
        raise ValueError('Unbekanntes Backup-Format. Erwartet: Finanz-Multitool Depot V8 (Version 1).')
    strategy = state.get('strategy')
    if strategy not in ('A', 'B'):
        raise ValueError('Strategie muss A oder B sein.')
    positions = state.get('positions')
    if not isinstance(positions, dict) or len(positions) > 100:
        raise ValueError('Ungültige Positionsliste (maximal 100).')
    cleaned = {}
    for symbol, data in positions.items():
        sym = mt_depot_ticker(symbol)
        if sym in cleaned or not isinstance(data, dict):
            raise ValueError('Doppelte oder ungültige Depotposition.')
        quantity = mt_depot_number(data.get('quantity'), f'{sym} Stückzahl', False)
        cost = mt_depot_number(data.get('avg_entry_eur', 0), f'{sym} Ø Einstand')
        cleaned[sym] = {'quantity': quantity, 'avg_entry_eur': cost}
    journal = state.get('journal', [])
    if not isinstance(journal, list) or len(journal) > 5000:
        raise ValueError('Ungültiges Handelsjournal (maximal 5.000 Einträge).')
    safe_journal = []
    for row in journal:
        if not isinstance(row, dict) or len(row) > 16:
            raise ValueError('Ungültiger Journaleintrag.')
        safe_journal.append({str(k)[:40]: str(v)[:200] if isinstance(v, str) else v
                             for k, v in row.items() if isinstance(v, (str, int, float, bool, type(None)))})
    acknowledged = state.get('acknowledged_signal')
    if acknowledged is not None and (not isinstance(acknowledged, str) or len(acknowledged) > 200):
        raise ValueError('Ungültiges letztes Signal.')
    return {'schema_version': _MT_DEPOT_SCHEMA, 'strategy': strategy,
            'cash_eur': mt_depot_number(state.get('cash_eur'), 'Cash'),
            'positions': cleaned, 'journal': safe_journal,
            'acknowledged_signal': acknowledged, 'updated_at': str(state.get('updated_at', ''))[:50]}


def mt_depot_backup_bytes(state):
    normalized = mt_depot_validate(state)
    return json.dumps(normalized, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')


def mt_depot_import_bytes(raw):
    if len(raw) > 1024 * 1024:
        raise ValueError('Backup ist größer als 1 MB.')
    try:
        obj = json.loads(raw.decode('utf-8-sig'))
    except (ValueError, UnicodeError) as exc:
        raise ValueError('Datei enthält kein gültiges JSON.') from exc
    return mt_depot_validate(obj)


def mt_depot_log(state, kind, symbol='', qty=0, price=0, fee=0, tax=0, note='', trade_date=None):
    state['journal'].append({
        'Zeitpunkt': str(trade_date or mt_depot_now()), 'Art': str(kind),
        'Ticker': str(symbol), 'Stückzahl': round(float(qty), 7),
        'Kurs EUR': round(float(price), 6), 'Gebühren EUR': round(float(fee), 2),
        'Steuer EUR': round(float(tax), 2), 'Hinweis': str(note)[:200]
    })
    if len(state['journal']) > 5000:
        state['journal'] = state['journal'][-5000:]
    state['updated_at'] = mt_depot_now()


def mt_depot_set_snapshot(state, rows, cash_eur):
    new_state = deepcopy(mt_depot_validate(state))
    new_positions = {}
    for row in rows:
        sym_raw = row.get('Ticker', '')
        if sym_raw is None or pd.isna(sym_raw) or not str(sym_raw).strip():
            continue
        sym = mt_depot_ticker(sym_raw)
        if sym in new_positions:
            raise ValueError(f'Ticker {sym} ist doppelt erfasst.')
        raw_qty = row.get('Stückzahl', 0)
        qty = mt_depot_number(raw_qty, f'{sym}: Stückzahl')
        cost = mt_depot_number(row.get('Ø Einstand EUR', 0), f'{sym}: Einstand')
        if qty > 0:
            new_positions[sym] = {'quantity': qty, 'avg_entry_eur': cost}
    if len(new_positions) > 100:
        raise ValueError('Zu viele Depotpositionen.')
    changed = (new_positions != new_state['positions'] or
               mt_depot_number(cash_eur, 'Cash') != new_state['cash_eur'])
    new_state['positions'] = new_positions
    new_state['cash_eur'] = mt_depot_number(cash_eur, 'Cash')
    if changed:
        mt_depot_log(new_state, 'BESTANDSABGLEICH', note='Manuelle Bestandseingabe; kein rekonstruierter Kauf/Verkauf')
    return new_state


def mt_depot_trade(state, side, ticker, quantity, price_eur, fees_eur=0, taxes_eur=0, trade_date=None):
    new_state = deepcopy(mt_depot_validate(state))
    sym = mt_depot_ticker(ticker)
    qty = mt_depot_number(quantity, 'Stückzahl', False)
    price = mt_depot_number(price_eur, 'Ausführungskurs EUR', False)
    fee = mt_depot_number(fees_eur, 'Gebühren EUR')
    tax = mt_depot_number(taxes_eur, 'Tatsächlicher Steuerabzug EUR')
    position = new_state['positions'].get(sym, {'quantity': 0.0, 'avg_entry_eur': 0.0})
    if side == 'KAUF':
        debit = qty * price + fee + tax
        if debit > new_state['cash_eur'] + 1e-7:
            raise ValueError(f'Nicht genug Cash für den Kauf (benötigt {debit:,.2f} €).')
        held = position['quantity']
        weighted = ((held * position['avg_entry_eur'] + qty * price + fee) / (held + qty))
        new_state['positions'][sym] = {'quantity': held + qty, 'avg_entry_eur': weighted}
        new_state['cash_eur'] = max(0.0, new_state['cash_eur'] - debit)
    elif side == 'VERKAUF':
        held = position['quantity']
        if qty > held + 1e-7:
            raise ValueError(f'Verkauf überschreitet die gehaltenen Stücke ({held:g}).')
        credit = qty * price - fee - tax
        if credit < 0:
            raise ValueError('Gebühren und Steuern übersteigen den Verkaufserlös.')
        remaining = held - qty
        if remaining <= 1e-7:
            new_state['positions'].pop(sym, None)
        else:
            new_state['positions'][sym] = {'quantity': remaining,
                                          'avg_entry_eur': position['avg_entry_eur']}
        new_state['cash_eur'] += credit
    else:
        raise ValueError('Unbekannte Transaktionsart.')
    mt_depot_log(new_state, side, sym, qty, price, fee, tax, trade_date=trade_date)
    return new_state


def mt_depot_cash(state, side, amount, trade_date=None):
    new_state = deepcopy(mt_depot_validate(state))
    value = mt_depot_number(amount, 'Betrag', False)
    if side == 'EINZAHLUNG':
        new_state['cash_eur'] += value
    elif side == 'AUSZAHLUNG':
        if value > new_state['cash_eur'] + 1e-7:
            raise ValueError('Auszahlung übersteigt verfügbares Cash.')
        new_state['cash_eur'] -= value
    else:
        raise ValueError('Unbekannte Cash-Buchung.')
    mt_depot_log(new_state, side, note=f'{value:.2f} EUR', trade_date=trade_date)
    return new_state


def mt_depot_plan(state, target_frame, prices_eur, tolerance=0.02, min_trade_eur=50):
    """Unverbindlicher Ziel-Ist-Abgleich nach EUR, nie eine Orderübermittlung."""
    state = mt_depot_validate(state)
    weights = {}
    if not target_frame.empty:
        for item in target_frame.to_dict('records'):
            sym = mt_depot_ticker(item['Ticker'])
            pct = float(item['Zielgewicht (%)'])
            if not math.isfinite(pct) or not 0 <= pct <= 100:
                raise ValueError('Ungültige Zielgewichte.')
            weights[sym] = pct / 100.0
    all_symbols = set(weights) | set(state['positions'])
    missing = sorted(sym for sym in all_symbols if not math.isfinite(prices_eur.get(sym, math.nan))
                     or prices_eur.get(sym, 0) <= 0)
    if missing:
        return [], math.nan, missing
    nav = state['cash_eur'] + sum(pos['quantity'] * prices_eur[sym]
                                  for sym, pos in state['positions'].items())
    if nav <= 0:
        return [], nav, []
    rows = []
    for sym in sorted(all_symbols):
        price = float(prices_eur[sym])
        held = state['positions'].get(sym, {}).get('quantity', 0.0)
        actual_eur = held * price
        target_eur = nav * weights.get(sym, 0.0)
        difference = target_eur - actual_eur
        material = max(float(min_trade_eur), nav * float(tolerance))
        if weights.get(sym, 0.0) == 0 and held > 0:
            action, quantity = 'VERKAUF', held
        elif held == 0 and weights.get(sym, 0.0) > 0:
            quantity = math.floor(max(0.0, difference) / price)
            action = 'KAUF' if quantity >= 1 else 'KAUF NICHT MÖGLICH (1 STÜCK ZU TEUER)'
        elif abs(difference) < material:
            action, quantity = 'HALTEN', 0.0
        elif difference > 0:
            quantity = math.floor(difference / price)
            action = 'KAUF' if quantity >= 1 else 'HALTEN (UNTER 1 AKTIE)'
        else:
            quantity = min(held, math.floor((-difference) / price))
            action = 'VERKAUF' if quantity >= 1 else 'HALTEN (UNTER 1 AKTIE)'
        rows.append({'Ticker': sym, 'Aktion': action,
                     'Bestand (Stück)': held, 'Indikativer Kurs (€)': round(price, 2),
                     'Ist-Wert (€)': round(actual_eur, 2), 'Ziel-Wert (€)': round(target_eur, 2),
                     'Differenz (€)': round(difference, 2), 'Stück grob': quantity})
    return rows, nav, []


@st.cache_data(ttl=900, max_entries=16, show_spinner=False)
def mt_depot_latest_quotes(tickers, asof_date):
    """Unadjusted Aktien-Schlusskurse für EURO-Indikationen, keine Live-Brokerquotes."""
    if not tickers:
        return {}, None
    selected = sorted({mt_depot_ticker(t) for t in tickers})
    raw = yf.download(selected, period='1mo', auto_adjust=False,
                      actions=False, progress=False, threads=False, group_by='column')
    close = extract_yf_field(raw, 'Close')
    if len(selected) == 1 and len(close.columns) == 1:
        close.columns = selected
    prices = {}
    max_date = pd.Timestamp(asof_date).normalize()
    for symbol in selected:
        if symbol not in close:
            continue
        s = close[symbol].loc[:max_date].dropna()
        if s.empty or (max_date - s.index[-1]).days > 8:
            continue
        px = float(s.iloc[-1])
        if np.isfinite(px) and px > 0:
            prices[symbol] = px
    fx = fetch_historical('EURUSD=X', years=1)
    fx = fx.loc[:max_date].dropna() if not fx.empty else pd.Series(dtype=float)
    if fx.empty or (max_date - fx.index[-1]).days > 8 or fx.iloc[-1] <= 0:
        return {}, None
    eurusd = float(fx.iloc[-1])
    return {t: value / eurusd for t, value in prices.items()}, eurusd

# ==========================================
# V8: STREAMLIT-DEPOT-OBERFLAECHE
# ==========================================
def mt_live_depot_controls():
    """V10.2: Zentrale manuelle Depotverwaltung statt verstecktem Seiten-Expander.

    Ein Formular für alle realen Buchungen, getrennt von Simulationstrades.
    Zustand wie bisher nur in st.session_state; JSON-Backup unverändert.
    """
    if 'mt_live_depot' not in st.session_state:
        st.session_state['mt_live_depot'] = mt_depot_empty()
        st.session_state['mt_live_revision'] = 0
    state = st.session_state['mt_live_depot']
    rev = st.session_state['mt_live_revision']
    st.subheader('📒 Mein echtes Depot & Handelstagebuch')
    st.caption('**Echte Buchungen**, unabhängig von den simulierten Trades. '
               'Keine automatische Orderübermittlung. Änderungen müssen als privates JSON gesichert werden.')
    status_a, status_b, status_c = st.columns(3)
    status_a.metric('Positionen', len(state['positions']))
    status_b.metric('Verfügbares Cash', format_de(state['cash_eur']))
    status_c.metric('Buchungen', len(state['journal']))
    section_book, section_holdings, section_backup = st.tabs(
        ['✏️ Neue Buchung', '📋 Bestand bearbeiten', '📚 Journal & Backup'])
    with section_book:
        st.caption('Hier dokumentierst du einen **tatsächlich ausgeführten** Trade '
                   'oder eine Einzahlung/Auszahlung. Keine Modellorder wird automatisch gebucht.')
        # Auswahl bewusst AUSSERHALB des Formulars: Streamlit-Forms aktualisieren
        # abhängige Felder erst beim Submit. Die Eingabefelder müssen aber sofort
        # beim Wechsel zwischen Kauf/Verkauf und Ein-/Auszahlung umschalten.
        kind = st.selectbox('Buchungstyp', ['KAUF', 'VERKAUF', 'EINZAHLUNG', 'AUSZAHLUNG'],
                            key=f'mt_live_kind_{rev}')
        with st.form(f'mt_live_one_booking_{rev}', clear_on_submit=True):
            if kind in ('KAUF', 'VERKAUF'):
                c1, c2, c3 = st.columns(3)
                with c1:
                    ticker = st.text_input('Ticker (z. B. MSFT)', key=f'mt_live_ticker_{rev}')
                with c2:
                    qty = st.number_input('Stückzahl', min_value=0.000001, value=1.0,
                                          format='%.6f', key=f'mt_live_qty_{rev}')
                with c3:
                    px = st.number_input('Ausführungskurs in € je Aktie',
                                         min_value=0.01, value=100.0, key=f'mt_live_px_{rev}')
                fee_c, tax_c = st.columns(2)
                with fee_c:
                    fee = st.number_input('Orderkosten (€)', min_value=0.0, value=0.0,
                                          key=f'mt_live_fee_{rev}')
                with tax_c:
                    tax = st.number_input('Einbehaltene Steuer (€)', min_value=0.0, value=0.0,
                                          key=f'mt_live_tax_{rev}')
            else:
                amount = st.number_input('Ein- / Auszahlungsbetrag (€)',
                                         min_value=0.01, value=500.0,
                                         key=f'mt_live_amount_{rev}')
            date = st.date_input('Tatsächliches Buchungsdatum', value=datetime.now().date(),
                                 key=f'mt_live_date_{rev}')
            submit = st.form_submit_button('✅ Buchung übernehmen', use_container_width=True)
        if submit:
            try:
                if kind in ('KAUF', 'VERKAUF'):
                    updated = mt_depot_trade(state, kind, ticker, qty, px,
                                             fees_eur=fee, taxes_eur=tax, trade_date=str(date))
                else:
                    updated = mt_depot_cash(state, kind, amount, trade_date=str(date))
                st.session_state['mt_live_depot'] = updated
                st.session_state['mt_live_revision'] += 1
                st.rerun()
            except ValueError as exc:
                st.error(f'Buchung konnte nicht übernommen werden: {exc}')
        if state['journal']:
            st.caption('**Letzte Buchungen:**')
            st.dataframe(pd.DataFrame(state['journal'][-5:]).iloc[::-1],
                         hide_index=True, use_container_width=True)
    with section_holdings:
        choice = st.selectbox('Welches Momentum-System ist deine Depot-Referenz?',
                              ['A', 'B'], index=0 if state['strategy'] == 'A' else 1,
                              format_func=lambda x: 'A – 5 Tage' if x == 'A' else 'B – 10 Tage',
                              key=f'mt_live_strategy_{rev}')
        if choice != state['strategy']:
            state['strategy'] = choice
            state['acknowledged_signal'] = None
            mt_depot_log(state, 'SYSTEMWECHSEL', note='Referenzsystem ' + choice)
            st.session_state['mt_live_revision'] += 1
            st.rerun()
        st.caption('Für eine **einmalige Bestandsaufnahme** oder Korrektur. '
                   'Spätere einzelne Käufe/Verkäufe bitte unter „Neue Buchung“ erfassen.')
        rows = [{'Ticker': t, 'Stückzahl': v['quantity'],
                 'Ø Einstand EUR': v['avg_entry_eur']}
                for t, v in sorted(state['positions'].items())]
        if not rows:
            rows = [{'Ticker': '', 'Stückzahl': 0.0, 'Ø Einstand EUR': 0.0}]
        with st.form(f'mt_live_snapshot_form_{rev}'):
            edited = st.data_editor(pd.DataFrame(rows), num_rows='dynamic', hide_index=True,
                                    use_container_width=True,
                                    column_config={
                                        'Ticker': st.column_config.TextColumn('Ticker'),
                                        'Stückzahl': st.column_config.NumberColumn('Stückzahl', min_value=0.0),
                                        'Ø Einstand EUR': st.column_config.NumberColumn(
                                            'Ø Einstand EUR', min_value=0.0, format='%.2f')
                                    }, key=f'mt_live_editor_{rev}')
            cash = st.number_input('Vorhandenes Cash (€)', min_value=0.0,
                                   value=float(state['cash_eur']), step=100.0,
                                   key=f'mt_live_cash_{rev}')
            saved = st.form_submit_button('Bestand übernehmen')
        if saved:
            try:
                st.session_state['mt_live_depot'] = mt_depot_set_snapshot(
                    state, edited.to_dict('records'), cash)
                st.session_state['mt_live_revision'] += 1
                st.rerun()
            except ValueError as exc:
                st.error(f'Bestand nicht übernommen: {exc}')
    with section_backup:
        st.warning('**Wichtig:** Streamlit Cloud speichert deine Depotdaten nur während '
                   'der Browsersitzung. Nach jeder Buchung ein privates JSON-Backup herunterladen!')
        st.download_button('⬇️ Depot und Journal sichern (JSON)',
                           data=mt_depot_backup_bytes(st.session_state['mt_live_depot']),
                           file_name='mein_momentum_depot.json', mime='application/json',
                           key=f'mt_live_backup_{rev}', use_container_width=True)
        upload = st.file_uploader('Gesichertes Depot wiederherstellen (JSON)',
                                  type=['json'], key=f'mt_live_restore_{rev}')
        if st.button('Backup importieren – ersetzt aktuellen Depotstand',
                     disabled=(upload is None), key=f'mt_live_import_{rev}'):
            try:
                restored = mt_depot_import_bytes(upload.getvalue())
                st.session_state['mt_live_depot'] = restored
                st.session_state['mt_live_revision'] += 1
                st.rerun()
            except (ValueError, TypeError) as exc:
                st.error(f'Backup konnte nicht importiert werden: {exc}')
        st.markdown('**Vollständiges echtes Handelstagebuch**')
        if state['journal']:
            journal = pd.DataFrame(state['journal'])
            st.dataframe(journal.iloc[::-1], hide_index=True, use_container_width=True)
            st.download_button('⬇️ Journal als CSV exportieren',
                               data=journal.to_csv(index=False).encode('utf-8-sig'),
                               file_name='momentum_echte_trades.csv', mime='text/csv',
                               key=f'mt_live_csv_{rev}')
        else:
            st.info('Noch keine Transaktionen dokumentiert.')


def mt_live_depot_comparison(holdings_5, holdings_10, kpis_5, kpis_10, last_close_date):
    state = st.session_state['mt_live_depot']
    use_a = state['strategy'] == 'A'
    frame = holdings_5 if use_a else holdings_10
    kpi = kpis_5 if use_a else kpis_10
    st.divider()
    st.subheader('🧭 Mein Trading-Assistent – Depot gegen neues Signal')
    st.caption('Nur **ein** Referenzsystem gleichzeitig. Zielportfolio und reale Bestände '
               'werden getrennt geführt. Alle Ordermengen sind unverbindliche Schätzungen.')
    symbol = 'A (wöchentlich)' if use_a else 'B (zweiwöchentlich)'
    target_date = kpi.get('last_signal_date') or 'unbekannt'
    next_session = kpi.get('next_execution') or 'unbekannt'
    st.write(f'**Ausgewähltes System:** {symbol} · **Letztes Signal:** {target_date} '
             f'· **Früheste Modell-Ausführung:** {next_session}')
    from hashlib import sha256
    fingerprint_payload = {'system': state['strategy'], 'date': target_date,
                           'weights': sorted((str(row['Ticker']), round(float(row['Zielgewicht (%)']), 5))
                                             for row in frame.to_dict('records'))}
    signature = sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()[:40]
    signature = state['strategy'] + ':' + signature
    if state.get('acknowledged_signal') == signature:
        st.info('Dieses Signal wurde bereits als geprüft markiert. '
                'Änderungen am Depot werden trotzdem erneut verglichen.')
    else:
        st.info('Dieses Signal wurde noch nicht als geprüft markiert. '
                'Eine Prüfung ist keine Kauf- oder Verkaufsbestätigung.')

    desired = set(frame['Ticker'].astype(str)) if not frame.empty else set()
    actual = set(state['positions'])
    a1, a2, a3 = st.columns(3)
    a1.metric('Aktuelle Depotpositionen', len(actual))
    a2.metric('Neue Zielaktien', len(desired - actual))
    a3.metric('Aus Ziel entfernt', len(actual - desired))
    if not actual and state['cash_eur'] <= 0:
        st.warning('Noch kein tatsächliches Depot eingetragen. Links unter '
                   '„Mein echtes Depot“ die Bestände und Cash erfassen oder ein Backup importieren.')
    else:
        if desired - actual:
            st.write('🟢 **Im Ziel, aber nicht im Depot:** ' + ', '.join(sorted(desired - actual)))
        if actual - desired:
            st.write('🔴 **Im Depot, aber nicht mehr im Ziel:** ' + ', '.join(sorted(actual - desired)))
        if not (desired - actual or actual - desired):
            st.success('Aktienauswahl stimmt überein. Positionsgrößen werden im nächsten Schritt geprüft.')

        wanted = sorted(desired | actual)
        try:
            with st.spinner('Lade zuletzt verfügbare unbereinigte Schlusskurse für den Depotabgleich ...'):
                quotes, fx = mt_depot_latest_quotes(tuple(wanted), str(last_close_date))
        except Exception as exc:
            LOG.warning('Depotpreise nicht verfügbar: %s', exc)
            quotes, fx = {}, None
        rows, equity, missing = mt_depot_plan(state, frame, quotes)
        if missing:
            st.warning('Für folgende Aktien fehlen verwendbare aktuelle EUR-Referenzkurse: '
                       + ', '.join(missing) + '. Deshalb wird **keine Stückzahlorder** geschätzt. '
                       'Der obige Tickervergleich ist davon nicht betroffen.')
        elif not wanted:
            st.info('Kein Aktienbestand und kein Aktiensignal. Eine etwaige Cashquote bleibt unverändert.')
        else:
            st.metric('Depotwert auf Basis letzter Schlusskurse (indikativ)', format_de(equity))
            st.caption(f'Unbereinigte USD-Schlusskurse bis {str(last_close_date)[:10]}, '
                       f'umgerechnet mit EUR/USD ca. {fx:.4f}. '
                       'Kein Echtzeitkurs, keine ausführbare Brokerquote. '
                       'Gewinnsteuern, Spreads und Gebühren sind in dieser IST-Bewertung nicht pauschal enthalten.')
            active_orders = [r for r in rows if r['Aktion'] in ('KAUF', 'VERKAUF')]
            if not active_orders:
                st.success('**Kein Handlungsbedarf nach aktueller Toleranz.** '
                           'Es bestehen keine ausreichend großen Abweichungen, '
                           'die sich mit ganzen Aktien abbilden lassen.')
            else:
                st.warning(f'**{len(active_orders)} mögliche Anpassungen** '
                           '(bitte Kurse, Kosten und verfügbare Mittel manuell prüfen).')
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption('Testschwelle: mindestens 50 € oder 2 % des Depotwerts für '
                       'Gewichtsanpassungen bestehender Aktien. Beim vollständigen Ausscheiden '
                       'einer Aktie wird unabhängig von der Schwelle VERKAUF angezeigt. '
                       'Die Stückzahl ist auf ganze Aktien gerundet, Käufe sind nicht '
                       'gegen verfügbare Liquidität optimiert. Verkaufsorders zuerst prüfen.')
    if st.button('✅ Signal als geprüft markieren (führt KEINE Order aus)',
                 key='mt_live_ack_signal', use_container_width=True):
        state['acknowledged_signal'] = signature
        mt_depot_log(state, 'SIGNAL_GEPRUEFT', note=f'{state["strategy"]} | {target_date}')
        st.session_state['mt_live_revision'] += 1
        st.rerun()
    st.caption('Die Prüfung wird im Depot-Backup mitgesichert. '
               'Nach tatsächlichem Handel müssen Käufe/Verkäufe links separat dokumentiert '
               'und der neue Depotstand erneut als JSON gespeichert werden.')

# ==========================================
# BLOCK 2: TAB 7 - S&P 500 DUAL-TRACKER (V2)
# Ersetzt den bisherigen kompletten `with tab7:`-Block.
# Beide Systeme A (5 Tage) und B (10 Tage) bleiben erhalten.
# ==========================================
if tab7.open:
    with tab7:
        # =============================
        # V6: EINMALIGE, AUSSCHLIESSLICH ANKLICKBARE HILFE
        # Die Momentum-Engine, Scores, Risiken und Backtests bleiben unverändert.
        # Vermeidet doppelte Texte bei Mausbewegung.
        # =============================
        _mt_glossary = {
            'Grundlagen': {
                'S&P 500': 'US-Aktienindex mit rund 500 großen börsennotierten Unternehmen. Er dient hier als Ausgangspunkt für die Aktienauswahl.',
                'SPY / Benchmark': 'SPY ist ein ETF, der den S&P 500 nachbildet. Benchmark bedeutet Vergleichsmaßstab: Hat das Trading-System besser abgeschnitten als ein passives SPY-Investment?',
                'Momentum': 'Kursdynamik: Aktien, die über mehrere Monate stark gestiegen sind, werden bevorzugt. Die Annahme ist, dass ein etablierter Trend manchmal weiterläuft – eine Garantie gibt es nicht.',
                'Ranking': 'Rangliste der Aktien nach Momentum-Score. Rang 1 hat unter den aktuell zulässigen Kandidaten den höchsten Score.',
                'Signal & Ausführung': 'Ein Signal entsteht nach dem Börsenschluss. Der Backtest rechnet eine Ausführung zum Schlusskurs der folgenden NYSE-Handelssitzung. Du handelst weiterhin selbst.',
                'Backtest': 'Historischer Probelauf einer Strategie auf vergangenen Kursdaten. Ein gutes Ergebnis beweist nicht, dass die Strategie künftig genauso abschneidet.',
                'Survivorship Bias': 'Verzerrung im Rückblick: Werden nur heute im Index enthaltene Firmen getestet, fehlen frühere Aussteiger. Historische Gewinne können dadurch zu optimistisch erscheinen.',
            },
            'Einstellungen': {
                'Aktien im Zielportfolio': 'Wie viele verschiedene Aktien das System maximal gleichzeitig auswählt. Bei 8 Aktien und voller Investition wären das zunächst je 12,5 %.',
                'Bestandsschutz bis Rang': 'Auch Hysterese genannt. Eine bereits gehaltene Aktie muss nicht sofort verkauft werden, wenn sie aus den Top 8 fällt. Bei Grenze 12 darf sie bis Rang 12 bleiben; das spart Umschichtungen.',
                'Schutz bei SPY unter SMA200': 'SMA200 = durchschnittlicher Schlusskurs der vergangenen 200 Handelstage. Liegt SPY darunter, reduziert der Filter die Aktienquote auf 50 % oder 0 %. Ohne Filter bleibt die Strategie investiert.',
                'Backtest-Zeitraum': 'Wie viele vergangene Jahre im Test ausgewertet werden. Das System lädt zusätzliche ältere Kurse für die Berechnung der Trendindikatoren; fehlende historische Daten können den Test verkürzen.',
                'Sektorlimit': 'Maximale Zahl gleichzeitig gewählter Aktien aus einer Branche, etwa Technologie oder Banken. Verhindert, dass viele Positionen fast dasselbe Risiko tragen.',
                'Cash / Aktienquote': 'Cash ist hier nicht investiertes Geld ohne Zins. Eine Aktienquote von 50 % bedeutet: ungefähr die Hälfte des Modellportfolios ist in Aktien angelegt, der Rest bleibt Cash.',
                '5 / 10 / 20 Tage Rotation': 'A prüft ungefähr wöchentlich, B alle zwei Börsenkalenderwochen. Ein Prüfzeitpunkt bedeutet nicht zwingend einen Kauf oder Verkauf.',
            },
            'Kennzahlen': {
                'Rendite letzte 252 Handelstage': 'Wertentwicklung des Modells über rund ein Börsenjahr. Beispiel: +12 % bedeutet aus 10.000 € wären rechnerisch 11.200 € geworden, vor Steuern und Kosten.',
                'Überrendite ggü. SPY (252T)': 'Differenz zur SPY-Rendite in Prozentpunkten. Beispiel: Strategie +15 %, SPY +10 % → +5 Prozentpunkte Überrendite. Das ist nicht automatisch risikobereinigtes Alpha.',
                'CAGR im gesamten Backtest': 'Annualisierte Wachstumsrate über den gesamten Test. Beispiel: 10.000 € auf 20.000 € in fünf Jahren ergibt rund 14,9 % CAGR, nicht 20 %.',
                'Maximaler Drawdown': 'Größter zwischenzeitlicher Verlust vom bisherigen Höchststand bis zu einem späteren Tief. Beispiel: Depot von 12.000 € auf 9.000 € → −25 %. Besonders wichtig für deine Verlusttoleranz.',
                'Sharpe Ratio (Cashzins 0 %)': 'Einfaches Maß für Rendite im Verhältnis zu Schwankungen. Größer ist grundsätzlich besser; eine Sharpe von 1 ist kein Versprechen. Hier wird ein risikofreier Zins von 0 % angenommen.',
                'Anteil positiver Signalzyklen': 'Anteil der Prüfintervalle mit positiver Portfoliorendite. 60 % bedeutet sechs von zehn Zeiträumen im Plus – nicht, dass 60 % der einzelnen Käufe gewonnen haben.',
                'Historischer Jahresmedian*': 'Mittlere Jahresrendite, wenn man die vollständigen historischen Kalenderjahre sortiert. Die Hälfte liegt darüber, die andere darunter. Keine Prognose.',
                'Umschichtungen insgesamt': 'Wie oft der Backtest eine tatsächliche Änderung der Depotgewichte ausgeführt hat. Es handelt sich um Rebalancing-Vorgänge, nicht um die Zahl einzelner Kauf- und Verkaufsorders.',
                'Turnover p. a.': 'Durchschnittlich pro Jahr umgeschichteter Depotanteil. 100 % bedeutet rechnerisch etwa einen vollständigen Depotumschlag pro Jahr; mehr Turnover heißt mehr Handelsaktivität.',
            },
            'Depot, Kosten & Steuern': {
                'Euro-Depot-Simulation': 'Rechnet Startkapital, Käufe, Verkäufe und Gebühren in Euro durch. Zur faireren Gegenüberstellung kann das Depot täglich zusätzlich so bewertet werden, als würden alle offenen Positionen verkauft und dabei Kosten und modellierte Steuern abgezogen.',
                'SPY netto / Vergleichsgrafiken': 'Brutto-Historie: Prozentperformance in USD ohne Kosten und Steuern. Euro-Nettovergleich: nach Gebühren und modellierten Steuern, wahlweise bei angenommener vollständiger Liquidation. SPY kauft am Starttag einmal und hält. Unterschiedliche Starttage von A und B erfordern separate SPY-Vergleiche.',
                'Gebühr je Order': 'Fixer Betrag pro ausgeführtem Kauf oder Verkauf. Zusätzlich kann eine prozentuale Belastung angegeben werden.',
                'Variable Handelskosten': 'Prozentualer Abschlag je Kauf/Verkauf, als vereinfachtes Modell für Spread, Slippage und weitere variable Kosten.',
                'Sparer-Pauschbetrag': 'Jährlich frei nutzbarer Anteil an bestimmten Kapitalerträgen. Bitte nur den tatsächlich für diese Simulation verfügbaren Rest eintragen.',
                'FIFO': 'First In, First Out: Beim Verkauf gelten die zuerst angeschafften Stücke einer Aktie modellhaft als zuerst verkauft.',
                'Steuerlich modellierte Rendite': 'Depotwert nach bereits simulierten Steuerabzügen. Auf noch nicht verkaufte Gewinne können bei späterem Verkauf weitere Steuern anfallen.',
                'Hypothetischer Liquidationswert': 'Der rechnerische Eurobetrag, wenn am letzten Testtag alle Aktien verkauft, Kosten bezahlt und zusätzliche Steuern modelliert würden.',
                'Handelsjournal': 'Liste aller simulierten Käufe, Verkäufe, Gebühren und modellierten Steuerzahlungen; keine realen Brokerdaten.',
                'Steuermodell-Grenzen': 'Adjusted-Close-Kurse bilden Dividenden/Corporate Actions nicht als echte Zahlung und Broker-Einstandskurse ab. Quellensteuer, steuerliche Einzelfälle, jahresbezogene Rechtsänderungen und Freistellungsaufträge bei anderen Instituten fehlen.',
            },
            'Aktientabelle': {
                'Ticker': 'Börsenkürzel einer Aktie. Beispiel: MSFT steht für Microsoft.',
                'Unternehmen': 'Name der Gesellschaft hinter dem Börsenkürzel.',
                'Score': 'Interne Vergleichszahl aus Momentum, Volatilität und Trendqualität. Ein höherer Score bedeutet einen besseren Platz im aktuellen Ranking – er ist keine Renditeprognose.',
                'Trend-Qualität (R²)': 'Wert von 0 bis 1, der beschreibt, wie gleichmäßig die logarithmierte Kursentwicklung in den letzten 90 Handelstagen einem geraden Trend folgt. Höher heißt gleichmäßiger; nicht automatisch sicherer.',
                'Rendite (5 Tage)': 'Kursveränderung über etwa fünf Handelstage bis zum letzten Signalschluss. Nur eine Rückschau, keine Prognose.',
                'Rendite (10 Tage)': 'Kursveränderung über etwa zehn Handelstage bis zum letzten Signalschluss. Nur eine Rückschau, keine Prognose.',
                'Rendite (T-130 bis T-5)': 'Kursgewinn/-verlust von vor 130 Handelstagen bis vor 5 Handelstagen. Die letzten 5 Tage werden ausgeklammert, um kurzfristige Ausschläge weniger stark zu gewichten.',
                'Volatilität': 'Historische Schwankungsstärke über 90 Handelstage, auf ein Jahr hochgerechnet. Hohe Werte bedeuten größere Kursschwankungen und häufig auch höheres Risiko.',
                'Kurs am Signaltag': 'Bereinigter Kurs, der für das letzte Signal genutzt wurde. Keine Zusage, dass du die Aktie heute zu diesem Preis kaufen kannst.',
                'Zielgewicht (%)': 'Geplanter prozentualer Depotanteil der Aktie nach dem Signal. Beispiel: 50 % Aktienquote bei acht Plätzen ergibt 6,25 % pro Aktie.',
                'Zielportfolio': 'Aktien, die das Modell nach dem letzten Signal halten möchte. Diese Liste entspricht NICHT automatisch deinem realen Brokerdepot.',
                'Tickerdifferenz': 'Einfacher Vergleich zwischen Modell und deinen eingegebenen Symbolen. Zeigt nur fehlende oder zusätzliche Ticker – keine konkreten Stückzahlen oder Ordervorschläge.',
            },
        }
        _mt_help = {name: description for chapter in _mt_glossary.values()
                    for name, description in chapter.items()}

        def _mt_kpi_table_with_help(dataframe):
            """KPI-Vergleich mit anklickbarer Hilfe für jede einzelne Kennzahl.

            Streamlit-Tabellen bieten Hilfen pro Spalte, nicht pro Datenzeile.
            Deshalb nutzen wir native Streamlit-Spalten und pro Zeile
            einen eigenen Popover; keine HTML/JS/CSS-Hover-Abhängigkeiten.
            """
            names = list(dataframe.columns)
            widths = [3.6, 2.15, 2.15]
            headers = st.columns(widths, gap='small')
            for col, text in zip(headers, names):
                col.markdown(f'**{text}**')
            st.divider()
            for _, item in dataframe.iterrows():
                cols = st.columns(widths, gap='small')
                title = str(item['Kennzahl'])
                with cols[0]:
                    # Einzigartiges Label je Kennzahl verhindert doppelte Widget-IDs.
                    # Kein help-Argument: ausschließlich per Klick öffnen.
                    explanation = _mt_help.get(title, 'Historische Kennzahl der Strategie.')
                    with st.popover(f'ⓘ {title}'):
                        st.markdown(f'**{title}**')
                        st.write(explanation)
                for j, name in enumerate(names[1:], 1):
                    cols[j].write(str(item[name]))

        def _mt_glossary_popover(label, terms):
            """Zusätzliche klickbare Erklärung bei Tabellen/Einstellungen."""
            with st.popover(label):
                for term in terms:
                    if term in _mt_help:
                        st.markdown(f'**{term}**')
                        st.write(_mt_help[term])

        st.markdown('### 🚀 S&P 500 Dual-Tracker V7: 5 Tage vs. 10 Tage')
        st.caption('Manuelles Trading | Brutto-Backtest und optionaler Euro-Netto-Backtest | Signale nach NYSE-Schluss, '
                   'simulierte Ausführung zum Schlusskurs der nächsten NYSE-Sitzung')
        st.caption('💡 **Hilfe:** Klicke auf ⓘ für die Erklärung. '
                   'Keine zusätzlichen Hover-Texte. Das 📖-Lexikon links funktioniert auch mobil.')

        left, right = st.columns([1, 2.8], gap='large')
        with left:
            st.markdown('#### Strategie-Einstellungen')
            _mt_glossary_popover('ⓘ Einstellungen erklärt', [
                'Aktien im Zielportfolio', 'Bestandsschutz bis Rang',
                'Schutz bei SPY unter SMA200', 'Backtest-Zeitraum', 'Sektorlimit',
                'Cash / Aktienquote', '5 / 10 / 20 Tage Rotation'
            ])
            positions = st.selectbox('Aktien im Zielportfolio', [5, 8, 10], index=1,
                                     key='mtv2_positions')
            hyst = st.selectbox('Bestandsschutz bis Rang', [10, 12, 15], index=1,
                                key='mtv2_hyst')
            regime = st.selectbox(
                'Schutz bei SPY unter SMA200',
                ['50%', 'aus', 'cash'],
                format_func=lambda s: {'50%': '50 % Zielquote', 'aus': 'Kein Schutzfilter',
                                       'cash': '100 % Cash'}[s],
                key='mtv2_regime'
            )
            st.caption('🛡️ **Was bedeutet der SPY-Schutz?** SPY ist der S&P-500-ETF. '
                       'SMA200 ist sein durchschnittlicher Schlusskurs der letzten 200 Handelstage. '
                       'Unterhalb dieser Linie werden **bei der nächsten regulären Signalprüfung** '
                       'je nach Einstellung nur 50 % oder 0 % in Aktien gehalten; der Rest bleibt Cash (0 % Zins). '
                       'Oberhalb der Linie wird wieder die normale Zielquote angestrebt. '
                       'Der Filter verhindert keine Verluste und kann Markterholungen verpassen. '
                       'Er ist **nicht** identisch mit dem separaten Risk Guard.')
            horizon = st.selectbox('Backtest-Zeitraum', [3, 5, 10, 15], index=3,
                                   format_func=lambda n: f'{n} Jahre (soweit Daten vorhanden)',
                                   key='mtv2_horizon')
            sectors = _mt_sectors()
            enough_sectors = len(sectors) >= 400
            sector_on = st.checkbox('Sektorlimit: max. 3 Aktien/Sektor', value=False,
                                    disabled=not enough_sectors, key='mtv2_sector')
            if not enough_sectors:
                st.caption('Sektormapping optional: Datei sp500_sectors.csv mit Spalten ticker,sector.')
            st.caption('👇 **Kosten und Steuern:** Die Einstellungen sind direkt hier sichtbar. '
                       'Nach „Beide Systeme berechnen“ erscheint die Nettoauswertung ganz oben rechts.')
            with st.expander('💶 Depot, Ordergebühren & Steuern', expanded=True):
                _mt_glossary_popover('ⓘ Depotberechnung erklärt', [
                    'Euro-Depot-Simulation', 'Gebühr je Order', 'Variable Handelskosten',
                    'Sparer-Pauschbetrag', 'FIFO', 'Steuerlich modellierte Rendite',
                    'Hypothetischer Liquidationswert', 'Steuermodell-Grenzen'
                ])
                net_enabled = st.checkbox('Euro-Nettosimulation anzeigen', value=True, key='v7_net_enabled')
                net_capital = st.number_input('Startkapital (€)', min_value=100.0,
                                              value=8000.0, step=500.0, key='v7_capital')
                net_fixed = st.number_input('Gebühr je Order (€)', min_value=0.0,
                                            value=1.0, step=0.5, key='v7_fixed')
                net_variable = st.number_input('Variable Handelskosten je Order (%)',
                                               min_value=0.0, max_value=10.0,
                                               value=0.1, step=0.05, key='v7_variable')
                net_tax_enabled = st.checkbox('Modellierte deutsche Kapitalertragsteuer abziehen',
                                              value=True, key='v7_tax_enabled')
                net_allowance = st.number_input('Noch verfügbarer jährlicher Sparer-Pauschbetrag (€)',
                                                min_value=0.0, max_value=2000.0,
                                                value=1000.0, step=100.0, key='v7_allowance')
                net_church = st.selectbox('Kirchensteuer in % der Kapitalertragsteuer',
                                          [0, 8, 9], key='v7_church')
                net_fractions = st.checkbox('Aktienbruchstücke bei der Simulation zulassen',
                                            value=False, key='v7_fractions')
                spy_exemption = st.checkbox('SPY-Benchmark als Aktienfonds: 30 % Teilfreistellung',
                                            value=True, key='v102_spy_exemption',
                                            help='Modellannahme für einen steuerlich qualifizierten Aktienfonds. '
                                                 'Kein Ersatz für Vorabpauschale, Ausschüttungen, Quellsteuer oder Steuerberatung.')
                st.caption('Ohne Bruchstücke werden ganze Aktien gehandelt; nicht investierbares '
                           'Restgeld bleibt Cash. Fixkosten und variable Kosten gelten pro Einzelorder.')
            st.caption('📒 **Echtes Handelstagebuch:** Jetzt direkt im Hauptbereich rechts oben – '
                       'mit einer gemeinsamen Buchungsmaske und privatem Backup.')

            st.caption('✅ Nach dem ersten Start werden geänderte Einstellungen automatisch neu berechnet.')
            run = st.button('🔄 Beide Systeme berechnen', type='primary',
                            use_container_width=True, key='mtv2_run')
            if run:
                st.session_state['run_momentum'] = True
            st.caption('A: wöchentliche Signale · B: alle zwei Kalenderwochen. '
                       'Gehandelt wird nur bei veränderter Zielauswahl oder wesentlicher Expositionsabweichung.')

            # Kompaktes Nachschlagewerk ausschließlich links im Momentum-Tab.
            with st.expander('📖 Fachbegriffe einfach erklärt', expanded=False):
                st.caption('Wähle einen Bereich und einen Begriff – auch auf dem Smartphone.')
                _mt_chapter = st.selectbox('Themenbereich', list(_mt_glossary),
                                           key='mtv4_help_section')
                _mt_word = st.selectbox('Was möchtest du verstehen?',
                                        list(_mt_glossary[_mt_chapter]),
                                        key='mtv4_help_term')
                st.info(f'**{_mt_word}**\n\n{_mt_glossary[_mt_chapter][_mt_word]}')
                st.caption('Hinweis: Die Erklärungen dienen zum Verständnis; sie sind '
                           'keine Renditeprognosen oder persönlichen Handelsempfehlungen.')

        with right:
            mt_live_depot_controls()
            st.divider()
            if not st.session_state.get('run_momentum', False):
                st.info('Klicke links auf „Beide Systeme berechnen“.')
            else:
                with st.spinner('Lade S&P-500-Daten. Bei rund 500 Aktien und 16 Jahren kann der Download umfangreich sein ...'):
                    tickers = fetch_sp500_universe()
                    if tickers:
                        # 1 Jahr zusätzliche Historie für SMA200 / Momentum-Warmup.
                        close_all, vol_all = fetch_sp500_data(tickers, years=horizon + 1)
                    else:
                        close_all, vol_all = pd.DataFrame(), pd.DataFrame()

                if close_all.empty or 'SPY' not in close_all:
                    st.error('Keine vollständige Datenbasis für einen Backtest verfügbar.')
                else:
                    if not os.path.isfile('sp500_membership_history.csv'):
                        st.warning('**Wichtig – Survivorship Bias:** Ohne vollständige Datei '
                                   '`sp500_membership_history.csv` enthält der historische Backtest '
                                   'die heute gelisteten S&P-500-Mitglieder und ist lediglich indikativ. '
                                   'Historische Überrenditen sind damit nicht verlässlich belegt.')
                    st.caption(f'Datenbasis: {close_all.index.min():%d.%m.%Y} bis '
                               f'{close_all.index.max():%d.%m.%Y} | '
                               f'{len(close_all.columns) - 1} Aktienkurse | Benchmark: SPY (USD, adjusted Close). '
                               'Cashzins 0 %, Brutto-Vergleich ohne Gebühren und Steuern. '
                               'Die Euro-Depot-Auswertung berücksichtigt deine Kosten-/Steuerparameter.')

                    kwargs = dict(hysterese_limit=hyst, n_positions=positions,
                                  market_mode=regime, backtest_years=horizon,
                                  sector_limit=3 if sector_on and enough_sectors else None)
                    with st.spinner('Berechne die zwei Backtests mit Ausführung am nächsten Handelstag ...'):
                        holdings_5, track_5, kpis_5 = calculate_momentum_strategy(
                            close_all, vol_all, holding_days=5, **kwargs)
                        holdings_10, track_10, kpis_10 = calculate_momentum_strategy(
                            close_all, vol_all, holding_days=10, **kwargs)

                    if not kpis_5 or not kpis_10 or track_5.empty or track_10.empty:
                        st.error('Mindestens ein System konnte nicht gültig berechnet werden. '
                                 'Siehe konkrete Fehlermeldung oben.')
                    else:
                        def fmt(num, digits=1, suffix=' %'):
                            if num is None or not np.isfinite(num):
                                return '–'
                            return f'{num:,.{digits}f}'.replace(',', 'X').replace('.', ',').replace('X', '.') + suffix

                        # V7.1 – EURO-NETTO-BACKTEST AN ERSTER STELLE, SIGNALLOGIK UNVERÄNDERT
                        st.divider()
                        st.subheader('💶 Euro-Depot: nach Handelskosten und modellierten Steuern')
                        _mt_glossary_popover('ⓘ Euro-Depot und Steuermodell erklärt', [
                            'Euro-Depot-Simulation', 'Steuerlich modellierte Rendite',
                            'Hypothetischer Liquidationswert', 'Handelsjournal',
                            'Steuermodell-Grenzen', 'FIFO'
                        ])
                        if net_enabled:
                            try:
                                with st.spinner('Berechne Euro-Depotwerte und einzelne Transaktionen ...'):
                                    fx_net = fetch_historical('EURUSD=X', years=horizon + 2)
                                    if fx_net.empty:
                                        raise ValueError('Historische EUR/USD-Kurse konnten nicht geladen werden.')
                                    net_args = dict(capital=net_capital, fixed_fee=net_fixed,
                                                    variable_fee_pct=net_variable,
                                                    saver_allowance=net_allowance, church_tax=net_church,
                                                    include_tax=net_tax_enabled, fractions=net_fractions)
                                    net5, trades5, info5 = simulate_momentum_eur_depot(
                                        close_all, kpis_5['execution_events'], fx_net, **net_args)
                                    net10, trades10, info10 = simulate_momentum_eur_depot(
                                        close_all, kpis_10['execution_events'], fx_net, **net_args)
                                    spy5, spy_info5 = simulate_spy_buy_hold_eur(
                                        close_all['SPY'], fx_net, net5.index, **net_args,
                                        equity_fund_exemption=0.30 if spy_exemption else 0.0)
                                    spy10, spy_info10 = simulate_spy_buy_hold_eur(
                                        close_all['SPY'], fx_net, net10.index, **net_args,
                                        equity_fund_exemption=0.30 if spy_exemption else 0.0)
                                report = pd.DataFrame([
                                    {'System': 'A – 5 Tage', 'Endwert (€)': info5['end_value'],
                                     'Rendite (%)': info5['return_pct'],
                                     'Rendite bei Verkauf (%)': (info5['liquidation_value'] / net_capital - 1)*100,
                                     'CAGR (%)': info5['cagr'],
                                     'Drawdown (%)': info5['max_drawdown'],
                                     'Gebühren (€)': info5['fees'], 'Steuern (€)': info5['taxes'],
                                     'Einzelorders': info5['orders'],
                                     'Liquidationswert (€)': info5['liquidation_value']},
                                    {'System': 'B – 10 Tage', 'Endwert (€)': info10['end_value'],
                                     'Rendite (%)': info10['return_pct'],
                                     'Rendite bei Verkauf (%)': (info10['liquidation_value'] / net_capital - 1)*100,
                                     'CAGR (%)': info10['cagr'],
                                     'Drawdown (%)': info10['max_drawdown'],
                                     'Gebühren (€)': info10['fees'], 'Steuern (€)': info10['taxes'],
                                     'Einzelorders': info10['orders'],
                                     'Liquidationswert (€)': info10['liquidation_value']},
                                    {'System': 'SPY Buy & Hold (Start A)',
                                     'Endwert (€)': spy_info5['end_value'],
                                     'Rendite (%)': spy_info5['return_pct'],
                                     'Rendite bei Verkauf (%)': spy_info5['return_after_sale_pct'],
                                     'CAGR (%)': spy_info5['cagr'],
                                     'Drawdown (%)': np.nan,
                                     'Gebühren (€)': spy_info5['buy_fee'],
                                     'Steuern (€)': 0.0,
                                     'Einzelorders': 1,
                                     'Liquidationswert (€)': spy_info5['liquidation_value']},
                                    {'System': 'SPY Buy & Hold (Start B)',
                                     'Endwert (€)': spy_info10['end_value'],
                                     'Rendite (%)': spy_info10['return_pct'],
                                     'Rendite bei Verkauf (%)': spy_info10['return_after_sale_pct'],
                                     'CAGR (%)': spy_info10['cagr'],
                                     'Drawdown (%)': np.nan,
                                     'Gebühren (€)': spy_info10['buy_fee'],
                                     'Steuern (€)': 0.0,
                                     'Einzelorders': 1,
                                     'Liquidationswert (€)': spy_info10['liquidation_value']}
                                ])
                                metric_a, metric_b = st.columns(2)
                                metric_a.metric('System A – Depotwert nach Abzügen',
                                                format_de(info5['end_value']),
                                                f"{format_de_num(info5['return_pct'], 1)} % Rendite")
                                metric_b.metric('System B – Depotwert nach Abzügen',
                                                format_de(info10['end_value']),
                                                f"{format_de_num(info10['return_pct'], 1)} % Rendite")
                                st.caption('**Direktvergleich:** Die folgende Tabelle zeigt '
                                           'Ordergebühren, modellierte Steuern, Netto-Rendite '
                                           'und Liquidationswert für beide Systeme.')
                                st.dataframe(report, use_container_width=True, hide_index=True,
                                             column_config={
                                                'Endwert (€)': st.column_config.NumberColumn(format='%.2f €'),
                                                'Rendite (%)': st.column_config.NumberColumn(format='%.2f %%'),
                                                'Rendite bei Verkauf (%)': st.column_config.NumberColumn(format='%.2f %%'),
                                                'CAGR (%)': st.column_config.NumberColumn(format='%.2f %%'),
                                                'Drawdown (%)': st.column_config.NumberColumn(format='%.2f %%'),
                                                'Gebühren (€)': st.column_config.NumberColumn(format='%.2f €'),
                                                'Steuern (€)': st.column_config.NumberColumn(format='%.2f €'),
                                                'Liquidationswert (€)': st.column_config.NumberColumn(format='%.2f €'),
                                             })
                                st.caption('Für SPY wird optional eine 30%-Teilfreistellung für qualifizierte '
                                           'Aktienfonds angenommen (§ 20 InvStG). Vorabpauschale, tatsächliche '
                                           'Ausschüttungsbesteuerung und Quellensteuer fehlen; es bleibt ein Modell.')
                                st.caption('Tabellenwerte Rendite/CAGR zeigen den **laufenden Depotwert**. '
                                           'Die Spalte „Rendite bei Verkauf“ und der Liquidationswert '
                                           'ziehen zusätzlich hypothetische finale Verkäufe ab. '
                                           'Für SPY beträgt die bisher **gezahlte** Steuer 0 €, '
                                           'weil das Buy-and-Hold-Modell nicht zwischendurch verkauft.')
                                st.markdown('##### 📈 Netto-Renditevergleich: System A, B und SPY')
                                st.caption('**Alle Linien mit gleichem Startkapital und denselben Gebühren-/Steuerannahmen.** '
                                           'SPY wird einmal am jeweiligen Modellstart gekauft und gehalten. '
                                           'Beim hypothetischen Verkauf werden für alle drei Varianten '
                                           'Verkaufsgebühren und mögliche weitere Steuern berücksichtigt.')
                                graph_basis = st.radio('Welche Netto-Bewertung anzeigen?',
                                                       ['Bei vollständigem Verkauf', 'Depotwert ohne Schlussverkauf'],
                                                       horizontal=True, key='v102_net_chart_basis')
                                graph_unit = st.radio('Skala der Grafik', ['Rendite (%)', 'Depotwert (€)'],
                                                      horizontal=True, key='v102_net_chart_unit')
                                liquidate = graph_basis == 'Bei vollständigem Verkauf'
                                values = pd.DataFrame(index=net5.index.union(net10.index).sort_values())
                                strategy_col = 'Liquidationswert (€)' if liquidate else 'Netto-Depot (€)'
                                spy_col = 'Liquidationswert (€)' if liquidate else 'Depotwert (€)'
                                values['System A – netto'] = net5[strategy_col]
                                values['System B – netto'] = net10[strategy_col]
                                values['SPY netto (Start A)'] = spy5[spy_col]
                                different_start = net5.index[0] != net10.index[0]
                                if different_start:
                                    values['SPY netto (Start B)'] = spy10[spy_col]
                                if graph_unit == 'Rendite (%)':
                                    values = (values / float(net_capital) - 1) * 100
                                fig_net = px.line(values, height=395,
                                    color_discrete_map={
                                        'System A – netto': '#e0782a',
                                        'System B – netto': '#007a33',
                                        'SPY netto (Start A)': '#757d85',
                                        'SPY netto (Start B)': '#b89e65'})
                                fig_net.update_layout(xaxis_title='Datum',
                                    yaxis_title='Nettorendite (%)' if graph_unit == 'Rendite (%)' else 'Euro (€)',
                                    legend_title='Nettovergleich')
                                st.plotly_chart(fig_net, use_container_width=True)
                                st.caption(f'**Zeitraum A:** {info5["start"]} bis {info5["end"]} · '
                                           f'**Zeitraum B:** {info10["start"]} bis {info10["end"]}. '
                                           + ('SPY hat deshalb zwei Startlinien für den fairen Vergleich.'
                                              if different_start else 'Beide Systeme und SPY haben denselben Starttag.'))
                                _spycol_a, _spycol_b = st.columns(2)
                                _spycol_a.metric('SPY netto bei Verkauf (Start A)',
                                    format_de(spy_info5['liquidation_value']),
                                    f"{format_de_num(spy_info5['return_after_sale_pct'], 1)} %")
                                _spycol_b.metric('SPY netto bei Verkauf (Start B)',
                                    format_de(spy_info10['liquidation_value']),
                                    f"{format_de_num(spy_info10['return_after_sale_pct'], 1)} %")
                                with st.popover('ⓘ Warum unterscheiden sich Netto- und Bruttografik?'):
                                    st.write('**Historische Performance weiter unten:** rein prozentuale Brutto-'
                                             'Backtests auf USD-Basis, ohne Gebühren, Steuern und ganzzahlige Orders.')
                                    st.write('**Euro-Nettovergleich hier:** simuliertes Startkapital in Euro; '
                                             'Währungskurs, Kauf-/Verkaufskosten und Steuermodell zählen. '
                                             'Das kann zu anderen Renditen und Abständen zum SPY führen.')
                                    st.write('**Depotwert ohne Schlussverkauf** enthält latente '
                                             'Steuern auf unrealisierte Gewinne noch nicht. '
                                             '**Bei vollständigem Verkauf** werden diese hypothetisch '
                                             'abgezogen – niemand verkauft tatsächlich.')
                                st.warning('**Keine brokergetreue Steuerabrechnung:** '
                                           'Die Kursreihe verwendet dividenden-/splitbereinigte Werte. '
                                           'Dividenden, Quellensteuer, genaue steuerliche Anschaffungskurse, '
                                           'Corporate Actions, mögliche Gesetzesänderungen über den '
                                           'Backtestzeitraum und persönliche Verlusttöpfe anderer Depots '
                                           'werden nicht vollständig abgebildet. Die Nettorendite ist '
                                           'eine Szenariorechnung, keine realisierte oder steuerlich geprüfte Rendite.')
                                st.caption('**Investment pro Position:** Bei vollständiger Aktienquote '
                                           f'anfangs ca. {format_de(net_capital / positions)} '
                                           'je Zielposition (vor Gebühren und Stückzahlrundung). '
                                           'Bei 50 % Aktienquote entsprechend halbiert.')
                                _n_a, _n_b = st.columns(2)
                                _n_a.metric('A – verbleibendes Cash', format_de(info5['cash']))
                                _n_b.metric('B – verbleibendes Cash', format_de(info10['cash']))
                                for sysname, tr, info in [('A – 5 Tage', trades5, info5),
                                                          ('B – 10 Tage', trades10, info10)]:
                                    with st.expander(f'📒 Handelsjournal {sysname} ({len(tr)} Vorgänge)'):
                                        st.caption('Steuerbuchungen sind separate Ereignisse; Steuererstattungen '
                                                   'werden innerhalb des Modelljahres gegengerechnet. '
                                                   'Verfügbare Liquidität und Ganzstückhandel können '
                                                   'Abweichungen vom idealen Zielgewicht verursachen.')
                                        if not tr.empty:
                                            view = tr.sort_values('Datum', ascending=False).head(200).copy()
                                            view['Datum'] = pd.to_datetime(view['Datum']).dt.strftime('%d.%m.%Y')
                                            st.dataframe(view, use_container_width=True, hide_index=True)
                                            st.download_button('CSV-Handelsjournal herunterladen',
                                                data=tr.to_csv(index=False).encode('utf-8-sig'),
                                                file_name=f'momentum_journal_{sysname[0]}.csv',
                                                mime='text/csv', key=f'v7_journal_{sysname[0]}')
                                        if info['position_values_eur']:
                                            positions_df = pd.DataFrame([
                                                {'Ticker': ticker, 'Stückzahl (modelliert)': info['positions'][ticker],
                                                 'Positionswert (€)': value}
                                                for ticker, value in info['position_values_eur'].items()
                                            ])
                                            st.markdown('**Modellierte letzte Depotpositionen**')
                                            st.dataframe(positions_df, use_container_width=True, hide_index=True)
                                        st.caption('Zusätzliche modellierte Steuer bei sofortiger Komplettveräußerung: '
                                                   f'{format_de(info["liquidation_tax_additional"])}. '
                                                   'Der Liquidationswert zieht auch Verkaufsgebühren ab.')
                            except Exception as exc:
                                LOG.exception('V7-Euro-Depot-Simulation fehlgeschlagen')
                                st.error(f'Euro-Depot-Simulation nicht verfügbar: {exc}')
                        else:
                            st.caption('Die Euro-Nettosimulation ist links deaktiviert. '
                                       'Die ursprünglichen Brutto-Backtests bleiben sichtbar.')

                        # V9 – UNABHAENGIGES FRUEHWARNSYSTEM; READ-ONLY UND OPT-IN-BACKTEST.
                        # Keine Veraenderung der Momentum-Scores, Zielholdings oder Depotorders.
                        st.divider()
                        with st.expander('🛡️ Momentum Risk Guard – Frühwarnsystem für Verlustphasen', expanded=True):
                            st.info('**Risikoampel: 0 Punkte = geringere Warnsignale (besser), '
                                    '100 Punkte = viele Warnsignale (schlechter).** '
                                    'Der Wert ist **keine Verlustwahrscheinlichkeit in Prozent**. '
                                    'Berechnung jeweils zum abgeschlossenen NYSE-Wochensignal.')
                            with st.popover('ⓘ Was bedeuten Risikoscore und Aktienquote?'):
                                st.write('Fünf Faktoren: Markttrend, Schwankungen, Marktbreite, '
                                         'Indexrückgang und plötzliche Erholung nach einem Einbruch. '
                                         'Höhere Scores sprechen für eine defensivere Aktienquote. '
                                         'Die Empfehlung ist ein Forschungsmodell und verändert '
                                         'deine Kauf-/Verkaufssignale noch nicht.')
                                st.write('Eine Reduzierung der Aktienquote senkt normalerweise das '
                                         'Marktrisiko, kann aber Erholungen verpassen und Verluste '
                                         'nicht ausschließen.')
                            try:
                                _guard = build_risk_guard(close_all, minimum_stocks=100)
                                _end = _guard.iloc[-1]
                                _last_signal = _end['Letzter_Signaltermin']
                                if pd.isna(_last_signal) or pd.isna(_guard.at[_last_signal, 'Risikoscore']):
                                    st.warning('Es liegt noch kein vollständiges Risikosignal vor. '
                                               'Prüfe insbesondere die verfügbare Kurs- und Marktbreite-Historie.')
                                else:
                                    _signal_row = _guard.loc[_last_signal]
                                    _score = float(_signal_row['Risikoscore'])
                                    _quota = float(_end['Zielaktienquote'])
                                    _state = str(_signal_row['Risikostufe'])
                                    _rgcols = st.columns(4)
                                    _rgcols[0].metric('Risiko (mehr = schlechter)', f'{_score:.0f} / 100')
                                    _rgcols[1].metric('Risikostufe', _state)
                                    _rgcols[2].metric('Modell-Aktienquote', f'{_quota:.0%}')
                                    _rgcols[3].metric('Letztes Wochensignal', _last_signal.strftime('%d.%m.%Y'))
                                    st.progress(min(100, max(0, round(_score))) / 100,
                                                text=f'{_state}: {_score:.0f} von 100 Risikopunkten')
                                    st.caption('**Ampel:** 0–24 = Normal · 25–49 = Erhöht · '
                                               '50–69 = Hoch · 70–100 = Sehr hoch. '
                                               '**Je höher der Wert, desto vorsichtiger sollte man sein.**')
                                    st.caption('Aktienquote ist ein **separater Testvorschlag**, '
                                               'keine automatische Änderung deines realen Depots. '
                                               'Bei niedrigerem Risiko wird die Quote erst nach '
                                               'zwei bestätigenden Wochensignalen erhöht.')
                                    _parts = [
                                        ('Markttrend', 'Trendpunkte', 25, 'SPY gegenüber 50- und 200-Tage-Durchschnitt'),
                                        ('Volatilität', 'Volatilitätspunkte', 20, 'Stärke der 20-Tage-Kursschwankungen'),
                                        ('Marktbreite', 'Breitepunkte', 20, 'Anteil der Aktien über ihrem SMA200'),
                                        ('Index-Drawdown', 'Drawdownpunkte', 20, 'Rückgang von einem 252-Tage-Hoch'),
                                        ('Trendwechsel', 'Reversalpunkte', 15, 'Erholung nach vorheriger Marktschwäche'),
                                    ]
                                    _details = pd.DataFrame([
                                        {'Frühindikator': title, 'Risikopunkte': float(_signal_row[key]),
                                         'Maximum': maximum, 'Bedeutung': detail}
                                        for title, key, maximum, detail in _parts
                                    ])
                                    st.dataframe(_details, hide_index=True, use_container_width=True)
                                    _last_260 = _guard.iloc[-260:]
                                    _fig = go.Figure()
                                    _fig.add_trace(go.Scatter(x=_last_260.index,
                                                              y=_last_260['Risikoscore_letztes_Signal'],
                                                              name='Wöchentlicher Risikoscore', mode='lines',
                                                              line=dict(color='#d98f37', width=2.5)))
                                    for _y, _clr in [(25, '#8b9d45'), (50, '#d09b38'), (70, '#c75959')]:
                                        _fig.add_hline(y=_y, line_dash='dot', line_color=_clr)
                                    _fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=15),
                                                       yaxis=dict(range=[0, 100], title='Risikopunkte'),
                                                       xaxis_title='Datum')
                                    st.plotly_chart(_fig, use_container_width=True)
                                    st.caption('Marktbreite beruht standardmäßig auf heutigen S&P-500-Mitgliedern. '
                                               'Ohne historische Mitgliedschaftsdaten sind Frühindikator und '
                                               'rückblickende Auswertung möglicherweise verzerrt.')

                                    if st.checkbox('Experiment: Originalstrategie mit/ohne Risk Guard vergleichen',
                                                   value=False, key='v9_compare_risk_guard'):
                                        if regime != 'aus':
                                            st.warning('Der bestehende SMA200-Marktschutz ist eingeschaltet. '
                                                       'Der zusätzliche Risk Guard reduziert die Quote nochmals. '
                                                       'Für einen Vergleich ohne Doppelabsicherung links '
                                                       '„Kein Schutzfilter“ auswählen.')
                                        _overlay5, _rep5 = simulate_risk_overlay(track_5, _guard, holding_days=5)
                                        _overlay10, _rep10 = simulate_risk_overlay(track_10, _guard, holding_days=10)
                                        _risk_comp = pd.concat([
                                            _rep5.assign(System='A – 5 Tage'),
                                            _rep10.assign(System='B – 10 Tage')
                                        ]).reset_index().rename(columns={'index': 'Variante'})
                                        st.dataframe(_risk_comp, hide_index=True, use_container_width=True)
                                        _oc = pd.DataFrame({
                                            'A – Original': _overlay5['Original'] - 100,
                                            'A – mit Risk Guard': _overlay5['Mit Risk Guard'] - 100,
                                            'B – Original': _overlay10['Original'] - 100,
                                            'B – mit Risk Guard': _overlay10['Mit Risk Guard'] - 100
                                        })
                                        _fig_cmp = px.line(_oc, height=340,
                                                           labels={'value':'Rendite (%)', 'index':'Datum'})
                                        st.plotly_chart(_fig_cmp, use_container_width=True)
                                        st.caption('Prototypen-Backtest auf den bestehenden täglichen '
                                                   'Strategierenditen: Quotenänderung erst nach dem '
                                                   'nächsten Handelstagsschluss wirksam. '
                                                   'Cashzins, zusätzliche Käufe/Verkäufe, Steuern, '
                                                   'Gebühren und Gewichtsdrift werden nicht neu '
                                                   'auf Transaktionsebene simuliert. '
                                                   'Die Euro-Nettoberechnung unten bleibt unverändert.')

                                    if st.checkbox('Rückblick: Häufigkeit negativer 20-Tage-Phasen prüfen',
                                                   value=False, key='v9_risk_outcomes'):
                                        _freq = diagnostic_forward_losses(track_5, _guard,
                                                                          holding_days=5, horizon=20)
                                        if _freq.empty:
                                            st.info('Noch nicht genügend historische Daten für die Auswertung.')
                                        else:
                                            st.dataframe(_freq, hide_index=True, use_container_width=True)
                                            st.caption('Nur nicht überlappende 20-Tage-Fenster. '
                                                       'Bei kleiner Stichprobe sind Prozentwerte instabil. '
                                                       '**Keine** kalibrierte 20-Tage-Prognose; '
                                                       'kein Out-of-Sample-Validierungsnachweis.')
                            except Exception as _risk_exc:
                                LOG.exception('Momentum Risk Guard konnte nicht erstellt werden')
                                st.error(f'Risiko-Frühwarnsystem nicht berechenbar: {_risk_exc}')

                        # 1. PERFORMANCE
                        st.subheader('📈 Historischer BRUTTO-Vergleich – USD, vor Gebühren & Steuern')
                        st.caption('**Unterschied zum oberen Diagramm:** Hier vergleichst du '
                                   'reine Strategie- und SPY-Total-Return-Näherungen in **USD-Prozent**. '
                                   'Keine Steuern, keine Ordergebühren, keine Ganzstück-Rundung '
                                   'und keine EUR/USD-Effekte. Das obere Netto-Diagramm '
                                   'simuliert dagegen ein **Euro-Depot nach Abzügen**. '
                                   'Daher sind die Linien nicht identisch.')
                        combo = pd.DataFrame(index=track_5.index.union(track_10.index))
                        combo['System A (5 Tage)'] = track_5['Momentum Rotation (5 Tage)']
                        combo['System B (10 Tage)'] = track_10['Momentum Rotation (10 Tage)']
                        combo['SPY Benchmark'] = track_5['S&P 500 Benchmark']
                        combo = combo.sort_index().ffill()
                        fig = px.line(combo, height=380, color_discrete_map={
                            'System A (5 Tage)': '#e0782a',
                            'System B (10 Tage)': '#007a33',
                            'SPY Benchmark': '#757d85'
                        })
                        fig.update_layout(xaxis_title='Datum', yaxis_title='Kumulierte Rendite (%)',
                                          legend_title='Portfolio / Benchmark',
                                          margin=dict(t=10, b=0, l=0, r=0))
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption('Signal: letzter NYSE-Handelstag einer Woche. '
                                   'Ausführung: nächster NYSE-Handelstag zu dessen Schlusskurs. '
                                   'Die 10-Tage-Variante ist an festen Kalenderwochen verankert.')

                        # 2. KPIs
                        st.markdown('#### 📊 Kennzahlen im direkten Vergleich')
                        label_map = [
                            ('Rendite letzte 252 Handelstage', 'ist_rendite', '%'),
                            ('Überrendite ggü. SPY (252T)', 'alpha', 'pp'),
                            ('CAGR im gesamten Backtest', 'cagr', '%'),
                            ('Maximaler Drawdown', 'max_drawdown', '%'),
                            ('Sharpe Ratio (Cashzins 0 %)', 'sharpe', 'x'),
                            ('Anteil positiver Signalzyklen', 'win_rate', '%'),
                            ('Historischer Jahresmedian*', 'prog_median', '%'),
                            ('Umschichtungen insgesamt', 'trades', 'int'),
                            ('Turnover p. a.', 'turnover_p_a', '%'),
                        ]
                        def cell(kpis, key, unit):
                            val = kpis.get(key, np.nan)
                            if unit == 'pp':
                                return fmt(val, 1, ' Prozentpunkte')
                            if unit == 'x':
                                return fmt(val, 2, '')
                            if unit == 'int':
                                return str(int(val)) if val is not None else '–'
                            return fmt(val)

                        table = pd.DataFrame([
                            {'Kennzahl': title,
                             'System A (5 Tage)': cell(kpis_5, key, unit),
                             'System B (10 Tage)': cell(kpis_10, key, unit)}
                            for title, key, unit in label_map
                        ])
                        st.caption('ⓘ **Klicke auf das Info-Symbol** neben einer Kennzahl, '
                                   'um genau eine Erklärung zu öffnen.')
                        _mt_kpi_table_with_help(table)
                        st.caption('* Median vollständig beobachteter Kalenderjahre, **keine** '
                                   'Vorhersage. Die historischen Backtest-Jahre sind '
                                   f'{kpis_5["start_date"]} bis {kpis_5["end_date"]}. '
                                   'Alle Renditen in USD, nicht als EUR-Anlegerrenditen interpretieren.')

                        # 3. NYSE-KALENDER: NÄCHSTE SIGNALE (DEUTSCHE LOKALZEIT)
                        st.divider()
                        st.subheader('⏱️ Nächste Signaltermine (Börsenkalender)')
                        with st.popover('ⓘ Was ist ein Signaltermin?'):
                            st.write(_mt_help['Signal & Ausführung'])
                        now_utc = pd.Timestamp.now(tz='UTC')
                        sched = mcal.get_calendar('NYSE').schedule(
                            start_date=(now_utc - pd.Timedelta(days=8)).date(),
                            end_date=(now_utc + pd.Timedelta(days=35)).date()
                        )
                        ends = pd.Series(sched.index, index=pd.DatetimeIndex(sched.index).to_period('W-FRI'))
                        weekly_ends = pd.DatetimeIndex(ends.groupby(level=0).last().values)

                        def next_signal(days):
                            anchor = pd.Timestamp('2020-01-06')
                            for session_date in weekly_ends:
                                date = pd.Timestamp(session_date).normalize()
                                week_num = (date - anchor).days // 7
                                if days == 10 and week_num % 2 != 0:
                                    continue
                                closing = sched.at[session_date, 'market_close']
                                if closing + pd.Timedelta(minutes=15) > now_utc:
                                    return closing.tz_convert('Europe/Berlin').strftime('%d.%m.%Y, %H:%M Uhr')
                            return 'Noch nicht verfügbar'

                        c1, c2 = st.columns(2)
                        c1.metric('System A – nächster Signalschluss', next_signal(5))
                        c2.metric('System B – nächster Signalschluss', next_signal(10))
                        st.caption('NYSE-Feiertage, verkürzte Handelstage und US-/EU-Zeitumstellungen '
                                   'werden berücksichtigt. Eine abgeschlossene Kursdatenlieferung '
                                   'kann kurz nach Börsenschluss verzögert eintreffen.')

                        # V10: KLARE DATIERUNG DER HANDELSPERIODEN UND HISTORISCHE AKTIENLISTEN
                        st.divider()
                        st.subheader('📆 Aktuelle und vergangene Momentum-Perioden')
                        st.caption('**Von–bis** bezeichnet die Gültigkeit zwischen zwei geplanten '
                                   'Ausführungen. Eine Ausführung wird im Backtest zum NYSE-'
                                   'Schlusskurs des Handelstags nach dem Signal angenommen. '
                                   'Ein Prüfzeitpunkt muss nicht zu einem Kauf oder Verkauf führen.')
                        lookback_periods = st.selectbox('Zurückliegende Perioden je System',
                                                        [4, 8, 12, 26], index=1,
                                                        key='mtv10_period_count')
                        latest_close = close_all.index[-1]
                        period_views = [
                            ('⚡ System A – 5 Tage', kpis_5, 5),
                            ('🛡️ System B – 10 Tage', kpis_10, 10),
                        ]
                        tabs_period = st.tabs([v[0] for v in period_views])
                        for tab_period, (_, stats_period, duration) in zip(tabs_period, period_views):
                            with tab_period:
                                periods_view = momentum_period_overview(
                                    stats_period, latest_close, duration, past_count=lookback_periods)
                                current = periods_view['active']
                                if current is None:
                                    st.warning('Aktuell kein ausgeführter Signalzyklus in der Datenbasis erkennbar.')
                                else:
                                    shown_to = (pd.Timestamp(current['to']).strftime('%d.%m.%Y')
                                                if current['to'] else 'noch offen')
                                    st.markdown(f"**Laufende Periode: "
                                                f"{pd.Timestamp(current['from']):%d.%m.%Y} bis {shown_to}**")
                                    st.caption(f"Signal: {pd.Timestamp(current['signal_date']):%d.%m.%Y} "
                                               f"| Aktienquote: {current['exposure_pct']:.1f} % "
                                               f"| Datenstand: {latest_close:%d.%m.%Y} "
                                               '(Periode kann vor ihrem vorgesehenen Enddatum laufen)')
                                    if current['tickers']:
                                        tickers_frame = pd.DataFrame({
                                            'Ticker': current['tickers'],
                                            'Unternehmen': [get_asset_name(t) for t in current['tickers']],
                                        })
                                        st.dataframe(tickers_frame, hide_index=True,
                                                     use_container_width=True)
                                    else:
                                        st.info('In dieser Periode: keine Aktienquote (Cash).')
                                upcoming = periods_view['pending']
                                if upcoming is not None:
                                    with st.expander('⏳ Neues Signal liegt vor – Ausführung noch ausstehend',
                                                     expanded=True):
                                        st.caption(f"Signal vom {pd.Timestamp(upcoming['signal_date']):%d.%m.%Y} "
                                                   f"| früheste modellierte Ausführung: "
                                                   f"{pd.Timestamp(upcoming['from']):%d.%m.%Y} (NYSE-Schluss)")
                                        st.write('**Neue Zielkandidaten:** ' +
                                                 (', '.join(upcoming['tickers']) if upcoming['tickers']
                                                  else 'Cash / keine Aktienquote'))
                                with st.expander(f'🗂️ Letzte {lookback_periods} abgeschlossene Perioden '
                                                 'mit Aktienbestand', expanded=False):
                                    if periods_view['past']:
                                        st.dataframe(pd.DataFrame([
                                            {
                                                'Von': pd.Timestamp(p['from']).strftime('%d.%m.%Y'),
                                                'Bis': (pd.Timestamp(p['to']).strftime('%d.%m.%Y')
                                                        if p['to'] else 'offen'),
                                                'Aktien': ', '.join(p['tickers']) if p['tickers'] else 'Cash',
                                                'Aktienquote (%)': round(p['exposure_pct'], 1),
                                            }
                                            for p in periods_view['past']
                                        ]), hide_index=True, use_container_width=True)
                                        choices = list(range(len(periods_view['past'])))
                                        selected_past = st.selectbox(
                                            'Vergangenen Zeitraum für Unternehmensnamen auswählen',
                                            choices,
                                            format_func=lambda k: (
                                                pd.Timestamp(periods_view['past'][k]['from']).strftime('%d.%m.%Y')
                                                + ' bis ' + pd.Timestamp(periods_view['past'][k]['to']).strftime('%d.%m.%Y')
                                            ),
                                            key=f'mtv10_past_detail_{duration}',
                                        )
                                        chosen = periods_view['past'][selected_past]
                                        st.caption('Signal vom ' +
                                                   pd.Timestamp(chosen['signal_date']).strftime('%d.%m.%Y') +
                                                   f" · Aktienquote: {chosen['exposure_pct']:.1f} %")
                                        if chosen['tickers']:
                                            st.dataframe(pd.DataFrame({
                                                'Ticker': chosen['tickers'],
                                                'Unternehmen': [get_asset_name(t) for t in chosen['tickers']],
                                            }), hide_index=True, use_container_width=True)
                                        else:
                                            st.write('Cash / keine Aktienpositionen.')
                                    else:
                                        st.caption('Noch keine abgeschlossenen Perioden im gewählten Fenster.')
                        st.caption('Hinweis: Gezeigt werden historisch berechnete Zielkandidaten '
                                   'pro Signalzyklus, nicht dein tatsächliches Brokerdepot. '
                                   'Bei unveränderten Kandidaten kann dieselbe Aktienliste '
                                   'in aufeinanderfolgenden Perioden stehen. '
                                   'Vergangenheitsdaten können nach Yahoo-Datenkorrekturen abweichen.')

                        # 4. AKTUELLE ZIELPORTFOLIOS UND MANUELLE SIGNALDIFFERENZEN
                        st.divider()
                        st.subheader('🏆 Letzte Zielportfolios und manuelle Handelssignale')
                        st.info('Das sind **modellierte Zielbestände**, nicht automatisch die echten '
                                'Positionen deines Brokers. Für konkrete Kauf-/Verkaufsdifferenzen '
                                'kannst du deine derzeit gehaltenen Ticker eingeben.')
                        depot_text = st.text_input(
                            'Eigene aktuell gehaltene Ticker (optional, kommagetrennt)',
                            value='', placeholder='z. B. AAPL, NVDA, MSFT',
                            key='mtv2_real_holdings'
                        )
                        owned = {_mt_symbol(x) for x in depot_text.replace(';', ',').split(',') if x.strip()}

                        def display_holdings(frame, days, stats):
                            st.markdown(f'#### {"⚡ System A" if days == 5 else "🛡️ System B"}: '
                                        f'{days}-Tage-Rotation')
                            if stats['last_signal_date']:
                                st.caption(f'Signal vom {stats["last_signal_date"]} | '
                                           f'ursprüngliche nächste Ausführung: {stats["next_execution"]} '
                                           f'(zum US-Schlusskurs) | '
                                           f'Aktienquote im Ziel: {fmt(stats["target_exposure"] * 100)}')
                            if frame.empty:
                                st.warning('Keine qualifizierten Aktien im aktuellen Zielportfolio (Cash).')
                            else:
                                formatted = frame.copy()
                                for c in ['Score', 'Trend-Qualität (R²)']:
                                    formatted[c] = formatted[c].map(lambda x: fmt(x, 3, ''))
                                for c in [f'Rendite ({days} Tage)', 'Rendite (T-130 bis T-5)',
                                          'Volatilität', 'Zielgewicht (%)']:
                                    formatted[c] = formatted[c].map(lambda x: fmt(x))
                                formatted['Aktueller Kurs'] = formatted['Aktueller Kurs'].map(lambda x: fmt(x, 2, ' $'))
                                formatted = formatted.rename(columns={'Aktueller Kurs': 'Kurs am Signaltag'})
                                st.caption('ⓘ Die Spalten werden ausschließlich über „Spalten erklärt“ erläutert.')
                                _mt_glossary_popover('ⓘ Spalten erklärt', list(formatted.columns))
                                st.dataframe(formatted, hide_index=True, use_container_width=True)

                            if depot_text.strip():
                                base = owned
                                reference = 'deinen eingegebenen Depot-Tickern'
                            else:
                                base = set(stats['simulated_tickers'])
                                reference = 'den simulierten Beständen (keine echten Depotdaten)'
                            desired = set(stats['target_tickers']) if stats['target_exposure'] > 0 else set()
                            buy = sorted(desired - base)
                            sell = sorted(base - desired)
                            stay = sorted(desired & base)
                            st.write(f'**Tickerdifferenz zu {reference}:**')
                            st.write('🟢 Neue Ticker: ' + (', '.join(buy) if buy else 'keine'))
                            st.write('🔴 Zu entfernende Ticker: ' + (', '.join(sell) if sell else 'keine'))
                            st.write('🔵 Weiter zu halten: ' + (', '.join(stay) if stay else 'keine'))
                            st.caption('Die Anzeige vergleicht nur Tickernamen. Auch bei gleichen Tickern '
                                       'können Gewichtsanpassungen nötig sein; ohne Stückzahlen '
                                       'und Depotwerte werden keine Stückzahl-Orders berechnet.')

                        display_holdings(holdings_5, 5, kpis_5)
                        st.divider()
                        display_holdings(holdings_10, 10, kpis_10)

                        mt_live_depot_comparison(holdings_5, holdings_10, kpis_5, kpis_10, close_all.index[-1])

                        # V7 – ZUSÄTZLICHE RISIKOGRAFIKEN / KALENDERJAHRE
                        st.divider()
                        st.subheader('📉 Risikoverlauf und jährliche Renditen')
                        wealth_curves = 100.0 + combo[['System A (5 Tage)',
                                                        'System B (10 Tage)', 'SPY Benchmark']]
                        dd = (wealth_curves / wealth_curves.cummax() - 1.0) * 100
                        fig_dd = px.line(dd, height=280, title='Rückgang vom bisherigen Höchststand (Drawdown)')
                        fig_dd.update_layout(xaxis_title='Datum', yaxis_title='Drawdown (%)',
                                             legend_title='Strategie')
                        st.plotly_chart(fig_dd, use_container_width=True)
                        st.caption('Drawdown: Wie stark das Depot zeitweise unter seinen früheren Höchstwert fiel. '
                                   'Die Grafik berücksichtigt im Vergleich oben noch keine Kosten/Steuern.')
                        annual_rows = []
                        for cal_year, portion in wealth_curves.groupby(wealth_curves.index.year):
                            before = wealth_curves.loc[wealth_curves.index < portion.index[0]]
                            basis = before.iloc[-1] if len(before) else wealth_curves.iloc[0]
                            yearly_pct = (portion.iloc[-1] / basis - 1) * 100
                            for label in wealth_curves.columns:
                                annual_rows.append({'Jahr': str(cal_year), 'Strategie': label,
                                                    'Rendite (%)': float(yearly_pct[label])})
                        if annual_rows:
                            fig_year = px.bar(pd.DataFrame(annual_rows), x='Jahr', y='Rendite (%)',
                                              color='Strategie', barmode='group',
                                              title='Kalenderjährliche Renditen – Rumpfjahre inklusive',
                                              color_discrete_map={
                                                  'System A (5 Tage)': '#e0782a',
                                                  'System B (10 Tage)': '#007a33',
                                                  'SPY Benchmark': '#757d85'})
                            st.plotly_chart(fig_year, use_container_width=True)
                        # Übersicht bewusst schlank: Sektorgrafik entfernt.
                        # Optionaler Sektorlimit-Filter in den Einstellungen bleibt erhalten.

                        # 5. OPTIONALER ROBUSTHEITSTEST BEI GLEICHEM DATENSTAND
                        _mt_glossary_popover('ⓘ Kennzahlen der Robustheitstests erklärt', [
                            'Aktien im Zielportfolio', 'CAGR im gesamten Backtest',
                            'Maximaler Drawdown', 'Sharpe Ratio (Cashzins 0 %)',
                            'Umschichtungen insgesamt', 'Schutz bei SPY unter SMA200',
                            '5 / 10 / 20 Tage Rotation'
                        ])
                        with st.expander('🔬 Sensitivität: 5 / 8 / 10 Aktien im 5-Tage-System'):
                            st.caption('Einzelparameter-Test bei unverändertem Score, Marktfilter und '
                                       'Handelskalender. Die Ergebnisse sind keine unabhängige '
                                       'Out-of-Sample-Validierung.')
                            if st.button('Test 5 / 8 / 10 Aktien durchführen', key='mtv2_sensitivity'):
                                comparison = []
                                with st.spinner('Berechne drei Varianten ...'):
                                    for n in (5, 8, 10):
                                        _h, _track, k = calculate_momentum_strategy(
                                            close_all, vol_all, holding_days=5,
                                            hysterese_limit=hyst, n_positions=n,
                                            market_mode=regime, backtest_years=horizon,
                                            sector_limit=3 if sector_on and enough_sectors else None
                                        )
                                        if k:
                                            comparison.append({
                                                'Aktien': n,
                                                'CAGR': fmt(k['cagr']),
                                                'Max. Drawdown': fmt(k['max_drawdown']),
                                                'Sharpe': fmt(k['sharpe'], 2, ''),
                                                'Umschichtungen': k['trades']
                                            })
                                if comparison:
                                    st.dataframe(pd.DataFrame(comparison), hide_index=True,
                                                 use_container_width=True,
                                                 column_config={
                                                     'Aktien': st.column_config.NumberColumn(
                                                         'Aktien'),
                                                     'CAGR': st.column_config.TextColumn(
                                                         'CAGR'),
                                                     'Max. Drawdown': st.column_config.TextColumn(
                                                         'Max. Drawdown'),
                                                     'Sharpe': st.column_config.TextColumn(
                                                         'Sharpe'),
                                                     'Umschichtungen': st.column_config.NumberColumn(
                                                         'Umschichtungen')
                                                 })

                        with st.expander('🔬 Sensitivität: 5 / 10 / 20 Tage Rotation'):
                            st.caption('Vergleicht einen, zwei oder vier feste NYSE-Kalenderwochen. '
                                       'Alle übrigen Strategieparameter bleiben identisch.')
                            if st.button('Rotationshorizonte testen', key='mtv2_rotation_test'):
                                tests = []
                                with st.spinner('Teste drei Rotationsrhythmen ...'):
                                    for duration in (5, 10, 20):
                                        _h, _t, k = calculate_momentum_strategy(
                                            close_all, vol_all, holding_days=duration,
                                            hysterese_limit=hyst, n_positions=positions,
                                            market_mode=regime, backtest_years=horizon,
                                            sector_limit=3 if sector_on and enough_sectors else None
                                        )
                                        if k:
                                            tests.append({'Rhythmus (Handelstage näherungsweise)': duration,
                                                          'CAGR': fmt(k['cagr']),
                                                          'Max. Drawdown': fmt(k['max_drawdown']),
                                                          'Sharpe': fmt(k['sharpe'], 2, ''),
                                                          'Umschichtungen': k['trades']})
                                if tests:
                                    st.dataframe(pd.DataFrame(tests), hide_index=True,
                                                 use_container_width=True,
                                                 column_config={
                                                     'Rhythmus (Handelstage näherungsweise)': st.column_config.NumberColumn(
                                                         'Rhythmus (Handelstage näherungsweise)'),
                                                     'CAGR': st.column_config.TextColumn(
                                                         'CAGR'),
                                                     'Max. Drawdown': st.column_config.TextColumn(
                                                         'Max. Drawdown'),
                                                     'Sharpe': st.column_config.TextColumn(
                                                         'Sharpe'),
                                                     'Umschichtungen': st.column_config.NumberColumn(
                                                         'Umschichtungen')
                                                 })

                        with st.expander('🛡️ Sensitivität: verschiedene Marktregimefilter'):
                            st.caption('Teste ohne Filter, mit 50-%-Aktienquote oder vollständigem Cash-Wechsel '
                                       'bei SPY unter SMA200. Kein Zinsertrag auf Cash.')
                            if st.button('Drei Marktregime-Varianten testen', key='mtv2_regime_test'):
                                tests = []
                                with st.spinner('Vergleiche Risikofilter ...'):
                                    for selected_regime in ('aus', '50%', 'cash'):
                                        _h, _t, k = calculate_momentum_strategy(
                                            close_all, vol_all, holding_days=5,
                                            hysterese_limit=hyst, n_positions=positions,
                                            market_mode=selected_regime, backtest_years=horizon,
                                            sector_limit=3 if sector_on and enough_sectors else None
                                        )
                                        if k:
                                            tests.append({'Marktfilter': {'aus': 'Ohne Filter',
                                                                          '50%': '50 % Aktienquote',
                                                                          'cash': '100 % Cash'}[selected_regime],
                                                          'CAGR': fmt(k['cagr']),
                                                          'Max. Drawdown': fmt(k['max_drawdown']),
                                                          'Sharpe': fmt(k['sharpe'], 2, ''),
                                                          'Umschichtungen': k['trades']})
                                if tests:
                                    st.dataframe(pd.DataFrame(tests), hide_index=True,
                                                 use_container_width=True,
                                                 column_config={
                                                     'Marktfilter': st.column_config.TextColumn(
                                                         'Marktfilter'),
                                                     'CAGR': st.column_config.TextColumn(
                                                         'CAGR'),
                                                     'Max. Drawdown': st.column_config.TextColumn(
                                                         'Max. Drawdown'),
                                                     'Sharpe': st.column_config.TextColumn(
                                                         'Sharpe'),
                                                     'Umschichtungen': st.column_config.NumberColumn(
                                                         'Umschichtungen')
                                                 })


    # ==========================================
    # TAB 8: FREI WÄHLBARER ASSET-CHART + SMA 50 / SMA 200
# ==========================================
if tab8.open:
    with tab8:
        st.markdown("### 📉 Asset-Chart: Kursverlauf mit SMA 50 und SMA 200")
        st.caption(
            "Wähle eine Aktie, einen ETF, einen Index, eine Kryptowährung oder einen "
            "Rohstoff-Future über sein Yahoo-Finance-Symbol. "
            "Die Linien zeigen bereinigte Tageskurse und gleitende Durchschnitte, "
            "keine garantierten Live-Börsenkurse."
        )

        with st.form("asset_sma_chart_form", clear_on_submit=False):
            chart_col_symbol, chart_col_period = st.columns([1.1, 1.0])
            with chart_col_symbol:
                chart_symbol_input = st.text_input(
                    "Asset / Ticker", value="SPY", key="asset_sma_symbol",
                    help="Beispiele: AAPL, SPY, NVDA, BTC-USD, GC=F, BMW.DE, ^GDAXI. "
                         "Bitte das Yahoo-Finance-Symbol und nicht WKN oder ISIN eingeben."
                )
            with chart_col_period:
                chart_period_input = st.selectbox(
                    "Zeitraum des Charts", CHART_PERIODS, index=3, key="asset_sma_period"
                )
            submit_chart = st.form_submit_button(
                "📈 Chart anzeigen / aktualisieren", type="primary", use_container_width=True
            )

        if submit_chart:
            chart_symbol_clean = str(chart_symbol_input).strip().upper()
            # Erlaubte Börsen-Sondersymbole, z. B. ^GSPC, GC=F, BRK-B, BMW.DE.
            import re
            if not re.fullmatch(r"[A-Z0-9^._=\-]{1,30}", chart_symbol_clean):
                st.session_state.pop("asset_sma_selection", None)
                st.error("Bitte ein gültiges Yahoo-Finance-Börsensymbol eingeben (z. B. AAPL oder BMW.DE).")
            else:
                st.session_state["asset_sma_selection"] = (
                    chart_symbol_clean, chart_period_input
                )

        selected_chart = st.session_state.get("asset_sma_selection")
        if selected_chart is None:
            st.info("Ticker eingeben und auf **Chart anzeigen** klicken. "
                    "So verursacht der zusätzliche Reiter beim normalen Start des Terminals keinen Datenabruf.")
        else:
            selected_symbol, selected_period = selected_chart
            with st.spinner(f"Lade Tageskurse für {selected_symbol} …"):
                raw_chart_prices = fetch_asset_chart_prices(selected_symbol, selected_period)
            chart_data = prepare_sma_chart(raw_chart_prices)

            if chart_data.empty:
                st.error(f"Für {selected_symbol} konnten keine Yahoo-Finance-Kursdaten geladen werden. "
                         "Bitte das Symbol prüfen oder den Abruf später wiederholen.")
            else:
                chart_view = restrict_sma_chart_period(chart_data, selected_period)
                latest = chart_data.iloc[-1]
                prior = chart_data["Kurs"].iloc[-2] if len(chart_data) > 1 else np.nan
                day_change = (latest["Kurs"] / prior - 1) * 100 if pd.notna(prior) and prior > 0 else np.nan
                distance_200 = (latest["Kurs"] / latest["SMA 200"] - 1) * 100 if pd.notna(latest["SMA 200"]) else np.nan

                st.markdown(f"#### {selected_symbol} · {selected_period}")
                metr_price, metr_s50, metr_s200, metr_distance = st.columns(4)
                metr_price.metric(
                    "Letzter gemeldeter Kurs", format_de_num(latest["Kurs"], 2),
                    f"{day_change:+.2f} %" if np.isfinite(day_change) else None,
                    help="Tageskurse gemäß Yahoo Finance. Die Notierungswährung hängt vom "
                         "Ticker ab; der jüngste Tageswert kann noch vorläufig sein."
                )
                metr_s50.metric(
                    "SMA 50", format_de_num(latest["SMA 50"], 2) if pd.notna(latest["SMA 50"]) else "–",
                    help="Durchschnitt der letzten 50 verfügbaren Tageskurswerte."
                )
                metr_s200.metric(
                    "SMA 200", format_de_num(latest["SMA 200"], 2) if pd.notna(latest["SMA 200"]) else "–",
                    help="Durchschnitt der letzten 200 verfügbaren Tageskurswerte."
                )
                metr_distance.metric(
                    "Abstand zum SMA 200",
                    f"{distance_200:+.2f} %" if np.isfinite(distance_200) else "–",
                    help="Prozentualer Abstand des letzten gemeldeten Kurses zum 200-Tage-Durchschnitt."
                )

                fig_sma = go.Figure()
                for col_name, col_color, col_width in (
                    ("Kurs", "#2E86C1", 2.5),
                    ("SMA 50", "#E39B30", 2.0),
                    ("SMA 200", "#B53E4A", 2.0),
                ):
                    fig_sma.add_trace(go.Scatter(
                        x=chart_view.index, y=chart_view[col_name],
                        mode="lines", name=col_name,
                        line=dict(color=col_color, width=col_width),
                        connectgaps=False,
                        hovertemplate="%{x|%d.%m.%Y}: %{y:,.2f}<extra>" + col_name + "</extra>"
                    ))
                fig_sma.update_layout(
                    height=510, hovermode="x unified",
                    xaxis_title="Datum", yaxis_title="Kurs (Notierungswährung des Assets)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    margin=dict(t=60, b=35, l=10, r=10),
                    xaxis=dict(rangeslider=dict(visible=False)),
                )
                st.plotly_chart(fig_sma, use_container_width=True,
                                config={"displaylogo": False, "scrollZoom": False})

                last_price_date = chart_data.index[-1]
                st.caption(
                    f"Kursdatenstand: {last_price_date.strftime('%d.%m.%Y')} · "
                    f"{len(chart_data)} verfügbare Tageswerte geladen. "
                    "Bereinigte Yahoo-Finance-Kurse (inklusive Anpassungen für Kapitalmaßnahmen); "
                    "keine Intraday-Livekurse. Bei Kryptowährungen zählt jeder Kalendertag "
                    "als Tageswert."
                )
                if (pd.Timestamp.now().normalize() - last_price_date).days > 10:
                    st.warning("Der zuletzt gemeldete Kurs liegt mehr als zehn Kalendertage zurück. "
                               "Bitte Ticker und Datenverfügbarkeit prüfen.")
                if pd.isna(latest["SMA 200"]):
                    st.warning("Für den SMA 200 sind mindestens 200 verfügbare Tageswerte nötig. "
                               f"Derzeit liegen nur {len(chart_data)} vor.")

                if pd.notna(latest["SMA 50"]) and pd.notna(latest["SMA 200"]):
                    trend_name = ("SMA 50 über SMA 200 (positives langfristiges Trendbild)"
                                  if latest["SMA 50"] > latest["SMA 200"]
                                  else "SMA 50 unter oder gleich SMA 200 (schwächeres Trendbild)")
                    st.info("**Technische Einordnung:** " + trend_name + ". "
                            + ("Der Kurs liegt oberhalb des SMA 200."
                               if latest["Kurs"] > latest["SMA 200"]
                               else "Der Kurs liegt nicht oberhalb des SMA 200.")
                            + " Diese nachlaufenden Indikatoren sind keine Kursprognose.")

                with st.expander("📖 SMA 50 und SMA 200 einfach erklärt"):
                    st.markdown(
                        "**SMA** steht für *Simple Moving Average* (einfacher gleitender Durchschnitt). "
                        "Der **SMA 50** ist der Mittelwert der letzten 50 Tageskurswerte, "
                        "der **SMA 200** der letzten 200. "
                        "Die Linien helfen, kurzfristige Schwankungen vom längerfristigen "
                        "Trend zu unterscheiden. Ein Schnittpunkt der beiden Linien wird "
                        "manchmal *Golden Cross* bzw. *Death Cross* genannt, "
                        "ist aber keine verlässliche Kauf- oder Verkaufsprognose."
                    )

                output_data = chart_view.copy()
                output_data.index.name = "Datum"
                csv_chart = output_data.to_csv(sep=";", decimal=",", float_format="%.6f")
                safe_file_symbol = selected_symbol.replace("^", "IDX_").replace("=", "_")
                st.download_button(
                    "⬇️ Chartdaten als CSV herunterladen", data=csv_chart.encode("utf-8-sig"),
                    file_name=f"Chart_SMA_{safe_file_symbol}.csv", mime="text/csv",
                    key="asset_sma_csv"
                )
