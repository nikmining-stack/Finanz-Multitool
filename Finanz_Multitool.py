"""Integriertes Finanz-Terminal – eigenständige Startdatei, bedarfsabhängige Tabs.

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
# ==== BEGIN INTERNER RECHENKERN: momentum_risk_guard ====
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

LEVEL_ORDER = ['Normal', 'Erhöht', 'Hoch', 'Sehr hoch']
RISK_EXPOSURES = {'Normal': 1.0, 'Erhöht': 0.75, 'Hoch': 0.50, 'Sehr hoch': 0.25}


def _weekly_nyse_signals(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Richtige letzte NYSE-Sitzung einer Kalenderwoche; keine Teilwoche."""
    index = pd.DatetimeIndex(index).tz_localize(None).normalize()
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=index.min(), end_date=index.max() + pd.Timedelta(days=7))
    dates = pd.DatetimeIndex(schedule.index).tz_localize(None).normalize()
    weeks = dates.to_period('W-FRI')
    true_last = pd.Series(dates, index=weeks).groupby(level=0).last()
    return pd.DatetimeIndex(true_last.values).intersection(index)


def _level(score):
    if not np.isfinite(score):
        return 'Daten fehlen'
    if score < 25:
        return 'Normal'
    if score < 50:
        return 'Erhöht'
    if score < 70:
        return 'Hoch'
    return 'Sehr hoch'


def _confirmed_weekly_exposure(raw_scores: pd.Series, signal_dates: pd.DatetimeIndex) -> pd.Series:
    """Risiko runter: sofort am Signal; Risiko hoch: erst nach zwei Wochenbestätigungen.

    Ein neues Wochenziel muss ZWEIMAL hintereinander die höhere Aktienquote
    vorschlagen. Dazwischen wird die bisherige Schutzquote beibehalten.
    Bei fehlenden Daten werden weder neue Quoten noch Neueinstiege abgeleitet.
    """
    out = pd.Series(np.nan, index=raw_scores.index, dtype=float)
    exposure = 1.0
    recovery_candidate = None
    recovery_count = 0
    valid_dates = set(signal_dates)
    for day, score in raw_scores.items():
        if day in valid_dates and np.isfinite(score):
            target = RISK_EXPOSURES[_level(score)]
            if target < exposure:
                exposure = target
                recovery_candidate = None
                recovery_count = 0
            elif target > exposure:
                if recovery_candidate is not None and abs(recovery_candidate - target) < 1e-9:
                    recovery_count += 1
                else:
                    recovery_candidate, recovery_count = target, 1
                if recovery_count >= 2:
                    exposure = target
                    recovery_candidate = None
                    recovery_count = 0
            else:
                recovery_candidate = None
                recovery_count = 0
        out.at[day] = exposure
    return out


def build_risk_guard(close_full: pd.DataFrame, minimum_stocks=100) -> pd.DataFrame:
    """Tägliche ex-ante-Risikofaktoren, aus den Kursen bis einschließlich T.

    close_full: abgeschlossene Adjusted-Close-Kurse für SPY und S&P-500-Aktien.
    Mindestens 200 auf 200-Tage-MA prüfbare Aktien sind in echten Datensätzen
    sinnvoll; minimum_stocks=100 erlaubt kleineren Universen zur Diagnose.
    Historische Auswahl mit HEUTIGEN Indexmitgliedern bleibt verzerrt.
    """
    if close_full is None or close_full.empty or 'SPY' not in close_full:
        raise ValueError('SPY und ein nicht-leerer Kursdatenblock sind erforderlich.')
    close = close_full.sort_index().copy()
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    if not close.index.is_unique:
        raise ValueError('Doppelte Kurstage im Risiko-Input.')
    if len(close) < 252:
        raise ValueError('Mindestens 252 abgeschlossene NYSE-Handelstage erforderlich.')
    spy = pd.to_numeric(close['SPY'], errors='coerce')
    spy = spy.where(spy > 0)
    if spy.isna().any():
        raise ValueError('SPY enthält fehlende/ungültige Preise. Nicht fortschreiben.')
    others = close.drop(columns=['SPY']).apply(pd.to_numeric, errors='coerce')
    if others.shape[1] < minimum_stocks:
        raise ValueError(f'Mindestens {minimum_stocks} Aktien für Marktbreite erforderlich.')
    others = others.where(others > 0)

    sma50 = spy.rolling(50, min_periods=50).mean()
    sma200 = spy.rolling(200, min_periods=200).mean()
    vol20 = spy.pct_change(fill_method=None).rolling(20, min_periods=20).std() * np.sqrt(252)
    peak252 = spy.rolling(252, min_periods=200).max()
    drawdown = spy / peak252 - 1.0
    perf63 = spy / spy.shift(63) - 1.0
    rebound10 = spy / spy.rolling(10, min_periods=10).min() - 1.0

    # Die Basisaktien müssen mindestens 200 echte Kurswerte besitzen,
    # ansonsten zählt eine Aktie NICHT in Zähler oder Nenner.
    sma200_stocks = others.rolling(200, min_periods=200).mean()
    eligibility = others.notna() & sma200_stocks.notna()
    sample_n = eligibility.sum(axis=1)
    n_above = ((others > sma200_stocks) & eligibility).sum(axis=1)
    breadth = (n_above / sample_n.replace(0, np.nan)).where(sample_n >= minimum_stocks)

    # Fünf transparente Stufenwerte, maximal 100 Punkte.
    trend = pd.Series(np.where(spy < sma200, 25.0,
                        np.where(spy < sma50, 12.5, 0.0)), index=close.index)
    volatility = pd.Series(np.where(vol20 > 0.30, 20.0,
                             np.where(vol20 > 0.20, 10.0, 0.0)), index=close.index)
    breadth_component = pd.Series(np.where(breadth < 0.35, 20.0,
                                     np.where(breadth < 0.50, 10.0, 0.0)), index=close.index)
    decline = pd.Series(np.where(drawdown <= -0.15, 20.0,
                          np.where(drawdown <= -0.08, 10.0, 0.0)), index=close.index)
    bounce_risk = pd.Series(np.where((perf63 < -0.10) & (rebound10 > 0.05) &
                                     (vol20 > 0.20), 15.0, 0.0), index=close.index)
    valid = (sma200.notna() & vol20.notna() & drawdown.notna() &
             perf63.notna() & breadth.notna())
    raw = (trend + volatility + breadth_component + decline + bounce_risk).where(valid)

    signals = _weekly_nyse_signals(close.index)
    exposure = _confirmed_weekly_exposure(raw, signals)
    weekly = pd.Series(close.index.isin(signals), index=close.index)
    last_signal = pd.Series(pd.NaT, index=close.index, dtype='datetime64[ns]')
    last_signal.loc[signals] = signals
    last_signal = last_signal.ffill()
    # Letzter VALIDIERTER Wochen-Score für den Handel, nicht tägliches Zwischenrauschen.
    weekly_score = raw.where(weekly).ffill()
    return pd.DataFrame({
        'SPY': spy,
        'SMA200': sma200,
        'Volatilität_20T': vol20,
        'Marktbreite': breadth,
        'Drawdown_SPY': drawdown,
        '63T_Rendite_SPY': perf63,
        'Erholung_10T': rebound10,
        'Trendpunkte': trend.where(valid),
        'Volatilitätspunkte': volatility.where(valid),
        'Breitepunkte': breadth_component.where(valid),
        'Drawdownpunkte': decline.where(valid),
        'Reversalpunkte': bounce_risk.where(valid),
        'Risikoscore': raw,
        'Risikoscore_letztes_Signal': weekly_score,
        'Risikostufe': raw.map(_level),
        'Ist_NYSE_Wochensignal': weekly,
        'Letzter_Signaltermin': last_signal,
        'Zielaktienquote': exposure,
        'Daten_vollständig': valid,
    }, index=close.index)


def simulate_risk_overlay(base_track: pd.DataFrame, guard: pd.DataFrame, holding_days: int):
    """Rückblickender RISIKO-Overlay auf der historischen Strategie-Kurve.

    Keine Veränderung der Momentum-Aktienauswahl. Der bestehende Portfoliorücklauf
    wird anteilig mit Cash (0 % Zinsen) kombiniert. Aktienquote erst nach dem
    nächsten Session-Close wirksam: Signal am Freitag, Ausführung Montag-Close,
    erster Return unter neuer Aktienquote am Dienstag (shift(2)).

    Vereinfacht: Zielaktienquote wird täglich konstant gehalten; KEIN realer
    Order-/Steuerbacktest, keine Steuern/Kosten, nur analytischer Vergleich.
    """
    if base_track.empty or guard.empty:
        raise ValueError('Portfolio-Track und Risikoverlauf erforderlich.')
    col = f'Momentum Rotation ({holding_days} Tage)'
    if col not in base_track:
        raise ValueError(f'{col} fehlt im Backtest.')
    index = pd.DatetimeIndex(base_track.index).normalize()
    idx = guard.index.intersection(index)
    if len(idx) != len(index):
        raise ValueError('Risikodaten decken nicht alle Track-Handelstage ab.')
    prices = (100.0 + base_track[col].astype(float)).to_numpy()
    if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
        raise ValueError('Ungültige Depotwertentwicklung.')
    baseline = pd.Series(prices, index=index)
    returns = baseline.pct_change(fill_method=None).fillna(0.0)
    # Die Quote vom Signal an T wird erst nach T+1-Close implementiert.
    exposures = guard['Zielaktienquote'].reindex(index).shift(2).fillna(1.0)
    adjusted = (1 + returns * exposures).cumprod() * 100
    adjusted.iloc[0] = 100.0
    original = baseline / baseline.iloc[0] * 100
    result = pd.DataFrame({
        'Original': original,
        'Mit Risk Guard': adjusted,
        'Aktive_Aktienquote': exposures,
        'Risikoscore': guard['Risikoscore_letztes_Signal'].reindex(index)
    }, index=index)
    def metric(series):
        years = (series.index[-1] - series.index[0]).days / 365.25
        dd = series / series.cummax() - 1
        return {'CAGR (%)': ((series.iloc[-1] / series.iloc[0]) ** (1 / years) - 1) * 100,
                'Max. Drawdown (%)': float(dd.min()) * 100,
                'Gesamtrendite (%)': (series.iloc[-1] / series.iloc[0] - 1) * 100}
    report = pd.DataFrame({'Original': metric(original), 'Mit Risk Guard': metric(adjusted)}).T
    return result, report


def diagnostic_forward_losses(base_track: pd.DataFrame, guard: pd.DataFrame, holding_days=5, horizon=20):
    """Rückblickende Trefferhäufigkeiten auf nicht überlappenden 20-Tage-Fenstern.

    KEINE geeichte Zukunftswahrscheinlichkeit: ausgewertete Historie,
    Überlebendenverzerrung und Regimeabhängigkeit können erheblich sein.
    """
    col = f'Momentum Rotation ({holding_days} Tage)'
    if col not in base_track or base_track.empty:
        raise ValueError('Backtest-Track fehlt.')
    wealth = (100 + base_track[col].astype(float)).rename('wealth')
    aligned = pd.DataFrame({'wealth': wealth, 'score': guard['Risikoscore_letztes_Signal'],
                            'signal': guard['Ist_NYSE_Wochensignal']}).dropna(subset=['wealth', 'score'])
    # 20 Handelstage auseinander, daher keine überlappenden Vorwärtsrenditen.
    selected = []
    positions = {stamp: i for i, stamp in enumerate(wealth.index)}
    prev_i = -horizon - 1
    for stamp in aligned.index[aligned['signal'].fillna(False)]:
        i = positions.get(stamp)
        if i is not None and i + horizon < len(wealth) and i - prev_i >= horizon:
            next_wealth = float(wealth.iloc[i + horizon])
            first_wealth = float(wealth.iloc[i])
            selected.append({'Datum': stamp, 'Risikostufe': _level(float(aligned.at[stamp, 'score'])),
                             'Rendite_20T': next_wealth / first_wealth - 1})
            prev_i = i
    if not selected:
        return pd.DataFrame(columns=['Risikostufe', 'Stichprobe', 'Negative 20T (%)', 'Mittlere 20T-Rendite (%)'])
    sample = pd.DataFrame(selected)
    result = sample.groupby('Risikostufe', observed=True)['Rendite_20T'].agg(
        Stichprobe='count', Negative_20T=lambda x: float((x < 0).mean() * 100),
        Mittlere_20T=lambda x: float(x.mean() * 100)).reset_index()
    result = result.rename(columns={'Negative_20T':'Negative 20T (%)',
                                    'Mittlere_20T':'Mittlere 20T-Rendite (%)'})
    return result.sort_values('Risikostufe', key=lambda col: col.map({s:i for i,s in enumerate(LEVEL_ORDER)}))

# ==== ENDE RECHENKERN: momentum_risk_guard ====

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


# ==== BEGIN INTERNER RECHENKERN: finanz_assets ====
import logging
import pandas as pd
import streamlit as st
LOG = logging.getLogger("finanz_terminal")

MOMENTUM_ETFS = {"QQQ", "SPY", "GLD", "COPX", "REMX", "CGW", "URNM", "PJP", "IXG", "IGF", "VHYL"}
S_AND_P_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"


@st.cache_data(ttl=86400, show_spinner=False)
def load_sp500_constituents():
    """Namen und aktuelle GICS-Sektoren aus öffentlicher Datenquelle.
    KEINE historische Mitgliedschaftsdatenbank.
    """
    try:
        raw = pd.read_csv(S_AND_P_CSV)
        if not {"Symbol", "Security"}.issubset(raw.columns) or len(raw) < 400:
            raise ValueError("Die S&P-500-CSV ist unvollständig.")
        raw["Symbol"] = raw["Symbol"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
        raw = raw.drop_duplicates("Symbol")
        return raw
    except Exception as exc:
        LOG.warning("S&P-500-Namensliste nicht verfügbar: %s", exc)
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def get_sp500_name_mapping():
    mapping = {
        "QQQ": "Invesco QQQ Trust (Nasdaq 100)", "SPY": "SPDR S&P 500 ETF Trust",
        "GLD": "SPDR Gold Shares", "COPX": "Global X Copper Miners ETF",
        "REMX": "VanEck Rare Earth Metals ETF", "CGW": "Invesco S&P Global Water ETF",
        "URNM": "Sprott Uranium Miners ETF", "PJP": "Invesco Pharmaceuticals ETF",
        "IXG": "iShares Global Financials ETF", "IGF": "iShares Global Infrastructure ETF",
        "VHYL": "Vanguard FTSE All-World Dividend ETF",
    }
    df = load_sp500_constituents()
    if not df.empty:
        mapping.update(dict(zip(df["Symbol"], df["Security"])))
    return mapping


def get_sp500_sector_mapping():
    df = load_sp500_constituents()
    if not df.empty and "GICS Sector" in df.columns:
        return dict(zip(df["Symbol"], df["GICS Sector"].fillna("")))
    return {}


def get_asset_name(ticker):
    if ticker is None or not str(ticker).strip():
        return ""
    t = str(ticker).strip().upper().replace(".", "-")
    return get_sp500_name_mapping().get(t, t)

# ==== ENDE RECHENKERN: finanz_assets ====


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

# Momentum-Berechnungen als unabhängiger Rechenkern in dieser Datei.
# Die UI wird erst beim Öffnen des entsprechenden Tabs ausgeführt.
# ==== BEGIN INTERNER RECHENKERN: momentum_engine ====
import os
import logging
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_market_calendars as mcal

MOMENTUM_LOG = logging.getLogger('momentum_v2')
MOMENTUM_ETFS = {"QQQ", "SPY", "GLD", "COPX", "REMX", "CGW", "URNM", "PJP", "IXG", "IGF", "VHYL"}


def _mt_symbol(ticker):
    return str(ticker).strip().upper().replace('.', '-')


def _mt_memberships():
    """Optionales vollständiges Datumspanel historischer Indexmitgliedschaften.

    CSV: ticker,start_date,end_date; end_date inklusive, leer = noch Mitglied.
    Ein vollständiger historischer Datenbestand inkl. delisteter Titel ist Voraussetzung
    für einen entsprechend aussagekräftigen survivorship-freien Backtest.
    """
    path = 'sp500_membership_history.csv'
    if not os.path.isfile(path):
        return None
    raw = pd.read_csv(path, dtype={'ticker': str, 'start_date': str, 'end_date': str})
    required = {'ticker', 'start_date', 'end_date'}
    if not required.issubset(raw.columns):
        raise ValueError(f'{path}: Spalten {sorted(required)} erforderlich.')
    out = {}
    for row in raw.itertuples(index=False):
        ticker = _mt_symbol(row.ticker)
        start = pd.to_datetime(row.start_date, errors='coerce')
        end = pd.to_datetime(row.end_date, errors='coerce')
        if ticker and pd.notna(start):
            out.setdefault(ticker, []).append((start.normalize(),
                                                 end.normalize() if pd.notna(end) else pd.Timestamp.max.normalize()))
    if not out:
        raise ValueError('Historische Mitgliedschaftsdatei ist leer oder ungültig.')
    return out


def _mt_sectors():
    """Sektormapping optional via vorhandener Hilfsfunktion oder lokaler CSV."""
    result = {}
    getter = globals().get('get_sp500_sector_mapping')
    if callable(getter):
        try:
            mapped = getter()
            if isinstance(mapped, dict):
                result.update({_mt_symbol(k): str(v) for k, v in mapped.items() if pd.notna(v)})
        except Exception as exc:
            MOMENTUM_LOG.warning('Sektormapping-Funktion: %s', exc)
    path = 'sp500_sectors.csv'
    if os.path.isfile(path):
        df = pd.read_csv(path)
        if {'ticker', 'sector'}.issubset(df.columns):
            for t, s in df[['ticker', 'sector']].dropna().itertuples(index=False, name=None):
                result[_mt_symbol(t)] = str(s)
    return result


@st.cache_data(ttl=86400, max_entries=2, show_spinner=False)
def fetch_sp500_universe():
    """Keine stille Umstellung auf nur 18 Mega-Caps bei Datenproblemen."""
    try:
        mapping = get_sp500_name_mapping()
        if not isinstance(mapping, dict):
            raise ValueError('get_sp500_name_mapping() lieferte kein Dictionary.')
        current = {_mt_symbol(t) for t in mapping if _mt_symbol(t) not in MOMENTUM_ETFS}
        if len(current) < 400:
            raise ValueError(f'Nur {len(current)} S&P-500-Ticker vorhanden (mind. 400 benötigt).')
        history = _mt_memberships()
        if history:
            current.update(history.keys())
        return sorted(current)
    except Exception as exc:
        st.error(f'S&P-500-Universum nicht zuverlässig geladen: {exc}')
        return []


def _mt_read_download(raw):
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0)
        level1 = raw.columns.get_level_values(1)
        if 'Close' in level0 and 'Volume' in level0:
            close, vol = raw['Close'], raw['Volume']
        elif 'Close' in level1 and 'Volume' in level1:
            close = raw.xs('Close', axis=1, level=1)
            vol = raw.xs('Volume', axis=1, level=1)
        else:
            return pd.DataFrame(), pd.DataFrame()
    elif {'Close', 'Volume'}.issubset(raw.columns):
        close, vol = raw[['Close']], raw[['Volume']]
    else:
        return pd.DataFrame(), pd.DataFrame()
    close, vol = close.copy(), vol.copy()
    close.columns = [_mt_symbol(c) for c in close.columns]
    vol.columns = [_mt_symbol(c) for c in vol.columns]
    close = close.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    vol = vol.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return close, vol


@st.cache_data(ttl=3600, max_entries=2, show_spinner=False)
def fetch_sp500_data(tickers, years=16):
    """Chunk-Download inkl. SPY, OHNE Vorwärtsauffüllen fehlender Aktienkurse.

    Verwendet auto_adjust=True (dividenden-/splitbereinigte Close-Reihe).
    Nur vollständig abgeschlossene NYSE-Sitzungen sind zulässig.
    """
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
    ticker_list = ['SPY'] + sorted({_mt_symbol(t) for t in tickers} - {'SPY'})
    start = (pd.Timestamp.now(tz='America/New_York').tz_localize(None)
             - pd.DateOffset(years=int(years))).strftime('%Y-%m-%d')
    close_parts, vol_parts = [], []
    for j in range(0, len(ticker_list), 50):
        batch = ticker_list[j:j + 50]
        try:
            raw = yf.download(
                batch, start=start, interval='1d', auto_adjust=True,
                actions=False, group_by='column', progress=False,
                threads=True, timeout=25
            )
            c, v = _mt_read_download(raw)
            if not c.empty:
                close_parts.append(c)
                vol_parts.append(v)
        except Exception as exc:
            MOMENTUM_LOG.exception('Yahoo-Download für Charge %s fehlgeschlagen: %s', j // 50, exc)
    if not close_parts:
        st.error('Keine brauchbaren Kursdaten heruntergeladen.')
        return pd.DataFrame(), pd.DataFrame()

    close = pd.concat(close_parts, axis=1)
    vol = pd.concat(vol_parts, axis=1)
    close = close.loc[:, ~close.columns.duplicated()]
    vol = vol.loc[:, ~vol.columns.duplicated()]
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    vol.index = pd.DatetimeIndex(vol.index).tz_localize(None).normalize()
    close = close.sort_index().loc[lambda x: ~x.index.duplicated(keep='last')]
    vol = vol.sort_index().loc[lambda x: ~x.index.duplicated(keep='last')]
    if 'SPY' not in close.columns:
        st.error('SPY-Benchmark fehlt: Backtest wird nicht simuliert.')
        return pd.DataFrame(), pd.DataFrame()

    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=close.index.min(), end_date=close.index.max())
    now_utc = pd.Timestamp.now(tz='UTC')
    finished = schedule.index[(schedule['market_close'] + pd.Timedelta(minutes=15)) <= now_utc]
    complete_dates = pd.DatetimeIndex(finished).normalize()
    common = close.index.intersection(complete_dates)
    close = close.reindex(common).sort_index()
    vol = vol.reindex(common).sort_index()
    valid_spy = close['SPY'].notna() & (close['SPY'] > 0)
    close = close.loc[valid_spy]
    vol = vol.reindex(close.index)

    cols = close.columns.intersection(vol.columns)
    close = close[cols].where(close[cols] > 0)
    vol = vol[cols].where(vol[cols] >= 0)
    if close.empty or 'SPY' not in close.columns:
        st.error('Keine abgeschlossenen NYSE-Börsentage mit gültigem SPY-Kurs gefunden.')
        return pd.DataFrame(), pd.DataFrame()
    # Offensichtlich unvollständige Downloads nicht als S&P-500-Scan ausgeben.
    usable = ['SPY'] + [t for t in close.columns if t != 'SPY' and close[t].notna().sum() >= 200]
    if len(usable) - 1 < 400:
        st.error(f'Yahoo lieferte nur {len(usable) - 1} brauchbare Aktienreihen. '
                 'Bitte später erneut laden; unvollständiger S&P-500-Test wurde verhindert.')
        return pd.DataFrame(), pd.DataFrame()
    return close[usable], vol[usable]


def _mt_signal_dates(index):
    """Letzte reguläre Handelssitzung JEDER NYSE-Kalenderwoche, inkl. Feiertage."""
    schedule = mcal.get_calendar('NYSE').schedule(
        start_date=index.min(), end_date=index.max() + pd.Timedelta(days=7)
    )
    sessions = pd.DatetimeIndex(schedule.index).normalize()
    periods = sessions.to_period('W-FRI')
    last_session_per_week = pd.Series(sessions, index=periods).groupby(level=0).last()
    return pd.DatetimeIndex(last_session_per_week.values).intersection(index)


def _mt_next_session(date):
    schedule = mcal.get_calendar('NYSE').schedule(
        start_date=pd.Timestamp(date) + pd.Timedelta(days=1),
        end_date=pd.Timestamp(date) + pd.Timedelta(days=12)
    )
    if schedule.empty:
        return None
    return pd.Timestamp(schedule.index[0]).strftime('%d.%m.%Y')


@st.cache_data(ttl=3600, max_entries=4, show_spinner=False)
def calculate_momentum_strategy(
    df_close_full, df_vol_full, holding_days=5, hysterese_limit=12,
    n_positions=8, market_mode='50%', backtest_years=15,
    sector_limit=None
):
    """Signal am abgeschlossenen NYSE-Wochenschluss,
    Ausführung zum NÄCHSTEN Handelstagsschluss (simulierte MOC-Order).
    holding_days=5: jede Woche; =10: jede zweite; =20: jede vierte Woche.
    Transaktionskosten = 0, Cashzins = 0.
    Rückgabe kompatibel: (holdings_df, tracking_df, kpis_dict).
    """
    try:
        if holding_days not in (5, 10, 20):
            raise ValueError('Kalenderwochen-Rotation unterstützt holding_days=5, 10 oder 20.')
        if df_close_full.empty or df_vol_full.empty or 'SPY' not in df_close_full:
            raise ValueError('Kursdaten oder SPY-Benchmark fehlen.')
        close = df_close_full.sort_index().copy()
        volume = df_vol_full.reindex(index=close.index, columns=close.columns).copy()
        spy = close['SPY'].astype(float)
        symbols = [s for s in close.columns if s != 'SPY']
        prices = close[symbols].astype(float)
        volumes = volume[symbols].astype(float)
        if len(close) < 260 or len(symbols) < n_positions:
            raise ValueError('Zu wenige Handelstage/Aktien für einen Momentum-Backtest.')

        sma20 = prices.rolling(20, min_periods=20).mean()
        sma50 = prices.rolling(50, min_periods=50).mean()
        sma200 = prices.rolling(200, min_periods=200).mean()
        v10 = volumes.rolling(10, min_periods=10).mean()
        v50 = volumes.rolling(50, min_periods=50).mean()
        vol90 = prices.pct_change(fill_method=None).rolling(90, min_periods=90).std() * np.sqrt(252)
        spy_sma200 = spy.rolling(200, min_periods=200).mean()
        mom = prices.shift(5) / prices.shift(130) - 1.0

        membership = _mt_memberships()
        sectors = _mt_sectors() if sector_limit is not None else {}
        index = close.index
        cutoff = index[-1] - pd.DateOffset(years=int(backtest_years))
        start_i = max(200, int(index.searchsorted(cutoff)))
        if start_i >= len(index) - 3:
            raise ValueError('Backtestzeitraum zu kurz.')

        calendar_signals = _mt_signal_dates(index)
        signal_index = {}
        anchor = pd.Timestamp('2020-01-06')
        for date in calendar_signals:
            i = int(index.get_indexer([date])[0])
            if i < start_i or i >= len(index):
                continue
            week_number = (pd.Timestamp(date).normalize() - anchor).days // 7
            if week_number % (holding_days // 5) == 0:
                signal_index[i] = date

        x = np.arange(90, dtype=float)
        xc = x - x.mean()
        xsum2 = np.dot(xc, xc)
        features_cache = {}

        def features_at(t):
            if t in features_cache:
                return features_cache[t]
            p = prices.iloc[t]
            mask = (
                (p > sma20.iloc[t]) & (sma20.iloc[t] > sma50.iloc[t]) &
                (sma50.iloc[t] > sma200.iloc[t]) &
                (v10.iloc[t] > v50.iloc[t]) & (mom.iloc[t] > 0) &
                np.isfinite(vol90.iloc[t]) & (vol90.iloc[t] > 1e-8)
            )
            eligible = list(p.index[mask.fillna(False)])
            if membership is not None:
                day = index[t]
                eligible = [s for s in eligible if any(a <= day <= b for a, b in membership.get(s, []))]
            if not eligible:
                features_cache[t] = ({}, {}, {})
                return features_cache[t]
            arr = prices[eligible].iloc[t - 89:t + 1].to_numpy(dtype=float)
            valid = np.isfinite(arr).all(axis=0) & (arr > 0).all(axis=0)
            eligible = [s for s, ok in zip(eligible, valid) if ok]
            if not eligible:
                features_cache[t] = ({}, {}, {})
                return features_cache[t]
            arr = np.log(arr[:, valid])
            arr = arr - arr.mean(axis=0)
            den = np.sqrt(xsum2 * np.sum(arr * arr, axis=0))
            corr = np.divide(xc @ arr, den, out=np.zeros(len(eligible)), where=den > 1e-15)
            r2vals = np.clip(corr ** 2, 0, 1)
            r2 = dict(zip(eligible, r2vals))
            scores = {}
            for symbol in eligible:
                score = (float(mom.at[index[t], symbol]) /
                         float(vol90.at[index[t], symbol])) * float(r2[symbol])
                if np.isfinite(score):
                    scores[symbol] = score
            features_cache[t] = (scores, r2, {s: float(mom.at[index[t], s]) for s in scores})
            return features_cache[t]

        def target_at(t, held_symbols):
            scores, r2, mom_values = features_at(t)
            ranked = sorted(scores, key=lambda sym: (-scores[sym], sym))
            selected = []
            sector_counts = {}

            def try_add(sym):
                if sym in selected:
                    return False
                sector = sectors.get(sym)
                # Bei fehlendem Sektor: Positionslimit nicht blind als erfüllt annehmen.
                if sector_limit is not None and (not sector or sector_counts.get(sector, 0) >= sector_limit):
                    return False
                selected.append(sym)
                if sector:
                    sector_counts[sector] = sector_counts.get(sector, 0) + 1
                return True

            for sym in held_symbols:
                if sym in ranked[:hysterese_limit]:
                    try_add(sym)
            for sym in ranked:
                if len(selected) == n_positions:
                    break
                try_add(sym)
            if market_mode == 'aus':
                exposure = 1.0
            else:
                bull = bool(spy.iloc[t] > spy_sma200.iloc[t])
                exposure = 1.0 if bull else (0.5 if market_mode == '50%' else 0.0)
            # Nicht besetzte Aktien-Slots bleiben Cash statt Klumpenrisiko.
            targets = {sym: exposure / n_positions for sym in selected if exposure > 0}
            return selected, targets, scores, r2, mom_values, exposure

        wealth = 100.0
        benchmark = 100.0
        weights = {}  # tatsächliche, täglich fortgeschriebene Aktiengewichte
        pending = None
        times, wealth_path, spy_path = [index[start_i]], [wealth], [benchmark]
        n_trades = 0
        execution_events = []  # Nur tatsächlich ausgeführte Zielgewichtsänderungen
        total_turnover = 0.0
        last_target = []
        last_signal_i = None
        last_scores, last_r2, last_mom = {}, {}, {}
        last_exposure = 0.0
        n_signals = 0
        signal_snapshots = []  # JEDES Signal, auch ohne tatsächlichen Trade.

        for i in range(start_i, len(index)):
            if i > start_i:
                # zuerst alte Aktien vom vorherigen Schluss zum heutigen Schluss bewerten
                growth = 1.0
                if weights:
                    prev_p = prices.iloc[i - 1]
                    now_p = prices.iloc[i]
                    returns = {}
                    for sym in weights:
                        p0, p1 = prev_p[sym], now_p[sym]
                        if not (np.isfinite(p0) and np.isfinite(p1) and p0 > 0 and p1 > 0):
                            raise ValueError(f'Fehlender Preis in offener Position {sym} am {index[i].date()}. '
                                             'Kein Auffüllen mit fiktiver 0%-Rendite.')
                        returns[sym] = float(p1 / p0 - 1)
                    growth += sum(w * returns[sym] for sym, w in weights.items())
                    if growth <= 0:
                        raise ValueError('Ungültiger Portfoliofaktor <= 0.')
                    weights = {s: w * (1 + returns[s]) / growth for s, w in weights.items()}
                wealth *= growth
                benchmark *= float(spy.iloc[i] / spy.iloc[i - 1])

                # erst JETZT Order vom Vortag am heutigen Close simulieren
                if pending is not None:
                    new_weights = pending
                    universe = set(weights) | set(new_weights)
                    turnover = 0.5 * (
                        sum(abs(weights.get(s, 0.0) - new_weights.get(s, 0.0)) for s in universe) +
                        abs((1 - sum(weights.values())) - (1 - sum(new_weights.values())))
                    )
                    if turnover > 1e-10:
                        n_trades += 1
                        total_turnover += turnover
                        execution_events.append({
                            'date': str(index[i].date()),
                            'signal_date': str(index[i - 1].date()),
                            'targets': dict(new_weights),
                        })
                        weights = new_weights.copy()
                    pending = None
                times.append(index[i])
                wealth_path.append(wealth)
                spy_path.append(benchmark)

            # NYSE-Wochenschluss: Signal erst nach Tagesclose; Ausführung nächste Sitzung
            if i in signal_index:
                n_signals += 1
                selected, targets, scores, r2vals, mom_values, exposure = target_at(i, list(weights))
                last_signal_i = i
                last_target = selected
                last_scores, last_r2, last_mom = scores, r2vals, mom_values
                last_exposure = exposure
                # Signal-/Periodenjournal: feste NYSE-Kalenderwochen statt
                # abhängig vom aktuellen Download verschobene Zählung.
                # Die Aktie wird erst zum nächsten Handelstagsschluss handelbar.
                effective = (pd.Timestamp(index[i + 1]) if i + 1 < len(index)
                             else _mt_following_session(index[i]))
                signal_snapshots.append({
                    'signal_date': str(pd.Timestamp(index[i]).date()),
                    'execution_date': str(effective.date()) if effective is not None else None,
                    'tickers': list(selected) if exposure > 0 else [],
                    'qualified_tickers': list(selected),
                    'scores': {sym: round(float(scores.get(sym, 0.0)), 8) for sym in selected},
                    'exposure_pct': float(exposure * len(selected) / n_positions * 100),
                })
                # Ohne Portfolio- oder deutliche Expositionsaenderung keine Order.
                # Keine kuenstliche woechentliche Gleichgewichtung der Bestandswerte.
                same_names = set(weights) == set(targets)
                same_exposure = abs(sum(weights.values()) - sum(targets.values())) < 0.05
                pending = None if same_names and same_exposure else targets

        # Falls letztes Signal bereits im Track ausgeführt wurde, ist pending None.
        # Die Zielpositionen bleiben für die Anzeige trotzdem erhalten.
        rows = []
        t_final = last_signal_i
        if t_final is not None:
            for symbol in last_target:
                price_now = float(prices.at[index[t_final], symbol])
                old_i = max(0, t_final - holding_days)
                old_price = prices.at[index[old_i], symbol]
                recent_ret = (price_now / old_price - 1) * 100 if pd.notna(old_price) and old_price > 0 else np.nan
                try:
                    name = get_asset_name(symbol)
                except Exception:
                    name = symbol
                rows.append({
                    'Ticker': symbol,
                    'Unternehmen': name,
                    'Score': last_scores.get(symbol, np.nan),
                    'Trend-Qualität (R²)': last_r2.get(symbol, np.nan),
                    f'Rendite ({holding_days} Tage)': recent_ret,
                    'Rendite (T-130 bis T-5)': last_mom.get(symbol, np.nan) * 100,
                    'Volatilität': float(vol90.at[index[t_final], symbol]) * 100,
                    'Aktueller Kurs': price_now,
                    'Zielgewicht (%)': last_exposure / n_positions * 100,
                })
        holdings_df = pd.DataFrame(rows)

        track = pd.DataFrame({
            f'Momentum Rotation ({holding_days} Tage)': np.asarray(wealth_path) - 100.0,
            'S&P 500 Benchmark': np.asarray(spy_path) - 100.0,
        }, index=pd.DatetimeIndex(times))
        port = pd.Series(wealth_path, index=times, dtype=float)
        bm = pd.Series(spy_path, index=times, dtype=float)
        one_year_i = max(0, len(port) - 253)
        trailing_1y = (port.iloc[-1] / port.iloc[one_year_i] - 1) * 100
        trailing_spy = (bm.iloc[-1] / bm.iloc[one_year_i] - 1) * 100
        elapsed_years = max(1e-6, (port.index[-1] - port.index[0]).days / 365.25)
        cagr = ((port.iloc[-1] / port.iloc[0]) ** (1 / elapsed_years) - 1) * 100
        drawdown = port / port.cummax() - 1
        daily_r = port.pct_change().dropna()
        daily_std = float(daily_r.std(ddof=1)) if len(daily_r) > 1 else np.nan
        sharpe = (float(daily_r.mean()) / daily_std * np.sqrt(252)
                  if np.isfinite(daily_std) and daily_std > 0 else np.nan)
        annual_returns = []
        for year, year_values in port.groupby(port.index.year):
            # 200+ Tage und keine angefangenen Rumpfjahre, sofern Handelstage fehlen
            if len(year_values) >= 200 and year_values.index[0].month == 1 and year_values.index[-1].month == 12:
                prev = port.loc[port.index < year_values.index[0]]
                if not prev.empty:
                    annual_returns.append((year_values.iloc[-1] / prev.iloc[-1] - 1) * 100)
        median_year = float(np.median(annual_returns)) if annual_returns else np.nan
        period_end_indices = [i for i in sorted(signal_index) if i > start_i]
        periodic = port.reindex(index[period_end_indices]).pct_change().dropna()
        winrate = float((periodic > 0).mean() * 100) if not periodic.empty else np.nan
        kpis = {
            'ist_rendite': trailing_1y,
            'bench_rendite': trailing_spy,
            'alpha': trailing_1y - trailing_spy,
            'win_rate': winrate,
            'prog_median': median_year,  # Legacy-Key; jetzt historischer Jahresmedian, KEINE Prognose!
            'prog_pess': float(np.percentile(annual_returns, 10)) if annual_returns else np.nan,
            'prog_opt': float(np.percentile(annual_returns, 90)) if annual_returns else np.nan,
            'cagr': cagr,
            'max_drawdown': float(drawdown.min() * 100),
            'sharpe': sharpe,
            'total_return': (port.iloc[-1] / port.iloc[0] - 1) * 100,
            'trades': n_trades,
            'turnover_p_a': total_turnover / elapsed_years * 100,
            'start_date': str(port.index[0].date()),
            'end_date': str(port.index[-1].date()),
            'last_signal_date': str(index[t_final].date()) if t_final is not None else None,
            'next_execution': _mt_next_session(index[t_final]) if t_final is not None else None,
            'target_tickers': last_target,
            'simulated_tickers': list(weights),
            'target_exposure': last_exposure * (len(last_target) / n_positions),
            'sector_filter_used': sector_limit is not None,
            'membership_history_used': membership is not None,
            'signals': n_signals,
            'costs_included': False,
            'execution_events': execution_events,
            'signal_snapshots': signal_snapshots,
        }
        return holdings_df, track, kpis
    except Exception as exc:
        MOMENTUM_LOG.exception('Momentum-Berechnung fehlgeschlagen')
        st.error(f'Momentum-Berechnung ({holding_days} Tage) fehlgeschlagen: {exc}')
        return pd.DataFrame(), pd.DataFrame(), {}

# -------------------------------------------------------------
# EURO-DEPOT-SIMULATION FÜR MOMENTUM
# -------------------------------------------------------------
def german_stock_tax_rate(church_tax_percent=0):
    """Abgeltungsteuer inkl. Soli und Kirchensteuer-Modell bei §32d EStG.
    Nur als vereinfachtes Steuerszenario, keine Steuerabrechnung.
    """
    church = float(church_tax_percent) / 100.0
    return (0.25 / (1.0 + 0.25 * church)) * (1.055 + church)


@st.cache_data(ttl=3600, max_entries=6, show_spinner=False)
def simulate_momentum_eur_depot(
    close_full, events, eurusd, capital=8000.0, fixed_fee=1.0,
    variable_fee_pct=0.1, saver_allowance=1000.0, church_tax=0,
    include_tax=True, fractions=False, min_order=10.0,
):
    """Simuliert echte Geldbeträge anhand der Zielgewichts-Events der Momentumstrategie.

    Aktienkurse sind auto-adjusted: daher nur *modellierte* Einstandskurse,
    nicht steuerlich richtige Broker-Einstandskurse. Keine tatsächliche
    Dividenden-/Split-/Quellensteuer- oder lückenlose Corporate-Action-Historie.
    Ein- und Auszahlungen von Fremdwährung werden jeweils am Handelstag in EUR
    umgerechnet; kein dauerhaftes Fremdwährungscashkonto. FIFO je Aktie.
    Vorhandene Steuergewinne und -verluste werden unterjährig verrechnet.
    Keine externen Kapitalflüsse; keine Fremdfinanzierung; Cashzins = 0.
    """
    if close_full is None or close_full.empty or 'SPY' not in close_full:
        raise ValueError('Ungültige Aktienkurse / SPY nicht verfügbar.')
    if capital <= 0 or fixed_fee < 0 or variable_fee_pct < 0 or min_order < 0:
        raise ValueError('Startkapital muss positiv und Kosten dürfen nicht negativ sein.')
    if not events:
        raise ValueError('Im Backtest wurden keine Umschichtungen ausgeführt.')
    idx = pd.DatetimeIndex(close_full.index).normalize()
    first_date = pd.Timestamp(events[0]['date']).normalize()
    # Vorgeschaltete Simulationsstartphase: letzter sichtbarer Track beginnt
    # vor erster Zielposition. Für konsistente Nettokurve Start am ersten Trade.
    idx = idx[idx >= first_date]
    closes = close_full.reindex(idx)
    fx = eurusd.copy().astype(float)
    fx.index = pd.DatetimeIndex(fx.index).tz_localize(None).normalize()
    fx = fx[~fx.index.duplicated(keep='last')]
    fx = fx.reindex(idx).ffill(limit=3)
    if fx.isna().any() or (fx <= 0).any():
        raise ValueError('EUR/USD fehlt für einzelne historische NYSE-Sitzungen. Keine künstliche Umrechnung.')
    event_map = {pd.Timestamp(e['date']).normalize(): e for e in events}
    names = sorted({t for e in events for t in e['targets']})
    missing_columns = [t for t in names if t not in closes.columns]
    if missing_columns:
        raise ValueError(f'Handelsdaten fehlen für: {", ".join(missing_columns[:8])}')
    # Kurslücken im historischen Download periodengerecht prüfen.
    # Vor dem Börsengang einer Aktie fehlen naturgemäß Kurse. Entscheidend ist,
    # dass die Aktie an ihrem Kauftag und während der Haltedauer handelbar ist.
    fee_rate = float(variable_fee_pct) / 100.0
    tax_rate = german_stock_tax_rate(church_tax)
    cash = float(capital)
    # FIFO-Lots: [{qty, eur_cost_per_unit}] inkl. Kaufkosten, wie steuerliche Anschaffungskosten
    lots = {}
    n_orders = 0
    total_fees = 0.0
    taxes_paid = 0.0
    realized_gain = 0.0
    journal = []
    series = []
    exposure_series = []
    liquidation_series = []
    carry_losses = 0.0
    year = int(idx[0].year)
    year_realized = 0.0
    year_tax_paid = 0.0

    def booked_qty(t):
        return sum(lot['qty'] for lot in lots.get(t, []))

    def year_tax_liability(net):
        return tax_rate * max(0.0, net - carry_losses - float(saver_allowance)) if include_tax else 0.0

    def rollover(new_year):
        nonlocal year, year_realized, year_tax_paid, carry_losses
        if year != new_year:
            carry_losses = max(0.0, carry_losses - year_realized)
            year = new_year
            year_realized = 0.0
            year_tax_paid = 0.0

    def mark_to_market(date, prices_eur):
        value = sum(booked_qty(t) * prices_eur[t] for t in lots)
        return cash + value, value

    for date in idx:
        rollover(int(date.year))
        px_eur = closes.loc[date] / float(fx.loc[date])
        required = {sym for sym in lots if booked_qty(sym) > 1e-10}
        if date in event_map:
            required.update(event_map[date]['targets'])
        invalid = [sym for sym in sorted(required)
                   if sym not in px_eur.index or not np.isfinite(px_eur[sym]) or px_eur[sym] <= 0]
        if invalid:
            raise ValueError(f'Kein handelbarer Kurs am {date:%d.%m.%Y}: '
                             f'{", ".join(invalid[:8])}. Simulation abgebrochen.')
        if date in event_map:
            desired = dict(event_map[date]['targets'])
            equity_before, _ = mark_to_market(date, px_eur)
            target_value = {sym: equity_before * max(0.0, float(weight))
                            for sym, weight in desired.items()}
            sales_this_event = []
            # 1) Verkäufe zuerst ausführen: genug EUR-Liquidität für Käufe und Steuern.
            for sym in sorted(list(lots)):
                qty = booked_qty(sym)
                if qty <= 1e-10:
                    continue
                unit = float(px_eur[sym])
                now_value = qty * unit
                want = target_value.get(sym, 0.0)
                delta = max(0.0, now_value - want)
                if delta < float(min_order) and sym in target_value:
                    continue
                sell_qty = min(qty, delta / unit)
                if not fractions and sell_qty < qty - 1e-8:
                    sell_qty = float(int(sell_qty))
                if sell_qty < 1e-10:
                    continue
                amount = sell_qty * unit
                fee = float(fixed_fee) + amount * fee_rate
                if amount <= fee:
                    continue
                remaining = sell_qty
                cost_basis = 0.0
                for lot in lots[sym]:
                    taken = min(lot['qty'], remaining)
                    lot['qty'] -= taken
                    remaining -= taken
                    cost_basis += taken * lot['cost']
                    if remaining <= 1e-8:
                        break
                lots[sym] = [item for item in lots[sym] if item['qty'] > 1e-8]
                if not lots[sym]:
                    del lots[sym]
                gain = amount - fee - cost_basis
                cash += amount - fee
                year_realized += gain
                realized_gain += gain
                total_fees += fee
                n_orders += 1
                sales_this_event.append(len(journal))
                journal.append({'Datum': date, 'Ticker': sym, 'Aktion': 'Verkauf',
                                'Stück': sell_qty, 'Kurs (€)': unit, 'Betrag (€)': amount,
                                'Gebühr (€)': fee, 'Realisierter Gewinn (€)': gain,
                                'Steuer (€)': 0.0})
            # 2) Unterjährige Modell-Steuer / Erstattung (inkl. Verlustverrechnung)
            tax_delta = year_tax_liability(year_realized) - year_tax_paid
            cash -= tax_delta
            taxes_paid += tax_delta
            year_tax_paid += tax_delta
            if abs(tax_delta) > 1e-8:
                # Eigener Journalposten vermeidet willkürliche Zuordnung der Steuer
                # zu einem einzelnen Trade, wenn mehrere Verkäufe vorliegen.
                journal.append({'Datum': date, 'Ticker': 'STEUER',
                                'Aktion': 'Steuerzahlung' if tax_delta > 0 else 'Steuererstattung',
                                'Stück': 0.0, 'Kurs (€)': np.nan, 'Betrag (€)': 0.0,
                                'Gebühr (€)': 0.0, 'Realisierter Gewinn (€)': 0.0,
                                'Steuer (€)': tax_delta})
            # 3) Restbudget auf die gewünschten Aktien verteilen.
            # Nach Steuern und Gebühren kann die effektive Quote vom Soll abweichen.
            equity_after_sales, _ = mark_to_market(date, px_eur)
            wanted_after_tax = {sym: equity_after_sales * max(0.0, float(w))
                                for sym, w in desired.items()}
            buy_list = sorted(wanted_after_tax, key=lambda t: -(wanted_after_tax[t] - booked_qty(t) * px_eur[t]))
            for sym in buy_list:
                unit = float(px_eur[sym])
                missing_value = max(0.0, wanted_after_tax[sym] - booked_qty(sym) * unit)
                if missing_value < float(min_order) or cash <= fixed_fee:
                    continue
                affordable = max(0.0, (cash - fixed_fee) / (1.0 + fee_rate))
                amount = min(missing_value, affordable)
                qty = amount / unit
                if not fractions:
                    qty = float(int(qty))
                if qty <= 1e-10:
                    continue
                amount = qty * unit
                fee = fixed_fee + fee_rate * amount
                if amount + fee > cash + 1e-7:
                    continue
                cash -= amount + fee
                lots.setdefault(sym, []).append({'qty': qty, 'cost': (amount + fee) / qty})
                total_fees += fee
                n_orders += 1
                journal.append({'Datum': date, 'Ticker': sym, 'Aktion': 'Kauf',
                                'Stück': qty, 'Kurs (€)': unit, 'Betrag (€)': amount,
                                'Gebühr (€)': fee, 'Realisierter Gewinn (€)': 0.0,
                                'Steuer (€)': 0.0})
        equity, invested = mark_to_market(date, px_eur)
        if cash < -0.01 or equity < -0.01:
            raise ValueError('Negative Kasse oder negativer Depotwert – unzulässige Konfiguration.')
        series.append(equity)
        exposure_series.append(invested / equity if equity > 0 else 0.0)
        # Liquidationsvergleich ohne tatsächliche Verkaufsorder: alle offenen
        # FIFO-Lots zum heutigen bereinigten Schlusskurs hypothetisch verkaufen.
        # Keine Beeinflussung der späteren Trades oder Cashbestände.
        liquidation_gains_today = 0.0
        liquidation_fees_today = 0.0
        for sym, open_lots in lots.items():
            qty = sum(item['qty'] for item in open_lots)
            if qty <= 1e-10:
                continue
            gross_sale = qty * float(px_eur[sym])
            fee = float(fixed_fee) + gross_sale * fee_rate
            basis = sum(item['qty'] * item['cost'] for item in open_lots)
            liquidation_fees_today += fee
            liquidation_gains_today += gross_sale - fee - basis
        additional_tax_today = (year_tax_liability(year_realized + liquidation_gains_today)
                                - year_tax_liability(year_realized))
        liquidation_series.append(equity - liquidation_fees_today - additional_tax_today)
    nav = pd.Series(series, index=idx, name='Nach Kosten & Steuer (€)')
    liquidation_nav = pd.Series(liquidation_series, index=idx,
                                name='Wert bei hypothetischem Verkauf (€)')
    br = pd.Series(float(capital) * (closes['SPY'].to_numpy() / fx.to_numpy()) /
                   (float(closes['SPY'].iloc[0]) / float(fx.iloc[0])),
                   index=idx, name='SPY ohne Kosten und Steuer (€)')
    current = {sym: booked_qty(sym) for sym in lots if booked_qty(sym) > 1e-10}
    final_px = (closes.iloc[-1] / fx.iloc[-1])
    current_values = {sym: q * float(final_px[sym]) for sym, q in current.items()}
    # Theoretische sofortige Komplettveräußerung, um latente Gewinne zu berücksichtigen.
    liquidation_gains = 0.0
    liquidation_fees = 0.0
    for sym, q in current.items():
        gross_sale = current_values[sym]
        sell_fee = fixed_fee + gross_sale * fee_rate
        liquidation_fees += sell_fee
        basis = sum(lot['qty'] * lot['cost'] for lot in lots[sym])
        liquidation_gains += gross_sale - sell_fee - basis
    current_year_tax = year_tax_liability(year_realized)
    liquidation_year_tax = year_tax_liability(year_realized + liquidation_gains)
    liquidation_incremental_tax = liquidation_year_tax - current_year_tax
    liquidation_value = float(liquidation_nav.iloc[-1])
    elapsed_years = max(1e-9, (idx[-1] - idx[0]).days / 365.25)
    dd = nav / nav.cummax() - 1.0
    info = {
        'start': str(idx[0].date()), 'end': str(idx[-1].date()),
        'end_value': float(nav.iloc[-1]), 'profit': float(nav.iloc[-1] - capital),
        'return_pct': float((nav.iloc[-1] / capital - 1) * 100),
        'cagr': float(((nav.iloc[-1] / capital) ** (1 / elapsed_years) - 1) * 100),
        'max_drawdown': float(dd.min() * 100),
        'fees': float(total_fees), 'taxes': float(taxes_paid),
        'realized_gain': float(realized_gain), 'orders': n_orders,
        'cash': float(cash), 'positions': current,
        'position_values_eur': current_values,
        'cash_ratio': float(cash / nav.iloc[-1]) if nav.iloc[-1] else np.nan,
        'exposure': float(exposure_series[-1]) if exposure_series else 0.0,
        'liquidation_value': liquidation_value,
        'liquidation_tax_additional': liquidation_incremental_tax,
        'tax_rate_pct': tax_rate * 100,
        'loss_carry': max(0.0, carry_losses - year_realized),
    }
    return pd.DataFrame({'Netto-Depot (€)': nav, 'SPY (EUR, brutto)': br,
                         'Liquidationswert (€)': liquidation_nav,
                         'Aktienquote (%)': np.array(exposure_series) * 100}), \
           pd.DataFrame(journal), info



# Keine Cache-Dekoration: `index` ist ein pandas.DatetimeIndex.
# Streamlit kann diesen in einigen Umgebungen nicht zuverlässig hashen.
# Ein Ausschluss von `index` aus dem Cache-Schlüssel per `_index` wäre
# fachlich falsch: System A und B können unterschiedliche Zeiträume
# simulieren, was zu vermischten Benchmark-Ergebnissen führen könnte.
# Diese reine Buy-and-Hold-Rechnung ist leichtgewichtig; die aufwendigen
# Datenabrufe und Momentum-Backtests bleiben separat gecacht.
def simulate_spy_buy_hold_eur(
    spy_prices, eurusd, index, capital=8000.0, fixed_fee=1.0,
    variable_fee_pct=0.1, saver_allowance=1000.0, church_tax=0,
    include_tax=True, fractions=False, equity_fund_exemption=0.30,
):
    """Passende SPY-Kauf-und-Halte-Benchmark mit einem Kauf am Modellstart.

    Zeigt (1) tatsächlichen Modell-Depotwert nach Kaufgebühr, (2) einen
    täglich hypothetisch liquidierten Wert nach Verkaufsgebühr und
    Steuern auf *positive* realisierte Gewinne im jeweiligen Jahr.
    Keine echten Verkäufe; gleiche simplified Tax-Satz/FIFO-Logik wie Strategie.
    """
    if capital <= 0 or fixed_fee < 0 or variable_fee_pct < 0:
        raise ValueError('SPY: Startkapital/Kosten ungültig.')
    if not 0 <= equity_fund_exemption <= 1:
        raise ValueError('SPY: Teilfreistellung muss zwischen 0 und 100 % liegen.')
    idx = pd.DatetimeIndex(index).tz_localize(None).normalize()
    if len(idx) < 2:
        raise ValueError('SPY-Benchmark benötigt mindestens zwei Handelstage.')
    prices = spy_prices.copy().astype(float)
    prices.index = pd.DatetimeIndex(prices.index).tz_localize(None).normalize()
    prices = prices.reindex(idx)
    fx = eurusd.copy().astype(float)
    fx.index = pd.DatetimeIndex(fx.index).tz_localize(None).normalize()
    fx = fx.loc[~fx.index.duplicated(keep='last')].reindex(idx).ffill(limit=3)
    if (not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any() or
        not np.isfinite(fx.to_numpy()).all() or (fx <= 0).any()):
        raise ValueError('SPY: historische Preis- oder EUR/USD-Daten unvollständig.')
    price_eur = prices / fx
    fee_rate = float(variable_fee_pct) / 100.0
    entry_unit = float(price_eur.iloc[0])
    budget = max(0.0, (float(capital) - float(fixed_fee)) / (1.0 + fee_rate))
    qty = budget / entry_unit
    if not fractions:
        qty = float(int(qty))
    if qty <= 0:
        raise ValueError('SPY: Bei Startkapital und Ganzstückhandel kein ETF-Anteil finanzierbar.')
    cost = qty * entry_unit
    buy_fee = fixed_fee + cost * fee_rate
    basis = cost + buy_fee
    cash = float(capital) - basis
    gross_sale = qty * price_eur
    sell_fee = fixed_fee + gross_sale * fee_rate
    gains_after_fees = gross_sale - sell_fee - basis
    tax_rate = german_stock_tax_rate(church_tax)
    # ETF-Annahme: §20 InvStG – 30% Teilfreistellung bei qualifiziertem
    # Aktienfonds; Vorabpauschale und Ausschüttungen hier nicht modelliert.
    tax = ((gains_after_fees * (1.0 - float(equity_fund_exemption))
            - float(saver_allowance)).clip(lower=0) * tax_rate
           if include_tax else pd.Series(0.0, index=idx))
    nav = (cash + gross_sale).rename('SPY Depotwert (€)')
    liquidation = (nav - sell_fee - tax).rename('SPY Liquidationswert (€)')
    result = pd.DataFrame({'Depotwert (€)': nav, 'Liquidationswert (€)': liquidation,
                           'Hypothetische Steuer (€)': tax}, index=idx)
    years = max(1e-9, (idx[-1] - idx[0]).days / 365.25)
    info = {'start': str(idx[0].date()), 'end': str(idx[-1].date()),
            'end_value': float(nav.iloc[-1]), 'liquidation_value': float(liquidation.iloc[-1]),
            'return_pct': float((nav.iloc[-1] / capital - 1) * 100),
            'return_after_sale_pct': float((liquidation.iloc[-1] / capital - 1) * 100),
            'cagr': float((nav.iloc[-1] / capital) ** (1 / years) * 100 - 100)
                    if nav.iloc[-1] > 0 else float('nan'),
            'cagr_after_sale': float((liquidation.iloc[-1] / capital) ** (1 / years) * 100 - 100)
                    if liquidation.iloc[-1] > 0 else float('nan'),
            'equity_fund_exemption': float(equity_fund_exemption),
            'buy_fee': float(buy_fee), 'hypothetical_sell_fee': float(sell_fee.iloc[-1]),
            'hypothetical_tax': float(tax.iloc[-1]), 'shares': qty, 'cash': cash}
    return result, info




def _mt_following_session(date):
    """NYSE-Sitzung unmittelbar NACH einem Signaltag, als Datum (ohne Uhrzeit)."""
    day = pd.Timestamp(date).normalize()
    schedule = mcal.get_calendar('NYSE').schedule(
        start_date=(day + pd.Timedelta(days=1)).date(),
        end_date=(day + pd.Timedelta(days=12)).date(),
    )
    return pd.Timestamp(schedule.index[0]).normalize() if len(schedule) else None


def _mt_next_rotating_signal(last_date, holding_days):
    """Naechster geplanter Wochenend-Signaltag fuer feste 5/10/20-Rotation."""
    if holding_days not in (5, 10, 20):
        raise ValueError('holding_days muss 5, 10 oder 20 sein')
    last = pd.Timestamp(last_date).normalize()
    schedule = mcal.get_calendar('NYSE').schedule(
        start_date=(last + pd.Timedelta(days=1)).date(),
        end_date=(last + pd.Timedelta(days=60)).date(),
    )
    sessions = pd.DatetimeIndex(schedule.index).normalize()
    if len(sessions) == 0:
        return None
    weeks = sessions.to_period('W-FRI')
    ends = pd.Series(sessions, index=weeks).groupby(level=0).last()
    anchor = pd.Timestamp('2020-01-06')
    for day in ends.values:
        candidate = pd.Timestamp(day).normalize()
        wn = (candidate - anchor).days // 7
        if candidate > last and wn % (holding_days // 5) == 0:
            return candidate
    return None


def _mt_previous_session(date):
    """Letzter NYSE-Handelstag VOR dem naechsten Ausfuehrungstermin."""
    day = pd.Timestamp(date).normalize()
    schedule = mcal.get_calendar('NYSE').schedule(
        start_date=(day - pd.Timedelta(days=12)).date(),
        end_date=(day - pd.Timedelta(days=1)).date(),
    )
    return pd.Timestamp(schedule.index[-1]).normalize() if len(schedule) else None


def momentum_period_overview(kpis, last_close_date, holding_days, past_count=8):
    """Laufender Zeitraum, letzte volle Signalzyklen, noch nicht ausgefuehrtes Signal.

    'von' ist der Schlusskurs des ersten nachfolgenden NYSE-Handelstags.
    'bis' ist der letzte Handelsschluss vor der naechsten planmaessigen Order.
    Bei unveraendertem Ranking kann ein Signalzyklus ohne Order stattfinden.
    """
    history = list(kpis.get('signal_snapshots', []))
    if not history:
        return {'active': None, 'past': [], 'pending': None}
    last_close = pd.Timestamp(last_close_date).normalize()
    periods = []
    for j, snap in enumerate(history):
        start_raw = snap.get('execution_date')
        if not start_raw:
            continue
        start = pd.Timestamp(start_raw).normalize()
        if j + 1 < len(history):
            next_raw = history[j + 1].get('execution_date')
            next_start = pd.Timestamp(next_raw).normalize() if next_raw else None
        else:
            next_signal = _mt_next_rotating_signal(snap['signal_date'], holding_days)
            next_start = _mt_following_session(next_signal) if next_signal is not None else None
        end = _mt_previous_session(next_start) if next_start is not None else None
        if end is not None and end < start:
            continue
        periods.append({
            'signal_date': snap['signal_date'],
            'from': str(start.date()),
            'to': str(end.date()) if end is not None else None,
            'next_execution': str(next_start.date()) if next_start is not None else None,
            'tickers': list(snap.get('tickers', [])),
            'qualified_tickers': list(snap.get('qualified_tickers', [])),
            'exposure_pct': float(snap.get('exposure_pct', 0.0)),
        })
    active = None
    past = []
    pending = None
    for p in periods:
        start = pd.Timestamp(p['from'])
        if start > last_close:
            pending = p
            break
        if p['to'] is None or last_close <= pd.Timestamp(p['to']):
            active = p
        else:
            past.append(p)
    # Falls letztes Signal vor dem Folgetermin noch aussteht, bleibt das
    # bis dahin tatsaechlich gueltige vorige Portfolio als 'aktiv' sichtbar.
    return {'active': active, 'past': list(reversed(past[-max(0, int(past_count)):])),
            'pending': pending}

# ==== ENDE RECHENKERN: momentum_engine ====


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
    Rückgabe-Signatur bleibt kompatibel zu den bestehenden Monte-Carlo-Reitern.
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


# ================================================================
# REALE RENDITE: RECHENKERN (ohne externe Daten / API-Abfragen)
# ================================================================

def rr_effective_tax_rate(capital_tax_pct=25.0, solidarity=True, church_pct=0.0):
    """DE-Standardmodell: KapESt mit Kirchensteueranrechnung (§ 32d EStG)."""
    k = float(church_pct) / 100.0
    base = float(capital_tax_pct) / 100.0
    if not 0 <= base <= 1 or k not in (0.0, 0.08, 0.09):
        raise ValueError("Ungültige Steuersätze")
    # Kirchensteuer mindert die Bemessung der KapESt, nicht nur ein additiver Zuschlag.
    income_tax_rate = base / (1.0 + base * k)
    return income_tax_rate * (1.0 + (0.055 if solidarity else 0.0) + k)


def rr_monthly_money_weighted_rate(start_capital, monthly_payment, months, ending_value):
    """Geldgewichtete Rendite p.a. aus Endwert und Einzahlungen am jeweiligen Monatsende.

    Keine Zeitreihe tagesgenauer Cashflows: alle Monate sind modellhaft gleich lang.
    Für positive Zahlungen ist der Endwert monoton in der Monatsrendite.
    """
    start_capital = float(start_capital)
    monthly_payment = float(monthly_payment)
    ending_value = float(ending_value)
    months = int(months)
    if months <= 0 or (start_capital <= 0 and monthly_payment <= 0):
        return float('nan')

    def future_value(rate):
        val = start_capital
        for _ in range(months):
            val = val * (1.0 + rate) + monthly_payment
            if val > 1e100:
                return float('inf')
        return val

    if ending_value <= 0 and monthly_payment == 0:
        return -1.0
    if ending_value < monthly_payment and monthly_payment > 0:
        # Verlust der am letzten Tag neu eingezahlten Sparrate lässt sich im Modell
        # (Einzahlung erst am Monatsende) nicht als Geldrendite darstellen.
        return float('nan')
    low, high = -0.999999, 1.0
    if future_value(low) > ending_value or future_value(high) < ending_value:
        return float('nan')
    for _ in range(85):
        mid = (low + high) / 2.0
        if future_value(mid) > ending_value:
            high = mid
        else:
            low = mid
    rm = (low + high) / 2.0
    return (1.0 + rm) ** 12 - 1.0


def rr_calculate(start_capital, annual_return_pct, years, inflation_pct=2.0,
                 monthly_payment=0.0, annual_product_cost_pct=0.0,
                 tax_mode='end', available_allowance=1000.0,
                 partial_exemption_pct=0.0, capital_tax_pct=25.0,
                 solidarity=True, church_pct=0.0, months=None):
    """Vergleichs-/Planungsmodell einer Geldanlage, kein Steuerbescheid.

    tax_mode='end'    : alle Gewinne erst bei Liquidation am Jahresende besteuert.
    tax_mode='annual' : 100 % zum Jahresende verkauft & sofort wieder angelegt;
                        jährlicher Pauschbetrag, vereinfachter Verlustvortrag.
    tax_mode='none'   : kein Steuerabzug.
    Monatliche Einzahlung am Monatsende. Jahreskosten proportional eingepreist.
    Kursrückgang führt zu keiner automatischen Steuererstattung.
    """
    capital = float(start_capital)
    r = float(annual_return_pct) / 100.0
    inflation = float(inflation_pct) / 100.0
    payment = float(monthly_payment)
    annual_cost = float(annual_product_cost_pct) / 100.0
    allowance = float(available_allowance)
    exemption = float(partial_exemption_pct) / 100.0
    total_months = int(months if months is not None else 12 * years)
    years = total_months / 12.0
    if not (capital > 0 and payment >= 0 and 1 <= total_months <= 600 and
            r >= -1 and inflation > -1 and 0 <= annual_cost < 1 and
            allowance >= 0 and 0 <= exemption <= 1 and
            tax_mode in ('end', 'annual', 'none')):
        raise ValueError("Ungültige Eingabewerte für die Renditeberechnung")

    tax_rate = rr_effective_tax_rate(capital_tax_pct, solidarity, church_pct)
    gross_monthly = (1.0 + r) ** (1.0 / 12.0) - 1.0
    net_cost_monthly = ((1.0 + r) * (1.0 - annual_cost)) ** (1.0 / 12.0) - 1.0
    gross = capital
    value = capital
    paid = capital
    tax_total = 0.0
    tax_loss_carry = 0.0
    rows = [{
        'Jahr': 0, 'Einzahlungen': capital, 'Brutto': capital,
        'Nach Produktkosten / vor Steuern': capital, 'Nach Steuern': capital,
        'Kaufkraft nach Steuern': capital, 'Im Jahr gezahlte Steuer': 0.0,
        'Steuern kumuliert': 0.0
    }]
    year_start = value
    year_paid = 0.0
    for m in range(1, total_months + 1):
        gross = gross * (1.0 + gross_monthly) + payment
        value = value * (1.0 + net_cost_monthly) + payment
        paid += payment
        year_paid += payment
        is_period_end = (m % 12 == 0) or (m == total_months)
        if not is_period_end:
            continue
        paid_tax = 0.0
        pre_current_tax = value
        if tax_mode == 'annual':
            # Vollständiger Verkauf an jedem Modelljahresende und am letzten
            # (ggf. unvollständigen) Modelljahr; Neubeginn der Einstands-Basis.
            gain = (value - year_start - year_paid) * (1.0 - exemption)
            if gain < 0:
                tax_loss_carry += -gain
            else:
                after_loss = max(0.0, gain - tax_loss_carry)
                tax_loss_carry = max(0.0, tax_loss_carry - gain)
                taxable = max(0.0, after_loss - allowance)
                paid_tax = taxable * tax_rate
                value -= paid_tax
                tax_total += paid_tax
            after_tax = value
        elif tax_mode == 'end':
            # Unverwendete Freibeträge früherer Jahre werden NICHT kumuliert.
            gain = max(0.0, value - paid) * (1.0 - exemption)
            paid_tax = max(0.0, gain - allowance) * tax_rate
            after_tax = value - paid_tax
        else:
            after_tax = value
        elapsed_years = m / 12.0
        rows.append({
            'Jahr': elapsed_years, 'Einzahlungen': paid, 'Brutto': gross,
            'Nach Produktkosten / vor Steuern': pre_current_tax,
            'Nach Steuern': after_tax,
            'Kaufkraft nach Steuern': after_tax / ((1.0 + inflation) ** elapsed_years),
            'Im Jahr gezahlte Steuer': paid_tax,
            'Steuern kumuliert': tax_total if tax_mode == 'annual' else paid_tax
        })
        year_start = value
        year_paid = 0.0

    frame = pd.DataFrame(rows)
    last = frame.iloc[-1]
    net_end = float(last['Nach Steuern'])
    gross_end = float(last['Brutto'])
    pretax_end = float(last['Nach Produktkosten / vor Steuern'])
    paid_total = float(last['Einzahlungen'])
    total_tax = float(last['Steuern kumuliert'])
    infl_factor = (1.0 + inflation) ** years
    # Gegenwert sämtlicher Einzahlungen in der Kaufkraft des Startzeitpunkts.
    # Beginn t=0 plus monatliche Einzahlungen am jeweiligen Monatsende.
    inflation_adjusted_payments = capital + sum(
        payment / (1.0 + inflation) ** (m / 12.0)
        for m in range(1, total_months + 1)
    )
    net_ann = rr_monthly_money_weighted_rate(capital, payment, total_months, net_end)
    real_ann = ((1.0 + net_ann) / (1.0 + inflation) - 1.0) if np.isfinite(net_ann) else float('nan')
    summary = {
        'end_gross': gross_end,
        'end_before_tax': pretax_end,
        'end_after_tax': net_end,
        'end_real': net_end / infl_factor,
        'paid_total': paid_total,
        'gain_nominal_net': net_end - paid_total,
        'gain_real': net_end / infl_factor - inflation_adjusted_payments,
        'real_value_paid': inflation_adjusted_payments,
        'total_taxes': total_tax,
        'cost_impact': gross_end - pretax_end if tax_mode != 'annual' else float('nan'),
        'net_annual_return': net_ann,
        'real_annual_return': real_ann,
        'effective_tax_rate': tax_rate if tax_mode != 'none' else 0.0,
        'tax_mode': tax_mode,
        'years': years,
    }
    return frame, summary


# ---------------------------------------------
# APP LAYOUT: BESTEHENDE ACHT TABS + REALE RENDITE
# ---------------------------------------------
st.title("📊 Integriertes Finanz-Terminal")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9=st.tabs([
    "🌐 Makro & Markt", "🧭 Thematic Trends", "📈 Backtesting", "🎲 MC Asset",
    "🥧 MC Portfolio", "💶 Steuern", "🚀 S&P 500 Dual-Tracker",
    "📉 Chart & SMA", "🧮 Reale Rendite",
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


# ================================================================
# SIGNAL-COCKPIT: REINE MODELLSIGNALE, KEINE BROKER-DEPOTBUCHHALTUNG
# Die Signalhistorie wird aus zeitlich korrekten Backtest-Snapshots gebildet.
# Eine explizit exportierbare Signal-Sicherung speichert Originalanzeigen,
# da Streamlit Cloud ohne externe Datenbank KEINE dauerhafte Speicherung bietet.
# ================================================================
import json
import re
from hashlib import sha256

_MT_SIGNAL_ARCHIVE_SCHEMA = 1


def mt_signal_record(system, kpis, settings):
    """Erzeugt einen eindeutigen Snapshot für System/Parameter/Signaltermin."""
    snaps = kpis.get('signal_snapshots') or []
    if not snaps:
        return None
    last = snaps[-1]
    source = {
        'system': str(system),
        'signal_date': str(last['signal_date']),
        'settings': settings,
    }
    key = sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]
    return {
        'id': key,
        'system': str(system),
        'signal_date': str(last['signal_date']),
        # Zeitpunkt der ERSTEN Speicherung, nicht des historischen Signals.
        'first_seen_at': datetime.now(ZoneInfo('Europe/Berlin')).isoformat(timespec='seconds'),
        'execution_date': last.get('execution_date'),
        'tickers': [str(x) for x in last.get('tickers', [])],
        'exposure_pct': round(float(last.get('exposure_pct', 0.0)), 6),
        'scores': {str(t): float(v) for t,v in (last.get('scores') or {}).items()},
        'settings': settings,
    }


def mt_archive_merge(archive, record):
    """First write wins: gleiche ID niemals stillschweigend überschreiben."""
    merged = dict(archive)
    if record is not None and record['id'] not in merged:
        merged[record['id']] = dict(record)
    return merged


def mt_validate_archive(data):
    if not isinstance(data, dict) or data.get('schema_version') != _MT_SIGNAL_ARCHIVE_SCHEMA:
        raise ValueError('Unbekanntes Signalarchiv-Format.')
    items = data.get('signals')
    if not isinstance(items, list) or len(items) > 1000:
        raise ValueError('Ungültige Anzahl von Signalen im Archiv.')
    valid = {}
    for item in items:
        if not isinstance(item, dict):
            raise ValueError('Ungültiger Signal-Datensatz.')
        for required in ('id', 'system', 'signal_date', 'tickers', 'exposure_pct', 'settings'):
            if required not in item:
                raise ValueError(f'Fehlendes Archivfeld: {required}')
        if not (isinstance(item['id'], str) and re.fullmatch(r'[0-9a-f]{24}',item['id'])):
            raise ValueError('Ungültige Signal-ID.')
        if item['system'] not in ('A','B') or not isinstance(item['tickers'], list):
            raise ValueError('Ungültiges System oder Aktienliste.')
        if len(item['tickers']) > 20 or not all(isinstance(t,str) and re.fullmatch(r'[A-Z0-9.\-]{1,16}',t) for t in item['tickers']):
            raise ValueError('Ungültige Ticker in der Signaldatei.')
        exp = float(item['exposure_pct'])
        if not np.isfinite(exp) or not 0 <= exp <= 100:
            raise ValueError('Ungültige Aktienquote in der Signaldatei.')
        if not isinstance(item['settings'], dict):
            raise ValueError('Ungültige Parametereinstellungen.')
        pd.Timestamp(item['signal_date'])
        if item.get('execution_date'):
            pd.Timestamp(item['execution_date'])
        valid[item['id']] = dict(item)
    return valid


def mt_signal_actions(previous, current):
    """Unterschied zwischen alten und neuen SOLL-Gewichten, kein Broker-Abgleich."""
    old_names = previous.get('tickers', []) if previous else []
    new_names = current.get('tickers', []) if current else []
    old_total = float(previous.get('exposure_pct', 0)) if previous else 0.0
    new_total = float(current.get('exposure_pct', 0)) if current else 0.0
    old_each = old_total / len(old_names) if old_names else 0.0
    new_each = new_total / len(new_names) if new_names else 0.0
    old = {str(t): old_each for t in old_names}
    new = {str(t): new_each for t in new_names}
    rows = []
    for symbol in sorted(set(old) | set(new)):
        a, b = old.get(symbol, 0.0), new.get(symbol, 0.0)
        if a > 0 and b == 0:
            action = 'Verkaufen'
        elif b > 0 and a == 0:
            action = 'Kaufen'
        elif abs(a-b) >= 0.1:
            action = 'Gewicht anpassen'
        else:
            action = 'Halten'
        rows.append({'Ticker': symbol, 'Aktion': action,
                     'Vorher (%)': round(a, 2), 'Neu (%)': round(b, 2),
                     'Differenz (Prozentpunkte)': round(b-a, 2)})
    priority = {'Verkaufen':0, 'Kaufen':1, 'Gewicht anpassen':2, 'Halten':3}
    return pd.DataFrame(sorted(rows,key=lambda row:(priority[row['Aktion']],row['Ticker'])),
                        columns=['Ticker','Aktion','Vorher (%)','Neu (%)','Differenz (Prozentpunkte)'])


def mt_signal_outcomes(close_full, snapshots, max_rows=100, risk_guard=None):
    """Konservative Diagnostik ab Ausführungskurs bis zur NÄCHSTEN Ausführung.

    Gemittelte Asset-Returns * Signalaktienquote; nicht als exakte Depotperformance
    zu interpretieren (u.a. intra-periodisches Gewichtedriften und Handelskosten).
    Noch offene Perioden sind explizit nicht als abgeschlossener Test enthalten.
    """
    columns = ['Signal','Ausführung','Ende','Status','Aktien','Aktienquote (%)',
               'Signalzyklus-Rendite (%)','Schlechtester Titel','Schlechtester Titel (%)',
               'Risikowert (0–100)']
    if close_full is None or close_full.empty or not snapshots:
        return pd.DataFrame(columns=columns)
    prices = close_full.sort_index()
    idx = pd.DatetimeIndex(prices.index).normalize()
    items = list(snapshots)[-max_rows:]
    rows=[]
    for i,snap in enumerate(items):
        start_raw = snap.get('execution_date')
        if not start_raw:
            continue
        begin = pd.Timestamp(start_raw).normalize()
        if begin not in idx:
            if begin > idx[-1]:
                rows.append({'Signal':snap['signal_date'],'Ausführung':str(begin.date()),
                             'Ende':'–','Status':'Ausführung ausstehend',
                             'Aktien':', '.join(snap.get('tickers') or []) or 'Cash',
                             'Aktienquote (%)':round(float(snap.get('exposure_pct',0)),1),
                             'Signalzyklus-Rendite (%)':np.nan,
                             'Schlechtester Titel':'–','Schlechtester Titel (%)':np.nan,
                             'Risikowert (0–100)':np.nan})
            continue
        if i+1 < len(items):
            next_raw = items[i+1].get('execution_date')
            end = pd.Timestamp(next_raw).normalize() if next_raw else idx[-1]
        else:
            end = idx[-1]
        if end > idx[-1]:
            end = idx[-1]
            complete = False
        else:
            complete = i+1 < len(items) and bool(items[i+1].get('execution_date'))
        if end not in idx or end < begin:
            continue
        tickers = list(snap.get('tickers') or [])
        exposure = float(snap.get('exposure_pct',0))/100.0
        indiv={}
        for ticker in tickers:
            if ticker not in prices.columns:
                continue
            v0, v1 = prices.at[begin,ticker], prices.at[end,ticker]
            if pd.notna(v0) and pd.notna(v1) and v0 > 0 and v1 > 0:
                indiv[ticker] = float(v1/v0-1)
        valid_cycle = len(indiv)==len(tickers) and end>begin
        cycle_return = exposure * sum(indiv.values()) / len(tickers) if tickers and valid_cycle else (0.0 if not tickers and end>begin else np.nan)
        worst = min(indiv,key=indiv.get) if indiv else '–'
        risk = np.nan
        if risk_guard is not None and not risk_guard.empty:
            day = pd.Timestamp(snap['signal_date']).normalize()
            if day in risk_guard.index:
                risk = risk_guard.at[day,'Risikoscore_letztes_Signal']
        rows.append({'Signal':snap['signal_date'], 'Ausführung':str(begin.date()),
                     'Ende':str(end.date()), 'Status':'Abgeschlossen' if complete else 'Laufend',
                     'Aktien':', '.join(tickers) if tickers else 'Cash',
                     'Aktienquote (%)':round(exposure*100,1),
                     'Signalzyklus-Rendite (%)':round(cycle_return*100,2) if np.isfinite(cycle_return) else np.nan,
                     'Schlechtester Titel':worst,
                     'Schlechtester Titel (%)':round(100*indiv[worst],2) if worst!='–' else np.nan,
                     'Risikowert (0–100)':round(float(risk),1) if pd.notna(risk) else np.nan})
    return pd.DataFrame(rows,columns=columns).iloc[::-1].reset_index(drop=True)

# ==========================================
# TAB 7 – S&P 500 DUAL-TRACKER
# Ersetzt den bisherigen kompletten `with tab7:`-Block.
# Beide Systeme A (5 Tage) und B (10 Tage) bleiben erhalten.
# ==========================================
if tab7.open:
    with tab7:
        # =============================
        # EINMALIGE, AUSSCHLIESSLICH ANKLICKBARE HILFE
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
                'Handelsjournal': 'Interne rechnerische Buchungen der Kosten-/Steuersimulation; es wird kein eigenes Broker-Handelstagebuch geführt.',
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
                'Zielportfolio': 'Aktien, die das Modell nach dem letzten abgeschlossenen Signal halten möchte. Kein Brokerabgleich.',
                'Signaldifferenz': 'Vergleich des neuen Zielportfolios mit dem vorherigen Modellsignal. Voraussetzung: Alle vergangenen Empfehlungen wurden tatsächlich umgesetzt. Stückzahlen und Broker-Ausführungskurse werden nicht geprüft.',
                'Signaldiagnostik': 'Historische Zielaktien und ihre nachfolgenden Renditen. Die Signalzyklen sind keine Vorhersagen, sondern eine rückblickende Qualitätssicherung.',
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

        st.markdown('### 🚀 S&P 500 Momentum – Trading-Cockpit')
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
            st.caption('🎯 **Trading-Cockpit:** Ein verbindliches System, reine Soll-Signale '
                       'und eine historische Qualitätskontrolle. Kein zusätzliches Broker-Handelsjournal.')
            active_system = st.radio('Verbindliches Handelssystem', ['A','B'], horizontal=True,
                                     format_func=lambda x: 'A – wöchentlich' if x=='A' else 'B – zweiwöchentlich',
                                     key='mt_active_system')
            st.caption('Die Auswahl gilt für diese Sitzung. Für dauerhafte Signalarchivierung '
                       'kannst du unten einen Snapshot exportieren und wieder importieren.')

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

                        # ===========================================================
                        # VERBINDLICHER SIGNAL-ASSISTENT, OHNE ECHTES BROKER-DEPOT
                        # ===========================================================
                        st.divider()
                        st.subheader('🎯 Trading-Cockpit: verbindliches Signal')
                        active_stats = kpis_5 if active_system == 'A' else kpis_10
                        active_snapshots = active_stats.get('signal_snapshots', [])
                        settings_id = {'n_positions':positions,'hysterese_limit':hyst,
                                       'market_mode':regime,'backtest_years':horizon,
                                       'sector_limit':3 if sector_on and enough_sectors else None}
                        current_record = mt_signal_record(active_system, active_stats, settings_id)
                        if current_record is None:
                            st.warning('Für das ausgewählte System liegt noch kein Signal vor.')
                        else:
                            if 'mt_signal_archive' not in st.session_state:
                                st.session_state['mt_signal_archive'] = {}
                            archive = st.session_state['mt_signal_archive']
                            if current_record['id'] not in archive:
                                archive[current_record['id']] = current_record
                            frozen = archive[current_record['id']]
                            if (frozen['tickers'] != current_record['tickers'] or
                                    abs(float(frozen['exposure_pct'])-float(current_record['exposure_pct']))>0.01):
                                st.error('⚠️ Die neu berechnete Aktienliste weicht vom gespeicherten '
                                         'Originalsignal bei gleicher Konfiguration ab. Das ursprüngliche '
                                         'Signal bleibt unverändert fixiert; bitte Datenqualität prüfen.')
                            prior_record = mt_signal_record(active_system,
                                {'signal_snapshots':active_snapshots[:-1]}, settings_id)
                            if prior_record is not None and prior_record['id'] not in archive:
                                archive[prior_record['id']] = prior_record
                            preceding = archive.get(prior_record['id']) if prior_record else None
                            plans = mt_signal_actions(preceding, frozen)
                            signal_dt = pd.Timestamp(frozen['signal_date'])
                            execute_dt = pd.Timestamp(frozen['execution_date']) if frozen.get('execution_date') else None
                            last_data_dt = pd.Timestamp(close_all.index[-1]).normalize()
                            is_pending = execute_dt is not None and execute_dt > last_data_dt
                            sys_name = 'A – 5 Tage' if active_system == 'A' else 'B – 10 Tage'
                            st.markdown(f'**Verbindliche Modellstrategie: {sys_name}**')
                            st.caption('Signal-Snapshot erstmals gesichert: '
                                       f"{frozen.get('first_seen_at', 'nicht dokumentiert')}" +
                                       ' (ein nachträglich berechnetes Signal ist kein Beleg für eine '
                                       'damals tatsächlich angezeigte Empfehlung).')
                            _a,_b,_c=st.columns(3)
                            _a.metric('Signal vom',signal_dt.strftime('%d.%m.%Y'))
                            _b.metric('Geplante Ausführung',execute_dt.strftime('%d.%m.%Y') if execute_dt is not None else '–')
                            _c.metric('Ziel-Aktienquote',f"{float(frozen['exposure_pct']):.1f} %")
                            if is_pending:
                                st.warning('**Neues Signal – Ausführung ausstehend.** '
                                           'Modellannahme: Kurs der nächsten regulären NYSE-Sitzung zum Schluss. '
                                           'Die tatsächliche Order musst du beim Broker platzieren.')
                            else:
                                st.info('**Letzter bereits simulierter Signalzyklus.** '
                                        'Kein zusätzliches, noch ausstehendes Signal. '
                                        'Nicht nachträglich zum historischen Ausführungspreis handeln.')
                            if plans.empty:
                                st.success('Keine Aktien im Zielportfolio (Cash).')
                            else:
                                changes = plans[plans['Aktion']!='Halten']
                                if changes.empty:
                                    st.success('Keine Änderung der Zielaktien zwischen den zwei letzten Signalen.')
                                else:
                                    st.markdown('**Änderungen zum vorherigen SOLL-Signal**')
                                    st.dataframe(changes,hide_index=True,use_container_width=True)
                                with st.expander('Vollständiges Zielportfolio mit beibehaltenen Positionen'):
                                    st.dataframe(plans,hide_index=True,use_container_width=True)
                            st.caption('**Voraussetzung:** Alle vorherigen Empfehlungen wurden tatsächlich '
                                       'umgesetzt. Die Anzeige kennt deinen Brokerbestand nicht. '
                                       'Positionsgrößen sind Soll-Gewichte; exakte Orders, Spreads und '
                                       'abweichende Broker-Kurse musst du vor Ausführung prüfen. '
                                       'Der separate Risk Guard ist nur Analyse; verbindlich ist '
                                       'ausschließlich der gewählte SPY-SMA200-Filter.')
                            with st.expander('🔒 Signal-Original sichern oder wiederherstellen'):
                                st.caption('Neue Signale werden während dieser Sitzung automatisch '
                                           'als Original fixiert. Streamlit Cloud speichert diese '
                                           'Information ohne externe Datenbank NICHT dauerhaft. '
                                           'Nutze den JSON-Export und importiere ihn bei einer späteren Sitzung.')
                                archive_package={'schema_version':_MT_SIGNAL_ARCHIVE_SCHEMA,
                                                 'signals':list(archive.values())}
                                st.download_button('📥 Originalsignale herunterladen (JSON)',
                                                   data=json.dumps(archive_package,ensure_ascii=False,indent=2).encode('utf-8'),
                                                   file_name='momentum_signale.json',mime='application/json',
                                                   key='mt_signal_export')
                                imported=st.file_uploader('Gespeichertes Signalarchiv laden',type=['json'],
                                                          key='mt_signal_import_file')
                                if st.button('Gespeicherte Signale übernehmen',
                                             disabled=imported is None,key='mt_signal_import_button'):
                                    try:
                                        loaded=json.loads(imported.getvalue().decode('utf-8-sig'))
                                        st.session_state['mt_signal_archive'].update(mt_validate_archive(loaded))
                                        st.rerun()
                                    except (ValueError,TypeError,UnicodeDecodeError) as exc:
                                        st.error(f'Ungültige Archivdatei: {exc}')
                            st.caption('Originalsignale werden nicht stillschweigend durch neu geladene '
                                       'historische Kurse überschrieben. Bei Parameteränderungen wird '
                                       'eine neue Signal-ID verwendet.')

                        # EURO-NETTO-BACKTEST AN ERSTER STELLE, SIGNALLOGIK UNVERÄNDERT
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
                                st.caption('Die detaillierte, interne FIFO-Transaktionsliste dient '
                                           'nur der Berechnung von Kosten und Steuern. '
                                           'Eine separate Depot- oder Handelsbuchführung entfällt.')
                            except Exception as exc:
                                LOG.exception('Euro-Depot-Simulation fehlgeschlagen')
                                st.error(f'Euro-Depot-Simulation nicht verfügbar: {exc}')
                        else:
                            st.caption('Die Euro-Nettosimulation ist links deaktiviert. '
                                       'Die ursprünglichen Brutto-Backtests bleiben sichtbar.')

                        # Unabhängiger Risk Guard: nur Analyse, nicht Teil des Handelssignals.
                        _guard = None
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

                        # KLARE DATIERUNG DER HANDELSPERIODEN UND HISTORISCHE AKTIENLISTEN
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

                        # 4. TRANSPARENTE MODELL-HOLDINGS UND SIGNALQUALITÄT
                        st.divider()
                        st.subheader('🏆 Zielaktien beider Strategien')
                        for _label, _hold, _days, _stat in [
                            ('System A – 5 Tage',holdings_5,5,kpis_5),
                            ('System B – 10 Tage',holdings_10,10,kpis_10)]:
                            with st.expander(_label,expanded=(_days==(5 if active_system=='A' else 10))):
                                st.caption(f"Letztes Signal: {_stat.get('last_signal_date','–')} · "
                                           f"Folgende Modell-Ausführung: {_stat.get('next_execution','–')} · "
                                           f"Aktienquote: {fmt(100*_stat.get('target_exposure',0))}")
                                if _hold.empty:
                                    st.info('Keine qualifizierten Positionen (Cash).')
                                else:
                                    _view=_hold.copy()
                                    _view['Unternehmen']=_view['Unternehmen'].astype(str)
                                    _view['Zielgewicht (%)']=_view['Zielgewicht (%)'].round(2)
                                    _view['Score']=_view['Score'].round(3)
                                    st.dataframe(_view[['Ticker','Unternehmen','Score','Zielgewicht (%)']],
                                                 hide_index=True,use_container_width=True)
                        st.caption('Nur das links gewählte System ist verbindlich. '
                                   'Die zweite Liste bleibt für den Strategie-Vergleich sichtbar.')

                        st.divider()
                        st.subheader('🔎 Signalqualität – welche Empfehlungen liefen schlecht?')
                        st.caption('Rein historische Diagnose auf Basis der **zum Signaltag verfügbaren** '
                                   'Informationen. Renditen werden erst ab dem modellierten '
                                   'Ausführungsschlusskurs gemessen. Offene Perioden zählen nicht '
                                   'als abgeschlossene Verlustsignale.')
                        _hist_count=st.selectbox('Anzahl historischer Signale',[10,20,50,100],index=1,
                                                  key='mt_signal_history_count')
                        _active_snaps=(kpis_5 if active_system=='A' else kpis_10).get('signal_snapshots',[])
                        _guard_history=_guard
                        _hist=mt_signal_outcomes(close_all,_active_snaps,max_rows=_hist_count+1,
                                                 risk_guard=_guard_history)
                        if _hist.empty:
                            st.info('Noch nicht genügend berechenbare Signalperioden vorhanden.')
                        else:
                            _closed=_hist[_hist['Status']=='Abgeschlossen'].dropna(
                                subset=['Signalzyklus-Rendite (%)'])
                            _k1,_k2,_k3=st.columns(3)
                            _k1.metric('Abgeschlossene Signalzyklen',len(_closed))
                            _k2.metric('Verlustzyklen',str(int((_closed['Signalzyklus-Rendite (%)']<0).sum())))
                            _k3.metric('Verlustzyklen (%)',
                                       f"{100*(_closed['Signalzyklus-Rendite (%)']<0).mean():.1f} %"
                                       if not _closed.empty else '–')
                            st.dataframe(_hist,hide_index=True,use_container_width=True)
                            if not _closed.empty:
                                with st.expander('Die fünf schwächsten abgeschlossenen Signalzyklen'):
                                    st.dataframe(_closed.nsmallest(5,'Signalzyklus-Rendite (%)'),
                                                 hide_index=True,use_container_width=True)
                            st.download_button('Signaldiagnostik als CSV',
                                               data=_hist.to_csv(index=False).encode('utf-8-sig'),
                                               file_name='momentum_signaldiagnose.csv',mime='text/csv',
                                               key='mt_signal_diagnostics_export')
                        st.caption('Die Periodenrendite ist eine **Näherung** aus den bereinigten '
                                   'Schlusskursen und dem Ziel-Aktienanteil. Sie ersetzt **nicht** '
                                   'die exakte Brutto-/Netto-Depotsimulation. Risikowert: '
                                   '0 = geringe Warnsignale, 100 = hohe Warnsignale.')

                        # ZUSÄTZLICHE RISIKOGRAFIKEN / KALENDERJAHRE
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


# ================================================================
# TAB 9: REALE RENDITE (NACH STEUERN UND INFLATION)
# ================================================================
if tab9.open:
    with tab9:
        st.markdown('### 🧮 Reale Rendite – was bleibt nach Steuern und Inflation?')
        st.caption(
            'Ein interaktiver Rechner für historische Renditen oder Zukunftsszenarien. '
            '„Real“ bedeutet hier: um den Kaufkraftverlust durch Inflation bereinigt. '
            'Alle Euro-Beträge und Sätze kannst du selbst anpassen.'
        )
        rr_left, rr_right = st.columns([1, 1.85], gap='large')
        with rr_left:
            st.markdown('#### 1. Investment und Rendite')
            rr_capital = st.number_input(
                'Angelegtes Startkapital (€)', min_value=1.0, value=10000.0,
                step=500.0, format='%.2f', key='rr_capital',
                help='Bereits investierter Anfangsbetrag. Nur Gewinne, nicht das eingesetzte Kapital, sind in diesem Modell steuerpflichtig.'
            )
            rr_unit = st.radio('Zeitraum eingeben in', ['Jahren', 'Monaten'],
                               horizontal=True, key='rr_time_unit')
            if rr_unit == 'Jahren':
                rr_months = 12 * st.slider(
                    'Betrachtungszeitraum (Jahre)', min_value=1, max_value=50,
                    value=10, key='rr_years',
                    help='Wie viele Jahre die Anlage betrachtet beziehungsweise modelliert wird.'
                )
            else:
                rr_months = st.number_input(
                    'Betrachtungszeitraum (Monate)', min_value=1, max_value=600,
                    value=18, step=1, key='rr_months'
                )
            rr_years = rr_months / 12.0
            rr_input_type = st.radio(
                'Wie möchtest du die Rendite eingeben?',
                ['Jahresrendite (%)', 'Gesamtrendite (%)', 'Tatsächlicher Endwert (€)'],
                key='rr_input_type',
                help='Eine Jahresrendite lässt sich prognostizieren. Eine Gesamtrendite oder ein Endwert eignen sich für ein bereits abgeschlossenes Investment.'
            )
            if rr_input_type == 'Jahresrendite (%)':
                rr_input_return = st.number_input(
                    'Angenommene / erzielte Rendite pro Jahr (%)',
                    min_value=-99.0, max_value=250.0, value=8.0,
                    step=0.5, key='rr_annual_input',
                    help='Gleichbleibende geometrische Jahresrendite als Szenario, vor optionalen jährlichen Produktkosten.'
                )
                rr_annual_return = float(rr_input_return)
            elif rr_input_type == 'Gesamtrendite (%)':
                rr_input_total = st.number_input(
                    'Erzielte / angenommene Gesamtrendite (%)',
                    min_value=-99.9, max_value=100000.0, value=100.0,
                    step=5.0, key='rr_total_input',
                    help='Gesamte Wertänderung über alle Jahre, ohne zwischenzeitliche Ein- und Auszahlungen.'
                )
                rr_annual_return = 100.0 * (
                    (1.0 + float(rr_input_total) / 100.0) ** (1.0 / rr_years) - 1.0
                )
            else:
                rr_end_actual = st.number_input(
                    'Endwert des Investments (€)', min_value=0.0,
                    value=18000.0, step=500.0, format='%.2f',
                    key='rr_final_value_input',
                    help='Wert vor modelliertem Steuerabzug. Nur geeignet, wenn es keine unterjährigen Ein- oder Auszahlungen gab.'
                )
                rr_annual_return = 100.0 * (
                    (float(rr_end_actual) / float(rr_capital)) ** (1.0 / rr_years) - 1.0
                )
            rr_allow_savings = rr_input_type == 'Jahresrendite (%)'
            rr_monthly = st.number_input(
                'Zusätzliche monatliche Einzahlung (€)', min_value=0.0,
                value=0.0, step=50.0, format='%.2f', key='rr_monthly',
                disabled=not rr_allow_savings,
                help='Monatliche Einzahlung jeweils am Monatsende. Bei Gesamtrendite/Endwert sind weitere Einzahlungen ohne genaue Zahlungshistorie nicht eindeutig rekonstruierbar.'
            )
            if not rr_allow_savings:
                st.caption('Bei Gesamtrendite/Endwert sind Sparraten deaktiviert. Die unterjährige Kurve wird gleichmäßig modelliert, nicht historisch rekonstruiert.')
            rr_product_fee = st.number_input(
                'Zusätzliche jährliche Produktkosten (%)',
                min_value=0.0, max_value=15.0, value=0.0,
                step=0.1, key='rr_fee', disabled=not rr_allow_savings,
                help='Nur Kosten ergänzen, die noch nicht in der eingegebenen Jahresrendite stecken (sonst Doppelzählung).'
            )

            st.markdown('#### 2. Steuern und Inflation')
            rr_inflation = st.number_input(
                'Durchschnittliche Inflation pro Jahr (%)',
                min_value=-5.0, max_value=30.0, value=2.0,
                step=0.25, key='rr_inflation',
                help='Konstante Annahme: mit steigender Inflation sinkt die Kaufkraft deines Endvermögens.'
            )
            rr_tax_label = st.selectbox(
                'Wann fallen Steuern an?',
                [
                    'Beim Verkauf am Ende (Standard)',
                    'Jedes Jahr verkaufen und wieder anlegen',
                    'Keine Steuern (Vergleich)',
                ], key='rr_tax_mode',
                help='Beim Verkauf nur ein Pauschbetrag im Verkaufsjahr. Bei jährlicher Realisierung fällt der Freibetrag modellhaft jedes Jahr neu an.'
            )
            rr_tax_mode = {
                'Beim Verkauf am Ende (Standard)': 'end',
                'Jedes Jahr verkaufen und wieder anlegen': 'annual',
                'Keine Steuern (Vergleich)': 'none',
            }[rr_tax_label]
            rr_asset_label = st.selectbox(
                'Art der Kapitalanlage',
                [
                    'Einzelaktien / Zinsen / Sonstige (0 %)',
                    'Aktienfonds / Aktien-ETF (30 % Teilfreistellung)',
                    'Mischfonds (15 % Teilfreistellung)',
                ], key='rr_asset_class',
                help='Die Fonds-Teilfreistellung betrifft qualifizierende Fonds, nicht gewöhnliche Einzelaktien. Bitte die Klassifizierung deines Fonds prüfen.'
            )
            rr_exemption = {
                'Einzelaktien / Zinsen / Sonstige (0 %)': 0.0,
                'Aktienfonds / Aktien-ETF (30 % Teilfreistellung)': 30.0,
                'Mischfonds (15 % Teilfreistellung)': 15.0,
            }[rr_asset_label]
            rr_allowance = st.number_input(
                'Noch verfügbarer Sparer-Pauschbetrag je Steuerjahr (€)',
                min_value=0.0, max_value=2000.0, value=1000.0,
                step=100.0, format='%.2f', key='rr_allowance',
                disabled=rr_tax_mode == 'none',
                help='In Deutschland regulär 1.000 € (einzeln), 2.000 € (zusammenveranlagt). Gib nur den Anteil ein, der für dieses Investment noch verfügbar ist. Nicht genutzte Beträge sind nicht auf spätere Jahre übertragbar.'
            )
            with st.expander('⚙️ Steuerdetails (optional)'):
                rr_capital_tax = st.number_input(
                    'Kapitalertragsteuersatz (%)', min_value=0.0,
                    max_value=60.0, value=25.0, step=0.5,
                    key='rr_capital_tax',
                    help='Deutscher Standard: 25 %. Sonderfälle und individuelle günstigere Besteuerung werden nicht automatisch ermittelt.'
                )
                rr_soli = st.checkbox('Solidaritätszuschlag berücksichtigen (5,5 % auf Steuer)',
                                      value=True, key='rr_soli')
                rr_church = st.selectbox('Kirchensteuersatz', [0, 8, 9],
                                         format_func=lambda n: 'Keine' if n == 0 else f'{n} %',
                                         key='rr_church')
                st.caption('Bei Kirchensteuerpflicht wird die Kapitalertragsteuer im deutschen Modell entsprechend gemindert.')

        with rr_right:
            try:
                rr_path, rr_result = rr_calculate(
                    start_capital=rr_capital,
                    annual_return_pct=rr_annual_return,
                    years=rr_years,
                    months=rr_months,
                    inflation_pct=rr_inflation,
                    monthly_payment=rr_monthly if rr_allow_savings else 0.0,
                    annual_product_cost_pct=rr_product_fee if rr_allow_savings else 0.0,
                    tax_mode=rr_tax_mode,
                    available_allowance=rr_allowance if rr_tax_mode != 'none' else 0.0,
                    partial_exemption_pct=rr_exemption,
                    capital_tax_pct=rr_capital_tax,
                    solidarity=rr_soli,
                    church_pct=rr_church,
                )
                st.markdown('#### Ergebnis: nominal und kaufkraftbereinigt')
                a, b, c = st.columns(3)
                a.metric('Endwert nach Steuern', format_de(rr_result['end_after_tax']),
                         help='Modellierter Endbetrag nach gegebenenfalls fälligen Kapitalertragsteuern, jedoch in Euro des Endjahrs.')
                b.metric('Kaufkraft in heutigen €', format_de(rr_result['end_real']),
                         help='Endwert nach Steuern geteilt durch den kumulierten Inflationsfaktor; Wert in heutiger Kaufkraft.')
                c.metric('Modellierte Steuern gesamt', format_de(rr_result['total_taxes']),
                         help='Steuerabzüge beim Schlussverkauf oder kumuliert bei jährlicher Gewinnrealisation.')

                a2, b2, c2 = st.columns(3)
                nominal_ann = rr_result['net_annual_return']
                real_ann = rr_result['real_annual_return']
                a2.metric('Nettorendite p.a.',
                          f"{nominal_ann*100:+.2f} %" if np.isfinite(nominal_ann) else '–',
                          help='Geldgewichtete, annualisierte Rendite nach modellierter Steuer. Bei Sparraten werden Zeitpunkt und Höhe der Einzahlungen berücksichtigt.')
                b2.metric('Reale Nettorendite p.a.',
                          f"{real_ann*100:+.2f} %" if np.isfinite(real_ann) else '–',
                          help='Geldgewichtete Jahresrendite nach Steuern, zusätzlich bereinigt um die angenommene Inflation.')
                c2.metric('Realer Vermögenszuwachs',
                          format_de(rr_result['gain_real']),
                          help='Kaufkraftbereinigter Endwert minus alle Einzahlungen, jeweils in Kaufkraft des Startzeitpunkts.')

                if rr_result['gain_real'] < -0.005:
                    st.warning('**Kaufkraftverlust:** Unter diesen Annahmen ist der reale Wertzuwachs negativ, auch wenn das Investment nominal gewachsen sein kann.')
                elif rr_result['gain_real'] > 0.005:
                    st.success('**Positiver realer Wertzuwachs:** Nach den gewählten Steuer- und Inflationsannahmen steigt die Kaufkraft gegenüber den Einzahlungen.')
                else:
                    st.info('Der reale Wertzuwachs liegt unter den aktuellen Annahmen ungefähr bei null.')

                st.markdown('#### Vermögensentwicklung')
                fig_rr = go.Figure()
                line_specs = [
                    ('Brutto', 'Ohne Produktkosten und Steuern', '#7E8A97', 'dash'),
                    ('Nach Steuern', 'Depotwert nach Steuern', '#168768', 'solid'),
                    ('Kaufkraft nach Steuern', 'Nach Steuern, in heutiger Kaufkraft', '#D38A2D', 'solid'),
                    ('Einzahlungen', 'Eingezahltes Kapital, nominal', '#5890C1', 'dot'),
                ]
                for col_name, label, color, dash in line_specs:
                    fig_rr.add_trace(go.Scatter(
                        x=rr_path['Jahr'], y=rr_path[col_name],
                        name=label, mode='lines+markers',
                        line=dict(color=color, width=2.5 if col_name != 'Einzahlungen' else 1.7, dash=dash),
                        marker=dict(size=4),
                        hovertemplate='Jahr %{x}: %{y:,.0f} €<extra>' + label + '</extra>',
                    ))
                fig_rr.update_layout(
                    height=420, xaxis_title='Anlagejahr', yaxis_title='Wert (€)',
                    hovermode='x unified', margin=dict(l=10, r=10, t=25, b=30),
                    legend=dict(orientation='h', yanchor='top', y=-0.22, x=0),
                )
                st.plotly_chart(fig_rr, use_container_width=True,
                                config={'displaylogo': False})

                st.markdown('#### Zahlen im Überblick')
                rr_table = pd.DataFrame([
                    {'Größe': 'Eingesetztes Kapital inkl. Sparraten', 'Wert (€)': rr_result['paid_total']},
                    {'Größe': 'Brutto-Endwert vor zusätzlichen Produktkosten und Steuern', 'Wert (€)': rr_result['end_gross']},
                    {'Größe': 'Modellierte Steuern', 'Wert (€)': rr_result['total_taxes']},
                    {'Größe': 'Endwert nach Steuern', 'Wert (€)': rr_result['end_after_tax']},
                    {'Größe': 'Kaufkraft des Endwerts in heutigen Euro', 'Wert (€)': rr_result['end_real']},
                    {'Größe': 'Nominaler Gewinn nach Steuer', 'Wert (€)': rr_result['gain_nominal_net']},
                    {'Größe': 'Realer Gewinn in heutiger Kaufkraft', 'Wert (€)': rr_result['gain_real']},
                ])
                rr_table['Wert (€)'] = rr_table['Wert (€)'].map(lambda v: f'{format_de_num(v, 2)} €')
                st.dataframe(rr_table, hide_index=True, use_container_width=True)
                with st.expander('📖 Was bedeuten die Kennzahlen und Annahmen?'):
                    st.markdown(
                        '**Nominal** ist die Euro-Rendite ohne Berücksichtigung des Kaufkraftverlusts. '
                        '**Real** berücksichtigt zusätzlich die Inflation. '
                        '**Nettorendite** berücksichtigt im Rechner die gewählten modellierten Steuern '
                        'und gegebenenfalls Produktkosten.\n\n'
                        '**Steuern am Ende:** Der gesamte bis zum Verkauf aufgelaufene Kursgewinn '
                        'wird einmal mit dem noch verfügbaren Pauschbetrag im Verkaufsjahr verrechnet. '
                        'Ein nicht ausgeschöpfter Freibetrag der Vorjahre wird nicht nachträglich gesammelt.\n\n'
                        '**Jährliche Realisierung:** Am Jahresende wird eine vollständige Veräußerung '
                        'mit sofortiger Wiederanlage angenommen. Erträge werden je Kalenderjahr '
                        'besteuert; steuerliche Verluste werden im vereinfachten Modell vorgetragen.\n\n'
                        '**Einzahlungen:** Weitere Sparbeträge werden am Monatsende angelegt. '
                        'Die annualisierte Nettorendite ist dann geldgewichtet, damit Einzahlungen '
                        'nicht fälschlich als Investmentgewinn zählen.\n\n'
                        '**Nicht enthalten:** Vorabpauschale von Fonds, Ausschüttungsbesteuerung, '
                        'Quellensteuer, individuelle Steuerbesonderheiten, tatsächliche Kauf-/Verkaufskurse, '
                        'separate Verlustverrechnungstöpfe und Gebühren des Brokers. '
                        'Im Fall bereits erzielter Gewinne wird der Kursverlauf nur modellhaft geglättet. '
                    'Bei einer bereits nach Steuern angegebenen Rendite bitte „Keine Steuern“ wählen.'
                    )
                st.caption(
                    f"Annahme: {rr_annual_return:+.2f} % p.a. vor ggf. zusätzlichen Produktkosten, "
                    f"{rr_inflation:.2f} % Inflation p.a., "
                    f"{rr_result['effective_tax_rate']*100:.3f} % kombinierter Steuersatz "
                    f"auf steuerpflichtige positive Erträge. Das ist eine vereinfachte "
                    'private Kapitalanlage-Modellrechnung nach deutschem Steuerrecht (Stand 2026), '
                    'keine Steuerberatung oder Renditeprognose.'
                )
                export_rr = rr_path.copy()
                export_rr.index.name = 'Zeile'
                st.download_button(
                    '⬇️ Berechnung als CSV herunterladen',
                    data=export_rr.to_csv(index=False, sep=';', decimal=',',
                                          float_format='%.4f').encode('utf-8-sig'),
                    file_name='Reale_Rendite_Berechnung.csv', mime='text/csv',
                    key='rr_csv_download'
                )
            except (ValueError, OverflowError, ZeroDivisionError) as exc:
                st.error(f'Berechnung nicht möglich: {exc}')
