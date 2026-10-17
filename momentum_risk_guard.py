"""Momentum Risk Guard: unabhängiger, regelbasierter Risiko-Overlay für V8/V9.

Kein trainiertes Vorhersagemodell; Scores sind heuristische Frühwarnindikatoren.
Nur abgeschlossene Tageskurse übergeben. Ausschließlich Signale zum tatsächlichen
NYSE-Wochenschluss; Änderungen werden konservativ erst NACH dem nächsten
Handelstagsschluss für tägliche Renditen berücksichtigt.

Kein Handel, keine Brokeranbindung, keine realen Steuer- oder Kostenberechnungen.
"""
from __future__ import annotations

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
