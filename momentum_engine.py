"""Eigenstaendiger Momentum-Rechenkern (V10).

Keine Charts und keine Streamlit-Tabs hier. Rechenfunktionen nutzen Cache,
werden aber erst aufgerufen, wenn der Momentum-Reiter geoeffnet wird.
"""
import os
import logging
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import pandas_market_calendars as mcal
from finanz_assets import get_asset_name, get_sp500_name_mapping, get_sp500_sector_mapping

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
    """V2: Signal am abgeschlossenen NYSE-Wochenschluss,
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
        MOMENTUM_LOG.exception('Momentum-V2-Berechnung fehlgeschlagen')
        st.error(f'Momentum-Berechnung ({holding_days} Tage) fehlgeschlagen: {exc}')
        return pd.DataFrame(), pd.DataFrame(), {}

# -------------------------------------------------------------
# V7: EURO-DEPOT-SIMULATION FÜR MOMENTUM
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
    """Simuliert echte Geldbeträge anhand der Zielgewichts-Events der V6-Strategie.

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
    # V7.1: Nicht über sämtliche historischen Zeilen hinweg NaN verlangen.
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



@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
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
