"""Gemeinsame Stammdaten für das Finanz-Multitool und den Momentum-Scanner."""
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


