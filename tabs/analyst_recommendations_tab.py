# tabs/analyst_recommendations_tab.py
import streamlit as st
import yfinance as yf
import pandas as pd

from data_fetchers import get_analyst_recommendations

def show_analyst_recommendations(ticker_input):
    st.subheader(f"Analyst Recommendations for {ticker_input}")
    ticker = yf.Ticker(ticker_input)
    
    rec_df = get_analyst_recommendations(ticker)
    if rec_df is None or rec_df.empty:
        st.write("No analyst recommendations available.")
    else:
        st.dataframe(rec_df)

    upgrades_downgrades = getattr(ticker, "upgrades_downgrades", pd.DataFrame())
    if not upgrades_downgrades.empty:
        st.write("### upgrades_downgrades")
        st.dataframe(upgrades_downgrades)

    eps_revisions = getattr(ticker, "eps_revisions", pd.DataFrame())
    if not eps_revisions.empty:
        st.write("### eps_revisions")
        st.dataframe(eps_revisions)

    earnings_estimate = getattr(ticker, "earnings_estimate", None)
    if earnings_estimate is not None:
        st.write("### earnings_estimate")
        st.dataframe(earnings_estimate)

    eps_trend = getattr(ticker, "eps_trend", None)
    if eps_trend is not None:
        st.write("### eps_trend")
        st.dataframe(eps_trend)
