# tabs/earnings_tab.py
import streamlit as st
import yfinance as yf

from data_fetchers import get_earnings

def show_earnings(ticker_input):
    st.subheader(f"Earnings for {ticker_input}")
    ticker = yf.Ticker(ticker_input)
    earnings_df = get_earnings(ticker)
    if earnings_df is None or earnings_df.empty:
        st.write("No earnings data available.")
    else:
        st.dataframe(earnings_df)
