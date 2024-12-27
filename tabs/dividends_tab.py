# tabs/dividends_tab.py
import streamlit as st
import yfinance as yf

from data_fetchers import get_dividends

def show_dividends(ticker_input):
    st.subheader(f"Dividend History for {ticker_input}")
    ticker = yf.Ticker(ticker_input)
    div_df = get_dividends(ticker)
    if div_df is None or div_df.empty:
        st.write("No dividend data available.")
    else:
        st.dataframe(div_df)
