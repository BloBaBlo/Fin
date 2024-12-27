# tabs/news_tab.py
import streamlit as st
import yfinance as yf

def show_news(ticker_input):
    st.subheader(f"News for {ticker_input}")
    ticker = yf.Ticker(ticker_input)
    news = ticker.news
    if news is not None:
        st.write("### news")
        st.dataframe(news)
    else:
        st.write("No news found for this ticker.")
