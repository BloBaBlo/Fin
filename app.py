import streamlit as st
import pandas as pd
import yfinance as yf
import json

import portfolio_tab

from ticker_manager import load_ticker_data, get_ticker_data, get_ticker_list

# Instead of `import tabs`, we import each tab directly
from tabs import (
    historical_data_tab,
    company_info_tab,
    news_tab,
    analyst_recommendations_tab,
    dividends_tab,
    earnings_tab,
    analysis_tab,
    search_tab,
)


st.set_page_config(page_title="My App",page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

st.title("📈 ")
st.sidebar.title("Settings")

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "AAPL"  # default



# Replace your current ticker data initialization with:
uploaded_file = st.sidebar.file_uploader("Upload a CSV file for preset tickers", type=["csv"])
if uploaded_file is not None:
    load_ticker_data(uploaded_file)

preset_tickers = get_ticker_list()
preset_tickers_data = get_ticker_data()

# -- Sidebar to select the ticker
input_method = st.sidebar.radio(
    "Choose Input Method:",
    ["Select from List", "Type Manually", "Search for a Ticker"]
)

if input_method == "Select from List":
    st.session_state.selected_ticker = st.sidebar.selectbox(
        "Choose a Stock/ETF Ticker:", 
        preset_tickers
    )
elif input_method == "Type Manually":
    st.session_state.selected_ticker = st.sidebar.text_input(
        "Enter a Stock/ETF Ticker (e.g. AAPL, MSFT, TSLA, SPY)", 
        value=st.session_state.selected_ticker
    )
elif input_method == "Search for a Ticker":
    search_keyword = st.sidebar.text_input("Enter a search keyword (e.g. 'Apple')", value="")
    if search_keyword:
        quotes = yf.Search(search_keyword, max_results=30).quotes
        if quotes:
            ticker_names = [quote['symbol'] + " - " + quote['shortname'] for quote in quotes]
            selected_quote = st.sidebar.selectbox("Select a Ticker:", ticker_names)
            st.session_state.selected_ticker = selected_quote.split(" - ")[0]  # Extract the ticker symbol
        else:
            st.sidebar.write("No results found. Try another keyword.")

ticker_input = st.session_state.selected_ticker

functions = [
    "Show Historical Data",
    "Portfolio", 
    "Show Company Info",
    "News",
    "Show Analyst Recommendations",
    "Show Dividends",
    "Show Earnings",
    "Analysis & Visualization",
    "Search",
]
selected_function = st.sidebar.radio("Select an operation:", functions)

if ticker_input:
    st.write(f"Selected Ticker: {ticker_input}")
else:
    st.write("Please select or enter a ticker.")

# -- Routing
if selected_function == "Portfolio":
    portfolio_tab.show_portfolio()

elif selected_function == "Show Historical Data":
    historical_data_tab.show_historical_data(ticker_input)

elif selected_function == "Show Company Info":
    company_info_tab.show_company_info(ticker_input)

elif selected_function == "News":
    news_tab.show_news(ticker_input)

elif selected_function == "Show Analyst Recommendations":
    analyst_recommendations_tab.show_analyst_recommendations(ticker_input)

elif selected_function == "Show Dividends":
    dividends_tab.show_dividends(ticker_input)

elif selected_function == "Show Earnings":
    earnings_tab.show_earnings(ticker_input)

elif selected_function == "Analysis & Visualization":
    analysis_tab.show_analysis_and_visualization(ticker_input)

elif selected_function == "Search":
    search_tab.show_search_page()

else:
    st.write("Please select an option from the sidebar.")

