# tabs/company_info_tab.py
import streamlit as st
import yfinance as yf
import pandas as pd

from data_fetchers import get_company_info

def show_company_info(ticker_input):
    st.subheader(f"Company Info for {ticker_input}")
    ticker = yf.Ticker(ticker_input)

    info_df = get_company_info(ticker)
    if info_df is None or info_df.empty:
        st.write("No fundamental company data available for this ticker. It might be an ETF or a ticker that doesn't provide this info.")
    else:
        st.dataframe(info_df)

    balance_sheet = ticker.balance_sheet
    if balance_sheet is not None:
        st.write("### balance_sheet")
        st.dataframe(balance_sheet)

    calendar = ticker.calendar
    if calendar is not None:
        st.write("### calendar")
        st.dataframe(calendar)

    cashflow = ticker.cashflow
    if cashflow is not None:
        st.write("### cashflow")
        st.dataframe(cashflow)

    institutional_holders = ticker.institutional_holders
    if institutional_holders is not None:
        st.write("### institutional_holders")
        st.dataframe(institutional_holders)

    major_holders = ticker.major_holders
    if major_holders is not None:
        st.write("### major_holders")
        st.dataframe(major_holders)

    mutualfund_holders = ticker.mutualfund_holders
    if mutualfund_holders is not None:
        st.write("### mutualfund_holders")
        st.dataframe(mutualfund_holders)

    insider_purchases = ticker.insider_purchases
    if insider_purchases is not None:
        st.write("### insider_purchases")
        st.dataframe(insider_purchases)

    insider_roster_holders = getattr(ticker, "insider_roster_holders", None)
    if insider_roster_holders is not None:
        st.write("### insider_roster_holders")
        st.dataframe(insider_roster_holders)

    insider_transactions = ticker.insider_transactions
    if insider_transactions is not None:
        st.write("### insider_transactions")
        st.dataframe(insider_transactions)
