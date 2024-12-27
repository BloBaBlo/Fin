# portfolio_tab.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from ticker_manager import get_ticker_data, get_ticker_list

def initialize_portfolio():
    """Initialize portfolio in session state if not present"""
    if 'portfolio_df' not in st.session_state:
        # Get data from ticker manager
        data = get_ticker_data()
        portfolio_rows = []
        
        # Convert existing data to portfolio format
        for sym, transactions in data.items():
            # We'll assume only 1 transaction stored (the last one)
            date_bought, buy_price, buy_quantity = transactions[-1]
            date_bought = pd.Timestamp(date_bought)
            
            current_price = get_current_price(sym)
            performance = (
                (current_price / buy_price - 1.0) * 100.0
                if pd.notna(current_price) and buy_price != 0
                else np.nan
            )
            
            portfolio_rows.append({
                'Ticker': sym,
                'Bought Date': date_bought.date(),
                'Buy Price': buy_price,
                'Quantity': buy_quantity,
                'Current Price': current_price,
                'Performance (%)': performance
            })
        
        # Create DataFrame with existing data
        st.session_state.portfolio_df = pd.DataFrame(portfolio_rows) if portfolio_rows else pd.DataFrame(
            columns=['Ticker', 'Bought Date', 'Buy Price', 'Quantity', 'Current Price', 'Performance (%)']
        )

def add_portfolio_entry(new_data):
    """Add a new entry to the portfolio dataframe"""
    if 'portfolio_df' not in st.session_state:
        initialize_portfolio()
    
    st.session_state.portfolio_df = pd.concat([
        st.session_state.portfolio_df,
        pd.DataFrame([new_data])
    ], ignore_index=True)

def get_current_price(symbol):
    """Fetch current price for a given ticker symbol"""
    try:
        ticker = yf.Ticker(symbol)
        current_info = ticker.info
        if "regularMarketPrice" in current_info and current_info["regularMarketPrice"] is not None:
            return current_info["regularMarketPrice"]
        hist = ticker.history(period="1d",  interval="1m")
        return hist["Close"].values[-1] if not hist.empty else np.nan
    except:
        return np.nan

def show_portfolio():
    """Display the portfolio overview with editing capabilities"""
    st.subheader("Portfolio Overview")
    
    # Initialize portfolio if needed
    initialize_portfolio()
    
    # Add new entry section
    st.subheader("Add New Position")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        new_ticker = st.text_input("Ticker Symbol", key="new_ticker").upper()
    with col2:
        new_date = st.date_input("Purchase Date", key="new_date", max_value=date.today())
    with col3:
        new_price = st.number_input("Purchase Price", key="new_price", min_value=0.0, format="%.2f")
    with col4:
        new_quantity = st.number_input("Quantity", key="new_quantity", min_value=0, step=1)
    
    if st.button("Add Position"):
        if new_ticker and new_price > 0 and new_quantity > 0:
            current_price = get_current_price(new_ticker)
            performance = ((current_price / new_price - 1.0) * 100.0) if pd.notna(current_price) and new_price != 0 else np.nan
            
            new_data = {
                'Ticker': new_ticker,
                'Bought Date': new_date,
                'Buy Price': new_price,
                'Quantity': new_quantity,
                'Current Price': current_price,
                'Performance (%)': performance
            }
            add_portfolio_entry(new_data)
            st.success(f"Added {new_ticker} to portfolio!")
        else:
            st.error("Please fill in all fields correctly")

    # Display and edit existing portfolio
    st.subheader("Current Portfolio")
    
    if st.session_state.portfolio_df.empty:
        st.info("Your portfolio is empty. Add positions using the form above.")
    else:
        # Update current prices and performance
        def update_prices():
            df = st.session_state.portfolio_df.copy()
            for idx, row in df.iterrows():
                current_price = get_current_price(row['Ticker'])
                df.at[idx, 'Current Price'] = current_price
                df.at[idx, 'Performance (%)'] = (
                    (current_price / row['Buy Price'] - 1.0) * 100.0
                    if pd.notna(current_price) and row['Buy Price'] != 0
                    else np.nan
                )
            return df
        
        updated_df = update_prices()

        # Define styling for the Performance (%) column
        def style_performance(val):
            color = 'red' if val < 0 else 'green' if val > 0 else 'black'
            return f'color: {color}'

        styled_df = updated_df.style.applymap(
            style_performance, subset=['Performance (%)']
        )

        # Display the styled dataframe
        st.dataframe(styled_df, use_container_width=True)
        
        # Add refresh button
        if st.button("Refresh Prices"):
            st.rerun()

        # Display portfolio summary
        total_investment = (updated_df['Buy Price'] * updated_df['Quantity']).sum()
        current_value = (updated_df['Current Price'] * updated_df['Quantity']).sum()
        total_return = ((current_value / total_investment - 1) * 100) if total_investment > 0 else 0

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Investment", f"${total_investment:,.2f}")
        with col2:
            st.metric("Current Value", f"${current_value:,.2f}")
        with col3:
            st.metric("Total Return", f"{total_return:.2f}%")

