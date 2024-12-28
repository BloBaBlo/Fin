# portfolio_tab.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from ticker_manager import get_ticker_data

# Make sure to install: pip install streamlit-aggrid
from st_aggrid import GridOptionsBuilder, AgGrid, DataReturnMode, GridUpdateMode
from st_aggrid import JsCode

def get_current_price(symbol):
    """Fetch current price, first price of the day, and formatted update date for a given ticker symbol."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1d", interval="1m")
        
        # If 1-day data is unavailable, try a 1-month period
        if hist.empty or "Close" not in hist:
            hist = ticker.history(period="1mo", interval="1m")
        
        if not hist.empty:
            # Extract the most recent price, first price of the day, and the latest timestamp
            latest_price = hist["Close"].iloc[-1]
            first_price_of_day = hist["Close"].iloc[0]
            update_date = hist.index[-1]  # Use index as date
            
            # Format the update_date to remove seconds and timezone
            formatted_update_date = update_date.strftime('%Y-%m-%d %H:%M')
            return latest_price, first_price_of_day, formatted_update_date
        else:
            return np.nan, np.nan, np.nan
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return np.nan, np.nan, np.nan


def initialize_portfolio():
    """Initialize portfolio in session state if not present."""
    if 'portfolio_df' not in st.session_state:
        data = get_ticker_data()
        portfolio_rows = []
        colors = []  # Store the color information
        
        # Convert existing data to portfolio format
        for sym, transactions in data.items():
            # Assume only the last transaction is stored
            date_bought, buy_price, buy_quantity, color, label = transactions[-1]
            date_bought = pd.Timestamp(date_bought)
            
            current_price, first_of_day, update_date = get_current_price(sym)
            performance = (
                (current_price / buy_price - 1.0) * 100.0
                if pd.notna(current_price) and buy_price != 0
                else np.nan
            )
            perf = (
                (current_price - buy_price) * buy_quantity
                if pd.notna(current_price) and buy_price != 0
                else np.nan
            )
            performance_day = (
                (current_price / first_of_day - 1.0) * 100.0
                if pd.notna(current_price) and first_of_day != 0
                else np.nan
            )
            
            portfolio_rows.append({
                'Ticker': sym,
                # Keep only the date (no time):
                'Bought Date': date_bought.date(),
                'Buy Price': buy_price,
                'Quantity': buy_quantity,
                'Current Price': current_price,
                'Perf (%)': performance,
                'Perf': perf,
                'day (%)': performance_day,
                'Update date': update_date,
                'Label': label,
            })
            colors.append(color)

        if portfolio_rows:
            portfolio_df = pd.DataFrame(portfolio_rows)
            portfolio_df['Color'] = colors  # Add color column
        else:
            portfolio_df = pd.DataFrame(
                columns=[
                    'Ticker', 'Bought Date', 'Buy Price', 'Quantity',
                    'Current Price', 'Perf (%)', "Perf", 'day (%)',
                    'Update date', 'Color', 'Label'
                ]
            )
        
        st.session_state.portfolio_df = portfolio_df


def add_portfolio_entry(new_data):
    """Add a new entry to the portfolio dataframe."""
    if 'portfolio_df' not in st.session_state:
        initialize_portfolio()
    
    st.session_state.portfolio_df = pd.concat([
        st.session_state.portfolio_df,
        pd.DataFrame([new_data])
    ], ignore_index=True)


def show_portfolio():
    """Display the portfolio overview with editing capabilities."""
    st.subheader("Portfolio Overview")
    
    # Initialize portfolio if needed
    initialize_portfolio()
    
    # Add new entry section
    st.subheader("Add New Position")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        new_ticker = st.text_input("Ticker Symbol", key="new_ticker").upper()
    with col2:
        new_date = st.date_input("Purchase Date", key="new_date", max_value=date.today())
    with col3:
        new_price = st.number_input("Purchase Price", key="new_price", min_value=0.0, format="%.2f")
    with col4:
        new_quantity = st.number_input("Quantity", key="new_quantity", min_value=0, step=1)
    with col5:
        new_label = st.text_input("Label", key="new_Label")
    
    new_color = st.color_picker("Select Row Color", key="new_color", value="#FFFFFF")
    
    if st.button("Add Position"):
        if new_ticker and new_price > 0 and new_quantity > 0:
            current_price, first_of_day, update_date = get_current_price(new_ticker)
            performance = (
                (current_price / new_price - 1.0) * 100.0
                if pd.notna(current_price) and new_price != 0
                else np.nan
            )
            performance_day = (
                (current_price / first_of_day - 1.0) * 100.0
                if pd.notna(current_price) and first_of_day != 0
                else np.nan
            )
            perf = (
                (current_price - new_price) * new_quantity
                if pd.notna(current_price) and new_price != 0
                else np.nan
            )
            new_data = {
                'Ticker': new_ticker,
                'Bought Date': new_date,
                'Buy Price': new_price,
                'Quantity': new_quantity,
                'Current Price': current_price,
                'Perf (%)': performance,
                'Perf': perf,
                'day (%)': performance_day,
                'Update date': update_date,
                'Label': new_label,
                'Color': new_color
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
                current_price, first_of_day, update_date = get_current_price(row['Ticker'])
                df.at[idx, 'Current Price'] = current_price
                df.at[idx, 'Perf (%)'] = (
                    (current_price / row['Buy Price'] - 1.0) * 100.0
                    if pd.notna(current_price) and row['Buy Price'] != 0
                    else np.nan
                )
                df.at[idx, 'Perf'] = (
                    (current_price - row["Buy Price"]) * row['Quantity']
                    if pd.notna(current_price) and row['Buy Price'] != 0
                    else np.nan
                )
                df.at[idx, 'day (%)'] = (
                    (current_price / first_of_day - 1.0) * 100.0
                    if pd.notna(current_price) and first_of_day != 0
                    else np.nan
                )
                df.at[idx, 'Update date'] = update_date
            return df
        
        updated_df = update_prices()
        
        # 1) Prepare the columns that we want to show:
        columns_to_keep = [
            "Ticker", "Bought Date", "Buy Price", "Quantity", 
            "Current Price", "Perf (%)", "Perf", 'day (%)', 
            'Update date', "Label"
        ]
        
        # 2) We'll make a copy that includes "Color" for row styling
        display_df = updated_df[columns_to_keep + ["Color"]].copy()

        # 3) Define a JS function for row background color from "Color"
        row_style_code = JsCode("""
        function(params) {
            if (params.data.Color) {
                return {
                    'background-color': params.data.Color,
                    'color': 'black'
                };
            }
        };
        """)

        # 4) Define a JS function for performance columns color
        performance_cell_style_code = JsCode("""
        function(params) {
            if (params.value < 0) {
                return { 'color': '#5E0007' };
            } else if (params.value > 0) {
                return { 'color': '#1E5400' };
            }
            return { 'color': 'black' };
        };
        """)

        # 5) Build the AgGrid options:
        gb = GridOptionsBuilder.from_dataframe(
            display_df,
            enableRowGroup=True,
            enableValue=True,
            enablePivot=True
        )

        # (a) Configure a default column to have a minimum width (so header is always readable)
        gb.configure_default_column(minWidth=50, resizable=True)

        # (b) Hide the "Color" column from view, but keep it for styling
        gb.configure_column("Color", header_name="Color", hide=True)

        # (c) Pin the "Ticker" column to the left
        gb.configure_column("Ticker", pinned='left')

        # (d) Performance columns with custom cellStyle
        for c in ["Perf (%)", "Perf", "day (%)"]:
            gb.configure_column(c, cellStyle=performance_cell_style_code)

        # (e) Format the date column "Bought Date" so it displays just YYYY-MM-DD
        gb.configure_column(
            "Bought Date",
            type=["customDateTimeFormat"],
            custom_format_string='yyyy-MM-dd'
        )

        # (f) For float columns, show only 2 decimals
        float_cols = [
            "Buy Price", "Quantity", "Current Price", 
            "Perf (%)", "Perf", "day (%)"
        ]
        # We'll use valueFormatter to ensure 2 decimals
        for col_name in float_cols:
            gb.configure_column(
                col_name,
                valueFormatter=JsCode("""
                function(params) {
                    if (params.value == null || isNaN(params.value)) {
                        return '';
                    }
                    return Number(params.value).toFixed(2);
                }
                """)
            )

        # Build final GridOptions
        grid_options = gb.build()

        # Attach the row style function for row background color
        grid_options["getRowStyle"] = row_style_code

        # 6) Render the table using AgGrid
        AgGrid(
            display_df,
            gridOptions=grid_options,
            data_return_mode=DataReturnMode.AS_INPUT,
            update_mode=GridUpdateMode.NO_UPDATE,
            fit_columns_on_grid_load=False,  # do NOT auto-fit, so we see minWidth
            allow_unsafe_jscode=True,        # needed to allow JsCode
            theme="alpine",                  # or 'streamlit', 'balham', etc.
            height=400
        )

        # Add refresh button
        if st.button("Refresh Prices"):
            st.rerun()

        # ===== Portfolio Summary =====
        total_investment = (updated_df['Buy Price'] * updated_df['Quantity']).sum()
        current_value = (updated_df['Current Price'] * updated_df['Quantity']).sum()
        total_return = (
            (current_value / total_investment - 1) * 100
            if total_investment > 0
            else 0
        )
        total_return_cash = (
            current_value - total_investment
            if total_investment > 0
            else 0
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Investment", f"${total_investment:,.2f}")
        with col2:
            st.metric("Current Value", f"${current_value:,.2f}")
        with col3:
            st.metric("Total Return", f"{total_return_cash:,.2f} ({total_return:.2f}%)")

        # Group by Label for summary
        grouped_summary = updated_df.groupby('Label').apply(
            lambda group: pd.Series({
                'Total Investment': (group['Buy Price'] * group['Quantity']).sum(),
                'Current Value': (group['Current Price'] * group['Quantity']).sum()
            })
        ).reset_index()

        # Add total return columns
        grouped_summary['Total Return Cash'] = (
            grouped_summary['Current Value'] - grouped_summary['Total Investment']
        )
        grouped_summary['Total Return (%)'] = (
            (grouped_summary['Current Value'] / grouped_summary['Total Investment'] - 1) * 100
        ).fillna(0)  # in case of zero total investment

        # Display grouped summary
        st.subheader("Portfolio Summary by Label")
        st.dataframe(grouped_summary, use_container_width=True)

