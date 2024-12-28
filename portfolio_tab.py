# portfolio_tab.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date
from ticker_manager import get_ticker_data

# st-aggrid for tables
from st_aggrid import GridOptionsBuilder, AgGrid, DataReturnMode, GridUpdateMode
from st_aggrid import JsCode

# For auto-refresh
from streamlit_autorefresh import st_autorefresh

# For charts
import altair as alt


###############################################################################
#                         1) HELPER FUNCTIONS
###############################################################################

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
            
            # Format update_date to remove seconds and timezone
            formatted_update_date = update_date.strftime('%Y-%m-%d %H:%M')
            return latest_price, first_price_of_day, formatted_update_date
        else:
            return np.nan, np.nan, np.nan
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return np.nan, np.nan, np.nan


def initialize_transactions():
    """
    Initialize a transactions DataFrame in session state if not present.
    This new DataFrame stores every single buy transaction (ticker, date, buy price, quantity, color, label).
    """
    if 'transactions_df' not in st.session_state:
        raw_data = get_ticker_data()
        
        transactions_rows = []
        for sym, transactions in raw_data.items():
            for t in transactions:  # each t = (date_bought, buy_price, buy_quantity, color, label)
                date_bought = pd.Timestamp(t[0])
                buy_price = t[1]
                quantity = t[2]
                color = t[3]
                label = t[4]
                transactions_rows.append({
                    'Ticker': sym,
                    'Date': date_bought.date(),
                    'Buy Price': buy_price,
                    'Quantity': quantity,
                    'Color': color,
                    'Label': label
                })
        
        if transactions_rows:
            transactions_df = pd.DataFrame(transactions_rows)
        else:
            transactions_df = pd.DataFrame(columns=['Ticker','Date','Buy Price','Quantity','Color','Label'])
        
        st.session_state.transactions_df = transactions_df


def aggregate_transactions():
    """
    Create a portfolio_df from the transactions_df by aggregating positions for each Ticker:
      - Summation of Quantities
      - Weighted Average Buy Price (Dollar-Cost Average)
      - We'll store the 'Color' and 'Label' from the *most recent* transaction for that ticker
    """
    df = st.session_state.transactions_df.copy()
    if df.empty:
        return pd.DataFrame(columns=['Ticker','Quantity','Buy Price','Color','Label'])
    
    grouped = df.groupby('Ticker', as_index=False)
    rows = []
    for sym, subdf in grouped:
        total_qty = subdf['Quantity'].sum()
        if total_qty == 0:
            avg_price = np.nan
        else:
            avg_price = (subdf['Buy Price'] * subdf['Quantity']).sum() / total_qty
        
        last_row = subdf.iloc[-1]
        color = last_row['Color']
        label = last_row['Label']
        
        rows.append({
            'Ticker': sym,
            'Quantity': total_qty,
            'Buy Price': avg_price,
            'Color': color,
            'Label': label
        })
    
    portfolio_df = pd.DataFrame(rows)
    return portfolio_df


def update_prices_and_performance(portfolio_df):
    """
    For each row in portfolio_df, fetch current prices and compute performance metrics.
    Return a new dataframe with these columns:
      - Current Price
      - Perf (%)
      - Perf (in $)
      - day (%)
      - Update date
    """
    df = portfolio_df.copy()
    df['Current Price'] = np.nan
    df['Perf (%)'] = np.nan
    df['Perf'] = np.nan
    df['day (%)'] = np.nan
    df['Update date'] = np.nan
    
    for idx, row in df.iterrows():
        sym = row['Ticker']
        buy_price = row['Buy Price']
        quantity = row['Quantity']
        
        if pd.isna(sym) or sym == '':
            continue
        
        current_price, first_of_day, update_date = get_current_price(sym)
        df.at[idx, 'Current Price'] = current_price
        df.at[idx, 'Update date'] = update_date
        
        if pd.notna(current_price) and buy_price != 0:
            perf_pct = (current_price / buy_price - 1.0) * 100.0
            df.at[idx, 'Perf (%)'] = perf_pct
            df.at[idx, 'Perf'] = (current_price - buy_price) * quantity
        else:
            df.at[idx, 'Perf (%)'] = np.nan
            df.at[idx, 'Perf'] = np.nan
        
        if pd.notna(current_price) and first_of_day != 0:
            df.at[idx, 'day (%)'] = (current_price / first_of_day - 1.0) * 100.0
        else:
            df.at[idx, 'day (%)'] = np.nan
    
    return df


###############################################################################
#                  2) MAIN SHOW_PORTFOLIO FUNCTION
###############################################################################

def show_portfolio():
    """
    Main function that displays the portfolio page, including:
      - Transactions input
      - The aggregated portfolio table (with edit/delete).
      - Auto-refresh functionality
      - Charting
    """
    st.subheader("Portfolio Overview")

    # 1) Initialize or retrieve transactions
    initialize_transactions()
    transactions_df = st.session_state.transactions_df
    
    # 2) Let the user add a new transaction
    st.subheader("Add New Position (Transaction)")
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

    if st.button("Add Transaction"):
        if new_ticker and new_price > 0 and new_quantity > 0:
            new_data = {
                'Ticker': new_ticker,
                'Date': new_date,
                'Buy Price': new_price,
                'Quantity': new_quantity,
                'Color': new_color,
                'Label': new_label
            }
            st.session_state.transactions_df = pd.concat([
                st.session_state.transactions_df,
                pd.DataFrame([new_data])
            ], ignore_index=True)
            st.success(f"Added new transaction for {new_ticker}!")
        else:
            st.error("Please fill in all fields correctly.")

    st.write("---")

    ############################################################################
    # 2.A) Show the TRANSACTIONS table with "Edit" and "Delete" possibilities
    ############################################################################
    st.subheader("All Transactions (Editable)")
    if not transactions_df.empty:
        gb_trans = GridOptionsBuilder.from_dataframe(transactions_df)
        # Make columns editable
        gb_trans.configure_default_column(editable=True, resizable=True)
        
        # We'll add a "Delete" button in an "Actions" column:
        delete_cell = JsCode("""
        class BtnCellRenderer {
            init(params) {
                this.params = params;
                this.eGui = document.createElement('div');
                this.eGui.innerHTML = `
                    <span>
                      <button id='deleteBtn' 
                              class='btn btn-danger btn-sm'>Delete</button>
                    </span>
                `;
                this.btnClickedHandler = this.btnClickedHandler.bind(this);
                this.eGui.querySelector('#deleteBtn').addEventListener('click', this.btnClickedHandler);
            }

            btnClickedHandler(e) {
                const rowIndex = this.params.node.rowIndex;
                window.deleteRow = rowIndex;  
            }

            getGui() {
                return this.eGui;
            }

            destroy() {}
        }
        """)
        gb_trans.configure_column(
            "Actions",
            cellRenderer=delete_cell,
            editable=False,
            resizable=False,
            filter=False,
            sortable=False
        )
        # If "Actions" not in df, add it
        if "Actions" not in transactions_df.columns:
            transactions_df["Actions"] = ""

        gridOptions_trans = gb_trans.build()
        grid_response = AgGrid(
            transactions_df,
            gridOptions=gridOptions_trans,
            data_return_mode=DataReturnMode.AS_INPUT,
            update_mode=GridUpdateMode.MODEL_CHANGED, 
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            theme="alpine",
            height=300
        )

        # 1) If user edited data, update session state
        edited_df = pd.DataFrame(grid_response["data"])
        st.session_state.transactions_df = edited_df

        # 2) Provide a manual approach to delete a row by index:
        st.write("**Delete a row by index**:")
        row_to_delete = st.number_input(
            "Row index to delete",
            min_value=0,
            max_value=len(edited_df)-1 if len(edited_df) > 0 else 0,
            value=0,
            step=1
        )
        delete_col1, delete_col2 = st.columns(2)
        with delete_col1:
            if st.button("Confirm Delete"):
                if 0 <= row_to_delete < len(edited_df):
                    st.session_state.transactions_df = edited_df.drop(index=row_to_delete).reset_index(drop=True)
                    st.success(f"Deleted row index {row_to_delete}")
                    st.rerun()

        # 3) **Add a Download CSV button** for the transactions
        csv_data = edited_df.to_csv(index=False)
        with delete_col2:
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="transactions.csv",
                mime="text/csv"
            )

    else:
        st.info("No transactions yet.")

    st.write("---")

    ############################################################################
    # 3) AGGREGATE the transactions to get current portfolio
    ############################################################################
    portfolio_df = aggregate_transactions()
    if portfolio_df.empty:
        st.warning("No aggregated portfolio yet (no transactions).")
        return

    updated_df = update_prices_and_performance(portfolio_df)

    st.subheader("Aggregated Portfolio (By Ticker, Weighted Average Basis)")

    # Let user choose whether to refresh automatically
    st.write("#### Automatic Price Refresh")
    auto_refresh = st.checkbox("Auto-Refresh Prices?", value=False)
    refresh_interval = st.number_input("Refresh Interval (seconds)", 5, 3600, 60, 5)
    if auto_refresh:
        count = st_autorefresh(interval=refresh_interval * 1000, limit=100000, key="price_autorefresh")
        st.caption(f"Auto-refreshed {count} times so far.")

    if st.button("Manual Refresh Prices"):
        st.rerun()  

    # Show the aggregated portfolio
    columns_to_show = [
        "Ticker", "Quantity", "Buy Price", "Current Price",
        "Perf (%)", "Perf", "day (%)", "Update date", "Label"
    ]
    display_portfolio = updated_df[columns_to_show + ["Color"]].copy()

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

    gb_pf = GridOptionsBuilder.from_dataframe(display_portfolio)
    gb_pf.configure_default_column(minWidth=50, resizable=True)
    gb_pf.configure_column("Color", hide=True)
    gb_pf.configure_column("Ticker", pinned='left')

    for col in ["Perf (%)", "Perf", "day (%)"]:
        gb_pf.configure_column(col, cellStyle=performance_cell_style_code)

    # Format floats to 2 decimals
    float_cols = ["Quantity", "Buy Price", "Current Price", "Perf (%)", "Perf", "day (%)"]
    for col in float_cols:
        gb_pf.configure_column(
            col,
            valueFormatter=JsCode("""
            function(params) {
                if (params.value == null || isNaN(params.value)) {
                    return '';
                }
                return Number(params.value).toFixed(2);
            }
            """)
        )

    grid_opts_pf = gb_pf.build()
    grid_opts_pf["getRowStyle"] = row_style_code

    AgGrid(
        display_portfolio,
        gridOptions=grid_opts_pf,
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=GridUpdateMode.NO_UPDATE,
        fit_columns_on_grid_load=False,
        allow_unsafe_jscode=True,
        theme="alpine",
        height=500
    )

    st.write("---")

    # ===== Portfolio Summary =====
    total_investment = (updated_df['Buy Price'] * updated_df['Quantity']).sum()
    current_value = (updated_df['Current Price'] * updated_df['Quantity']).sum()
    total_return_cash = (current_value - total_investment) if total_investment > 0 else 0
    total_return_pct = ((current_value / total_investment - 1) * 100) if total_investment > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Investment", f"${total_investment:,.2f}")
    with col2:
        st.metric("Current Value", f"${current_value:,.2f}")
    with col3:
        st.metric("Total Return", f"{total_return_cash:,.2f} ({total_return_pct:.2f}%)")

    st.subheader("Portfolio Summary by Label")
    grouped_summary = updated_df.groupby('Label').apply(
        lambda group: pd.Series({
            'Total Investment': (group['Buy Price'] * group['Quantity']).sum(),
            'Current Value': (group['Current Price'] * group['Quantity']).sum()
        })
    ).reset_index()

    grouped_summary['Total Return Cash'] = grouped_summary['Current Value'] - grouped_summary['Total Investment']
    grouped_summary['Total Return (%)'] = (
        (grouped_summary['Current Value'] / grouped_summary['Total Investment'] - 1) * 100
    ).fillna(0)

    st.dataframe(grouped_summary, use_container_width=True)

    ############################################################################
    # 3) CHARTS & VISUALIZATIONS
    ############################################################################
    st.subheader("Charts & Visualizations")

    # 3.A) Portfolio breakdown by label: Pie or bar chart
    if not grouped_summary.empty:
        # Pie chart
        st.markdown("**Portfolio Breakdown by Label (Pie Chart)**")
        chart_data = grouped_summary[['Label', 'Current Value']].copy()
        chart_data['Current Value'] = chart_data['Current Value'].fillna(0)

        pie_chart = alt.Chart(chart_data).mark_arc().encode(
            theta='Current Value',
            color='Label',
            tooltip=['Label','Current Value']
        )
        st.altair_chart(pie_chart, use_container_width=True)

        # Bar chart
        st.markdown("**Portfolio Breakdown by Label (Bar Chart)**")
        bar_chart = alt.Chart(chart_data).mark_bar().encode(
            x='Label',
            y='Current Value',
            color='Label',
            tooltip=['Label','Current Value']
        )
        st.altair_chart(bar_chart, use_container_width=True)

    ###########################################################################
    # (1) Compare Ticker Performance on a Bar Chart
    ###########################################################################
    """
    **(1) Compare Ticker Performance**
    A simple bar chart comparing each ticker's `Perf (%)`. 
    """

    if not updated_df.empty:
        # We'll take columns: Ticker, Perf (%)
        # Drop rows with no perf data
        chart_df_perf = updated_df[['Ticker', 'Perf (%)']].dropna().copy()
        
        st.markdown("**Compare Ticker Performance (%)**")
        if chart_df_perf.empty:
            st.write("No performance data to compare.")
        else:
            # Convert to two decimals
            chart_df_perf['Perf (%)'] = chart_df_perf['Perf (%)'].round(2)
            
            perf_bar = alt.Chart(chart_df_perf).mark_bar().encode(
                x=alt.X('Ticker', sort='-y'),
                y='Perf (%):Q',
                color='Ticker',
                tooltip=['Ticker', 'Perf (%)']
            ).properties(
                title="Performance (%) by Ticker"
            )
            st.altair_chart(perf_bar, use_container_width=True)
    else:
        st.info("No data in updated_df to show Ticker Performance chart.")


    ###########################################################################
    # (2) Distribution of Current Prices (Histogram)
    ###########################################################################
    """
    **(2) Distribution of Current Prices (Histogram)**
    See how current prices of your tickers (for those you hold) are distributed.
    """

    if not updated_df.empty:
        price_df = updated_df[['Ticker', 'Current Price']].dropna().copy()
        
        st.markdown("**Distribution of Current Prices**")
        if price_df.empty:
            st.write("No current prices to show.")
        else:
            # For a histogram, we only need the numeric column
            hist_chart = alt.Chart(price_df).mark_bar().encode(
                x=alt.X('Current Price:Q', bin=alt.Bin(maxbins=20)),
                y='count()',
                tooltip=['count()']
            ).properties(
                title="Histogram of Current Prices"
            )
            st.altair_chart(hist_chart, use_container_width=True)
    else:
        st.info("No data in updated_df to show Current Price histogram.")


    ###########################################################################
    # (3) Portfolio Value Over Time (Line Chart)
    ###########################################################################
    """
    **(3) Portfolio Value Over Time**  
    Create a synthetic “portfolio value” time-series by summing daily close prices * quantity for each ticker.
    If a fetched Close price is 0 (indicating holiday or missing data), we use the last positive price.
    """

    import datetime

    st.markdown("**Portfolio Value Over Time (last 30 days)**")

    if not updated_df.empty:
        # We'll do a quick approach: 
        # 1. For each ticker in updated_df, fetch last 30 days' daily data from Yahoo
        # 2. Replace any Close=0 with the last known positive price (forward fill)
        # 3. Multiply the adjusted Close by the quantity
        # 4. Sum across tickers by date

        tickers_list = updated_df['Ticker'].unique().tolist()
        end_date = datetime.datetime.today()
        start_date = end_date - datetime.timedelta(days=30)
        
        # Empty list to hold each ticker's daily value
        all_values = []
        
        for tkr in tickers_list:
            qty = updated_df.loc[updated_df['Ticker'] == tkr, 'Quantity'].values[0]
            if qty <= 0:
                continue
            
            # Fetch daily data 
            try:
                df_tkr = yf.download(tkr, start=start_date, end=end_date, progress=False)
                if not df_tkr.empty:
                    # Keep only 'Close'
                    df_tkr = df_tkr[['Close']].reset_index()
                    df_tkr['Date'] = pd.to_datetime(df_tkr['Date']).dt.date
                    df_tkr['Ticker'] = tkr

                    # Replace 0 with NaN and forward fill to take last positive price
                    df_tkr['Close'] = df_tkr['Close'].replace(0, np.nan).ffill()

                    # Now compute daily value
                    df_tkr['Value'] = df_tkr['Close'] * qty
                    all_values.append(df_tkr[['Date','Ticker','Value']])
            except Exception as e:
                print(f"Could not fetch data for {tkr}: {e}")
                continue
        
        if all_values:
            combined_df = pd.concat(all_values, ignore_index=True)
            # Group by Date to sum across all tickers
            daily_sum = combined_df.groupby('Date', as_index=False)['Value'].sum()
            daily_sum = daily_sum.sort_values('Date')

            # Now let's chart the daily_sum with Altair
            line_chart = alt.Chart(daily_sum).mark_line().encode(
                x='Date:T',
                y='Value:Q',
                tooltip=['Date:T','Value:Q']
            ).properties(
                title="Total Portfolio Value (last 30 days)"
            )
            st.altair_chart(line_chart, use_container_width=True)
        else:
            st.write("No valid daily price data to create a portfolio timeseries.")
    else:
        st.info("No data in updated_df to create a 'Portfolio Value Over Time' chart.")
        
    # ==========================================================================
    # 3.E) Additional Charts
    # ==========================================================================
    st.markdown("---")
    st.subheader("More Charts")

    # (4) Bubble Chart: Ticker vs Perf (%) with circle size = Quantity
    """
    **(4) Bubble Chart**: X-axis = Ticker, Y-axis = Perf (%), Circle size = Quantity
    """
    if not updated_df.empty:
        bubble_df = updated_df[['Ticker','Perf (%)','Quantity']].dropna().copy()
        if bubble_df.empty:
            st.write("Not enough data for bubble chart.")
        else:
            # We might convert Ticker to an ordered categorical to keep X sorted
            bubble_df['Perf (%)'] = bubble_df['Perf (%)'].round(2)
            bubble_chart = alt.Chart(bubble_df).mark_circle().encode(
                x=alt.X('Ticker', sort=None),
                y='Perf (%):Q',
                size='Quantity:Q',
                color='Ticker:N',
                tooltip=['Ticker','Perf (%)','Quantity']
            ).properties(
                title="Bubble Chart: Ticker vs Perf (%) [size=Quantity]"
            )
            st.altair_chart(bubble_chart, use_container_width=True)
    else:
        st.info("No data in updated_df to show Bubble Chart.")


    # (5) Box Plot: Perf (%) by Label
    """
    **(5) Box Plot**: Distribution of Perf (%) by Label
    """
    if not updated_df.empty:
        box_df = updated_df[['Label','Perf (%)']].dropna().copy()
        if box_df.empty:
            st.write("Not enough data to show box plot.")
        else:
            box_df['Perf (%)'] = box_df['Perf (%)'].round(2)
            box_plot = alt.Chart(box_df).mark_boxplot().encode(
                x='Label:N',
                y='Perf (%):Q',
                color='Label'
            ).properties(
                title="Distribution of Perf (%) by Label"
            )
            st.altair_chart(box_plot, use_container_width=True)
    else:
        st.info("No data in updated_df to show Box Plot.")

