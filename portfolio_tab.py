# portfolio_tab.py

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, datetime, timedelta
from ticker_manager import get_ticker_data

# st-aggrid for tables
from st_aggrid import GridOptionsBuilder, AgGrid, DataReturnMode, GridUpdateMode
from st_aggrid import JsCode

# For auto-refresh
from streamlit_autorefresh import st_autorefresh

# For charts
import altair as alt

# For data backup and restore
import io
import json

# For PDF report generation
from fpdf import FPDF

# For email notifications
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# For Excel export
import xlsxwriter

###############################################################################
#                         1) HELPER FUNCTIONS
###############################################################################

# -------------------
# Email Notification
# -------------------
def send_email(subject, body, to_email, from_email, smtp_server, smtp_port, smtp_user, smtp_password):
    """
    Send an email notification.
    """
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

# -------------------
# Fetch Ticker History with Original Caching
# -------------------
@st.cache_data(ttl=300, show_spinner=True)
def fetch_ticker_history(symbol, period="1d", interval="1m"):
    """Fetch historical data for a ticker symbol with caching."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if hist.empty or "Close" not in hist:
            hist = ticker.history(period="1mo", interval="1m")
        return hist
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def get_current_price(symbol):
    """
    Fetch current price, second latest price, first price of the day,
    and formatted update date for a given ticker symbol.
    """
    hist = fetch_ticker_history(symbol)
    if not hist.empty:
        try:
            latest_price = hist["Close"].iloc[-1]
        except IndexError:
            latest_price = np.nan
        try:
            second_latest_price = hist["Close"].iloc[-2]
        except IndexError:
            second_latest_price = latest_price
        first_price_of_day = hist["Close"].iloc[0]
        update_date = hist.index[-1]
        formatted_update_date = update_date.strftime('%Y-%m-%d %H:%M')
        return latest_price, second_latest_price, first_price_of_day, formatted_update_date
    else:
        return np.nan, np.nan, np.nan, np.nan

def initialize_transactions():
    """
    Initialize a transactions DataFrame in session state if not present.
    This DataFrame stores all buy transactions (ticker, date, buy price, quantity, color, label).
    """
    if 'transactions_df' not in st.session_state:
        raw_data = get_ticker_data()
        transactions_rows = [
            {
                'Ticker': sym,
                'Date': pd.Timestamp(t[0]).date(),
                'Buy Price': t[1],
                'Quantity': t[2],
                'Color': t[3],
                'Label': t[4]
            }
            for sym, transactions in raw_data.items()
            for t in transactions
        ]
        if transactions_rows:
            transactions_df = pd.DataFrame(transactions_rows)
        else:
            transactions_df = pd.DataFrame(columns=['Ticker','Date','Buy Price','Quantity','Color','Label'])
        st.session_state.transactions_df = transactions_df

def aggregate_transactions():
    """
    Aggregate transactions to create a portfolio DataFrame by Ticker:
      - Total Quantity
      - Weighted Average Buy Price (Dollar-Cost Averaging)
      - Most recent Color and Label
    """
    df = st.session_state.transactions_df.copy()
    if df.empty:
        return pd.DataFrame(columns=['Ticker','Quantity','Buy_Price','Color','Label'])
    
    portfolio = df.groupby('Ticker').agg(
        Quantity=pd.NamedAgg(column='Quantity', aggfunc='sum'),
        Buy_Price_Sum=pd.NamedAgg(column='Buy Price', aggfunc=lambda x: (x * df.loc[x.index, 'Quantity']).sum()),
        Color=pd.NamedAgg(column='Color', aggfunc='last'),
        Label=pd.NamedAgg(column='Label', aggfunc='last')
    ).reset_index()
    
    portfolio['Buy_Price'] = portfolio.apply(
        lambda row: row['Buy_Price_Sum'] / row['Quantity'] if row['Quantity'] != 0 else np.nan,
        axis=1
    )
    portfolio = portfolio.drop(columns=['Buy_Price_Sum'])
    return portfolio

@st.cache_data(ttl=300, show_spinner=True)
def fetch_daily_portfolio_value(tickers, quantities, start_date, end_date):
    """
    Fetch daily close prices for each ticker and calculate portfolio value over time.
    """
    all_values = []
    for tkr, qty in zip(tickers, quantities):
        if qty <= 0:
            continue
        try:
            df_tkr = yf.download(tkr, start=start_date, end=end_date, progress=False)
            if not df_tkr.empty:
                df_tkr = df_tkr[['Close']].reset_index()
                df_tkr['Date'] = df_tkr['Date'].dt.date
                df_tkr['Close'] = df_tkr['Close'].replace(0, np.nan).ffill()
                df_tkr['Value'] = df_tkr['Close'] * qty
                all_values.append(df_tkr[['Date','Value']])
        except Exception as e:
            st.warning(f"Could not fetch data for {tkr}: {e}")
            continue
    if all_values:
        combined_df = pd.concat(all_values, ignore_index=True)
        daily_sum = combined_df.groupby('Date').sum().reset_index()
        daily_sum = daily_sum.sort_values('Date')
        return daily_sum
    return pd.DataFrame()

def update_prices_and_performance(portfolio_df):
    """
    Update portfolio DataFrame with current prices and performance metrics.
    Renamed 'Perf (%)' to 'Perf_Pct' to avoid issues with special characters.
    """
    if portfolio_df.empty:
        return portfolio_df

    portfolio_df = portfolio_df.copy()
    portfolio_df[['Current Price', 'Second Latest Price', 'First Price of Day', 'Update date']] = portfolio_df['Ticker'].apply(
        lambda sym: pd.Series(get_current_price(sym))
    )

    # Renamed 'Perf (%)' to 'Perf_Pct'
    portfolio_df['Perf_Pct'] = ((portfolio_df['Current Price'] / portfolio_df['Buy_Price']) - 1) * 100

    portfolio_df['day_Pct'] = ((portfolio_df['Current Price'] / portfolio_df['First Price of Day']) - 1) * 100
    portfolio_df['last_Pct'] = ((portfolio_df['Current Price'] / portfolio_df['Second Latest Price']) - 1) * 100

    # Additional Metrics
    portfolio_df['Market Value'] = portfolio_df['Current Price'] * portfolio_df['Quantity']
    portfolio_df['Total Investment'] = portfolio_df['Buy_Price'] * portfolio_df['Quantity']
    portfolio_df['Return'] = portfolio_df['Market Value'] - portfolio_df['Total Investment']

    return portfolio_df

def validate_transaction_data(ticker, price, quantity, date_bought):
    """Validate the transaction data before adding to the DataFrame."""
    errors = []
    if not ticker:
        errors.append("Ticker symbol cannot be empty.")
    if price <= 0:
        errors.append("Purchase price must be greater than zero.")
    if quantity <= 0:
        errors.append("Quantity must be greater than zero.")
    if date_bought > date.today():
        errors.append("Purchase date cannot be in the future.")
    return errors

def backup_transactions(transactions_df):
    """Backup the transactions DataFrame to a CSV in memory."""
    return transactions_df.to_csv(index=False).encode('utf-8')

def restore_transactions(uploaded_file):
    """Restore the transactions DataFrame from an uploaded CSV file."""
    try:
        df = pd.read_csv(uploaded_file)
        required_columns = {'Ticker','Date','Buy Price','Quantity','Color','Label'}
        if not required_columns.issubset(df.columns):
            st.error("Uploaded CSV does not contain the required columns.")
            return None
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        return df
    except Exception as e:
        st.error(f"Error restoring transactions: {e}")
        return None

def get_unique_labels():
    """Retrieve unique labels from transactions for dropdowns."""
    df = st.session_state.transactions_df
    return df['Label'].dropna().unique().tolist()

# --------------------------------------------
# Improved Alert System Helper Functions
# --------------------------------------------

def initialize_alerts():
    """Initialize alerts DataFrame in session state."""
    if 'alerts_df' not in st.session_state:
        st.session_state.alerts_df = pd.DataFrame(columns=['Ticker', 'Condition', 'Threshold', 'Created At', 'Email'])

def add_alert(ticker, condition, threshold, email):
    """Add a new alert to the alerts DataFrame."""
    new_alert = {
        'Ticker': ticker,
        'Condition': condition,
        'Threshold': threshold,
        'Created At': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Email': email
    }
    st.session_state.alerts_df = pd.concat([
        st.session_state.alerts_df,
        pd.DataFrame([new_alert])
    ], ignore_index=True)

def delete_alert(index):
    """Delete an alert by its index."""
    st.session_state.alerts_df = st.session_state.alerts_df.drop(index).reset_index(drop=True)

def backup_alerts(alerts_df):
    """Backup the alerts DataFrame to a JSON in memory."""
    return alerts_df.to_json(orient='records').encode('utf-8')

def restore_alerts(uploaded_file):
    """Restore the alerts DataFrame from an uploaded JSON file."""
    try:
        df = pd.read_json(uploaded_file)
        required_columns = {'Ticker', 'Condition', 'Threshold', 'Created At', 'Email'}
        if not required_columns.issubset(df.columns):
            st.error("Uploaded JSON does not contain the required columns.")
            return None
        return df
    except Exception as e:
        st.error(f"Error restoring alerts: {e}")
        return None

def check_and_trigger_alerts(updated_df, alerts_df):
    """Check for alert conditions and trigger notifications."""
    triggered_alerts = []
    for idx, alert in alerts_df.iterrows():
        ticker = alert['Ticker']
        condition = alert['Condition']
        threshold = alert['Threshold']
        email = alert['Email']
        current_perf = updated_df.loc[updated_df['Ticker'] == ticker, 'Perf_Pct'].values[0]
        if condition == "Above" and current_perf > threshold:
            message = f"🚨 **{ticker}** performance is {current_perf:.2f}%, which is above {threshold}%."
            triggered_alerts.append((message, email))
        elif condition == "Below" and current_perf < threshold:
            message = f"⚠️ **{ticker}** performance is {current_perf:.2f}%, which is below {threshold}%."
            triggered_alerts.append((message, email))
    
    for alert_message, recipient in triggered_alerts:
        st.markdown(alert_message)
        # Send Email Notification
        subject = "Portfolio Alert Notification"
        body = alert_message
        # SMTP Configuration - replace with your SMTP server details
        smtp_details = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'smtp_user': 'your_email@gmail.com',
            'smtp_password': 'your_password',
            'from_email': 'your_email@gmail.com'
        }
        send_email(subject, body, recipient, smtp_details['from_email'],
                   smtp_details['smtp_server'], smtp_details['smtp_port'],
                   smtp_details['smtp_user'], smtp_details['smtp_password'])

###############################################################################
#                  2) MAIN SHOW_PORTFOLIO FUNCTION
###############################################################################

def show_portfolio():
    """
    Main function to display the portfolio page, including:
      - Transaction inputs
      - Editable transactions table
      - Aggregated portfolio table with performance metrics
      - Auto-refresh functionality
      - Interactive charts and visualizations
      - Backup and Restore functionality
      - Performance alerts
      - Risk metrics and report exporting
    """

    # Initialize transactions and alerts
    initialize_transactions()
    initialize_alerts()
    transactions_df = st.session_state.transactions_df
    alerts_df = st.session_state.alerts_df

    # Sidebar for navigation and settings
    with st.sidebar:
        st.header("Settings")
        theme_color = st.color_picker("Select Theme Color", value="#1f77b4", help="Choose the primary color for the app.")
        st.markdown(
            f"""
            <style>
            .css-1d391kg {{
                background-color: {theme_color};
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        # Dark Mode Toggle
        dark_mode = st.checkbox("Enable Dark Mode", value=False, help="Toggle dark mode for the application.")
        if dark_mode:
            st.markdown(
                """
                <style>
                body {
                    background-color: #2e2e2e;
                    color: #ffffff;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        st.markdown("---")

    # 1) Add New Transaction
    st.header("🆕 Add New Transaction")
    with st.form("Add Transaction Form", clear_on_submit=True):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            # Improved with tooltip for better guidance
            new_ticker = st.text_input("Ticker Symbol", placeholder="e.g., AAPL", help="Enter the stock ticker symbol.").upper()
        with col2:
            new_date = st.date_input("Purchase Date", max_value=date.today(), help="Select the date of purchase.")
        with col3:
            new_price = st.number_input("Purchase Price ($)", min_value=0.0, format="%.2f", help="Enter the purchase price per share.")
        with col4:
            new_quantity = st.number_input("Quantity", min_value=1, step=1, help="Enter the number of shares purchased.")
        with col5:
            label_options = get_unique_labels()
            new_label = st.selectbox("Label", options=label_options + ["Add New Label"], index=0, help="Assign a label to categorize the transaction.")
            if new_label == "Add New Label":
                new_label = st.text_input("Enter New Label", placeholder="e.g., Tech Stocks", help="Enter a new label name.")
        with col6:
            new_color = st.color_picker("Row Color", value="#FFFFFF", help="Choose a color to highlight the transaction row.")
        submitted = st.form_submit_button("➕ Add Transaction")

    if submitted:
        errors = validate_transaction_data(new_ticker, new_price, new_quantity, new_date)
        if errors:
            for error in errors:
                st.error(f"⚠️ {error}")
        else:
            if not new_label:
                new_label = "Unlabeled"
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
            st.success(f"✅ Added transaction for **{new_ticker}**!")

    st.markdown("---")

    # 2) Editable Transactions Table
    st.header("📋 All Transactions")
    if not transactions_df.empty:
        # Define AgGrid options
        gb_trans = GridOptionsBuilder.from_dataframe(transactions_df)
        gb_trans.configure_default_column(editable=True, resizable=True, sortable=True, filter=True)
        
        # Add Delete button using custom cell renderer
        delete_button = JsCode("""
        class BtnCellRenderer {
            init(params) {
                this.params = params;
                this.eGui = document.createElement('button');
                this.eGui.innerHTML = '🗑️';
                this.eGui.style.border = 'none';
                this.eGui.style.background = 'transparent';
                this.eGui.style.cursor = 'pointer';
                this.eGui.addEventListener('click', () => {
                    const rowIndex = params.rowIndex;
                    const confirmDelete = confirm("Are you sure you want to delete this transaction?");
                    if (confirmDelete) {
                        params.api.updateRowData({ remove: [params.node.data] });
                        // Trigger Streamlit to rerun
                        window.location.reload();
                    }
                });
            }
            getGui() {
                return this.eGui;
            }
            refresh() {}
        }
        """)
        gb_trans.configure_column("Actions", headerName="", cellRenderer=delete_button, width=50, sortable=False, filter=False)
        if "Actions" not in transactions_df.columns:
            transactions_df["Actions"] = ""
        gridOptions_trans = gb_trans.build()
        
        # Render AgGrid
        grid_response = AgGrid(
            transactions_df,
            gridOptions=gridOptions_trans,
            data_return_mode=DataReturnMode.AS_INPUT,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            theme="streamlit",
            height=400
        )

        # Update session state with edited data
        edited_df = pd.DataFrame(grid_response["data"])
        edited_df = edited_df.dropna(subset=['Ticker'])  # Remove rows where Ticker is deleted
        st.session_state.transactions_df = edited_df.drop(columns=['Actions'], errors='ignore').reset_index(drop=True)

        # Backup and Restore Transactions
        st.subheader("🔒 Backup and Restore Transactions")
        backup_col1, backup_col2 = st.columns(2)
        with backup_col1:
            csv_data = backup_transactions(st.session_state.transactions_df)
            st.download_button(
                label="📥 Download Transactions Backup (CSV)",
                data=csv_data,
                file_name="transactions_backup.csv",
                mime="text/csv"
            )
        with backup_col2:
            uploaded_file = st.file_uploader("📤 Upload Transactions Backup (CSV)", type=["csv"])
            if uploaded_file is not None:
                restored_df = restore_transactions(uploaded_file)
                if restored_df is not None:
                    st.session_state.transactions_df = restored_df
                    st.success("✅ Transactions successfully restored!")

        # Bulk Upload Transactions
        st.subheader("📤 Bulk Upload Transactions")
        bulk_upload_file = st.file_uploader("Upload Transactions CSV", type=["csv"], key="bulk_upload")
        if bulk_upload_file is not None:
            try:
                bulk_df = pd.read_csv(bulk_upload_file)
                required_columns = {'Ticker','Date','Buy Price','Quantity','Color','Label'}
                if not required_columns.issubset(bulk_df.columns):
                    st.error("Uploaded CSV does not contain the required columns.")
                else:
                    bulk_df['Date'] = pd.to_datetime(bulk_df['Date']).dt.date
                    st.session_state.transactions_df = pd.concat([
                        st.session_state.transactions_df,
                        bulk_df
                    ], ignore_index=True)
                    st.success("✅ Bulk transactions successfully added!")
            except Exception as e:
                st.error(f"Error uploading bulk transactions: {e}")

        # Download Transactions CSV
        csv_data = st.session_state.transactions_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Transactions CSV",
            data=csv_data,
            file_name="transactions.csv",
            mime="text/csv"
        )
    else:
        st.info("No transactions recorded yet. Add your first transaction above!")

    st.markdown("---")

    # 3) Aggregate Transactions to Portfolio
    portfolio_df = aggregate_transactions()
    if portfolio_df.empty:
        st.warning("No transactions to aggregate into a portfolio.")
        return

    # Update Prices and Performance
    updated_df = update_prices_and_performance(portfolio_df)

    # 4) Auto-Refresh Functionality
    st.header("🔄 Price Refresh Settings")
    with st.expander("Configure Auto-Refresh"):
        auto_refresh = st.checkbox("Enable Auto-Refresh", value=False, help="Automatically refresh prices at set intervals.")
        refresh_interval = st.number_input("Refresh Interval (seconds)", min_value=10, max_value=3600, value=60, step=10, help="Set the interval for auto-refresh.")
        if auto_refresh:
            count = st_autorefresh(interval=refresh_interval * 1000, limit=100000, key="price_autorefresh")
            st.caption(f"Auto-refreshed {count} times so far.")
    if st.button("🔄 Manual Refresh"):
        st.experimental_rerun()

    st.markdown("---")

    # 5) Display Aggregated Portfolio
    st.header("📊 Aggregated Portfolio")
    display_columns = [
        "Ticker", "Quantity", "Buy_Price", "Current Price",
        "Perf_Pct", "day_Pct", "last_Pct", "Update date", "Label", "Market Value", "Total Investment", "Return"
    ]
    display_df = updated_df[display_columns + ["Color"]].copy()

    # Define row styling based on Color
    row_style = JsCode("""
    function(params) {
        if (params.data.Color) {
            return { 'background-color': params.data.Color };
        }
        return {};
    }
    """)

    # Define cell styling for performance metrics
    perf_style = JsCode("""
    function(params) {
        if (params.value < 0) {
            return { 'color': '#FF3333' };
        } else if (params.value > 0) {
            return { 'color': '#33AA33' };
        }
        return { 'color': 'black' };
    }
    """)

    # Configure AgGrid for portfolio display
    gb_pf = GridOptionsBuilder.from_dataframe(display_df)
    gb_pf.configure_default_column(resizable=True, sortable=True, filter=True)
    gb_pf.configure_column("Color", hide=True)
    gb_pf.configure_column("Ticker", pinned='left')

    for col in ["Perf_Pct", "day_Pct", "last_Pct", "Return"]:
        gb_pf.configure_column(col, cellStyle=perf_style)

    # Format numeric columns
    float_cols = ["Buy_Price", "Current Price", "Perf_Pct", "day_Pct", "last_Pct", "Market Value", "Total Investment", "Return"]
    for col in float_cols:
        gb_pf.configure_column(
            col,
            type=["numericColumn"],
            valueFormatter=JsCode("""
            function(params) {
                if (params.value == null || isNaN(params.value)) {
                    return '';
                }
                return params.value.toFixed(2);
            }
            """)
        )

    # Additional feature: Sort by Market Value
    gb_pf.configure_column("Market Value", sort='desc')

    gridOptions_pf = gb_pf.build()
    gridOptions_pf["getRowStyle"] = row_style

    # Render AgGrid for portfolio
    AgGrid(
        display_df,
        gridOptions=gridOptions_pf,
        data_return_mode=DataReturnMode.AS_INPUT,
        update_mode=GridUpdateMode.NO_UPDATE,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=True,
        theme="streamlit",
        height=500
    )

    # 6) Portfolio Summary Metrics and Risk Analysis
    st.header("📈 Portfolio Summary & Risk Metrics")
    total_investment = (updated_df['Total Investment']).sum()
    current_value = (updated_df['Market Value']).sum()
    total_return_cash = (updated_df['Return']).sum()
    total_return_pct = ((current_value / total_investment - 1) * 100) if total_investment > 0 else 0

    # Risk Metrics
    # Calculate standard deviation of portfolio
    portfolio_returns = []
    for ticker in updated_df['Ticker']:
        hist = fetch_ticker_history(ticker, period="1y", interval="1d")
        if not hist.empty:
            daily_returns = hist['Close'].pct_change().dropna()
            portfolio_returns.append(daily_returns)
    if portfolio_returns:
        combined_returns = pd.concat(portfolio_returns, axis=1)
        combined_returns.fillna(0, inplace=True)
        portfolio_std = combined_returns.mean(axis=1).std() * np.sqrt(252)  # Annualized standard deviation
        sharpe_ratio = (combined_returns.mean().mean() * 252) / portfolio_std if portfolio_std != 0 else np.nan
    else:
        portfolio_std = np.nan
        sharpe_ratio = np.nan

    # Additional Risk Metrics: Beta and VaR
    # Assuming benchmark as S&P 500
    benchmark = fetch_ticker_history("^GSPC", period="1y", interval="1d")
    if not benchmark.empty:
        benchmark_returns = benchmark['Close'].pct_change().dropna()
        portfolio_returns_combined = combined_returns.mean(axis=1).dropna()
        benchmark_returns = benchmark_returns.reindex(portfolio_returns_combined.index).dropna()
        common_index = portfolio_returns_combined.index.intersection(benchmark_returns.index)
        portfolio_returns_final = portfolio_returns_combined.loc[common_index]
        benchmark_returns_final = benchmark_returns.loc[common_index]
        covariance = np.cov(portfolio_returns_final, benchmark_returns_final)[0][1]
        benchmark_variance = np.var(benchmark_returns_final)
        beta = covariance / benchmark_variance if benchmark_variance != 0 else np.nan
    else:
        beta = np.nan

    # Value at Risk (VaR) at 95% confidence
    VaR = np.percentile(combined_returns.mean(axis=1), 5) * np.sqrt(252) * current_value

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Investment", f"${total_investment:,.2f}")
    col2.metric("Current Value", f"${current_value:,.2f}")
    col3.metric("Total Return", f"${total_return_cash:,.2f} ({total_return_pct:.2f}%)")
    col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "N/A")
    col5.metric("Beta", f"{beta:.2f}" if not np.isnan(beta) else "N/A")

    st.subheader("📊 Portfolio Risk Metrics")
    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        st.metric("Annualized Volatility (Std Dev)", f"{portfolio_std:.2f}%" if not np.isnan(portfolio_std) else "N/A")
    with risk_col2:
        st.metric("Value at Risk (VaR 95%)", f"${VaR:,.2f}" if not np.isnan(VaR) else "N/A")

    # Portfolio Summary by Label
    st.subheader("📊 Summary by Label")
    grouped_summary = updated_df.groupby('Label').agg(
        Total_Investment=pd.NamedAgg(column='Total Investment', aggfunc='sum'),
        Current_Value=pd.NamedAgg(column='Market Value', aggfunc='sum'),
        Total_Return_Cash=pd.NamedAgg(column='Return', aggfunc='sum')
    ).reset_index()
    grouped_summary['Total_Return_%'] = ((grouped_summary['Current_Value'] / grouped_summary['Total_Investment']) - 1) * 100
    st.dataframe(grouped_summary.style.format({
        'Total_Investment': "${:,.2f}",
        'Current_Value': "${:,.2f}",
        'Total_Return_Cash': "${:,.2f}",
        'Total_Return_%': "{:.2f}%"
    }), use_container_width=True)

    st.markdown("---")

    # 7) Charts & Visualizations
    st.header("📊 Charts & Visualizations")

    # 7.A) Portfolio Breakdown by Label
    with st.container():
        st.subheader("📌 Portfolio Breakdown by Label")
        if not grouped_summary.empty:
            chart_data = grouped_summary[['Label', 'Current_Value']].fillna(0)

            # Treemap
            treemap_col1, treemap_col2 = st.columns(2)
            with treemap_col1:
                st.markdown("**Treemap**")
                treemap = alt.Chart(chart_data).mark_rect().encode(
                    alt.X('Label:N', title=None, axis=None),
                    alt.Y('Current_Value:Q', title=None),
                    alt.Color('Current_Value:Q', scale=alt.Scale(scheme='greens')),
                    tooltip=['Label', 'Current_Value']
                ).properties(
                    width=350,
                    height=350
                ).interactive()
                st.altair_chart(treemap, use_container_width=True)

            # Dynamic Pie Chart with Filtering
            with treemap_col2:
                st.markdown("**Interactive Pie Chart**")
                pie_chart = alt.Chart(chart_data).mark_arc().encode(
                    theta=alt.Theta(field='Current_Value', type='quantitative'),
                    color=alt.Color(field='Label', type='nominal'),
                    tooltip=['Label', 'Current_Value']
                ).properties(
                    width=350,
                    height=350
                ).interactive()
                st.altair_chart(pie_chart, use_container_width=True)
        else:
            st.info("No data available for Portfolio Breakdown charts.")

    # 7.B) Compare Ticker Performance
    with st.container():
        st.subheader("📈 Compare Ticker Performance (%)")
        chart_df_perf = updated_df[['Ticker', 'Perf_Pct']].dropna()
        if not chart_df_perf.empty:
            chart_df_perf['Perf_Pct'] = chart_df_perf['Perf_Pct'].round(2)
            perf_bar = alt.Chart(chart_df_perf).mark_bar().encode(
                x=alt.X('Ticker', sort='-y'),
                y=alt.Y('Perf_Pct', title='Performance (%)'),
                color='Ticker',
                tooltip=['Ticker', 'Perf_Pct']
            ).properties(
                width=700,
                height=400
            ).interactive()
            st.altair_chart(perf_bar, use_container_width=True)
        else:
            st.info("No performance data to display.")

    # 7.C) Portfolio Value Over Time
    with st.container():
        st.subheader("📈 Portfolio Value Over Time")
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)  # Extended to 1 year for better trend analysis
        portfolio_value_df = fetch_daily_portfolio_value(
            updated_df['Ticker'].tolist(),
            updated_df['Quantity'].tolist(),
            start_date=start_date.date(),
            end_date=end_date.date()
        )
        if not portfolio_value_df.empty:
            # Adding moving average for forecasting
            portfolio_value_df['30-Day MA'] = portfolio_value_df['Value'].rolling(window=30).mean()

            line_chart = alt.Chart(portfolio_value_df).mark_line(point=True).encode(
                x=alt.X('Date:T', title='Date'),
                y=alt.Y('Value:Q', title='Portfolio Value ($)'),
                tooltip=['Date', 'Value']
            ).properties(
                width=700,
                height=400,
                title="Total Portfolio Value (Last 365 Days)"
            )
            ma_line = alt.Chart(portfolio_value_df).mark_line(strokeDash=[5,5], color='orange').encode(
                x='Date:T',
                y='30-Day MA:Q',
                tooltip=['Date', '30-Day MA']
            )
            combined_chart = line_chart + ma_line
            st.altair_chart(combined_chart, use_container_width=True)
        else:
            st.info("No data available to display Portfolio Value Over Time.")

    # 7.D) Bubble Chart: Ticker vs Perf (%) with Size = Position Value
    with st.container():
        st.subheader("🟠 Bubble Chart: Ticker vs Performance (%)")
        bubble_df = updated_df[['Ticker','Perf_Pct','Quantity','Current Price']].dropna()
        if not bubble_df.empty:
            bubble_df['Position_Value'] = bubble_df['Quantity'] * bubble_df['Current Price']
            bubble_chart = alt.Chart(bubble_df).mark_circle().encode(
                x=alt.X('Ticker', sort=None, title='Ticker'),
                y=alt.Y('Perf_Pct', title='Performance (%)'),
                size=alt.Size('Position_Value', title='Position Value ($)', scale=alt.Scale(range=[100, 1000])),
                color='Ticker',
                tooltip=['Ticker', 'Perf_Pct', 'Quantity', 'Current Price', 'Position_Value']
            ).properties(
                width=700,
                height=400,
                title="Ticker vs Performance (%) with Position Value"
            ).interactive()
            st.altair_chart(bubble_chart, use_container_width=True)
        else:
            st.info("No data available for Bubble Chart.")

    # 7.E) Box Plot: Performance (%) by Label
    with st.container():
        st.subheader("📦 Box Plot: Performance (%) by Label")
        box_df = updated_df[['Label','Perf_Pct']].dropna()
        if not box_df.empty:
            box_plot = alt.Chart(box_df).mark_boxplot().encode(
                x=alt.X('Label:O', title='Label'),
                y=alt.Y('Perf_Pct:Q', title='Performance (%)'),
                color='Label'
            ).properties(
                width=700,
                height=400,
                title="Distribution of Performance (%) by Label"
            )
            st.altair_chart(box_plot, use_container_width=True)
        else:
            st.info("No performance data available for Box Plot.")

    # 7.F) Histogram: Distribution of Performance (%)
    with st.container():
        st.subheader("📊 Histogram: Distribution of Performance (%)")
        hist_df = updated_df[['Perf_Pct']].dropna()
        if not hist_df.empty:
            histogram = alt.Chart(hist_df).mark_bar().encode(
                alt.X('Perf_Pct', bin=alt.Bin(maxbins=20), title='Performance (%)'),
                y=alt.Y('count()', title='Number of Tickers'),
                tooltip=['count()']
            ).properties(
                width=700,
                height=400,
                title="Distribution of Performance (%)"
            )
            st.altair_chart(histogram, use_container_width=True)
        else:
            st.info("No performance data available for Histogram.")

    # 7.G) Scatter Plot: Quantity vs. Performance (%)
    with st.container():
        st.subheader("🔍 Scatter Plot: Quantity vs. Performance (%)")
        scatter_df = updated_df[['Ticker', 'Quantity', 'Perf_Pct', 'Current Price']].dropna()
        scatter_df = scatter_df[scatter_df['Quantity'] < 500]  # Filter out outliers
        if not scatter_df.empty:
            scatter_plot = alt.Chart(scatter_df).mark_circle(size=60).encode(
                x=alt.X('Quantity', title='Quantity Held'),
                y=alt.Y('Perf_Pct', title='Performance (%)'),
                color='Ticker',
                tooltip=['Ticker', 'Quantity', 'Perf_Pct', 'Current Price']
            ).properties(
                width=700,
                height=400,
                title="Quantity vs. Performance (%)"
            ).interactive()
            st.altair_chart(scatter_plot, use_container_width=True)
        else:
            st.info("No data available for Scatter Plot.")

    # 7.H) Heatmap: Correlation Between Stock Performances
    with st.container():
        st.subheader("🔥 Heatmap: Correlation Between Stock Performances")
        tickers = updated_df['Ticker'].tolist()
        if tickers:
            correlation_df = pd.DataFrame()
            for ticker in tickers:
                hist = fetch_ticker_history(ticker, period="1y", interval="1d")
                if not hist.empty:
                    hist['Return'] = hist['Close'].pct_change()
                    correlation_df[ticker] = hist['Return']
            if not correlation_df.empty:
                corr_matrix = correlation_df.corr()
                corr_df_melted = corr_matrix.reset_index().melt('index')
                corr_df_melted.columns = ['Ticker 1', 'Ticker 2', 'Correlation']
                heatmap = alt.Chart(corr_df_melted).mark_rect().encode(
                    x=alt.X('Ticker 1:O', title='Ticker'),
                    y=alt.Y('Ticker 2:O', title='Ticker'),
                    color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='viridis'), title='Correlation'),
                    tooltip=['Ticker 1', 'Ticker 2', alt.Tooltip('Correlation:Q', format=".2f")]
                ).properties(
                    width=700,
                    height=700,
                    title="Correlation Heatmap of Stock Performances"
                )
                st.altair_chart(heatmap, use_container_width=True)
            else:
                st.info("Insufficient data to generate correlation heatmap.")
        else:
            st.info("No tickers available for correlation analysis.")

    # 7.I) Candlestick Chart for Selected Ticker
    with st.container():
        st.subheader("📉 Candlestick Chart for Selected Ticker")
        selected_ticker = st.selectbox("Select Ticker", options=updated_df['Ticker'].unique().tolist(), help="Choose a ticker to view its candlestick chart.")
        if selected_ticker:
            hist = fetch_ticker_history(selected_ticker, period="3mo", interval="1d")
            if not hist.empty:
                hist = hist.reset_index()
                hist['Date'] = pd.to_datetime(hist['Date'])
                candlestick = alt.Chart(hist).mark_rule().encode(
                    x='Date:T',
                    y='Low:Q',
                    y2='High:Q',
                    tooltip=['Date', 'Low', 'High']
                ) + alt.Chart(hist).mark_bar().encode(
                    x='Date:T',
                    y='Open:Q',
                    y2='Close:Q',
                    color=alt.condition("datum.Open > datum.Close",
                                        alt.value("#FF3333"), alt.value("#33AA33")),
                    tooltip=['Date', 'Open', 'Close']
                ).properties(
                    width=700,
                    height=400,
                    title=f"Candlestick Chart for {selected_ticker}"
                )
                st.altair_chart(candlestick, use_container_width=True)
            else:
                st.info(f"No historical data available for {selected_ticker}.")

    # 8) Performance Alerts
    st.header("🔔 Performance Alerts")
    st.write("Set up alerts for stock performances.")

    with st.expander("📑 Configure Alerts"):
        if updated_df.empty:
            st.info("No data available to set alerts.")
        else:
            alert_col1, alert_col2, alert_col3, alert_col4 = st.columns(4)
            with alert_col1:
                alert_ticker = st.selectbox("Select Ticker for Alert", options=updated_df['Ticker'].unique().tolist(), key='alert_ticker')
            with alert_col2:
                alert_condition = st.selectbox("Condition", options=["Above", "Below"], key='alert_condition')
            with alert_col3:
                alert_threshold = st.number_input("Threshold (%)", value=0.0, step=0.1, key='alert_threshold', help="Set the performance threshold for the alert.")
            with alert_col4:
                alert_email = st.text_input("Notification Email", value="", help="Enter the email to receive alert notifications.")
            alert_submit = st.button("🔔 Set Alert")
            if alert_submit:
                if not alert_email:
                    st.error("Please provide a valid email address for notifications.")
                else:
                    add_alert(alert_ticker, alert_condition, alert_threshold, alert_email)
                    st.success(f"✅ Alert set for **{alert_ticker}**: {alert_condition} {alert_threshold}%")

    if not alerts_df.empty:
        st.subheader("📋 Current Alerts")
        # Define AgGrid options for alerts
        gb_alerts = GridOptionsBuilder.from_dataframe(alerts_df)
        gb_alerts.configure_default_column(editable=False, resizable=True, sortable=True, filter=True)
        
        # Add Delete button using custom cell renderer
        delete_alert_button = JsCode("""
        class BtnCellRenderer {
            init(params) {
                this.params = params;
                this.eGui = document.createElement('button');
                this.eGui.innerHTML = '🗑️';
                this.eGui.style.border = 'none';
                this.eGui.style.background = 'transparent';
                this.eGui.style.cursor = 'pointer';
                this.eGui.addEventListener('click', () => {
                    const rowIndex = params.rowIndex;
                    const confirmDelete = confirm("Are you sure you want to delete this alert?");
                    if (confirmDelete) {
                        params.api.updateRowData({ remove: [params.node.data] });
                        // Trigger Streamlit to rerun
                        window.location.reload();
                    }
                });
            }
            getGui() {
                return this.eGui;
            }
            refresh() {}
        }
        """)
        gb_alerts.configure_column("Actions", headerName="", cellRenderer=delete_alert_button, width=50, sortable=False, filter=False)
        if "Actions" not in alerts_df.columns:
            alerts_df["Actions"] = ""
        gridOptions_alerts = gb_alerts.build()
        
        # Render AgGrid
        alerts_response = AgGrid(
            alerts_df,
            gridOptions=gridOptions_alerts,
            data_return_mode=DataReturnMode.AS_INPUT,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            theme="streamlit",
            height=200
        )

        # Update session state with edited alerts
        edited_alerts_df = pd.DataFrame(alerts_response["data"])
        edited_alerts_df = edited_alerts_df.dropna(subset=['Ticker'])  # Remove rows where Alert is deleted
        st.session_state.alerts_df = edited_alerts_df.drop(columns=['Actions'], errors='ignore').reset_index(drop=True)

        # Backup and Restore Alerts
        st.subheader("🔒 Backup and Restore Alerts")
        backup_col_a, backup_col_b = st.columns(2)
        with backup_col_a:
            json_data = backup_alerts(st.session_state.alerts_df)
            st.download_button(
                label="📥 Download Alerts Backup (JSON)",
                data=json_data,
                file_name="alerts_backup.json",
                mime="application/json"
            )
        with backup_col_b:
            uploaded_alerts_file = st.file_uploader("📤 Upload Alerts Backup (JSON)", type=["json"])
            if uploaded_alerts_file is not None:
                restored_alerts_df = restore_alerts(uploaded_alerts_file)
                if restored_alerts_df is not None:
                    st.session_state.alerts_df = restored_alerts_df
                    st.success("✅ Alerts successfully restored!")

    else:
        st.info("No alerts set yet. Configure your first alert above.")

    # Trigger Alerts
    st.subheader("⚠️ Triggered Alerts")
    if not alerts_df.empty:
        check_and_trigger_alerts(updated_df, alerts_df)
    else:
        st.info("No alerts to trigger.")

    st.markdown("---")

    # 9) Notifications for High/Low Performers
    st.header("📢 Notifications")
    if not updated_df.empty:
        high_perf_threshold = st.slider("High Performance Threshold (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0, help="Set the threshold to identify high-performing stocks.")
        low_perf_threshold = st.slider("Low Performance Threshold (%)", min_value=-100.0, max_value=0.0, value=-20.0, step=1.0, help="Set the threshold to identify low-performing stocks.")
        
        high_perf = updated_df[updated_df['Perf_Pct'] >= high_perf_threshold]
        low_perf = updated_df[updated_df['Perf_Pct'] <= low_perf_threshold]
        
        if not high_perf.empty:
            st.success(f"🎉 The following stocks have exceeded a performance of {high_perf_threshold}%:")
            st.table(high_perf[['Ticker', 'Perf_Pct']])
        else:
            st.info(f"No stocks have exceeded a performance of {high_perf_threshold}%.")

        if not low_perf.empty:
            st.error(f"⚠️ The following stocks have fallen below a performance of {low_perf_threshold}%:")
            st.table(low_perf[['Ticker', 'Perf_Pct']])
        else:
            st.info(f"No stocks have fallen below a performance of {low_perf_threshold}%.")

    else:
        st.info("No data available for Notifications.")

    # 10) Report Exporting
    st.markdown("---")
    st.header("📝 Export Portfolio Report")
    report_col1, report_col2 = st.columns(2)
    with report_col1:
        if st.button("📄 Download PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="Portfolio Report", ln=True, align='C')
            pdf.ln(10)
            
            # Add Portfolio Summary
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt="Portfolio Summary", ln=True)
            pdf.set_font("Arial", size=12)
            summary_text = (
                f"Total Investment: ${total_investment:,.2f}\n"
                f"Current Value: ${current_value:,.2f}\n"
                f"Total Return: ${total_return_cash:,.2f} ({total_return_pct:.2f}%)\n"
                f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
                f"Beta: {beta:.2f}\n"
                f"Annualized Volatility: {portfolio_std:.2f}%\n"
                f"Value at Risk (VaR 95%): ${VaR:,.2f}"
            )
            for line in summary_text.split('\n'):
                pdf.cell(200, 10, txt=line, ln=True)
            pdf.ln(10)
            
            # Add Top Performers
            if not updated_df.empty:
                top_performers = updated_df.sort_values(by='Perf_Pct', ascending=False).head(5)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, txt="Top Performing Stocks", ln=True)
                pdf.set_font("Arial", size=12)
                for idx, row in top_performers.iterrows():
                    pdf.cell(200, 10, txt=f"{row['Ticker']}: {row['Perf_Pct']:.2f}%", ln=True)
                pdf.ln(10)
            
            # Add Bottom Performers
            if not updated_df.empty:
                bottom_performers = updated_df.sort_values(by='Perf_Pct').head(5)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 10, txt="Bottom Performing Stocks", ln=True)
                pdf.set_font("Arial", size=12)
                for idx, row in bottom_performers.iterrows():
                    pdf.cell(200, 10, txt=f"{row['Ticker']}: {row['Perf_Pct']:.2f}%", ln=True)
                pdf.ln(10)
            
            # Generate PDF and provide download
            pdf_data = pdf.output(dest='S').encode('latin1')  # Get PDF as bytes
            st.download_button(
                label="📥 Download Report",
                data=pdf_data,
                file_name="portfolio_report.pdf",
                mime="application/pdf"
            )
    with report_col2:
        st.info("Click the button to download a PDF report of your portfolio summary and top/bottom performers.")

   
