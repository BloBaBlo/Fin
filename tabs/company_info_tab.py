# tabs/company_info_tab.py

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def show_company_info(ticker_input):
    st.title("Company Information Dashboard")

    if ticker_input:
        # Instantiate the Ticker object
        ticker = yf.Ticker(ticker_input)
        di = ticker.info
        longname = di.get("longName", ticker_input)

        st.subheader(f"{longname}")
        # --- 1-Year Price History + % Performance ---
        price_data = ticker.history(period="1y")
        if not price_data.empty:
            # Compute % Change relative to the first day in the data
            price_data["PercentChange"] = (price_data["Close"] / price_data["Close"].iloc[0] - 1) * 100

            # Create subplot with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Price trace
            fig.add_trace(
                go.Scatter(
                    x=price_data.index, 
                    y=price_data["Close"], 
                    name="Close Price",
                    line=dict(color='blue')
                ),
                secondary_y=False
            )
            
            # % Change trace
            fig.add_trace(
                go.Scatter(
                    x=price_data.index, 
                    y=price_data["PercentChange"], 
                    name="Price Performance (%)",
                    line=dict(color='green')
                ),
                secondary_y=True
            )
            
            fig.update_layout(
                title=f"{ticker_input} - 1 Year Price History & % Performance",
                legend=dict(x=0, y=1.15, orientation="h")  # Position legend above
            )
            fig.update_xaxes(title_text="Date")
            fig.update_yaxes(title_text="Price (USD)", secondary_y=False)
            fig.update_yaxes(title_text="Performance (%)", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No historical price data available for charting.")

        # Create Streamlit tabs
        tab_info, tab_bs, tab_cal, tab_cf, tab_inst_hold, tab_maj_hold, tab_mf_hold, tab_insider, tab_insider_tx = st.tabs([
            "Company Info",
            "Balance Sheet",
            "Calendar",
            "Cashflow",
            "Institutional Holders",
            "Major Holders",
            "Mutual Fund Holders",
            "Insider Purchases/Roster",
            "Insider Transactions"
        ])

        # --- Tab 1: Company Info ---
        with tab_info:
            st.subheader(f"Company Info for {ticker_input}")
            info_data = ticker.info
            
            if not info_data or info_data is None:
                st.warning("No fundamental company data available. It might be an ETF or a ticker that doesn't provide info.")
            else:
                # Display high-level fields in columns
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Industry**: {info_data.get('industry', 'N/A')}")
                    st.markdown(f"**Sector**: {info_data.get('sector', 'N/A')}")
                    st.markdown(f"**Country**: {info_data.get('country', 'N/A')}")
                    st.markdown(f"**Phone**: {info_data.get('phone', 'N/A')}")
                    st.markdown(f"**Website**: {info_data.get('website', 'N/A')}")

                with col2:
                    st.markdown("**Long Business Summary**:")
                    st.write(info_data.get("longBusinessSummary", "N/A"))

                # ---- A. Key Metrics Bar Chart ----
                metrics = {
                    "Market Cap": info_data.get("marketCap", None),
                    "Beta": info_data.get("beta", None),
                    "Forward PE": info_data.get("forwardPE", None),
                    "Trailing PE": info_data.get("trailingPE", None),
                    "Dividend Rate": info_data.get("dividendRate", None),
                    "Dividend Yield": info_data.get("dividendYield", None),
                    "Profit Margins": info_data.get("profitMargins", None),
                    "Return On Assets": info_data.get("returnOnAssets", None),
                    "Return On Equity": info_data.get("returnOnEquity", None),
                }
                metrics_filtered = {k: v for k, v in metrics.items() if v is not None}
                
                if metrics_filtered:
                    st.markdown("### Key Metrics (Chart)")
                    metrics_df = pd.DataFrame(list(metrics_filtered.items()), columns=["Metric", "Value"])
                    fig_metrics = px.bar(
                        metrics_df, 
                        x="Metric", 
                        y="Value", 
                        title="Key Metrics", 
                        text="Value"
                    )
                    fig_metrics.update_layout(xaxis_title="", yaxis_title="")
                    fig_metrics.update_traces(textposition='outside')
                    st.plotly_chart(fig_metrics, use_container_width=True)
                else:
                    st.info("No numeric metrics available to chart.")

                # ---- A1. Additional Visual Statistical Analysis ----
                # Profitability Ratios
                profitability_metrics = {
                    "Profit Margins": info_data.get("profitMargins", None),
                    "Return On Assets": info_data.get("returnOnAssets", None),
                    "Return On Equity": info_data.get("returnOnEquity", None)
                }
                profitability_filtered = {k: v for k, v in profitability_metrics.items() if v is not None}
                if profitability_filtered:
                    st.markdown("### Profitability Ratios")
                    profit_df = pd.DataFrame(list(profitability_filtered.items()), columns=["Metric", "Value"])
                    fig_profit = px.bar(
                        profit_df,
                        x="Metric",
                        y="Value",
                        text="Value",
                        color="Metric",
                        title="Profitability Ratios"
                    )
                    fig_profit.update_layout(xaxis_title="", yaxis_title="")
                    fig_profit.update_traces(textposition='outside')
                    st.plotly_chart(fig_profit, use_container_width=True)

                # Dividend Info
                dividend_metrics = {
                    "Dividend Rate": info_data.get("dividendRate", None),
                    "Dividend Yield": info_data.get("dividendYield", None),
                    "Payout Ratio": info_data.get("payoutRatio", None),
                }
                dividend_filtered = {k: v for k, v in dividend_metrics.items() if v is not None}
                if dividend_filtered:
                    st.markdown("### Dividend Information")
                    div_df = pd.DataFrame(list(dividend_filtered.items()), columns=["Metric", "Value"])
                    fig_div = px.bar(
                        div_df,
                        x="Metric",
                        y="Value",
                        text="Value",
                        color="Metric",
                        title="Dividend Stats"
                    )
                    fig_div.update_layout(xaxis_title="", yaxis_title="")
                    fig_div.update_traces(textposition='outside')
                    st.plotly_chart(fig_div, use_container_width=True)

                # ---- B. Governance & Risk Indicators ----
                st.subheader("Governance & Risk Indicators")
                risk_fields = {
                    "Audit Risk": info_data.get("auditRisk", None),
                    "Board Risk": info_data.get("boardRisk", None),
                    "Compensation Risk": info_data.get("compensationRisk", None),
                    "Shareholder Rights Risk": info_data.get("shareHolderRightsRisk", None),
                    "Overall Risk": info_data.get("overallRisk", None),
                }
                risk_filtered = {k: v for k, v in risk_fields.items() if v is not None}

                if risk_filtered:
                    risk_df = pd.DataFrame(list(risk_filtered.items()), columns=["RiskType", "Score"])
                    fig_risk = px.bar(
                        risk_df,
                        x="RiskType",
                        y="Score",
                        text="Score",
                        color="RiskType",
                        title="Risk Scores"
                    )
                    fig_risk.update_layout(xaxis_title="", yaxis_title="Score (Lower is Better)")
                    fig_risk.update_traces(textposition="outside")
                    st.plotly_chart(fig_risk, use_container_width=True)
                else:
                    st.info("No governance & risk data available.")

                # ---- C. Growth Indicators ----
                st.subheader("Growth Indicators")
                growth_fields = {
                    "Earnings Growth": info_data.get("earningsGrowth", None),
                    "Revenue Growth": info_data.get("revenueGrowth", None),
                }
                growth_filtered = {k: v for k, v in growth_fields.items() if v is not None}
                if growth_filtered:
                    growth_df = pd.DataFrame(list(growth_filtered.items()), columns=["Metric", "Value"])
                    fig_growth = px.bar(
                        growth_df,
                        x="Metric",
                        y="Value",
                        text="Value",
                        color="Metric",
                        title="Growth Rates"
                    )
                    fig_growth.update_layout(xaxis_title="", yaxis_title="")
                    fig_growth.update_traces(textposition='outside')
                    st.plotly_chart(fig_growth, use_container_width=True)
                else:
                    st.info("No growth information available.")

                # ---- D. Price vs. Target ----
                st.subheader("Price vs. Target")
                price_data_dict = {
                    "Current Price": info_data.get("currentPrice", None),
                    "Target Low": info_data.get("targetLowPrice", None),
                    "Target Median": info_data.get("targetMedianPrice", None),
                    "Target Mean": info_data.get("targetMeanPrice", None),
                    "Target High": info_data.get("targetHighPrice", None),
                }
                price_filtered = {k: v for k, v in price_data_dict.items() if v is not None}
                if price_filtered:
                    df_price = pd.DataFrame(list(price_filtered.items()), columns=["Type", "Value"])
                    fig_price = px.bar(
                        df_price,
                        x="Type",
                        y="Value",
                        text="Value",
                        color="Type",
                        title="Current vs. Target Prices"
                    )
                    fig_price.update_layout(xaxis_title="", yaxis_title="USD")
                    fig_price.update_traces(textposition='outside')
                    st.plotly_chart(fig_price, use_container_width=True)
                else:
                    st.info("No price or target data available.")

                # ---- E. Company Officers & Compensation ----
                st.subheader("Company Officers & Compensation")
                officers = info_data.get("companyOfficers", [])
                if officers:
                    officers_df = pd.DataFrame(officers)
                    # Ensure these columns exist; fill if missing
                    keep_cols = ["name", "title", "age", "totalPay"]
                    for c in keep_cols:
                        if c not in officers_df.columns:
                            officers_df[c] = None

                    officers_df = officers_df[keep_cols].fillna(0)
                    st.dataframe(officers_df.style.format({"totalPay": "{:,.0f}"}))

                    # Bar chart of totalPay
                    officers_sorted = officers_df.sort_values("totalPay", ascending=False)
                    fig_officers = px.bar(
                        officers_sorted,
                        x="name",
                        y="totalPay",
                        color="title",
                        text="totalPay",
                        title="Officer Compensation (Total Pay in USD)"
                    )
                    fig_officers.update_layout(xaxis_title="Officer", yaxis_title="Total Pay (USD)")
                    fig_officers.update_traces(textposition='outside')
                    st.plotly_chart(fig_officers, use_container_width=True)
                else:
                    st.info("No officer data available.")

                # ---- F. Additional Data Visualizations ----
                st.subheader("Additional Trading & Ownership Statistics")

                # 1. Daily Price Range
                daily_range_fields = {
                    "Previous Close": info_data.get("previousClose", None),
                    "Open": info_data.get("open", None),
                    "Day High": info_data.get("dayHigh", None),
                    "Day Low": info_data.get("dayLow", None),
                    "Regular Mkt Open": info_data.get("regularMarketOpen", None),
                    "Regular Mkt Day High": info_data.get("regularMarketDayHigh", None),
                    "Regular Mkt Day Low": info_data.get("regularMarketDayLow", None),
                }
                daily_range_filtered = {k: v for k, v in daily_range_fields.items() if v is not None}
                if daily_range_filtered:
                    st.markdown("### Daily Price Range")
                    daily_range_df = pd.DataFrame(list(daily_range_filtered.items()), columns=["Label", "Value"])
                    fig_daily_range = px.bar(
                        daily_range_df,
                        x="Label",
                        y="Value",
                        text="Value",
                        title="Daily Price Range (USD)"
                    )
                    fig_daily_range.update_traces(textposition="outside")
                    st.plotly_chart(fig_daily_range, use_container_width=True)
                else:
                    st.info("No daily price range data available.")

                # 2. Volume Stats
                volume_fields = {
                    "Volume": info_data.get("volume", None),
                    "Average Volume": info_data.get("averageVolume", None),
                    "Avg Volume (10d)": info_data.get("averageVolume10days", None),
                    "Avg Daily Vol (10d)": info_data.get("averageDailyVolume10Day", None),
                }
                volume_filtered = {k: v for k, v in volume_fields.items() if v is not None}
                if volume_filtered:
                    st.markdown("### Volume Statistics")
                    volume_df = pd.DataFrame(list(volume_filtered.items()), columns=["Volume Metric", "Value"])
                    fig_volume = px.bar(
                        volume_df,
                        x="Volume Metric",
                        y="Value",
                        text="Value",
                        title="Trading Volume Stats"
                    )
                    fig_volume.update_traces(textposition='outside')
                    st.plotly_chart(fig_volume, use_container_width=True)
                else:
                    st.info("No volume data available.")

                # 3. Short Interest Stats
                short_fields = {
                    "Shares Short": info_data.get("sharesShort", None),
                    "Shares Short (Prior Mo)": info_data.get("sharesShortPriorMonth", None),
                    "Short Ratio": info_data.get("shortRatio", None),
                    "Short % of Float": info_data.get("shortPercentOfFloat", None),
                }
                short_filtered = {k: v for k, v in short_fields.items() if v is not None}
                if short_filtered:
                    st.markdown("### Short Interest Statistics")
                    short_df = pd.DataFrame(list(short_filtered.items()), columns=["Short Metric", "Value"])
                    fig_short = px.bar(
                        short_df,
                        x="Short Metric",
                        y="Value",
                        text="Value",
                        title="Short Interest"
                    )
                    fig_short.update_traces(textposition='outside')
                    st.plotly_chart(fig_short, use_container_width=True)

                # 4. Share & Ownership Stats
                shares_fields = {
                    "Float Shares": info_data.get("floatShares", None),
                    "Shares Outstanding": info_data.get("sharesOutstanding", None),
                    "Implied Shares Outstanding": info_data.get("impliedSharesOutstanding", None),
                    "Held % Insiders": info_data.get("heldPercentInsiders", None),
                    "Held % Institutions": info_data.get("heldPercentInstitutions", None),
                }
                shares_filtered = {k: v for k, v in shares_fields.items() if v is not None}
                if shares_filtered:
                    st.markdown("### Shares & Ownership")
                    shares_df = pd.DataFrame(list(shares_filtered.items()), columns=["Metric", "Value"])
                    fig_shares = px.bar(
                        shares_df,
                        x="Metric",
                        y="Value",
                        text="Value",
                        title="Shares & Ownership"
                    )
                    fig_shares.update_traces(textposition='outside')
                    st.plotly_chart(fig_shares, use_container_width=True)
                
                # ---- G. Full Info Dictionary (for reference) ----
                st.markdown("### Full Info Dictionary")
                st.json(info_data)

        # --- Tab 2: Balance Sheet ---
        with tab_bs:
            st.subheader("Balance Sheet")
            balance_sheet = ticker.balance_sheet
            if balance_sheet is not None and not balance_sheet.empty:
                st.dataframe(balance_sheet)

                # Example: Show a line chart for a few key rows (if they exist)
                key_rows = ["Total Assets", "Total Liab", "Total Stockholder Equity"]
                existing_rows = balance_sheet.index.intersection(key_rows)
                
                if not existing_rows.empty:
                    st.markdown("#### Key Items Over Time")
                    bs_selected_df = balance_sheet.loc[existing_rows].T
                    fig_bs = px.line(
                        bs_selected_df, 
                        x=bs_selected_df.index, 
                        y=bs_selected_df.columns,
                        markers=True,
                        title="Key Balance Sheet Items Over Time"
                    )
                    fig_bs.update_layout(xaxis_title="Date", yaxis_title="Amount")
                    st.plotly_chart(fig_bs, use_container_width=True)
            else:
                st.warning("No balance sheet data available.")

        # --- Tab 3: Calendar ---
        with tab_cal:
            st.subheader("Calendar")
            calendar = ticker.calendar
            if isinstance(calendar, pd.DataFrame):
            # If it's already a DataFrame, just display it
                if not calendar.empty:
                    st.dataframe(calendar)
                else:
                    st.warning("Calendar DataFrame is empty.")
            elif isinstance(calendar, dict) and calendar:
                # If it's a non-empty dict, convert to DataFrame
                calendar_df = pd.DataFrame.from_dict(calendar, orient="index", columns=["Value"])
                st.dataframe(calendar_df)
                
                # Optional: Try to visualize numeric fields
                numeric_fields = calendar_df["Value"].apply(lambda x: isinstance(x, (int, float)))
                cal_numeric_df = calendar_df[numeric_fields].reset_index()
                cal_numeric_df.columns = ["Field", "Value"]
                
                if not cal_numeric_df.empty:
                    fig_cal = px.bar(
                        cal_numeric_df,
                        x="Field",
                        y="Value",
                        title="Calendar Numeric Data",
                        text="Value"
                    )
                    fig_cal.update_layout(xaxis_title="", yaxis_title="")
                    fig_cal.update_traces(textposition='outside')
                    st.plotly_chart(fig_cal, use_container_width=True)
                else:
                    st.info("No numeric calendar data available for plotting.")

            else:
                # If it's neither a DataFrame nor a dict, just show a warning
                st.warning("No calendar data available or unsupported format.")

        # --- Tab 4: Cashflow ---
        with tab_cf:
            st.subheader("Cashflow")
            cashflow = ticker.cashflow
            if cashflow is not None and not cashflow.empty:
                st.dataframe(cashflow)

                # Example chart: Operating Cash Flow & Free Cash Flow over time
                cf_rows = ["Operating Cash Flow", "Free Cash Flow"]
                existing_rows = cashflow.index.intersection(cf_rows)
                if not existing_rows.empty:
                    st.markdown("#### Key Cash Flow Items Over Time")
                    cf_selected_df = cashflow.loc[existing_rows].T
                    fig_cf = px.line(
                        cf_selected_df, 
                        x=cf_selected_df.index, 
                        y=cf_selected_df.columns,
                        markers=True,
                        title="Operating Cash Flow & Free Cash Flow"
                    )
                    fig_cf.update_layout(xaxis_title="Date", yaxis_title="Amount")
                    st.plotly_chart(fig_cf, use_container_width=True)
            else:
                st.warning("No cashflow data available.")

        # --- Tab 5: Institutional Holders ---
        with tab_inst_hold:
            st.subheader("Institutional Holders")
            institutional_holders = ticker.institutional_holders
            if institutional_holders is not None and not institutional_holders.empty:
                st.dataframe(institutional_holders)
            else:
                st.warning("No institutional holder data available.")

        # --- Tab 6: Major Holders ---
        with tab_maj_hold:
            st.subheader("Major Holders")
            major_holders = ticker.major_holders
            
            if major_holders is not None and not major_holders.empty:
                st.dataframe(major_holders)

                # Attempt to parse it for a pie chart if the columns are interpretable
                if "Breakdown" in major_holders.columns and "Value" in major_holders.columns:
                    major_numeric = major_holders[pd.to_numeric(major_holders["Value"], errors="coerce").notnull()]
                    if not major_numeric.empty:
                        fig_mh = px.pie(
                            major_numeric,
                            names="Breakdown",
                            values="Value",
                            title="Major Holders Breakdown"
                        )
                        st.plotly_chart(fig_mh, use_container_width=True)
            else:
                st.warning("No major holder data available.")

        # --- Tab 7: Mutual Fund Holders ---
        with tab_mf_hold:
            st.subheader("Mutual Fund Holders")
            mutualfund_holders = ticker.mutualfund_holders
            if mutualfund_holders is not None and not mutualfund_holders.empty:
                st.dataframe(mutualfund_holders)
                
                # Example: bar chart for the top 5 holders by Value
                if "Value" in mutualfund_holders.columns:
                    top_5_mf = mutualfund_holders.nlargest(5, "Value")
                    fig_mf = px.bar(
                        top_5_mf,
                        x="Holder",
                        y="Value",
                        text="pctHeld",
                        title="Top 5 Mutual Fund Holders by Value"
                    )
                    fig_mf.update_layout(xaxis_title="", yaxis_title="Value")
                    fig_mf.update_traces(textposition='outside')
                    st.plotly_chart(fig_mf, use_container_width=True)
            else:
                st.warning("No mutual fund holder data available.")

        # --- Tab 8: Insider Purchases & Roster ---
        with tab_insider:
            st.subheader("Insider Purchases & Roster")
            insider_purchases = ticker.insider_purchases
            insider_roster_holders = getattr(ticker, "insider_roster_holders", None)

            # Insider Purchases
            st.write("#### Insider Purchases")
            if insider_purchases is not None and not insider_purchases.empty:
                st.dataframe(insider_purchases)
                # (Optional) Could create a chart of total shares purchased vs. sold
            else:
                st.warning("No insider purchase data available.")

            # Insider Roster Holders
            st.write("#### Insider Roster Holders")
            if insider_roster_holders is not None and not insider_roster_holders.empty:
                st.dataframe(insider_roster_holders)
            else:
                st.warning("No insider roster holder data available.")

        # --- Tab 9: Insider Transactions ---
        with tab_insider_tx:
            st.subheader("Insider Transactions")
            insider_transactions = ticker.insider_transactions
            if insider_transactions is not None and not insider_transactions.empty:
                st.dataframe(insider_transactions)

                # Example: chart total shares sold vs. purchased
                if "Shares" in insider_transactions.columns and "Text" in insider_transactions.columns:
                    # Simplistic approach: classify transaction type
                    def classify_tx(text):
                        text_lower = str(text).lower()
                        if "sale" in text_lower:
                            return "Sale"
                        elif "purchase" in text_lower or "gift" in text_lower:
                            return "Buy/Gift"
                        else:
                            return "Other"

                    insider_transactions["TransactionType"] = insider_transactions["Text"].apply(classify_tx)
                    tx_chart_data = insider_transactions.groupby("TransactionType")["Shares"].sum().reset_index()

                    fig_tx = px.bar(
                        tx_chart_data, 
                        x="TransactionType", 
                        y="Shares", 
                        text="Shares", 
                        title="Insider Transactions (Shares Summary)"
                    )
                    fig_tx.update_traces(textposition='outside')
                    st.plotly_chart(fig_tx, use_container_width=True)
            else:
                st.warning("No insider transactions data available.")
