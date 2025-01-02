# tabs/company_info_tab.py

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode


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
            st.subheader("📊 Balance Sheet")

            balance_sheet = ticker.balance_sheet

            if balance_sheet is not None and not balance_sheet.empty:
                # Transpose for better readability
                bs_display = balance_sheet.transpose()
                bs_display.index = pd.to_datetime(bs_display.index).strftime('%Y-%m-%d')

                # Styling the DataFrame

                # --- Section: Custom Visualization ---
                st.markdown("### 🔍 Custom Balance Sheet Visualization")

                # Retrieve all available items
                all_items = balance_sheet.index.tolist()

                # Define default key items that exist in the current DataFrame
                key_default_items = ["Total Assets", "Total Debt", "Stockholders Equity"]

                # Ensure default items exist in the options
                existing_default_items = [item for item in key_default_items if item in all_items]

                selected_items = st.multiselect(
                    "Select Balance Sheet Items to Visualize",
                    options=all_items,
                    default=existing_default_items
                )

                if selected_items:
                    bs_custom_df = balance_sheet.loc[selected_items].transpose()
                    bs_custom_df.index = pd.to_datetime(bs_custom_df.index).strftime('%Y-%m-%d')

                    # Choose visualization type
                    viz_type = st.selectbox(
                        "Select Visualization Type",
                        options=["Line Chart", "Bar Chart", "Area Chart"],
                        index=0
                    )

                    if viz_type == "Line Chart":
                        fig_custom = px.line(
                            bs_custom_df,
                            x=bs_custom_df.index,
                            y=bs_custom_df.columns,
                            markers=True,
                            title="Selected Balance Sheet Items Over Time",
                            labels={"value": "Amount (USD)", "index": "Date"}
                        )
                    elif viz_type == "Bar Chart":
                        fig_custom = px.bar(
                            bs_custom_df,
                            x=bs_custom_df.index,
                            y=bs_custom_df.columns,
                            title="Selected Balance Sheet Items Over Time",
                            labels={"value": "Amount (USD)", "index": "Date"},
                            barmode='group'
                        )
                    else:  # Area Chart
                        fig_custom = px.area(
                            bs_custom_df,
                            x=bs_custom_df.index,
                            y=bs_custom_df.columns,
                            title="Selected Balance Sheet Items Over Time",
                            labels={"value": "Amount (USD)", "index": "Date"}
                        )

                    fig_custom.update_layout(xaxis_title="Date", yaxis_title="Amount (USD)")
                    st.plotly_chart(fig_custom, use_container_width=True)

                st.markdown("---")

                # --- Section: Financial Ratios ---
                st.markdown("### 📈 Key Financial Ratios")

                    # Function to calculate financial ratios
                def calculate_financial_ratios(bs):
                    ratios = pd.DataFrame(index=bs.columns)

                    try:
                        # Extract necessary items using the provided balance_sheet data
                        total_assets = bs.loc["Total Assets"]
                        # Use "Total Liabilities Net Minority Interest" as total liabilities
                        total_liabilities = bs.loc["Total Liabilities Net Minority Interest"]
                        total_equity = bs.loc["Stockholders Equity"]
                        current_assets = bs.loc["Current Assets"]
                        current_liabilities = bs.loc["Current Liabilities"]
                        inventory = bs.loc["Inventory"]
                        cash = bs.loc["Cash And Cash Equivalents"]
                        long_term_debt = bs.loc["Long Term Debt"]
                        short_term_debt = bs.loc["Current Debt"]

                        # Calculate Ratios
                        ratios["Debt to Equity"] = total_liabilities / total_equity
                        ratios["Current Ratio"] = current_assets / current_liabilities
                        ratios["Quick Ratio"] = (current_assets - inventory) / current_liabilities
                        ratios["Debt to Assets"] = total_liabilities / total_assets
                        ratios["Cash Ratio"] = cash / current_liabilities
                        ratios["Debt to Capital"] = (long_term_debt + short_term_debt) / (long_term_debt + short_term_debt + total_equity)
                        ratios["Equity Ratio"] = total_equity / total_assets

                        return ratios
                    except KeyError as e:
                        st.error(f"Missing data for ratio calculation: {e}")
                        return None

                # Calculate the financial ratios
                ratios_df = calculate_financial_ratios(balance_sheet)

                if ratios_df is not None:
                    # Transpose for better readability
                    ratios_display = ratios_df.transpose()
                    ratios_display = ratios_display.round(2)

                    # Ensure all column names are strings to prevent TypeError
                    ratios_display.columns = [str(col) for col in ratios_display.columns]

                    # Display Ratios in AgGrid
                    st.markdown("#### Financial Ratios Table")
                    gb_ratios = GridOptionsBuilder.from_dataframe(ratios_display)
                    gb_ratios.configure_pagination(paginationAutoPageSize=True)
                    gb_ratios.configure_side_bar()
                    gb_ratios.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
                    grid_options_ratios = gb_ratios.build()

                    AgGrid(
                        ratios_display,
                        gridOptions=grid_options_ratios,
                        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                        update_mode=GridUpdateMode.MODEL_CHANGED,
                        enable_enterprise_modules=True,
                        height=300,
                        fit_columns_on_grid_load=True
                    )

                    # Visualization of Ratios
                    st.markdown("#### Financial Ratios Over Time")

                    # Select ratios to visualize
                    available_ratios = ratios_display.index.tolist()
                    # Define default ratios ensuring they exist in available_ratios
                    default_ratios = ["Debt to Equity", "Current Ratio", "Quick Ratio"]
                    existing_default_ratios = [ratio for ratio in default_ratios if ratio in available_ratios]

                    selected_ratios = st.multiselect(
                        "Select Financial Ratios to Visualize",
                        options=available_ratios,
                        default=existing_default_ratios
                    )

                    if selected_ratios:
                        ratios_plot_df = ratios_display.loc[selected_ratios].transpose()
                        # Convert index to string if it's a Timestamp
                        if isinstance(ratios_plot_df.index[0], pd.Timestamp):
                            ratios_plot_df.index = ratios_plot_df.index.strftime('%Y-%m-%d')
                        else:
                            ratios_plot_df.index = ratios_plot_df.index.astype(str)

                        fig_ratios = px.line(
                            ratios_plot_df,
                            x=ratios_plot_df.index,
                            y=ratios_plot_df.columns,
                            markers=True,
                            title="Selected Financial Ratios Over Time",
                            labels={"value": "Ratio", "index": "Date"}
                        )
                        fig_ratios.update_layout(xaxis_title="Date", yaxis_title="Ratio")
                        st.plotly_chart(fig_ratios, use_container_width=True)


                st.markdown("---")

                # --- Section: Detailed Insights ---
                st.markdown("### 📝 Detailed Insights")

                with st.expander("🔍 View Detailed Balance Sheet Items"):
                    st.dataframe(bs_display.style.format("${:,.2f}"))

                        # --- Section: Financial Ratios Explanations ---
                with st.expander("ℹ️ How Are These Ratios Calculated?"):
                    st.markdown("""
        ### 📖 **Understanding Key Financial Ratios**

        Financial ratios are essential tools for evaluating a company's financial health, efficiency, and performance. They provide insights into various aspects of a business by analyzing relationships between different items in the financial statements. Below is an explanation of each ratio displayed in the **Key Financial Ratios** section, including their sources from the balance sheet and what they indicate about the company.

        ---

        #### 1. **Debt to Equity Ratio**

        - **Formula**:
        \[
        \text{Debt to Equity} = \frac{\text{Total Liabilities Net Minority Interest}}{\text{Stockholders Equity}}
        \]
        
        - **Source**:
        - **Numerator**: `Total Liabilities Net Minority Interest`
        - **Denominator**: `Stockholders Equity`
        
        - **Interpretation**:
        - **Purpose**: Measures the company's financial leverage and indicates the proportion of debt used to finance the company's assets relative to equity.
        - **Insight**: 
            - A **higher ratio** suggests that the company is heavily financed by debt, which may imply higher financial risk.
            - A **lower ratio** indicates a more conservative approach with less reliance on debt.

        ---

        #### 2. **Current Ratio**

        - **Formula**:
        \[
        \text{Current Ratio} = \frac{\text{Current Assets}}{\text{Current Liabilities}}
        \]
        
        - **Source**:
        - **Numerator**: `Current Assets`
        - **Denominator**: `Current Liabilities`
        
        - **Interpretation**:
        - **Purpose**: Assesses the company's ability to meet its short-term obligations with its short-term assets.
        - **Insight**:
            - A **ratio above 1** indicates that the company has more current assets than current liabilities, suggesting good short-term financial health.
            - A **ratio below 1** may signal potential liquidity issues.

        ---

        #### 3. **Quick Ratio**

        - **Formula**:
        \[
        \text{Quick Ratio} = \frac{\text{Current Assets} - \text{Inventory}}{\text{Current Liabilities}}
        \]
        
        - **Source**:
        - **Numerator**: `Current Assets` minus `Inventory`
        - **Denominator**: `Current Liabilities`
        
        - **Interpretation**:
        - **Purpose**: Evaluates the company's ability to meet its short-term obligations without relying on the sale of inventory.
        - **Insight**:
            - A **higher quick ratio** indicates better liquidity and a stronger position to cover liabilities.
            - A **lower quick ratio** may suggest reliance on inventory sales to meet obligations, which could be risky if inventory is not quickly convertible to cash.

        ---

        #### 4. **Debt to Assets Ratio**

        - **Formula**:
        \[
        \text{Debt to Assets} = \frac{\text{Total Liabilities Net Minority Interest}}{\text{Total Assets}}
        \]
        
        - **Source**:
        - **Numerator**: `Total Liabilities Net Minority Interest`
        - **Denominator**: `Total Assets`
        
        - **Interpretation**:
        - **Purpose**: Indicates the proportion of a company's assets that are financed through debt.
        - **Insight**:
            - A **higher ratio** signifies greater leverage and higher financial risk.
            - A **lower ratio** suggests that the company relies more on equity financing, which is generally less risky.

        ---

        #### 5. **Cash Ratio**

        - **Formula**:
        \[
        \text{Cash Ratio} = \frac{\text{Cash And Cash Equivalents}}{\text{Current Liabilities}}
        \]
        
        - **Source**:
        - **Numerator**: `Cash And Cash Equivalents`
        - **Denominator**: `Current Liabilities`
        
        - **Interpretation**:
        - **Purpose**: Measures the company's ability to pay off its current liabilities with only its most liquid assets.
        - **Insight**:
            - A **higher cash ratio** indicates a strong liquidity position.
            - A **lower cash ratio** may imply potential challenges in covering short-term obligations without additional financing.

        ---

        #### 6. **Debt to Capital Ratio**

        - **Formula**:
        \[
        \text{Debt to Capital} = \frac{\text{Long Term Debt} + \text{Current Debt}}{\text{Long Term Debt} + \text{Current Debt} + \text{Stockholders Equity}}
        \]
        
        - **Source**:
        - **Numerator**: `Long Term Debt` + `Current Debt`
        - **Denominator**: `Long Term Debt` + `Current Debt` + `Stockholders Equity`
        
        - **Interpretation**:
        - **Purpose**: Evaluates the proportion of debt used in the company's capital structure relative to its total capital.
        - **Insight**:
            - A **higher ratio** suggests higher financial leverage and potential risk.
            - A **lower ratio** indicates a more balanced or equity-heavy capital structure, which is generally less risky.

        ---

        #### 7. **Equity Ratio**

        - **Formula**:
        \[
        \text{Equity Ratio} = \frac{\text{Stockholders Equity}}{\text{Total Assets}}
        \]
        
        - **Source**:
        - **Numerator**: `Stockholders Equity`
        - **Denominator**: `Total Assets`
        
        - **Interpretation**:
        - **Purpose**: Shows the proportion of a company's assets financed by shareholders' equity.
        - **Insight**:
            - A **higher equity ratio** indicates greater financial stability and lower reliance on debt.
            - A **lower equity ratio** suggests higher financial leverage and increased financial risk.

        ---

        ### 🔍 **Why These Ratios Matter**

        - **Financial Health Assessment**: These ratios collectively provide a snapshot of the company's financial stability, liquidity, and leverage.
        
        - **Investment Decisions**: Investors use these ratios to determine the risk and potential return of investing in a company.
        
        - **Creditworthiness Evaluation**: Lenders assess these ratios to decide whether to extend credit or loans to the company.
        
        - **Operational Efficiency**: Management uses these insights to make informed decisions about financing, investing, and operational strategies.

        ---

        ### 📌 **Key Takeaways**

        - **Leverage Ratios** (e.g., Debt to Equity, Debt to Assets) assess the level of debt relative to equity and assets, indicating financial risk.
        
        - **Liquidity Ratios** (e.g., Current Ratio, Quick Ratio, Cash Ratio) evaluate the company's ability to meet short-term obligations, reflecting short-term financial health.
        
        - **Capital Structure Ratios** (e.g., Debt to Capital, Equity Ratio) analyze how a company finances its overall operations and growth through different sources of funds.



        """)
            else:
                st.warning("⚠️ No balance sheet data available.")

        # --- Tab 3: Calendar ---

        with tab_cal:
            st.subheader("Calendar")
            calendar = ticker.calendar

            if isinstance(calendar, pd.DataFrame):
                # If it's already a DataFrame, just display it
                if not calendar.empty:
                    # Attempt to format numeric columns, leave others as is
                    numeric_cols = calendar.select_dtypes(include=['number']).columns
                    if not numeric_cols.empty:
                        formatted_calendar = calendar.copy()
                        formatted_calendar[numeric_cols] = formatted_calendar[numeric_cols].applymap(lambda x: f"{x:,.2f}")
                        st.dataframe(formatted_calendar)
                    else:
                        st.dataframe(calendar)
                else:
                    st.warning("Calendar DataFrame is empty.")
            elif isinstance(calendar, dict) and calendar:
                # If it's a non-empty dict, process accordingly

                # Separate the data into categories
                key_dates = {}
                earnings_estimates = {}
                revenue_estimates = {}
                other_info = {}

                for key, value in calendar.items():
                    if "Date" in key:
                        key_dates[key] = value
                    elif "Earnings" in key:
                        earnings_estimates[key] = value
                    elif "Revenue" in key:
                        revenue_estimates[key] = value
                    else:
                        other_info[key] = value

                # Display Key Dates
                st.markdown("### Key Dates")
                if key_dates:
                    key_dates_df = pd.DataFrame.from_dict(key_dates, orient='index', columns=['Date'])
                    key_dates_df.index.name = 'Event'
                    key_dates_df.reset_index(inplace=True)

                    # Convert dates to string for better display
                    def format_date(x):
                        if isinstance(x, list):
                            return ', '.join([d.strftime('%Y-%m-%d') if isinstance(d, (datetime, pd.Timestamp)) else str(d) for d in x])
                        elif isinstance(x, (datetime, pd.Timestamp)):
                            return x.strftime('%Y-%m-%d')
                        else:
                            return str(x)

                    key_dates_df['Date'] = key_dates_df['Date'].apply(format_date)
                    st.table(key_dates_df.style.format({"Date": lambda x: x}))
                else:
                    st.info("No key dates available.")

                # Display Earnings Estimates
                st.markdown("### Earnings Estimates")
                if earnings_estimates:
                    earnings_df = pd.DataFrame.from_dict(earnings_estimates, orient='index', columns=['Value'])
                    earnings_df.index.name = 'Metric'
                    earnings_df.reset_index(inplace=True)

                    # Format numeric values; leave non-numeric as is
                    def format_earnings(x):
                        if isinstance(x, (int, float)):
                            return f"${x:,.2f}"
                        return x

                    earnings_df['Value'] = earnings_df['Value'].apply(format_earnings)
                    st.table(earnings_df)

                    # Visualize Earnings Estimates
                    earnings_plot_df = pd.DataFrame(earnings_estimates, index=[0]).T.reset_index()
                    earnings_plot_df.columns = ['Metric', 'Value']
                    # Filter only numeric values
                    earnings_plot_df = earnings_plot_df[earnings_plot_df['Value'].apply(lambda x: isinstance(x, (int, float)))]
                    if not earnings_plot_df.empty:
                        fig_earnings = px.bar(
                            earnings_plot_df,
                            x='Metric',
                            y='Value',
                            title='Earnings Estimates',
                            text=earnings_plot_df['Value'].apply(lambda x: f"${x:,.2f}"),
                            labels={'Value': 'Amount (USD)', 'Metric': 'Estimate'}
                        )
                        fig_earnings.update_traces(textposition='outside')
                        fig_earnings.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                        st.plotly_chart(fig_earnings, use_container_width=True)
                    else:
                        st.info("No numeric earnings estimates available for plotting.")
                else:
                    st.info("No earnings estimates available.")

                # Display Revenue Estimates
                st.markdown("### Revenue Estimates")
                if revenue_estimates:
                    revenue_df = pd.DataFrame.from_dict(revenue_estimates, orient='index', columns=['Value'])
                    revenue_df.index.name = 'Metric'
                    revenue_df.reset_index(inplace=True)

                    # Format numeric values; leave non-numeric as is
                    def format_revenue(x):
                        if isinstance(x, (int, float)):
                            return f"${x:,.2f}"
                        return x

                    revenue_df['Value'] = revenue_df['Value'].apply(format_revenue)
                    st.table(revenue_df)

                    # Visualize Revenue Estimates
                    revenue_plot_df = pd.DataFrame(revenue_estimates, index=[0]).T.reset_index()
                    revenue_plot_df.columns = ['Metric', 'Value']
                    # Filter only numeric values
                    revenue_plot_df = revenue_plot_df[revenue_plot_df['Value'].apply(lambda x: isinstance(x, (int, float)))]
                    if not revenue_plot_df.empty:
                        fig_revenue = px.bar(
                            revenue_plot_df,
                            x='Metric',
                            y='Value',
                            title='Revenue Estimates',
                            text=revenue_plot_df['Value'].apply(lambda x: f"${x:,.0f}"),
                            labels={'Value': 'Amount (USD)', 'Metric': 'Estimate'}
                        )
                        fig_revenue.update_traces(textposition='outside')
                        fig_revenue.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
                        st.plotly_chart(fig_revenue, use_container_width=True)
                    else:
                        st.info("No numeric revenue estimates available for plotting.")
                else:
                    st.info("No revenue estimates available.")

                # Display Other Information
                if other_info:
                    st.markdown("### Additional Information")
                    other_df = pd.DataFrame.from_dict(other_info, orient='index', columns=['Value'])
                    other_df.index.name = 'Detail'
                    other_df.reset_index(inplace=True)

                    # Format numeric values; leave non-numeric as is
                    def format_other(x):
                        if isinstance(x, (int, float)):
                            return f"${x:,.2f}"
                        elif isinstance(x, (datetime, pd.Timestamp)):
                            return x.strftime('%Y-%m-%d')
                        return x

                    other_df['Value'] = other_df['Value'].apply(format_other)
                    st.table(other_df)
                else:
                    st.info("No additional calendar information available.")

                # Visualizations

                # Timeline of Key Dates
                st.markdown("### Timeline of Key Dates")
                if key_dates:
                    timeline_events = []
                    for event, date in key_dates.items():
                        if isinstance(date, list):
                            for d in date:
                                timeline_events.append({'Event': event, 'Date': d})
                        else:
                            timeline_events.append({'Event': event, 'Date': date})
                    timeline_df = pd.DataFrame(timeline_events)
                    # Ensure 'Date' column is in datetime format
                    try:
                        timeline_df['Date'] = pd.to_datetime(timeline_df['Date'])
                    except Exception as e:
                        st.error(f"Error converting dates: {e}")
                        timeline_df['Date'] = pd.to_datetime(timeline_df['Date'], errors='coerce')
                    timeline_df = timeline_df.dropna(subset=['Date']).sort_values('Date')

                    if not timeline_df.empty:
                        # Create a scatter plot to represent events on a timeline
                        fig_timeline = px.scatter(
                            timeline_df,
                            x='Date',
                            y=[1] * len(timeline_df),  # Single y-value for timeline
                            text='Event',
                            title='Upcoming Key Dates',
                            labels={'Date': 'Date', 'y': 'Event'},
                            hover_data=['Event']
                        )
                        fig_timeline.update_traces(mode='markers+text', textposition='top center', marker=dict(size=12))
                        fig_timeline.update_layout(
                            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                            xaxis=dict(showgrid=True),
                            showlegend=False,
                            height=300
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    else:
                        st.info("No valid key dates available for timeline visualization.")
                else:
                    st.info("No key dates available for timeline visualization.")

                # Download Calendar Data
                st.markdown("### Download Calendar Data")

                # Convert the calendar dict to a flat dictionary for download
                def flatten_calendar(cal_dict):
                    flattened = {}
                    for key, value in cal_dict.items():
                        if isinstance(value, list):
                            flattened[key] = ', '.join([v.strftime('%Y-%m-%d') if isinstance(v, (datetime, pd.Timestamp)) else str(v) for v in value])
                        elif isinstance(value, (datetime, pd.Timestamp)):
                            flattened[key] = value.strftime('%Y-%m-%d')
                        elif isinstance(value, (int, float)):
                            flattened[key] = value
                        else:
                            flattened[key] = str(value)
                    return flattened

                flattened_calendar = flatten_calendar(calendar)
                calendar_download_df = pd.DataFrame(list(flattened_calendar.items()), columns=['Event', 'Value'])
                csv_calendar = calendar_download_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Calendar Data as CSV",
                    data=csv_calendar,
                    file_name='calendar_data.csv',
                    mime='text/csv',
                )

            else:
                # If it's neither a DataFrame nor a dict, just show a warning
                st.warning("No calendar data available or unsupported format.")

        # --- Tab 4: Cashflow ---
        with tab_cf:
            st.subheader("Cashflow")
            cashflow = ticker.cashflow

            if cashflow is not None and not cashflow.empty:
                # Display the full cashflow dataframe with better formatting
                st.dataframe(cashflow.style.format("{:,.2f}").applymap(
                    lambda v: 'background-color: lightgrey' if pd.isnull(v) else ''
                ))

                # Summary Metrics
                st.markdown("### Summary Metrics")
                # Define key metrics to include in the summary
                summary_rows = [
                    "Operating Cash Flow", 
                    "Investing Cash Flow", 
                    "Financing Cash Flow", 
                    "Free Cash Flow", 
                    "Changes In Cash",
                    "Net Income From Continuing Operations",
                    "Depreciation And Amortization",
                    "Deferred Tax"
                ]
                existing_summary = cashflow.index.intersection(summary_rows)
                if not existing_summary.empty:
                    summary_df = cashflow.loc[existing_summary].T
                    summary_df = summary_df.rename_axis("Date").reset_index()
                    summary_df = summary_df.set_index("Date")
                    st.dataframe(summary_df.style.format("{:,.2f}"))
                else:
                    st.warning("No summary metrics available.")

                # Key Cash Flow Items Over Time
                st.markdown("#### Key Cash Flow Items Over Time")
                cf_rows = [
                    "Operating Cash Flow", 
                    "Free Cash Flow", 
                    "Investing Cash Flow", 
                    "Financing Cash Flow",
                    "Net Income From Continuing Operations",
                    "Depreciation And Amortization"
                ]
                existing_rows = cashflow.index.intersection(cf_rows)
                if not existing_rows.empty:
                    cf_selected_df = cashflow.loc[existing_rows].T
                    cf_selected_df.index = pd.to_datetime(cf_selected_df.index)
                    fig_cf = px.line(
                        cf_selected_df, 
                        x=cf_selected_df.index, 
                        y=cf_selected_df.columns,
                        markers=True,
                        title="Key Cash Flow Items Over Time"
                    )
                    fig_cf.update_layout(xaxis_title="Date", yaxis_title="Amount (USD)", hovermode="x unified")
                    st.plotly_chart(fig_cf, use_container_width=True)

                # Detailed Section: Operating, Investing, Financing Cash Flows
                st.markdown("### Detailed Cash Flow Sections")

                # Operating Cash Flow
                if "Operating Cash Flow" in cashflow.index:
                    st.markdown("#### Operating Cash Flow")
                    op_cf = cashflow.loc["Operating Cash Flow"].dropna()
                    fig_op_cf = px.bar(
                        op_cf,
                        x=op_cf.index,
                        y=op_cf.values,
                        labels={'x': 'Date', 'y': 'Operating Cash Flow (USD)'},
                        title="Operating Cash Flow Over Time",
                        text_auto=True
                    )
                    fig_op_cf.update_layout(xaxis_title="Date", yaxis_title="Amount (USD)")
                    st.plotly_chart(fig_op_cf, use_container_width=True)

                # Investing Cash Flow
                if "Investing Cash Flow" in cashflow.index:
                    st.markdown("#### Investing Cash Flow")
                    inv_cf = cashflow.loc["Investing Cash Flow"].dropna()
                    fig_inv_cf = px.bar(
                        inv_cf,
                        x=inv_cf.index,
                        y=inv_cf.values,
                        labels={'x': 'Date', 'y': 'Investing Cash Flow (USD)'},
                        title="Investing Cash Flow Over Time",
                        text_auto=True
                    )
                    fig_inv_cf.update_layout(xaxis_title="Date", yaxis_title="Amount (USD)")
                    st.plotly_chart(fig_inv_cf, use_container_width=True)

                # Financing Cash Flow
                if "Financing Cash Flow" in cashflow.index:
                    st.markdown("#### Financing Cash Flow")
                    fin_cf = cashflow.loc["Financing Cash Flow"].dropna()
                    fig_fin_cf = px.bar(
                        fin_cf,
                        x=fin_cf.index,
                        y=fin_cf.values,
                        labels={'x': 'Date', 'y': 'Financing Cash Flow (USD)'},
                        title="Financing Cash Flow Over Time",
                        text_auto=True
                    )
                    fig_fin_cf.update_layout(xaxis_title="Date", yaxis_title="Amount (USD)")
                    st.plotly_chart(fig_fin_cf, use_container_width=True)

                # Net Cash Flow
                if "Changes In Cash" in cashflow.index:
                    st.markdown("#### Net Cash Flow")
                    net_cf = cashflow.loc["Changes In Cash"].dropna()
                    fig_net_cf = px.line(
                        net_cf,
                        x=net_cf.index,
                        y=net_cf.values,
                        markers=True,
                        title="Net Cash Flow Over Time"
                    )
                    fig_net_cf.update_layout(xaxis_title="Date", yaxis_title="Net Cash Flow (USD)")
                    st.plotly_chart(fig_net_cf, use_container_width=True)

                # Changes in Working Capital
                st.markdown("### Changes in Working Capital")
                working_capital_rows = [
                    "Change In Working Capital",
                    "Change In Other Working Capital",
                    "Change In Payables And Accrued Expense",
                    "Change In Inventory",
                    "Change In Receivables"
                ]
                existing_wc = cashflow.index.intersection(working_capital_rows)
                if not existing_wc.empty:
                    wc_df = cashflow.loc[existing_wc].T
                    wc_df.index = pd.to_datetime(wc_df.index)

                    # Create individual line charts for each working capital component
                    for component in existing_wc:
                        st.markdown(f"#### {component}")
                        fig_wc = px.line(
                            wc_df[component],
                            x=wc_df.index,
                            y=wc_df[component],
                            markers=True,
                            title=f"{component} Over Time"
                        )
                        fig_wc.update_layout(xaxis_title="Date", yaxis_title="Amount (USD)")
                        st.plotly_chart(fig_wc, use_container_width=True)

                    # Optionally, provide a collapsible table for detailed data
                    with st.expander("View Detailed Working Capital Changes"):
                        st.dataframe(wc_df.style.format("{:,.2f}"))
                else:
                    st.warning("No working capital data available.")

                # Capital Expenditure & Stock-Based Compensation
                st.markdown("### Capital Expenditure & Stock-Based Compensation")
                capex = cashflow.loc["Capital Expenditure"].dropna() if "Capital Expenditure" in cashflow.index else pd.Series()
                sbc = cashflow.loc["Stock Based Compensation"].dropna() if "Stock Based Compensation" in cashflow.index else pd.Series()

                if not capex.empty or not sbc.empty:
                    fig_capex_sbc = go.Figure()

                    if not capex.empty:
                        fig_capex_sbc.add_trace(
                            go.Bar(
                                x=capex.index,
                                y=capex.values,
                                name="Capital Expenditure",
                                marker_color='indianred',
                                yaxis='y1'
                            )
                        )
                    if not sbc.empty:
                        fig_capex_sbc.add_trace(
                            go.Scatter(
                                x=sbc.index,
                                y=sbc.values,
                                name="Stock-Based Compensation",
                                mode='lines+markers',
                                marker=dict(color='blue'),
                                yaxis='y2'
                            )
                        )

                    # Create dual y-axes
                    fig_capex_sbc.update_layout(
                        title="Capital Expenditure & Stock-Based Compensation Over Time",
                        xaxis=dict(title='Date'),
                        yaxis=dict(
                            title="Capital Expenditure (USD)",
                            titlefont=dict(color='indianred'),
                            tickfont=dict(color='indianred')
                        ),
                        yaxis2=dict(
                            title="Stock-Based Compensation (USD)",
                            titlefont=dict(color='blue'),
                            tickfont=dict(color='blue'),
                            anchor='x',
                            overlaying='y',
                            side='right'
                        ),
                        legend=dict(x=0.01, y=0.99),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_capex_sbc, use_container_width=True)

                    # Provide a collapsible table for detailed data
                    with st.expander("View Detailed CapEx & SBC Data"):
                        insights_df = pd.DataFrame({
                            "Capital Expenditure": capex,
                            "Stock Based Compensation": sbc
                        })
                        st.dataframe(insights_df.style.format("{:,.2f}"))
                else:
                    st.warning("No Capital Expenditure or Stock-Based Compensation data available.")

                # Additional Insights: Other Key Metrics
                st.markdown("### Additional Key Metrics")
                additional_metrics = [
                    "Net Other Financing Charges",
                    "Cash Dividends Paid",
                    "Common Stock Dividend Paid",
                    "Net Common Stock Issuance",
                    "Net Issuance Payments Of Debt",
                    "Net Short Term Debt Issuance",
                    "Net Long Term Debt Issuance",
                    "Long Term Debt Payments",
                    "Net Other Investing Changes",
                    "Net Investment Purchase And Sale",
                    "Sale Of Investment",
                    "Purchase Of Investment",
                    "Net Business Purchase And Sale",
                    "Purchase Of Business",
                    "Net PPE Purchase And Sale",
                    "Depreciation And Amortization",
                    "Net Income From Continuing Operations",
                    "Deferred Tax",
                    "Deferred Income Tax"
                ]
                existing_additional = cashflow.index.intersection(additional_metrics)
                if not existing_additional.empty:
                    additional_df = cashflow.loc[existing_additional].T
                    additional_df.index = pd.to_datetime(additional_df.index)
                    st.markdown("#### Additional Cash Flow Items Over Time")
                    fig_additional = px.line(
                        additional_df, 
                        x=additional_df.index, 
                        y=additional_df.columns,
                        markers=True,
                        title="Additional Cash Flow Items Over Time"
                    )
                    fig_additional.update_layout(xaxis_title="Date", yaxis_title="Amount (USD)", hovermode="x unified")
                    st.plotly_chart(fig_additional, use_container_width=True)

                    # Provide a collapsible table for detailed additional metrics
                    with st.expander("View Detailed Additional Metrics"):
                        st.dataframe(additional_df.style.format("{:,.2f}"))
                else:
                    st.warning("No additional cash flow metrics available.")

                # Download Cashflow Data
                st.markdown("### Download Cashflow Data")
                csv = cashflow.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Cashflow as CSV",
                    data=csv,
                    file_name='cashflow_data.csv',
                    mime='text/csv',
                )

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
