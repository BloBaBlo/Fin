import yfinance as yf
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_analyst_recommendations_summary(ticker):
    """
    Fetches a summary of analyst recommendations for a given ticker using yfinance.
    Returns a pd.DataFrame with columns like 'period', 'strongBuy', 'buy', etc.
    """
    try:
        rec_summary_df = ticker.recommendations

        if rec_summary_df is None or rec_summary_df.empty:
            logger.info("No analyst recommendations summary data found in yfinance.")
            return pd.DataFrame()

        # Reset index to ensure 'Period' is a column
        rec_summary_df = rec_summary_df.reset_index()

        # Standardize column names by capitalizing appropriately
        rec_summary_df.columns = [col.strip().title() for col in rec_summary_df.columns]

        logger.info("Successfully fetched analyst recommendations summary from yfinance.")
        logger.debug(f"Recommendations Summary DataFrame:\n{rec_summary_df.head()}")

        return rec_summary_df

    except Exception as e:
        logger.error(f"Error fetching analyst recommendations summary from yfinance: {e}")
        return pd.DataFrame()


def get_upgrades_downgrades(ticker):
    """
    Fetches detailed upgrades and downgrades for a given ticker using yfinance.
    Returns a pd.DataFrame with columns like 'Firm', 'ToGrade', 'FromGrade', 'Action', etc.
    """
    try:
        upgrades_downgrades_df = ticker.get_upgrades_downgrades()

        if upgrades_downgrades_df is None or upgrades_downgrades_df.empty:
            logger.info("No upgrades/downgrades data found in yfinance.")
            return pd.DataFrame()

        # Reset index to ensure 'GradeDate' is a column
        upgrades_downgrades_df = upgrades_downgrades_df.reset_index()

        # Standardize column names
        upgrades_downgrades_df.columns = [col.strip().title() for col in upgrades_downgrades_df.columns]

        # Rename 'Gradedate' or 'GradeDate' to 'Date'
        if 'Gradedate' in upgrades_downgrades_df.columns:
            upgrades_downgrades_df.rename(columns={'Gradedate': 'Date'}, inplace=True)
        elif 'GradeDate' in upgrades_downgrades_df.columns:
            upgrades_downgrades_df.rename(columns={'GradeDate': 'Date'}, inplace=True)

        logger.info("Successfully fetched upgrades/downgrades data from yfinance.")
        logger.debug(f"Upgrades/Downgrades DataFrame:\n{upgrades_downgrades_df.head()}")

        return upgrades_downgrades_df

    except Exception as e:
        logger.error(f"Error fetching upgrades/downgrades from yfinance: {e}")
        return pd.DataFrame()


def get_sustainability(ticker):
    """
    Fetches sustainability (ESG) scores for a given ticker using yfinance.
    Returns a dictionary of ESG-related data.
    """
    try:
        sustainability_data = ticker.get_sustainability()

        if sustainability_data is None or sustainability_data.empty:
            logger.info("No sustainability data found in yfinance.")
            return {}

        logger.info("Successfully fetched sustainability data from yfinance.")
        logger.debug(f"Sustainability Data:\n{sustainability_data}")

        return sustainability_data.to_dict()

    except Exception as e:
        logger.error(f"Error fetching sustainability data from yfinance: {e}")
        return {}


def get_analyst_price_targets(ticker):
    """
    Fetches analyst price targets for a given ticker using yfinance.
    Returns a dictionary with keys like 'current', 'low', 'high', 'mean', 'median'.
    """
    try:
        price_targets = ticker.get_analyst_price_targets()

        if price_targets is None or not isinstance(price_targets, dict):
            logger.info("No analyst price targets data found in yfinance.")
            return {}

        logger.info("Successfully fetched analyst price targets from yfinance.")
        logger.debug(f"Analyst Price Targets: {price_targets}")

        return price_targets

    except Exception as e:
        logger.error(f"Error fetching analyst price targets from yfinance: {e}")
        return {}


def get_earnings_estimate(ticker):
    """
    Fetches earnings estimates for a given ticker using yfinance.
    Returns a pd.DataFrame with columns like 'numberOfAnalysts', 'avg', 'low', etc.
    """
    try:
        earnings_estimate_df = ticker.get_earnings_estimate()

        if earnings_estimate_df is None or earnings_estimate_df.empty:
            logger.info("No earnings estimates data found in yfinance.")
            return pd.DataFrame()

        logger.info("Successfully fetched earnings estimates from yfinance.")
        logger.debug(f"Earnings Estimates DataFrame:\n{earnings_estimate_df.head()}")

        return earnings_estimate_df

    except Exception as e:
        logger.error(f"Error fetching earnings estimates from yfinance: {e}")
        return pd.DataFrame()


def get_revenue_estimate(ticker):
    """
    Fetches revenue estimates for a given ticker using yfinance.
    Returns a pd.DataFrame with columns like 'numberOfAnalysts', 'avg', 'low', etc.
    """
    try:
        revenue_estimate_df = ticker.get_revenue_estimate()

        if revenue_estimate_df is None or revenue_estimate_df.empty:
            logger.info("No revenue estimates data found in yfinance.")
            return pd.DataFrame()

        logger.info("Successfully fetched revenue estimates from yfinance.")
        logger.debug(f"Revenue Estimates DataFrame:\n{revenue_estimate_df.head()}")

        return revenue_estimate_df

    except Exception as e:
        logger.error(f"Error fetching revenue estimates from yfinance: {e}")
        return pd.DataFrame()


def get_eps_revisions(ticker):
    """
    Fetches EPS revisions for a given ticker using yfinance.
    Returns a pd.DataFrame with columns like 'upLast7days', 'downLast30days', etc.
    """
    try:
        eps_revisions_df = ticker.get_eps_revisions()

        if eps_revisions_df is None or eps_revisions_df.empty:
            logger.info("No EPS revisions data found in yfinance.")
            return pd.DataFrame()

        logger.info("Successfully fetched EPS revisions from yfinance.")
        logger.debug(f"EPS Revisions DataFrame:\n{eps_revisions_df.head()}")

        return eps_revisions_df

    except Exception as e:
        logger.error(f"Error fetching EPS revisions from yfinance: {e}")
        return pd.DataFrame()


def get_growth_estimates(ticker):
    """
    Fetches growth estimates for a given ticker using yfinance.
    Returns a pd.DataFrame with columns like 'stock', 'industry', 'sector', 'index'.
    """
    try:
        growth_estimates_df = ticker.get_growth_estimates()

        if growth_estimates_df is None or growth_estimates_df.empty:
            logger.info("No growth estimates data found in yfinance.")
            return pd.DataFrame()

        logger.info("Successfully fetched growth estimates from yfinance.")
        logger.debug(f"Growth Estimates DataFrame:\n{growth_estimates_df.head()}")

        return growth_estimates_df

    except Exception as e:
        logger.error(f"Error fetching growth estimates from yfinance: {e}")
        return pd.DataFrame()

# news_page.py

import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf


def show_analyst_recommendations(ticker_input):
    """
    Streamlit page to display multiple financial datasets and visualizations 
    (recommendations, upgrades/downgrades, ESG, earnings, revenue, etc.) 
    for a specified ticker.
    """
    st.title("📰 Financial News and Analysis Dashboard")
    
    # If user has provided a ticker symbol
    if ticker_input:
        ticker = yf.Ticker(ticker_input.strip())
        
        # Attempt fetching data
        rec_summary_df = get_analyst_recommendations_summary(ticker)
        upgrades_downgrades_df = get_upgrades_downgrades(ticker)
        sustainability_data = get_sustainability(ticker)
        price_targets = get_analyst_price_targets(ticker)
        earnings_estimate_df = get_earnings_estimate(ticker)
        revenue_estimate_df = get_revenue_estimate(ticker)
        eps_revisions_df = get_eps_revisions(ticker)
        growth_estimates_df = get_growth_estimates(ticker)
        
        st.header(f"**Data & Analysis for:** {ticker_input.upper()}")
        
        # Create tabs for organized layout
        tabs = st.tabs([
            "Analyst Recommendations",
            "Upgrades/Downgrades",
            "Sustainability",
            "Price Targets",
            "Earnings Estimates",
            "Revenue Estimates",
            "EPS Revisions",
            "Growth Estimates"
        ])
        
        # --------------------------------------------------------------------
        # 1. Analyst Recommendations Tab
        # --------------------------------------------------------------------
        with tabs[0]:
            st.subheader("🔍 Analyst Recommendations Summary")
            if not rec_summary_df.empty:
                # Display raw table
                st.dataframe(rec_summary_df, use_container_width=True)
                
                # Melt the DataFrame for better bar chart display
                if all(col in rec_summary_df.columns for col in 
                       ["Period", "Strongbuy", "Buy", "Hold", "Sell", "Strongsell"]):
                    rec_melted = rec_summary_df.melt(
                        id_vars=['Period'],
                        value_vars=['Strongbuy', 'Buy', 'Hold', 'Sell', 'Strongsell'],
                        var_name='Recommendation',
                        value_name='Count'
                    )
                    
                    # Define color mapping
                    color_map = {
                        'Strongbuy': 'darkgreen',
                        'Buy': 'green',
                        'Hold': 'blue',
                        'Sell': 'red',
                        'Strongsell': 'darkred'
                    }
                    
                    # Create a grouped bar chart
                    fig = px.bar(
                        rec_melted,
                        x='Period',
                        y='Count',
                        color='Recommendation',
                        barmode='group',
                        title='Analyst Recommendations Over Periods',
                        color_discrete_map=color_map,
                        labels={'Count': 'Number of Recommendations'}
                    )
                    fig.update_layout(xaxis_title='Period', yaxis_title='Number of Recommendations')
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Could not plot recommendations (missing expected columns).")
            else:
                st.info("No analyst recommendations summary available.")
                
        # --------------------------------------------------------------------
        # 2. Upgrades/Downgrades Tab
        # --------------------------------------------------------------------
        with tabs[1]:
            st.subheader("⬆️⬇️ Upgrades and Downgrades")
            if not upgrades_downgrades_df.empty:
                # Ensure 'Date' is a datetime column for better sorting
                if 'Date' in upgrades_downgrades_df.columns:
                    upgrades_downgrades_df['Date'] = pd.to_datetime(
                        upgrades_downgrades_df['Date'], errors='coerce'
                    )
                    # Drop rows where the date couldn't be parsed
                    upgrades_downgrades_df.dropna(subset=['Date'], inplace=True)
                    
                    # Sort by Date descending to show the most recent first
                    upgrades_downgrades_df.sort_values(by='Date', ascending=False, inplace=True)
                    upgrades_downgrades_df.reset_index(drop=True, inplace=True)
                
                # Display the table
                st.dataframe(upgrades_downgrades_df, use_container_width=True)
                
                # Create a timeline scatter plot
                if 'Date' in upgrades_downgrades_df.columns:
                    fig_upgrade = px.scatter(
                        upgrades_downgrades_df,
                        x='Date',
                        y='Firm',
                        color='Action',
                        title='Upgrades and Downgrades Over Time',
                        labels={'Action': 'Action'},
                        hover_data=['Tograde', 'Fromgrade'],
                        color_discrete_sequence=px.colors.qualitative.Set1
                    )
                    fig_upgrade.update_traces(marker=dict(size=10, symbol='circle'))
                    fig_upgrade.update_layout(yaxis={'categoryorder':'category descending'})
                    st.plotly_chart(fig_upgrade, use_container_width=True)
                else:
                    st.error("'Date' column is missing in Upgrades/Downgrades data.")
            else:
                st.info("No upgrades or downgrades data available.")
        
        # --------------------------------------------------------------------
        # 3. Sustainability Tab
        # --------------------------------------------------------------------
        with tabs[2]:
            st.subheader("🌱 Sustainability (ESG) Scores")
            if sustainability_data:
                # Convert dictionary to DataFrame
                sustainability_df = pd.DataFrame.from_dict(
                    sustainability_data, orient='index', columns=['Value']
                )
                st.table(sustainability_df)
                
                # Plot ESG scores if available
                environment_score = sustainability_data.get('environmentScore', None)
                social_score = sustainability_data.get('socialScore', None)
                governance_score = sustainability_data.get('governanceScore', None)
                
                if None not in [environment_score, social_score, governance_score]:
                    esg_scores_plot = pd.DataFrame({
                        'ESG Category': ['Environment', 'Social', 'Governance'],
                        'Score': [environment_score, social_score, governance_score]
                    })
                    
                    fig_esg = px.bar(
                        esg_scores_plot,
                        x='ESG Category',
                        y='Score',
                        color='ESG Category',
                        title='ESG Scores',
                        labels={'Score': 'Score'},
                        color_discrete_sequence=px.colors.qualitative.Prism
                    )
                    fig_esg.update_layout(showlegend=False)
                    st.plotly_chart(fig_esg, use_container_width=True)
                else:
                    st.info("Some ESG scores are missing, so no chart is displayed.")
            else:
                st.info("No sustainability data available.")
        
        # --------------------------------------------------------------------
        # 4. Price Targets Tab
        # --------------------------------------------------------------------
        with tabs[3]:
            st.subheader("💰 Analyst Price Targets")
            if price_targets:
                # Display the main metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Current", price_targets.get('current', 'N/A'))
                col2.metric("Low", price_targets.get('low', 'N/A'))
                col3.metric("High", price_targets.get('high', 'N/A'))
                col4.metric("Mean", price_targets.get('mean', 'N/A'))
                col5.metric("Median", price_targets.get('median', 'N/A'))
                
                # Optionally create a small bar chart
                price_data = {
                    'Target': [
                        'Current', 'Low', 'High', 'Mean', 'Median'
                    ],
                    'Value': [
                        price_targets.get('current', 0),
                        price_targets.get('low', 0),
                        price_targets.get('high', 0),
                        price_targets.get('mean', 0),
                        price_targets.get('median', 0),
                    ]
                }
                df_price_targets = pd.DataFrame(price_data)
                
                fig_price = px.bar(
                    df_price_targets,
                    x='Target',
                    y='Value',
                    color='Target',
                    title='Analyst Price Targets',
                    labels={'Value': 'Price ($)'},
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_price.update_layout(showlegend=False)
                st.plotly_chart(fig_price, use_container_width=True)
            else:
                st.info("No analyst price targets available.")
        
        # --------------------------------------------------------------------
        # 5. Earnings Estimates Tab
        # --------------------------------------------------------------------
        with tabs[4]:
            st.subheader("📈 Earnings Estimates")
            if not earnings_estimate_df.empty:
                st.dataframe(earnings_estimate_df, use_container_width=True)
                
                # Example line chart comparing avg, low, and high per row
                # We assume each row is a period (0q, +1q, 0y, +1y, etc.)
                # We'll add a 'Period' column for clarity:
                periods = list(earnings_estimate_df.index)
                earnings_estimate_df['Period'] = periods
                
                # Melt to get (Period, estimate_type, value)
                earnings_melted = earnings_estimate_df.melt(
                    id_vars=['Period'],
                    value_vars=['avg', 'low', 'high'],
                    var_name='Estimate Type',
                    value_name='EPS'
                )
                
                fig_earnings = px.line(
                    earnings_melted,
                    x='Period',
                    y='EPS',
                    color='Estimate Type',
                    title='Earnings Estimates Over Periods',
                    markers=True
                )
                fig_earnings.update_layout(xaxis_title='Period', yaxis_title='Estimate')
                st.plotly_chart(fig_earnings, use_container_width=True)
            else:
                st.info("No earnings estimates available.")
        
        # --------------------------------------------------------------------
        # 6. Revenue Estimates Tab
        # --------------------------------------------------------------------
        with tabs[5]:
            st.subheader("📊 Revenue Estimates")
            if not revenue_estimate_df.empty:
                st.dataframe(revenue_estimate_df, use_container_width=True)
                
                # Similar approach: add a Period column
                periods = list(revenue_estimate_df.index)
                revenue_estimate_df['Period'] = periods
                
                # Melt to get (Period, estimate_type, value)
                revenue_melted = revenue_estimate_df.melt(
                    id_vars=['Period'],
                    value_vars=['avg', 'low', 'high'],
                    var_name='Estimate Type',
                    value_name='Revenue'
                )
                
                fig_revenue = px.line(
                    revenue_melted,
                    x='Period',
                    y='Revenue',
                    color='Estimate Type',
                    title='Revenue Estimates Over Periods',
                    markers=True
                )
                fig_revenue.update_layout(xaxis_title='Period', yaxis_title='Revenue')
                st.plotly_chart(fig_revenue, use_container_width=True)
            else:
                st.info("No revenue estimates available.")
        
        # --------------------------------------------------------------------
        # 7. EPS Revisions Tab
        # --------------------------------------------------------------------
        with tabs[6]:
            st.subheader("📉📈 EPS Revisions")
            if not eps_revisions_df.empty:
                st.dataframe(eps_revisions_df, use_container_width=True)
                
                # Convert None to 0 for plotting
                eps_revisions_df = eps_revisions_df.fillna(0)
                
                # Melt the DataFrame
                # If the index has period-like data (e.g., 0q, +1q, etc.), we can keep it
                periods = list(eps_revisions_df.index)
                eps_revisions_df['Period'] = periods
                
                # We assume columns: 'upLast7days', 'upLast30days', 'downLast7days', 'downLast30days'
                revision_columns = ['upLast7days', 'upLast30days', 'downLast7days', 'downLast30days']
                available_cols = [c for c in revision_columns if c in eps_revisions_df.columns]
                
                if available_cols:
                    eps_revisions_melted = eps_revisions_df.melt(
                        id_vars=['Period'],
                        value_vars=available_cols,
                        var_name='Revision Type',
                        value_name='Count'
                    )
                    
                    fig_eps_rev = px.bar(
                        eps_revisions_melted,
                        x='Period',
                        y='Count',
                        color='Revision Type',
                        title='EPS Revisions by Period',
                        barmode='group'
                    )
                    fig_eps_rev.update_layout(xaxis_title='Period', yaxis_title='Count')
                    st.plotly_chart(fig_eps_rev, use_container_width=True)
                else:
                    st.info("Expected columns for EPS Revisions not found.")
            else:
                st.info("No EPS revisions available.")
        
        # --------------------------------------------------------------------
        # 8. Growth Estimates Tab
        # --------------------------------------------------------------------
        with tabs[7]:
            st.subheader("📈 Growth Estimates")
            if not growth_estimates_df.empty:
                st.dataframe(growth_estimates_df, use_container_width=True)
                
                # We'll assume columns: ['stock', 'industry', 'sector', 'index']
                # Each row might represent a period (0q, +1q, etc.)
                # For a quick view, let's try to melt and plot a bar chart
                # We can add a 'Period' column from the index
                periods = list(growth_estimates_df.index)
                growth_estimates_df['Period'] = periods
                
                # Melt
                growth_melted = growth_estimates_df.melt(
                    id_vars=['Period'],
                    var_name='Category',
                    value_name='Growth'
                )
                
                # Filter out the 'Period' row if it's in columns
                # or we can just plot them as is
                fig_growth = px.bar(
                    growth_melted,
                    x='Period',
                    y='Growth',
                    color='Category',
                    barmode='group',
                    title='Growth Estimates by Category'
                )
                fig_growth.update_layout(xaxis_title='Period', yaxis_title='Growth Rate')
                st.plotly_chart(fig_growth, use_container_width=True)
            else:
                st.info("No growth estimates available.")
        
        # --------------------------------------------------------------------
        # Additional Insights Section
        # --------------------------------------------------------------------
        st.markdown("---")
        with st.expander("🔍 Additional Insights & Distribution"):
            # Recommendations Insights
            if not rec_summary_df.empty and all(
                col in rec_summary_df.columns 
                for col in ["Strongbuy", "Buy", "Hold", "Sell", "Strongsell"]
            ):
                total_recs = rec_summary_df[["Strongbuy", "Buy", "Hold", "Sell", "Strongsell"]].sum()
                
                # Show some metrics
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Strong Buy", int(total_recs['Strongbuy']))
                col2.metric("Buy", int(total_recs['Buy']))
                col3.metric("Hold", int(total_recs['Hold']))
                col4.metric("Sell", int(total_recs['Sell']))
                col5.metric("Strong Sell", int(total_recs['Strongsell']))
                
                # Pie Chart for distribution
                fig_pie = px.pie(
                    names=total_recs.index,
                    values=total_recs.values,
                    title='Overall Recommendations Distribution',
                    color=total_recs.index,
                    color_discrete_map={
                        'Strongbuy': 'darkgreen',
                        'Buy': 'green',
                        'Hold': 'blue',
                        'Sell': 'red',
                        'Strongsell': 'darkred'
                    }
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Upgrades/Downgrades Insights
            if not upgrades_downgrades_df.empty and 'Action' in upgrades_downgrades_df.columns:
                action_counts = upgrades_downgrades_df['Action'].value_counts()
                st.write("### Upgrades/Downgrades Summary")
                for action_type in action_counts.index:
                    st.metric(action_type.capitalize(), int(action_counts[action_type]))
            
            # Quick highlight for price targets, if available
            if price_targets:
                st.write("### Analyst Price Targets Overview:")
                st.json(price_targets)
                
            st.caption("All data is fetched via Yahoo Finance API (yfinance). "
                       "These insights are for informational purposes only and not investment advice.")
    else:
        st.warning("Please enter a valid ticker symbol in the sidebar to proceed.")
