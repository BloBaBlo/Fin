# data_fetchers.py
import streamlit as st
import pandas as pd
import numpy as np

import yfinance as yf

# ----------------------
# Keep your "bought" dictionary here
bought = {
    "ESEH.PA":  [(pd.Timestamp('2024-12-19'), 19.38)],
    "SXRT.DE":  [(pd.Timestamp('2024-12-19'), 183.2)],
    "WPEA.PA":  [(pd.Timestamp('2024-12-19'), 5.83)],
    "EXA1.AS":  [(pd.Timestamp('2024-12-19'), 8.71)],
    "EFRN.DE":  [(pd.Timestamp('2024-12-19'), 5.15)],
    "CSSPX.MI":  [(pd.Timestamp('2024-12-19'), 614)], 
    "MWRD.PA":  [(pd.Timestamp('2024-12-19'), 133.01)], 
    "0JDT.L":  [(pd.Timestamp('2024-12-19'), 49.2)], 
    "XONA.DE":  [(pd.Timestamp('2024-12-19'), 102.06)], 
    "5QQ0.F":  [(pd.Timestamp('2024-12-19'), 0.462)],
    "EQ":  [(pd.Timestamp('2024-12-19'), 0.763)], 
    "ALATA.PA":  [(pd.Timestamp('2024-12-19'), 0.1341)], 
    "OPTN":  [(pd.Timestamp('2024-12-19'), 0.528)], 
    "4CQA.MU":  [(pd.Timestamp('2024-12-19'), 0.1682)], 
    "VJF.HM":  [(pd.Timestamp('2024-12-19'), 0.0021)], 
    "BNE.MU":  [(pd.Timestamp('2024-12-19'), 0.0704)], 
    "COH.BE":  [(pd.Timestamp('2024-12-19'), 0.0517)], 
    "B4T0.BE":  [(pd.Timestamp('2024-12-19'), 0.0055)], 
    "ID4.F":  [(pd.Timestamp('2024-12-19'), 0.0847)],
    "3KJB.BE":  [(pd.Timestamp('2024-12-19'), 0.536)], 
    "5LA1.BE":  [(pd.Timestamp('2024-12-19'), 0.623)], 
    "0IU0.BE":  [(pd.Timestamp('2024-12-19'), 0.0759)], 
    "D4G.BE":  [(pd.Timestamp('2024-12-19'), 0.0079)], 
    "ATO.PA":  [(pd.Timestamp('2024-12-19'), 0.0033)], 
    "IS3Q.DE":  [(pd.Timestamp('2024-12-19'), 68.09)], 
    "EXI2.DE":  [(pd.Timestamp('2024-12-19'), 92.75)], 
    "XDWT.DE":  [(pd.Timestamp('2024-12-19'), 93.19)], 
    "IS3C.DE":  [(pd.Timestamp('2024-12-19'), 67.13)],
    "DBXJ.DE":  [(pd.Timestamp('2024-12-19'), 77.08)], 
    "IS3K.DE":  [(pd.Timestamp('2024-12-19'), 82.65)], 
    "SXRR.DE":  [(pd.Timestamp('2024-12-19'), 4.78)], 
    "IBCD.DE":  [(pd.Timestamp('2024-12-19'), 96.67)], 
    "XRP-USD":  [(pd.Timestamp('2024-12-19'), 2.523)], 
    "SHIB-USD":  [(pd.Timestamp('2024-12-19'), 0.0000304599)], 
    "XCN18679-USD":  [(pd.Timestamp('2024-12-19'), 0.00271)], 
    "GRT6719-USD":  [(pd.Timestamp('2024-12-19'), 0.333)], 
    "WLD-USD":  [(pd.Timestamp('2024-12-19'), 3.867)], 
    "ETH-USD":  [(pd.Timestamp('2024-12-19'), 3860.44)], 
    "SOL-USD": [(pd.Timestamp('2024-12-19'), 216.246)], 
}

# ----------------------
def get_historical_data(ticker_obj, period="1y", interval="1d"):
    try:
        data = ticker_obj.history(period=period, interval=interval)
        return data
    except Exception as e:
        st.error(f"Error retrieving historical data: {e}")
        return pd.DataFrame()

def get_company_info(ticker_obj):
    try:
        info = ticker_obj.info
        if not info or len(info) == 0:
            return None
        info_df = pd.DataFrame.from_dict(info, orient='index', columns=['Value'])
        return info_df
    except Exception as e:
        st.error(f"Error retrieving company info: {e}")
        return None

def get_analyst_recommendations(ticker_obj):
    try:
        rec = ticker_obj.recommendations
        if rec is None or rec.empty:
            return None
        return rec
    except Exception as e:
        st.error(f"Error retrieving recommendations: {e}")
        return None

def get_dividends(ticker_obj):
    try:
        div = ticker_obj.dividends
        if div.empty:
            return None
        return pd.DataFrame(div, columns=["Dividends"])
    except Exception as e:
        st.error(f"Error retrieving dividends: {e}")
        return None

def get_earnings(ticker_obj):
    try:
        earnings = ticker_obj.earnings
        if earnings is None or earnings.empty:
            return None
        return earnings
    except Exception as e:
        st.error(f"Error retrieving earnings: {e}")
        return None

# ----------------------
def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.0):
    if daily_returns.empty:
        st.error("No daily returns available to calculate Sharpe Ratio.")
        return np.nan
    mean_return = daily_returns.mean() * 252
    std_return = daily_returns.std() * np.sqrt(252)
    if std_return == 0:
        return np.nan
    return (mean_return - risk_free_rate) / std_return

def calculate_max_drawdown(hist_data):
    if 'Close' not in hist_data.columns or hist_data.empty:
        st.error("Historical data not available or missing 'Close' column for max drawdown calculation.")
        return np.nan
    roll_max = hist_data['Close'].cummax()
    daily_drawdown = hist_data['Close'] / roll_max - 1.0
    return daily_drawdown.min()

def additional_statistics(hist_data, risk_free_rate=0.0):
    if hist_data.empty or 'Close' not in hist_data.columns:
        st.error("Insufficient data for computing additional statistics.")
        return pd.DataFrame()

    daily_returns = hist_data['Close'].pct_change().dropna()
    if daily_returns.empty:
        st.error("Not enough data to compute daily returns for additional stats.")
        return pd.DataFrame()

    annualized_return = daily_returns.mean() * 252
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    max_dd = calculate_max_drawdown(hist_data)
    sharpe_ratio = calculate_sharpe_ratio(daily_returns, risk_free_rate=risk_free_rate)

    stats = pd.DataFrame({
        "Annualized Return": [annualized_return],
        "Annualized Volatility": [annualized_volatility],
        "Max Drawdown": [max_dd],
        "Sharpe Ratio": [sharpe_ratio]
    })
    return stats

def annotate_performance_with_points_and_time(ax, hist_data, points):
    # Performance over time
    start_price = hist_data['Close'].iloc[0]
    end_price = hist_data['Close'].iloc[-1]
    performance_time = (end_price / start_price - 1) * 100
    time_text = f"Performance over time: {performance_time:.2f}%"
    ax.text(0.02, 0.95, time_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    # Annotate the 'bought' points
    last_price = end_price
    for point_date, point_price in points:
        performance = (last_price / point_price - 1) * 100
        ax.axhline(point_price, color='red', 
                   label=f"{point_price} : Performance={performance:.2f}%")
