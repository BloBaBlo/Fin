# tabs/analysis_tab.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from data_fetchers import (
    get_historical_data, 
    additional_statistics
)
from patterns import (
    detect_trend, detect_pennant, detect_flag, detect_wedge, 
    detect_triangle, detect_cup_and_handle, detect_head_and_shoulders, 
    detect_double_top
)

def show_analysis_and_visualization(ticker_input):
    st.subheader(f"Analysis & Visualization for {ticker_input}")
    ticker = yf.Ticker(ticker_input)

    period = st.sidebar.selectbox("Analysis Period:", ["1y", "5y", "10y", "max"], index=0)
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (Annualized)",
        value=0.0,
        format="%.4f"
    )
    hist_data = get_historical_data(ticker, period=period, interval="1d")

    if hist_data.empty:
        st.write("No data available for this ticker and period.")
        return

    st.write("**Closing Price Statistics:**")
    st.write(hist_data['Close'].describe().to_frame("Close"))

    # Rolling means
    st.write("### Rolling Mean Analysis")
    window1 = st.sidebar.slider("Short-Term Window (days)", min_value=5, max_value=50, value=20)
    window2 = st.sidebar.slider("Long-Term Window (days)", min_value=50, max_value=200, value=50)
    _plot_rolling_means(hist_data, window1, window2)

    # Returns Analysis
    st.write("### Returns Analysis")
    daily_returns = _plot_returns_analysis(hist_data)

    # Benchmark Comparison
    st.write("### Benchmark Comparison")
    benchmark_ticker = st.sidebar.text_input("Enter a Benchmark Ticker (e.g. SPY)", value="")
    if benchmark_ticker:
        benchmark = yf.Ticker(benchmark_ticker)
        benchmark_hist = get_historical_data(benchmark, period=period, interval="1d")
        if not benchmark_hist.empty:
            benchmark_returns = benchmark_hist['Close'].pct_change().dropna()
            _analyze_against_benchmark(daily_returns, benchmark_returns, ticker_input, benchmark_ticker)
        else:
            st.write("No data for the selected benchmark ticker.")

    st.write("### Additional Statistics (for Analysis Period)")
    stats = additional_statistics(hist_data, risk_free_rate=risk_free_rate)
    st.dataframe(stats.style.format("{:.4f}"))

    # Pivot Points
    st.write("### Technical Indicators & Pivot Points")
    pivot_freq = st.selectbox("Pivot Frequency:", ["Daily", "Weekly", "Monthly"], index=0)
    pivot_type = st.selectbox("Pivot Type:", ["Classic", "Fibonacci", "Camarilla", "Woodie"], index=0)

    freq_map = {"Daily": 'D', "Weekly": 'W', "Monthly": 'M'}
    pivots = _compute_pivot_points(hist_data, pivot_type=pivot_type, freq=freq_map[pivot_freq])
    pivot_df = pd.DataFrame([pivots])
    st.write("**Computed Pivot Points:**")
    st.dataframe(pivot_df.style.format("{:.2f}"))

    if pivot_type == "Classic":
        touches = _count_pivot_touches(hist_data, pivots)
        touches_df = pd.DataFrame([touches], index=['Touches'])
        st.write("**Number of times price touched pivot levels (±0.1%):**")
        st.dataframe(touches_df)

    show_pivots = st.checkbox("Show Pivot Points", value=True)
    show_adl = st.checkbox("Show Accumulation/Distribution Line (ADL)", value=True)
    show_kama = st.checkbox("Show Kaufman's Adaptive Moving Average (KAMA)", value=True)

    _plot_technical_indicators(hist_data, show_pivots, show_adl, show_kama, pivots)

    st.write("**Closing Price Statistics:**")
    st.write(hist_data['Close'].describe().to_frame("Close"))

    # Pattern Detection
    st.write("### Pattern Detection (Demo)")
    detect_trend_ = st.checkbox("Detect Trend", value=True)
    detect_pennants_ = st.checkbox("Detect Pennants", value=False)
    detect_flags_ = st.checkbox("Detect Flags", value=False)
    detect_wedges_ = st.checkbox("Detect Wedges", value=False)
    detect_triangles_ = st.checkbox("Detect Triangles", value=False)
    detect_cup_handle_ = st.checkbox("Detect Cup and Handle", value=False)
    detect_head_shoulders_ = st.checkbox("Detect Head and Shoulders", value=False)
    detect_double_top_bottom_ = st.checkbox("Detect Double Top/Bottom", value=False)

    detected_patterns = []
    if detect_trend_:
        t = detect_trend(hist_data)
        detected_patterns.append(t)
    if detect_pennants_ and detect_pennant(hist_data):
        detected_patterns.append("Pennant")
    if detect_flags_ and detect_flag(hist_data):
        detected_patterns.append("Flag")
    if detect_wedges_ and detect_wedge(hist_data):
        detected_patterns.append("Wedge")
    if detect_triangles_:
        tri = detect_triangle(hist_data)
        if tri:
            detected_patterns.append(tri)
    if detect_cup_handle_ and detect_cup_and_handle(hist_data):
        detected_patterns.append("Cup and Handle")
    if detect_head_shoulders_ and detect_head_and_shoulders(hist_data):
        detected_patterns.append("Head and Shoulders")
    if detect_double_top_bottom_:
        dtb = detect_double_top(hist_data)
        if dtb:
            detected_patterns.append(dtb)

    _plot_pattern_detection(hist_data, ticker_input, detected_patterns)
    st.write("**Note:** This pattern detection is highly simplified and not reliable. It's for demonstration only.")


# --------------------------------------------------------------------------------
# Internal helper functions for analysis_tab.py
# --------------------------------------------------------------------------------
def _plot_rolling_means(hist_data, window1=20, window2=50):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist_data.index, hist_data['Close'], label='Close', alpha=0.5)
    ax.plot(hist_data['Close'].rolling(window1).mean(), label=f'{window1}-day SMA', linewidth=1)
    ax.plot(hist_data['Close'].rolling(window2).mean(), label=f'{window2}-day SMA', linewidth=1)
    ax.set_title(f'Price with {window1}-day and {window2}-day SMA')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    st.pyplot(fig)


def _plot_returns_analysis(hist_data):
    daily_returns = hist_data['Close'].pct_change().dropna()
    cumulative_returns = (1 + daily_returns).cumprod()

    st.write("**Daily Returns Statistics:**")
    st.write(daily_returns.describe().to_frame("Daily Returns"))

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(daily_returns.index, daily_returns, label='Daily Returns')
    ax1.axhline(0, color='red', linewidth=1)
    ax1.set_title('Daily Returns Over Time')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Daily Return")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Returns')
    ax2.set_title('Cumulative Returns')
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Cumulative Growth (1 = 100%)")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.hist(daily_returns, bins=30, alpha=0.7)
    ax3.set_title('Distribution of Daily Returns')
    ax3.set_xlabel("Daily Return")
    ax3.set_ylabel("Frequency")
    st.pyplot(fig3)

    return daily_returns


def _analyze_against_benchmark(main_returns, benchmark_returns, main_ticker, benchmark_ticker):
    if main_returns.empty or benchmark_returns.empty:
        st.write("Not enough data for correlation analysis.")
        return

    combined = pd.DataFrame({
        'Main': main_returns,
        'Benchmark': benchmark_returns
    }).dropna()
    correlation = combined['Main'].corr(combined['Benchmark'])

    st.write(f"**Correlation with Benchmark ({benchmark_ticker}):** {correlation:.4f}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(combined['Benchmark'], combined['Main'], alpha=0.5)
    ax.set_title('Daily Returns Correlation')
    ax.set_xlabel(f'{benchmark_ticker} Daily Returns')
    ax.set_ylabel(f'{main_ticker} Daily Returns')
    m, b = np.polyfit(combined['Benchmark'], combined['Main'], 1)
    ax.plot(combined['Benchmark'], m*combined['Benchmark']+b, color='red')
    st.pyplot(fig)


def _compute_pivot_points(hist_data, pivot_type="Classic", freq='D'):
    if freq == 'W':
        ohlc = hist_data.resample('W').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    elif freq == 'M':
        ohlc = hist_data.resample('M').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'})
    else:
        ohlc = hist_data.copy()

    last_bar = ohlc.iloc[-2] if freq != 'D' else ohlc.iloc[-1]
    high = last_bar['High']
    low = last_bar['Low']
    close = last_bar['Close']
    pivot = (high + low + close)/3

    if pivot_type == "Classic":
        r1 = 2*pivot - low
        s1 = 2*pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2*(pivot - low)
        s3 = low - 2*(high - pivot)

    elif pivot_type == "Fibonacci":
        r1 = pivot + 0.382*(high - low)
        s1 = pivot - 0.382*(high - low)
        r2 = pivot + 0.618*(high - low)
        s2 = pivot - 0.618*(high - low)
        r3 = pivot + 1.0*(high - low)
        s3 = pivot - 1.0*(high - low)

    elif pivot_type == "Camarilla":
        diff = high - low
        r1 = close + 1.1*(diff)/12
        s1 = close - 1.1*(diff)/12
        r2 = close + 1.1*(diff)*2/12
        s2 = close - 1.1*(diff)*2/12
        r3 = close + 1.1*(diff)*3/12
        s3 = close - 1.1*(diff)*3/12

    elif pivot_type == "Woodie":
        pivot = (high + low + 2*close)/4
        r1 = (2*pivot)-low
        s1 = (2*pivot)-high
        r2 = pivot + (high-low)
        s2 = pivot - (high-low)
        r3 = high + 2*(pivot - low)
        s3 = low - 2*(high - pivot)

    return {
        'Pivot': pivot,
        'R1': r1, 'R2': r2, 'R3': r3,
        'S1': s1, 'S2': s2, 'S3': s3
    }

def _count_pivot_touches(hist_data, pivots):
    close = hist_data['Close']
    touches = {}
    for level_name, level_value in pivots.items():
        threshold = level_value * 0.001
        condition = (close >= level_value - threshold) & (close <= level_value + threshold)
        touches[level_name] = condition.sum()
    return touches

def _plot_technical_indicators(hist_data, show_pivots=True, show_adl=True, show_kama=True, pivots=None):
    import pandas_ta as ta

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(hist_data.index, hist_data['Close'], label='Close', alpha=0.7)

    if show_pivots and pivots is not None:
        for level_name, level_value in pivots.items():
            ax.axhline(y=level_value, linestyle='--', linewidth=1, 
                       label=f"{level_name}: {level_value:.2f}")

    if show_kama:
        kama = ta.kama(hist_data['Close'], length=30)
        ax.plot(hist_data.index, kama, label='KAMA (30)', color='orange', linewidth=1.5)

    ax.set_title('Technical Indicators')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    st.pyplot(fig)

    if show_adl:
        adl = _accumulation_distribution_line(hist_data)
        fig2, ax2 = plt.subplots(figsize=(10,3))
        ax2.plot(adl.index, adl, label='ADL', color='blue')
        ax2.set_title('Accumulation/Distribution Line')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('ADL')
        ax2.legend()
        st.pyplot(fig2)

def _accumulation_distribution_line(hist_data):
    df = hist_data.copy()
    df['CLV'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    df['CLV'].fillna(0, inplace=True)
    df['ADL'] = df['CLV'] * df['Volume']
    df['ADL'] = df['ADL'].cumsum()
    return df['ADL']

def _plot_pattern_detection(hist_data, ticker_input, detected_patterns):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(hist_data.index, hist_data['Close'], label='Close', color='blue')
    ax.set_title(f"{ticker_input} Price with Pattern Detection")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")

    if detected_patterns:
        pattern_text = ", ".join(detected_patterns)
        ax.text(0.02, 0.95, f"Detected: {pattern_text}", 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', 
                bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='gray'))
    else:
        ax.text(0.02, 0.95, "No Patterns Detected", 
                transform=ax.transAxes, fontsize=12, 
                verticalalignment='top', 
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))

    ax.legend()
    st.pyplot(fig)
