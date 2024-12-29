import streamlit as st
import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
import pandas as pd
import numpy as np

# Assuming these functions/objects come from your own modules:
# - get_historical_data: Returns a DataFrame with columns [Open, High, Low, Close, Volume]
# - additional_statistics: Returns a DataFrame with some stats
# - bought: A dictionary that maps tickers to a list of tuples (date, price)
from data_fetchers import (
    get_historical_data,
    additional_statistics
)

from ticker_manager import get_ticker_data, get_ticker_list

###############################################################################
# Additional Utility
###############################################################################
def compute_on_balance_volume(hist_data):
    """
    Compute the On-Balance Volume (OBV).
    OBV is a cumulative indicator that uses volume flow 
    to predict changes in stock price.
    """
    data = hist_data.copy()
    data['Change'] = data['Close'].diff()
    # Sign of the volume
    data['Direction'] = 0
    data.loc[data['Change'] > 0, 'Direction'] = 1
    data.loc[data['Change'] < 0, 'Direction'] = -1

    data['Signed_Volume'] = data['Volume'] * data['Direction']
    data['OBV'] = data['Signed_Volume'].cumsum()
    return data['OBV']

###############################################################################
# Existing Utilities (Unchanged)
###############################################################################
def annotate_performance_with_points_and_time(ax, hist_data, points):
    """
    1) Overall performance from the first to last data point in hist_data.
    2) Horizontal line & label for each 'bought' point in `points`.
       points is expected to be a list of (date, price).
    """
    if hist_data.empty:
        return

    start_price = hist_data['Close'].iloc[0]
    end_price   = hist_data['Close'].iloc[-1]
    performance_time = (end_price / start_price - 1) * 100

    time_text = f"{end_price:.2f} - Performance over time: {performance_time:.2f}%"
    ax.text(
        0.02, 0.95,
        time_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    for _, point_price, _, _, _ in points:
        perf_from_buy = (end_price / point_price - 1) * 100
        ax.axhline(y=point_price, color='red', linestyle='--', alpha=0.7)
        
        color = "green" if perf_from_buy > 0 else "red"
        
        ax.text(
            ax.get_xlim()[0],
            point_price,
            f"{point_price} : Perf={perf_from_buy:.2f}%",
            color=color,
            verticalalignment='center',
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor=color)
        )

def compute_bollinger_bands(hist_data, window=20, num_std=2):
    rolling_mean = hist_data['Close'].rolling(window).mean()
    rolling_std  = hist_data['Close'].rolling(window).std()
    upper_band   = rolling_mean + (num_std * rolling_std)
    lower_band   = rolling_mean - (num_std * rolling_std)
    return rolling_mean, upper_band, lower_band

def compute_rsi(hist_data, period=14):
    delta = hist_data['Close'].diff()
    gain  = delta.where(delta > 0, 0.0)
    loss  = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(com=(period - 1), min_periods=period).mean()
    avg_loss = loss.ewm(com=(period - 1), min_periods=period).mean()

    rs  = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def compute_macd(hist_data, fast=12, slow=26, signal=9):
    fast_ema   = hist_data['Close'].ewm(span=fast, adjust=False).mean()
    slow_ema   = hist_data['Close'].ewm(span=slow, adjust=False).mean()

    macd_line   = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist   = macd_line - signal_line

    return macd_line, signal_line, macd_hist

def compute_daily_returns(hist_data):
    return hist_data['Close'].pct_change() * 100

def compute_stochastic_oscillator(hist_data, k_period=14, d_period=3):
    low_min   = hist_data['Low'].rolling(k_period).min()
    high_max  = hist_data['High'].rolling(k_period).max()

    stoch_k = 100.0 * (hist_data['Close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(d_period).mean()
    return stoch_k, stoch_d

def compute_keltner_channels(hist_data, ema_period=20, atr_period=10, multiplier=2):
    typical_price = (hist_data['High'] + hist_data['Low'] + hist_data['Close']) / 3
    center_line   = typical_price.ewm(span=ema_period, adjust=False).mean()

    prev_close = hist_data['Close'].shift(1)
    tr1        = hist_data['High'] - hist_data['Low']
    tr2        = (hist_data['High'] - prev_close).abs()
    tr3        = (hist_data['Low']  - prev_close).abs()
    true_range = tr1.combine(tr2, max).combine(tr3, max)

    atr = true_range.ewm(span=atr_period, adjust=False).mean()

    upper_channel = center_line + (multiplier * atr)
    lower_channel = center_line - (multiplier * atr)
    return center_line, upper_channel, lower_channel

def compute_aroon(hist_data, period=14):
    aroon_up   = pd.Series(dtype=float, index=hist_data.index)
    aroon_down = pd.Series(dtype=float, index=hist_data.index)

    for i in range(period, len(hist_data)):
        window_highs = hist_data['High'].iloc[i - period + 1 : i + 1]
        window_lows  = hist_data['Low'].iloc[i - period + 1 : i + 1]

        days_since_highest = (period - 1) - window_highs.argmax()
        days_since_lowest  = (period - 1) - window_lows.argmin()

        aroon_up.iloc[i]   = 100.0 * (period - days_since_highest) / period
        aroon_down.iloc[i] = 100.0 * (period - days_since_lowest)  / period

    return aroon_up, aroon_down

def rma(series, period):
    alpha  = 1.0 / period
    rma_ser= pd.Series(dtype=float, index=series.index)
    rma_ser.iloc[0] = series.iloc[0]
    for i in range(1, len(series)):
        rma_ser.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * rma_ser.iloc[i - 1]
    return rma_ser

def compute_adx(hist_data, period=14):
    prev_close = hist_data['Close'].shift(1)

    # True Range
    tr1 = hist_data['High'] - hist_data['Low']
    tr2 = (hist_data['High'] - prev_close).abs()
    tr3 = (hist_data['Low']  - prev_close).abs()
    true_range = tr1.combine(tr2, max).combine(tr3, max)

    # +DM, -DM
    plus_dm  = (hist_data['High'] - hist_data['High'].shift(1)).where(lambda x: x > 0, 0.0)
    minus_dm = (hist_data['Low'].shift(1) - hist_data['Low']).where(lambda x: x > 0, 0.0)

    # RMA
    atr      = rma(true_range.fillna(0), period)
    plus_sm  = rma(plus_dm.fillna(0),    period)
    minus_sm = rma(minus_dm.fillna(0),   period)

    plus_di  = 100.0 * (plus_sm / atr.replace(0, np.nan)) 
    minus_di = 100.0 * (minus_sm / atr.replace(0, np.nan)) 

    dx       = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_line = rma(dx.fillna(0), period)

    return plus_di, minus_di, adx_line

def compute_ichimoku(hist_data):
    """
    Basic Ichimoku Cloud calculation:
      - Tenkan-sen (Conversion Line): (9-day high + 9-day low) / 2
      - Kijun-sen (Base Line): (26-day high + 26-day low) / 2
      - Senkou Span A (Leading Span A): (Tenkan + Kijun)/2
      - Senkou Span B (Leading Span B): (52-day high + 52-day low)/2
      - Chikou Span (Lagging Span): Close shifted 26 days back
    """
    high_9  = hist_data['High'].rolling(9).max()
    low_9   = hist_data['Low'].rolling(9).min()
    tenkan  = (high_9 + low_9) / 2

    high_26 = hist_data['High'].rolling(26).max()
    low_26  = hist_data['Low'].rolling(26).min()
    kijun   = (high_26 + low_26) / 2

    # Leading Span A: average of Tenkan & Kijun
    span_a  = (tenkan + kijun) / 2

    # 52-day
    high_52 = hist_data['High'].rolling(52).max()
    low_52  = hist_data['Low'].rolling(52).min()
    span_b  = (high_52 + low_52) / 2

    # Chikou Span: close shifted 26 days back
    chikou  = hist_data['Close'].shift(-26)

    return tenkan, kijun, span_a, span_b, chikou

###############################################################################
# Main Function (show_historical_data) with Enhanced Checkboxes
###############################################################################
def show_historical_data(ticker_input):
    """
    Panels layout is dynamic now:
      Panel 0: Always Candles (+ performance annotations).
      Optional panels (volume, bollinger, RSI, MACD, etc.) are each 
      shown only if the corresponding checkbox is selected.
    """
    if not ticker_input:
        st.write("No ticker provided.")
        return

    ticker = yf.Ticker(ticker_input)
    di = ticker.info
    longname = di.get("longName", ticker_input)

    st.subheader(f"Historical Data for {ticker_input}:")
    st.subheader(f"{longname}")

    # -----------------------
    # Sidebar controls
    # -----------------------
    period = st.sidebar.selectbox(
        "Period:",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "5y", "10y", "max"],
        index=5
    )
    interval = st.sidebar.selectbox(
        "Interval:",
        ["1m", "5m", "15m", "30m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
        index=5
    )
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (Annualized)",
        value=0.0,
        format="%.4f"
    )

    # Checkboxes for each panel (except the candlestick panel = 0)
    show_volume         = st.sidebar.checkbox("Show Volume (Panel)", value=True)
    show_bollinger      = st.sidebar.checkbox("Show Bollinger Bands (Panel)", value=False)
    show_rsi            = st.sidebar.checkbox("Show RSI (Panel)", value=False)
    show_macd           = st.sidebar.checkbox("Show MACD (Panel)", value=False)
    show_daily_returns  = st.sidebar.checkbox("Show Daily % Returns (Panel)", value=False)
    show_stochastic     = st.sidebar.checkbox("Show Stochastic Oscillator (Panel)", value=False)
    show_keltner        = st.sidebar.checkbox("Show Keltner Channels (Panel)", value=False)
    show_aroon          = st.sidebar.checkbox("Show Aroon (Panel)", value=False)
    show_adx            = st.sidebar.checkbox("Show ADX (Panel)", value=False)
    show_ichimoku       = st.sidebar.checkbox("Show Ichimoku (Panel)", value=False)
    show_obv            = st.sidebar.checkbox("Show OBV (Panel)", value=False)

    # Fetch data
    hist_data = get_historical_data(ticker, period=period, interval=interval)
    if hist_data.empty:
        st.write("No historical data available for this ticker.")
        return

    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in required_cols:
        if col not in hist_data.columns:
            st.write(f"Missing '{col}' in the data. Cannot plot candlestick.")
            return

    if len(hist_data) < 52:
        st.warning("Not enough data to compute some (e.g. 52-day) indicators. "
                   "Try selecting a longer period for more complete indicators.")

    # Points to annotate from 'bought'
    bought = get_ticker_data()
    if ticker_input in bought:
        points_to_annotate = bought[ticker_input]
    else:
        points_to_annotate = []

    # Helper to avoid adding empty / all-NaN series
    def valid_series(series):
        if series is None:
            return False
        return not series.dropna().empty

    # --------------------------------------------------------
    # We will dynamically assign panel numbers based on checkboxes
    # panel=0 is always the candlestick panel
    # --------------------------------------------------------
    panel_index = 1  # Start at 1 because 0 is the candlestick panel
    volume_panel = None
    bollinger_panel = None
    rsi_panel = None
    macd_panel = None
    daily_returns_panel = None
    stochastic_panel = None
    keltner_panel = None
    aroon_panel = None
    adx_panel = None
    ichimoku_panel = None
    obv_panel = None

    # We'll also build the `panel_ratios` list dynamically.
    # The first item in the list will correspond to panel=0 (candlesticks),
    # then each subsequent panel that is shown gets an entry.
    panel_ratios = [4]  # bigger for candlesticks

    if show_volume:
        volume_panel = panel_index
        panel_index += 1
        panel_ratios.append(1)

    if show_bollinger:
        bollinger_panel = panel_index
        panel_index += 1
        panel_ratios.append(2)

    if show_rsi:
        rsi_panel = panel_index
        panel_index += 1
        panel_ratios.append(2)

    if show_macd:
        macd_panel = panel_index
        panel_index += 1
        panel_ratios.append(2)

    if show_daily_returns:
        daily_returns_panel = panel_index
        panel_index += 1
        panel_ratios.append(2)

    if show_stochastic:
        stochastic_panel = panel_index
        panel_index += 1
        panel_ratios.append(2)

    if show_keltner:
        keltner_panel = panel_index
        panel_index += 1
        panel_ratios.append(2)

    if show_aroon:
        aroon_panel = panel_index
        panel_index += 1
        panel_ratios.append(2)

    if show_adx:
        adx_panel = panel_index
        panel_index += 1
        panel_ratios.append(2)

    if show_ichimoku:
        ichimoku_panel = panel_index
        panel_index += 1
        panel_ratios.append(2)

    if show_obv:
        obv_panel = panel_index
        panel_index += 1
        panel_ratios.append(2)

    # The total number of panels
    num_panels = panel_index

    # --------------------------------------------------------
    # Build the list of addplots
    # --------------------------------------------------------
    addplots = []

    # Bollinger
    if bollinger_panel is not None:
        bb_mid, bb_up, bb_low = compute_bollinger_bands(hist_data, window=20, num_std=2)
        if valid_series(bb_mid):
            addplots.append(mpf.make_addplot(bb_mid, panel=bollinger_panel, color='blue', 
                                             linestyle='dashed', label='BB Mid'))
        if valid_series(bb_up):
            addplots.append(mpf.make_addplot(bb_up, panel=bollinger_panel, color='grey', 
                                             label='BB Upper'))
        if valid_series(bb_low):
            addplots.append(mpf.make_addplot(bb_low, panel=bollinger_panel, color='grey', 
                                             label='BB Lower'))

    # RSI
    if rsi_panel is not None:
        rsi_series = compute_rsi(hist_data, period=14)
        if valid_series(rsi_series):
            addplots.append(mpf.make_addplot(rsi_series, panel=rsi_panel, color='purple', 
                                             ylim=(0,100), ylabel='RSI', label='RSI'))

    # MACD
    if macd_panel is not None:
        macd_line, macd_signal, macd_hist = compute_macd(hist_data)
        if valid_series(macd_line):
            addplots.append(mpf.make_addplot(macd_line,   panel=macd_panel, color='green',  label='MACD Line'))
        if valid_series(macd_signal):
            addplots.append(mpf.make_addplot(macd_signal, panel=macd_panel, color='orange', label='Signal'))
        if valid_series(macd_hist):
            addplots.append(mpf.make_addplot(macd_hist,   panel=macd_panel, type='bar', color='gray', 
                                             alpha=0.5, label='MACD Hist'))

    # Daily Returns
    if daily_returns_panel is not None:
        daily_returns = compute_daily_returns(hist_data)
        if valid_series(daily_returns):
            addplots.append(
                mpf.make_addplot(daily_returns, panel=daily_returns_panel, type='bar', 
                                 color='teal', alpha=0.6, label='Daily % Returns')
            )

    # Stochastic
    if stochastic_panel is not None:
        stoch_k, stoch_d = compute_stochastic_oscillator(hist_data, k_period=14, d_period=3)
        if valid_series(stoch_k):
            addplots.append(mpf.make_addplot(stoch_k, panel=stochastic_panel, color='brown', 
                                             ylim=(0,100), label='Stoch %K'))
        if valid_series(stoch_d):
            addplots.append(mpf.make_addplot(stoch_d, panel=stochastic_panel, color='red', 
                                             ylim=(0,100), label='Stoch %D'))

    # Keltner Channels
    if keltner_panel is not None:
        kelt_center, kelt_up, kelt_low = compute_keltner_channels(hist_data, 
                                                                 ema_period=20, 
                                                                 atr_period=10, 
                                                                 multiplier=2)
        if valid_series(kelt_center):
            addplots.append(mpf.make_addplot(kelt_center, panel=keltner_panel, color='blue', 
                                             linestyle='dotted', label='Kelt Ctr'))
        if valid_series(kelt_up):
            addplots.append(mpf.make_addplot(kelt_up, panel=keltner_panel, color='grey', label='Kelt Up'))
        if valid_series(kelt_low):
            addplots.append(mpf.make_addplot(kelt_low, panel=keltner_panel, color='grey', label='Kelt Low'))

    # Aroon
    if aroon_panel is not None:
        aroon_up, aroon_down = compute_aroon(hist_data, period=14)
        if valid_series(aroon_up):
            addplots.append(mpf.make_addplot(aroon_up, panel=aroon_panel, color='green', 
                                             label='Aroon Up', ylim=(0,100)))
        if valid_series(aroon_down):
            addplots.append(mpf.make_addplot(aroon_down, panel=aroon_panel, color='red', 
                                             label='Aroon Down', ylim=(0,100)))

    # ADX
    if adx_panel is not None:
        plus_di, minus_di, adx_line = compute_adx(hist_data, period=14)
        if valid_series(plus_di):
            addplots.append(mpf.make_addplot(plus_di, panel=adx_panel, color='lime', 
                                             label='+DI', ylim=(0,100)))
        if valid_series(minus_di):
            addplots.append(mpf.make_addplot(minus_di, panel=adx_panel, color='red',  
                                             label='-DI', ylim=(0,100)))
        if valid_series(adx_line):
            addplots.append(mpf.make_addplot(adx_line, panel=adx_panel, color='blue', 
                                             label='ADX',  ylim=(0,100)))

    # Ichimoku
    if ichimoku_panel is not None:
        tenkan, kijun, span_a, span_b, chikou = compute_ichimoku(hist_data)
        if valid_series(tenkan):
            addplots.append(mpf.make_addplot(tenkan, panel=ichimoku_panel, color='orange', label='Tenkan-sen'))
        if valid_series(kijun):
            addplots.append(mpf.make_addplot(kijun, panel=ichimoku_panel, color='blue',   label='Kijun-sen'))
        if valid_series(span_a):
            addplots.append(mpf.make_addplot(span_a, panel=ichimoku_panel, color='green',  label='Span A'))
        if valid_series(span_b):
            addplots.append(mpf.make_addplot(span_b, panel=ichimoku_panel, color='red',    label='Span B'))
        if valid_series(chikou):
            addplots.append(mpf.make_addplot(chikou, panel=ichimoku_panel, color='gray',   label='Chikou Span'))

    # OBV
    if obv_panel is not None:
        obv_data = compute_on_balance_volume(hist_data)
        if valid_series(obv_data):
            addplots.append(mpf.make_addplot(obv_data, panel=obv_panel, color='darkblue', label='OBV'))

    # --------------------------------------------------------
    # Prepare mpf.plot parameters
    # --------------------------------------------------------
    # - volume=True only if show_volume is True; otherwise volume=False
    # - volume_panel = volume_panel if it exists
    # - num_panels = total panels
    # - panel_ratios = dynamically built above
    # - Candlestick is always in panel=0
    # --------------------------------------------------------
    mpf_volume = True if volume_panel is not None else False
    mpf_volume_panel = volume_panel if volume_panel is not None else 1  # fallback

    fig, axlist = mpf.plot(
        hist_data,
        type='candle',
        mav=(5, 20, 50),
        volume=mpf_volume,
        volume_panel=mpf_volume_panel,
        num_panels=num_panels,
        panel_ratios=panel_ratios,
        style='charles',
        addplot=addplots,
        returnfig=True,
        figsize=(16, 5 + 2*num_panels),  # a rough formula for figure height
        title=f"{longname} Price Chart"
    )

    # Annotate performance on the candlestick panel (panel=0)
    annotate_performance_with_points_and_time(axlist[0], hist_data, points_to_annotate)

    # Add legends to each panel
    for ax in axlist:
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc='best')

    # Display in Streamlit
    st.pyplot(fig)

    # Show the raw historical data
    st.write("### Raw Historical Data")
    st.write(hist_data)

    # Additional stats
    stats = additional_statistics(hist_data, risk_free_rate=risk_free_rate)
    st.write("### Additional Statistical Analysis")
    st.dataframe(stats.style.format("{:.4f}"))

    # Definitions of Each Panel
    st.write("## Definitions of Panels / Indicators")
    st.markdown("""
**Panel 0**: Candlestick Chart (Open, High, Low, Close) + annotations.

Then the following panels appear only if selected:

- **Volume**: Shares/contracts traded per interval.
- **Bollinger Bands**: 20-day SMA ± standard deviations.
- **RSI**: Momentum oscillator (0–100).
- **MACD**: Moving Average Convergence/Divergence (+ signal + histogram).
- **Daily % Returns**: Day-to-day % change in close.
- **Stochastic**: %K, %D measure comparing close to recent min/max.
- **Keltner Channels**: EMA-based channel using ATR for band width.
- **Aroon**: Time since recent highs/lows in a given period.
- **ADX**: Trend strength (+DI, -DI, and ADX).
- **Ichimoku**: Cloud-based indicator (Tenkan, Kijun, Span A/B, Chikou).
- **OBV**: On-Balance Volume (cumulative volume flow).
""")

