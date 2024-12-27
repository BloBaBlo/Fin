# patterns.py

import numpy as np
import pandas as pd
from scipy.stats import linregress
import streamlit as st

def detect_trend(hist_data):
    if hist_data.empty or 'Close' not in hist_data.columns:
        return "No Data"

    y = hist_data['Close'].values
    x = np.arange(len(y))
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    if slope > 0:
        return "Uptrend"
    elif slope < 0:
        return "Downtrend"
    else:
        return "Sideways"

def detect_pennant(hist_data):
    if len(hist_data) < 30:
        return False
    segment = hist_data[-30:]
    highs = segment['High']
    lows = segment['Low']
    # Check if highs are generally decreasing and lows are generally increasing
    if highs.iloc[-1] < highs.iloc[0] and lows.iloc[-1] > lows.iloc[0]:
        return True
    return False

def detect_flag(hist_data):
    if len(hist_data) < 30:
        return False
    segment = hist_data[-30:]
    returns = segment['Close'].pct_change().cumsum()
    # Check for run-up (example threshold)
    if returns.iloc[-1] > 0.05:
        # Check if recent range is stable, indicating a channel
        ranges = segment['High'] - segment['Low']
        if len(ranges) > 10 and np.std(ranges) < (0.1 * np.mean(ranges)):
            return True
    return False

def detect_wedge(hist_data):
    if len(hist_data) < 60:
        return False
    segment = hist_data[-60:]
    highs = segment['High']
    lows = segment['Low']
    # Example: falling wedge if both highs and lows trend downward
    if highs.iloc[-1] < highs.iloc[0] and lows.iloc[-1] < lows.iloc[0]:
        return True
    return False

def detect_triangle(hist_data):
    if len(hist_data) < 60:
        return None
    segment = hist_data[-60:]
    highs = segment['High']
    lows = segment['Low']

    # Symmetrical Triangle
    if highs.iloc[-1] < highs.iloc[0] and lows.iloc[-1] > lows.iloc[0]:
        return "Symmetrical Triangle"

    # Ascending Triangle: flat top, rising lows (just a naive check)
    top_range = highs.max() - highs.min()
    if top_range < 0.02 * highs.mean() and lows.iloc[-1] > lows.iloc[0]:
        return "Ascending Triangle"

    # Descending Triangle: flat bottom, falling highs (naive check)
    bottom_range = lows.max() - lows.min()
    if bottom_range < 0.02 * lows.mean() and highs.iloc[-1] < highs.iloc[0]:
        return "Descending Triangle"

    return None

def detect_cup_and_handle(hist_data):
    if len(hist_data) < 100:
        return False
    segment = hist_data[-100:]
    closes = segment['Close'].values

    mid = len(closes) // 2
    # Very naive "U shape" detection
    if closes[mid] == closes.min() and closes[0] > closes[mid] and closes[-1] > closes[mid]:
        # handle: last 10 days a slight dip?
        last10 = closes[-10:]
        if len(last10) == 10 and last10[-1] < last10[0]:
            return True
    return False

def detect_head_and_shoulders(hist_data):
    if len(hist_data) < 60:
        return False
    segment = hist_data[-60:]
    closes = segment['Close']
    peaks = (closes.shift(1) < closes) & (closes.shift(-1) < closes)
    peak_positions = np.where(peaks)[0]

    if len(peak_positions) < 3:
        return False

    # Very naive approach: check if there's a "middle" highest peak
    sorted_peaks = sorted(peak_positions, key=lambda i: closes.iloc[i], reverse=True)
    if len(sorted_peaks) >= 3:
        return True
    return False

def detect_double_top(hist_data):
    if len(hist_data) < 50:
        return None
    segment = hist_data[-50:]
    closes = segment['Close']

    # Find peaks
    peaks = (closes.shift(1) < closes) & (closes.shift(-1) < closes)
    peak_values = closes[peaks]

    if len(peak_values) >= 2:
        top_two = peak_values.nlargest(2)
        if len(top_two) == 2 and abs(top_two.iloc[0] - top_two.iloc[1]) < 0.01 * top_two.mean():
            return "Double Top"

    # Find troughs
    troughs = (closes.shift(1) > closes) & (closes.shift(-1) > closes)
    trough_values = closes[troughs]
    if len(trough_values) >= 2:
        bottom_two = trough_values.nsmallest(2)
        if len(bottom_two) == 2 and abs(bottom_two.iloc[0] - bottom_two.iloc[1]) < 0.01 * bottom_two.mean():
            return "Double Bottom"

    return None
