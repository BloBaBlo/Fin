import streamlit as st
import json
import pandas as pd
from typing import Dict, Optional

def initialize_ticker_data():
    """Initialize ticker data in session state if not already present"""
    if 'preset_tickers_data' not in st.session_state:
        st.session_state.preset_tickers_data = {}
    if 'preset_tickers' not in st.session_state:
        st.session_state.preset_tickers = []

def load_ticker_data_from_json(uploaded_file) -> Optional[Dict]:
    """Load ticker data from uploaded JSON file and store in session state"""
    if uploaded_file is not None:
        try:
            preset_tickers_data = json.load(uploaded_file)
            if isinstance(preset_tickers_data, dict):
                st.session_state.preset_tickers_data = preset_tickers_data
                st.session_state.preset_tickers = list(preset_tickers_data.keys())
                return preset_tickers_data
            else:
                st.sidebar.error("Invalid JSON format. Please upload a dictionary with tickers as keys.")
        except json.JSONDecodeError:
            st.sidebar.error("Error reading JSON file. Please ensure it's a valid JSON.")
    return None

def load_ticker_data(uploaded_file) -> Optional[Dict]:
    """Load ticker data from a CSV file and convert it to the required JSON format"""
    if uploaded_file is not None:
        try:
            # Read CSV file into a DataFrame
            df = pd.read_csv(uploaded_file)
            
            # Ensure the necessary columns exist
            required_columns = ['Ticker', 'Bought Date', 'Buy Price', 'Quantity', 'Color', 'Label','ISIN']
            if not all(col in df.columns for col in required_columns):
                st.sidebar.error("Invalid CSV format. Ensure it contains columns: 'Ticker', 'Bought Date', 'Buy Price', 'Quantity'.",)
                return None
            
            df['ISIN'] = df['ISIN'].apply(lambda x: None if pd.isna(x) or str(x).strip() == '' else x)
            # Convert DataFrame to the desired JSON format
            preset_tickers_data = {
                row['Ticker']: [[row['ISIN'], row['Bought Date'], row['Buy Price'], row['Quantity'], row['Color'], row['Label']]]
                for _, row in df.iterrows()
            }
            
            # Save to session state
            st.session_state.preset_tickers_data = preset_tickers_data
            st.session_state.preset_tickers = list(preset_tickers_data.keys())
            return preset_tickers_data
        
        except Exception as e:
            st.sidebar.error(f"Error reading CSV file: {e}")
    return None

def get_ticker_data() -> Dict:
    """Retrieve ticker data from session state"""
    initialize_ticker_data()
    return st.session_state.preset_tickers_data

def get_ticker_list() -> list:
    """Retrieve list of tickers from session state"""
    initialize_ticker_data()
    return st.session_state.preset_tickers



import requests
import re

from datetime import datetime

def get_tradegate_data(isin):
    url = f'https://www.tradegate.de/orderbuch.php?lang=en&isin={isin}'
    response = requests.get(url)  # Fetch the webpage
    html = response.text  # Get the HTML content

    # Regex patterns to extract data
    bid_regex = r'<strong id="bid">([\d.,]+)</strong>'
    last_regex = r'<strong id="last">([\d.,]+)</strong>'
    low_regex = r'<td id="low" class="longprice">([\d.,]+)</td>'
    date_regex = r'<span id="rt_datum">([\d/]+)</span> @\s*<span id="rt_zeit"[^>]*>([\d:]+)</span>'

    # Extract data
    bid_match = re.search(bid_regex, html)
    last_match = re.search(last_regex, html)
    low_match = re.search(low_regex, html)
    date_match = re.search(date_regex, html)

    # Parse matches
    bid_price = bid_match.group(1) if bid_match else None
    last_price = last_match.group(1) if last_match else None
    low_price = low_match.group(1) if low_match else None

    

    try : 
        low_price= float(low_price)
    except:
        low_price = bid_price

    try : 
        last_price= float(last_price)
    except:
        last_price = bid_price

    # Parse and format date and time
    if date_match:
        date_str = date_match.group(1)  # e.g., "02/01/2025"
        time_str = date_match.group(2)  # e.g., "14:05:20"
        combined_str = f"{date_str} {time_str}"
        update_date = datetime.strptime(combined_str, '%d/%m/%Y %H:%M:%S')
        formatted_update_date = update_date.strftime('%Y-%m-%d %H:%M')
    else:
        formatted_update_date = None

    try : 
        bid_price = float(bid_price)
    except:
        return None, None, None, formatted_update_date
    return bid_price, last_price, low_price, formatted_update_date

