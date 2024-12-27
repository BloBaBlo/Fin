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
            required_columns = ['Ticker', 'Bought Date', 'Buy Price', 'Quantity']
            if not all(col in df.columns for col in required_columns):
                st.sidebar.error("Invalid CSV format. Ensure it contains columns: 'Ticker', 'Bought Date', 'Buy Price', 'Quantity'.")
                return None
            
            # Convert DataFrame to the desired JSON format
            preset_tickers_data = {
                row['Ticker']: [[row['Bought Date'], row['Buy Price'], row['Quantity']]]
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

