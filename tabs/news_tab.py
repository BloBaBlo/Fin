# tabs/news_tab.py
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime
import logging
from dateutil import tz
import math

# Configure logging
logging.basicConfig(level=logging.ERROR, filename='app_errors.log',
                    format='%(asctime)s:%(levelname)s:%(message)s')

@st.cache_data(ttl=3600)  # Cache news data for 1 hour
def fetch_news(ticker):
    ticker_obj = yf.Ticker(ticker)
    return ticker_obj.news

def show_news(ticker_input):
    st.subheader(f"News for {ticker_input.upper()}")
    news = fetch_news(ticker_input)

    if news:
        st.write("### Latest News")
        
        # Pagination Setup
        news_per_page = 5
        total_pages = math.ceil(len(news) / news_per_page)
        page = st.number_input('Page', min_value=1, max_value=total_pages, value=1, step=1)
        start_idx = (page - 1) * news_per_page
        end_idx = start_idx + news_per_page
        paginated_news = news[start_idx:end_idx]
        
        for item in paginated_news:
            content = item.get('content', {})
            if not isinstance(content, dict):
                content = {}
            
            # Extract relevant fields with default values
            title = content.get('title', 'No Title')
            summary = content.get('summary', 'No Summary Available')
            pub_date = content.get('pubDate', 'No Publication Date')
            provider = content.get('provider', {}).get('displayName', 'Unknown Provider')
            url = content.get('canonicalUrl', {}).get('url', '#')
            
            # Handle thumbnail safely
            thumbnail_data = content.get('thumbnail')
            if isinstance(thumbnail_data, dict):
                thumbnail_url = thumbnail_data.get('originalUrl', None)
            else:
                thumbnail_url = None

            # Format publication date
            try:
                pub_date_dt = datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%SZ")
                from_zone = tz.tzutc()
                to_zone = tz.tzlocal()
                pub_date_dt = pub_date_dt.replace(tzinfo=from_zone).astimezone(to_zone)
                pub_date = pub_date_dt.strftime("%B %d, %Y %H:%M %Z")
            except (ValueError, TypeError):
                pub_date = 'Unknown Date'

            # Create a container for each news item
            with st.container():
                if thumbnail_url:
                    cols = st.columns([1, 3])
                    with cols[0]:
                        try:
                            st.image(thumbnail_url, width=100)
                        except Exception as e:
                            logging.error(f"Failed to load image: {thumbnail_url} with error: {e}")
                            st.write("![No Image Available](https://via.placeholder.com/100)")
                    with cols[1]:
                        st.markdown(f"#### [{title}]({url})")
                        st.markdown(f"**Provider:** {provider} | **Published:** {pub_date}")
                        st.markdown(f"{summary}")
                else:
                    st.markdown(f"#### [{title}]({url})")
                    st.markdown(f"**Provider:** {provider} | **Published:** {pub_date}")
                    st.markdown(f"{summary}")
                
                st.markdown("---")  # Separator
    else:
        st.write("No news found for this ticker.")

