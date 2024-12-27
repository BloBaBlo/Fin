# tabs/search_tab.py
import streamlit as st
import yfinance as yf

def show_search_page():
    st.subheader("Search for sector technology / financial-services / consumer-cyclical / healthcare")
    st.subheader("communication-services / industrials / consumer-defensive / energy")
    st.subheader("real-estate / basic-materials / utilities")

    search_inputs = st.sidebar.text_input("Enter a field", value="technology")
    search_button = st.sidebar.button("Search")

    st.subheader("Search for Industry")
    search_inputs2 = st.sidebar.text_input("Enter a field", value="semiconductors")
    search_button2 = st.sidebar.button("Search 2")

    if search_button or search_button2:
        if search_button:
            search_input = yf.Sector(search_inputs)
        else:
            search_input = yf.Industry(search_inputs2)

        key = getattr(search_input, "key", None)
        if key:
            st.write("### Key")
            st.write(key)

        symbol = getattr(search_input, "symbol", None)
        if symbol:
            st.write("### symbol")
            st.write(symbol)

        try:
            tick = search_input.ticker
            if tick:
                st.write("### ticker")
                st.write(tick)
        except:
            st.write("No Ticker")

        overview = getattr(search_input, "overview", None)
        if overview:
            st.write("### overview")
            st.write(overview)

        top_companies = getattr(search_input, "top_companies", None)
        if top_companies is not None:
            st.write("### top_companies")
            st.write(top_companies)

        research_reports = getattr(search_input, "research_reports", None)
        if research_reports:
            st.write("### research_reports")
            st.write(research_reports)

        try:
            top_etfs = search_input.top_etfs
            if top_etfs:
                st.write("### top_etfs")
                st.write(top_etfs)
        except:
            pass

        try:
            top_mutual_funds = search_input.top_mutual_funds
            if top_mutual_funds:
                st.write("### top_mutual_funds")
                st.write(top_mutual_funds)
        except:
            pass

        try:
            industries = search_input.industries
            if industries is not None:
                st.write("### industries")
                st.write(industries)
        except:
            pass

        try:
            sector_key = search_input.sector_key
            if sector_key:
                st.write("### sector_key")
                st.write(sector_key)
        except:
            pass

        try:
            sector_name = search_input.sector_name
            if sector_name is not None:
                st.write("### sector_name")
                st.write(sector_name)
        except:
            pass

        try:
            top_performing_companies = search_input.top_performing_companies
            if top_performing_companies:
                st.write("### top_performing_companies")
                st.write(top_performing_companies)
        except:
            pass

        try:
            top_growth_companies = search_input.top_growth_companies
            if top_growth_companies is not None:
                st.write("### top_growth_companies")
                st.write(top_growth_companies)
        except:
            pass
