import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd


def show_search_page():
    st.title("📊 Sector and Industry Explorer")

    st.sidebar.subheader("Search Options")
    # Define sectors
    sectors = [
        "technology", "financial-services", "consumer-cyclical", "healthcare",
        "communication-services", "industrials", "consumer-defensive", "energy",
        "real-estate", "basic-materials", "utilities"
    ]

    # Initialize session state for industries
    if "indus" not in st.session_state:
        st.session_state.indus = ["semiconductors"]

    # Sector selection
    sector = st.sidebar.selectbox("Select a Sector:", sectors)
    if st.sidebar.button("Search Sector"):
        try:
            sector_data = yf.Sector(sector)
            st.session_state.indus = sector_data.industries.index
        except Exception as e:
            st.error(f"Error fetching sector data: {e}")

    # Industry selection
    industry = st.sidebar.selectbox("Select an Industry:", st.session_state.indus)
    if st.sidebar.button("Search Industry"):
        try:
            industry_data = yf.Industry(industry)
        except Exception as e:
            st.error(f"Error fetching industry data: {e}")
            industry_data = None

    # Tabs for organized visualization
    tabs = st.tabs(["Sector Overview", "Industry Overview",  "ETFs & Funds", "Growth & Performance"])

    # Sector Overview Tab
    with tabs[0]:
        st.header("📈 Sector Overview")
        if 'sector_data' in locals():
            overview = sector_data.overview

            # Display key metrics
            st.metric("Market Cap", f"${overview['market_cap']:,}")
            st.metric("Companies Count", overview["companies_count"])
            st.metric("Industries Count", overview["industries_count"])
            st.metric("Employee Count", overview["employee_count"])

            # Description
            st.write("**Description:**")
            st.write(overview["description"])

            # Industries visualization
            st.subheader("Industries in Sector")
            industries = sector_data.industries
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(industries["name"], industries["market weight"], color="skyblue")
            plt.xticks(rotation=45, ha="right")
            plt.title("Industries by Market Weight")
            st.pyplot(fig)

        else:
            st.write("Search for a sector to view details.")

    # Industry Overview Tab
    with tabs[1]:
        st.header("📊 Industry Overview")
        if 'industry_data' in locals():
            overview = industry_data.overview

            # Display key metrics
            st.metric("Market Cap", f"${overview['market_cap']:,}")
            st.metric("Companies Count", overview["companies_count"])

            # Description
            st.write("**Description:**")
            st.write(overview["description"])

            # Top Companies Visualization
            st.subheader("Top Companies in Industry")
            top_companies = industry_data.top_companies
            top_company_names = top_companies["name"][:20]
            top_company_symbols = top_companies.index[:20]
            market_weights = top_companies["market weight"][:20]
            ratings = top_companies["rating"][:20]

            # Create a bar chart for Top Companies
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(top_company_symbols, market_weights, color="skyblue")
            ax.set_title("Top Companies by Market Weight", fontsize=16)
            ax.set_xlabel("Market Weight", fontsize=12)
            ax.set_ylabel("Company Symbols", fontsize=12)

            # Add annotations for company names and ratings
            for bar, name, rating in zip(bars, top_company_names, ratings):
                width = bar.get_width()
                ax.text(
                    width + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{name} ({rating})",
                    va="center", fontsize=10
                )

            st.pyplot(fig)
        else:
            st.write("Search for an industry to view top companies.")

    # ETFs & Funds Tab
    with tabs[2]:
        st.header("💼 ETFs & Mutual Funds")
        if 'sector_data' in locals():
            # Top ETFs
            st.subheader("Top ETFs")
            for etf, name in sector_data.top_etfs.items():
                st.markdown(f"- **{etf}**: {name}")

            # Top Mutual Funds
            st.subheader("Top Mutual Funds")
            for fund, name in sector_data.top_mutual_funds.items():
                st.markdown(f"- **{fund}**: {name if name else 'N/A'}")
        else:
            st.write("Search for a sector to view ETFs and funds.")

    # Growth & Performance Tab
    with tabs[3]:
        st.header("📈 Growth & Performance")
        if 'industry_data' in locals():
            # Top Growth Companies
            st.subheader("Top Growth Companies")
            top_growth = industry_data.top_growth_companies
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(top_growth["name"], top_growth[" growth estimate"], color="green")
            plt.xticks(rotation=45, ha="right")
            plt.title("Top Growth Companies")
            plt.ylabel("Growth Estimate")
            st.pyplot(fig)

            # Top Performing Companies
            st.subheader("Top Performing Companies")
            top_performing = industry_data.top_performing_companies
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(top_performing["name"], top_performing["ytd return"], color="purple")
            plt.xticks(rotation=45, ha="right")
            plt.title("Top Performing Companies")
            plt.ylabel("YTD Return")
            st.pyplot(fig)
        else:
            st.write("Search for an industry to view growth and performance data.")


# Run the app
if __name__ == "__main__":
    show_search_page()
