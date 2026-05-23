# combined_app.py — Final Unified Streamlit App
import streamlit as st

# Import all tab modules
from session_data import get_data_cache
from Tab1_code import create_tab1_interface
from Tab2_code import create_tab2_interface
from Tab3_code import create_tab3_interface
from Tab3b_code import create_tab3b_interface
from Tab7_validation import create_tab7_interface
from Tab8_summary import create_tab8_interface
from Tab9_backtest_insights import create_tab9_interface
from Tab10_expert_calls import create_tab10_interface

def main():
    st.set_page_config(
        page_title="NSE Alpha Console", 
        layout="wide", 
        initial_sidebar_state="auto" 
    )
    st.title("📈 NSE Stock Prediction and Analysis System")

    # Initialize centralized cache once
    _ = get_data_cache()

    # Define tabs consistent with roadmap
    tabs_labels = [
        "📊 Stock Screener",
        "📈 Performance Analysis",
        "🤖 Base Models",
        "🧪 Advanced Models",
        "🔬 Single Rolling Validation",
        "📊 Validation Summary",
        "💼 Portfolio Backtest & Insights",
        "🎯 Alpha Console (Expert Analyst)"
    ]

    tabs = st.tabs(tabs_labels)

    with tabs[0]:
        create_tab1_interface()
    with tabs[1]:
        create_tab2_interface()
    with tabs[2]:
        create_tab3_interface()
    with tabs[3]:
        create_tab3b_interface()
    with tabs[4]:
        create_tab7_interface()
    with tabs[5]:
        create_tab8_interface()
    with tabs[6]:
        create_tab9_interface()
    with tabs[7]:
        create_tab10_interface()

if __name__ == "__main__":
    main()