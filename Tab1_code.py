# Tab1_code.py — Live Dynamic Universe Screener
import streamlit as st
import pandas as pd
from master_scraper import build_exhaustive_dataset

def create_tab1_interface():
    st.header("📊 Dynamic Market Screener & Alpha Universe Generator")
    st.markdown("Extracts live NSE data, Yahoo Finance fundamentals, and Google RSS catalysts. Filter the entire market down to high-probability swing trade candidates.")
    
    # --- STEP 1: EXTRACTION ENGINE ---
    with st.expander("📡 Step 1: Live Market Extraction Engine", expanded=True):
        st.info("⚠️ Scraping deep fundamentals and 15-day news buckets for the entire exchange takes time. Use a limit for faster testing.")
        col_ex1, col_ex2 = st.columns([1, 2])
        limit_val = col_ex1.number_input("Limit Stocks Scraped (Leave 0 for full exchange)", min_value=0, max_value=2500, value=50)
        
        if col_ex2.button("🚀 Generate Live Master Universe", use_container_width=True):
            limit_arg = limit_val if limit_val > 0 else None
            with st.spinner("Scraping Fundamentals, Solvency Ratios, and Temporal News..."):
                try:
                    live_df = build_exhaustive_dataset(limit=limit_arg)
                    if not live_df.empty:
                        st.session_state['master_universe'] = live_df
                        st.success(f"Successfully generated master universe of {len(live_df)} stocks!")
                    else:
                        st.error("Extraction failed. Check network or try again.")
                except Exception as e:
                    st.error(f"Scraper Error: {str(e)}")

    # --- STEP 2: EXPERT FILTERING CONSOLE ---
    if 'master_universe' in st.session_state:
        df = st.session_state['master_universe'].copy()
        
        st.markdown("---")
        st.subheader("⚙️ Step 2: Expert Institutional Screener")
        st.write("Refine your universe using institutional-grade parameters to ensure high liquidity and strong momentum.")
        
        # Use a form to prevent constant page reloading while dragging sliders
        with st.form("expert_screener_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### 💧 Liquidity & Price")
                min_price, max_price = st.slider("Price Range (₹)", 10.0, 10000.0, (20.0, 300.0))
                min_vol = st.number_input("Min Average Volume", min_value=0, value=500000, step=100000, help="Ensure enough liquidity for easy entry/exit without slippage.")
                max_drop_52 = st.slider("Max Drop from 52W High (%)", 0, 100, 25, help="Filters out falling knives. Stocks near 52W highs have better momentum.")
            
            with col2:
                st.markdown("### 🏛️ Quality & Valuation")
                min_roe = st.slider("Minimum ROE (%)", -50.0, 100.0, 15.0, help="Return on Equity > 15% usually indicates strong business moats.")
                max_pe = st.number_input("Maximum P/E Ratio (Trailing)", min_value=1.0, max_value=500.0, value=45.0, help="Filters out wildly overvalued companies.")
                max_debt = st.number_input("Maximum Debt-to-Equity", min_value=0.0, max_value=10.0, value=1.0, help="Avoid highly leveraged companies.")
            
            with col3:
                st.markdown("### 🔥 Catalysts")
                min_score = st.slider("Min 30-Day Catalyst Score", 0, 10, 1, help="Ensures the stock has recent news flow (earnings, contracts, upgrades).")
                
                st.markdown("<br><br>", unsafe_allow_html=True)
                submit_filters = st.form_submit_button("🧪 Apply Expert Filters", use_container_width=True)

        # Execute filter logic when button is pressed OR if we already have a generated list
        if submit_filters or 'selected_symbols' in st.session_state:
            f_df = df.copy()
            
            # Type safety
            for col in ['LTP (₹)', 'Avg Volume', '52W High', 'ROE (%)', 'P/E Ratio (Trailing)', 'Debt to Equity', '30D_Action_Score']:
                if col in f_df.columns:
                    f_df[col] = pd.to_numeric(f_df[col], errors='coerce')
            
            # Apply Liquidity & Price Filters
            if 'LTP (₹)' in f_df.columns:
                f_df = f_df[(f_df['LTP (₹)'] >= min_price) & (f_df['LTP (₹)'] <= max_price)]
            if 'Avg Volume' in f_df.columns:
                f_df = f_df[f_df['Avg Volume'] >= min_vol]
            if '52W High' in f_df.columns and 'LTP (₹)' in f_df.columns:
                drop_pct = ((f_df['52W High'] - f_df['LTP (₹)']) / (f_df['52W High'] + 1e-8)) * 100
                f_df = f_df[drop_pct <= max_drop_52]
                
            # Apply Quality & Valuation Filters
            if 'ROE (%)' in f_df.columns:
                f_df = f_df[f_df['ROE (%)'] >= min_roe]
            if 'P/E Ratio (Trailing)' in f_df.columns:
                # Strip out negative P/E (unprofitable) and respect the max bound
                f_df = f_df[(f_df['P/E Ratio (Trailing)'] > 0) & (f_df['P/E Ratio (Trailing)'] <= max_pe)]
            if 'Debt to Equity' in f_df.columns:
                f_df['Debt to Equity'] = f_df['Debt to Equity'].fillna(0) 
                f_df = f_df[f_df['Debt to Equity'] <= max_debt]
                
            # Apply Catalyst Filters
            if '30D_Action_Score' in f_df.columns:
                f_df = f_df[f_df['30D_Action_Score'] >= min_score]

            # Save the final universe safely to session state for downstream tabs
            st.session_state['selected_symbols'] = f_df['SYMBOL'].tolist()
            
            st.markdown("---")
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric("Initial Master Universe", len(df))
            col_res2.metric("Stocks Passing Filters", len(f_df))
            
            if len(f_df) == 0:
                st.error("🚨 Filter combinations are too strict! 0 stocks passed. Try loosening the Liquidity (Volume) or P/E bounds.")
            else:
                st.success("✅ Qualified Trading Universe Generated. The ML models in the next tabs will now pull exclusively from this alpha list.")
                
                display_cols = ['SYMBOL', 'NAME OF COMPANY', 'LTP (₹)', '1D Change (%)', 'Avg Volume', 'ROE (%)', 'P/E Ratio (Trailing)', 'Debt to Equity', '30D_Action_Score']
                available_cols = [c for c in display_cols if c in f_df.columns]
                
                st.dataframe(
                    f_df[available_cols].style.format({
                        'LTP (₹)': '₹{:.2f}',
                        '1D Change (%)': '{:.2f}%',
                        'Avg Volume': '{:,.0f}',
                        'ROE (%)': '{:.2f}%',
                        'P/E Ratio (Trailing)': '{:.1f}',
                        'Debt to Equity': '{:.2f}'
                    }).background_gradient(subset=['1D Change (%)', 'ROE (%)'], cmap='RdYlGn'),
                    use_container_width=True
                )
                
                csv = f_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Alpha Universe to CSV",
                    data=csv,
                    file_name="Alpha_Universe_Filtered.csv",
                    mime="text/csv"
                )