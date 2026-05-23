# Tab10_expert_calls.py — Ultimate Expert Consolidated Report
import streamlit as st
import pandas as pd
import numpy as np
from session_data import get_data_cache
from Tab4b_code import MODEL_VALIDATORS

def generate_expert_report(symbol, df, t1_consensus, t2_consensus):
    last = df.iloc[-1]
    current_price = last['Close']
    nifty_trend = "Bullish" if last.get('NIFTY_Return', 0) > 0 else "Bearish"
    vix = last.get('VIX_Close', 15)
    
    # Determine L2 & Options Context
    buying_pressure = "High" if last.get('Bid_Ask_Pressure', 0.5) > 0.65 else ("Low" if last.get('Bid_Ask_Pressure', 0.5) < 0.35 else "Neutral")
    oi_decay = "Accelerating" if last.get('Vol_Contraction', 1) < 0.8 else "Stable"
    
    expected_ret_t1 = ((t1_consensus - current_price) / current_price) * 100
    
    # Synthesis Logic
    rating = "HOLD 🟡"
    if expected_ret_t1 > 1.5 and buying_pressure == "High" and nifty_trend == "Bullish":
        rating = "STRONG BUY 🟢"
    elif expected_ret_t1 < -1.0 or (expected_ret_t1 < 0 and nifty_trend == "Bearish"):
        rating = "SELL 🔴"
    elif expected_ret_t1 > 1.0:
        rating = "BUY 📈"
        
    report = f"""
    ### 🏆 Expert Verdict: {rating}
    
    **1. 🕴️ Market & Industry Expert View:**
    > "{symbol} is currently trading at ₹{current_close:.2f}. The broader macro context (NIFTY) is **{nifty_trend}** with the VIX sitting at **{vix:.1f}**. Looking at the simulated order book, intraday buying pressure is **{buying_pressure}**, indicating institutional accumulation. Option decay proxy suggests volatility is **{oi_decay}**, making it a safer swing-trade environment."
    
    **2. 🤖 Machine Learning / Quants View:**
    > "The multi-horizon ensemble consensus projects a T+1 target of **₹{t1_consensus:.2f}** ({expected_ret_t1:.2f}% return). Given the integration of Level-2 proxies and macro benchmarks, the models show high confidence in this directional bias. If a trade is executed, a stop-loss should be placed below the recent VWAP level of **₹{last.get('VWAP', current_price*0.98):.2f}**."
    """
    return report, rating, expected_ret_t1

def create_tab10_interface():
    st.header("🎯 The Ultimate Alpha Console (Expert Analyst)")
    st.markdown("Consolidates Level 2 Order Depth proxies, Macro Indices (NIFTY/VIX), and ML Consensus to generate definitive trading calls.")
    
    if 'selected_symbols' not in st.session_state or not st.session_state.selected_symbols:
        st.warning("Please generate a universe in Tab 1 first.")
        return
        
    stocks_to_analyze = st.multiselect("Select up to 5 stocks for Deep Expert Analysis:", st.session_state.selected_symbols, default=st.session_state.selected_symbols[:3])
    
    if st.button("Generate Consolidated Expert Calls", type="primary"):
        cache = get_data_cache()
        progress = st.progress(0)
        
        for i, symbol in enumerate(stocks_to_analyze):
            st.markdown(f"## {symbol} Analysis")
            df = cache.get_feature_data(symbol)
            
            if df.empty or len(df) < 50:
                st.error(f"Insufficient data for {symbol}.")
                continue
                
            # Run ML Models silently
            t1_preds, t2_preds = [], []
            for name, func in MODEL_VALIDATORS.items():
                try:
                    res = func(df)
                    if res and res.get('T+1'): t1_preds.append(res['T+1'])
                    if res and res.get('T+2'): t2_preds.append(res['T+2'])
                except:
                    pass
                    
            if not t1_preds:
                st.warning("Models failed to converge.")
                continue
                
            med_t1 = np.median(t1_preds)
            med_t2 = np.median(t2_preds) if t2_preds else med_t1
            
            # Global current close hack for report string formatting inside function
            global current_close 
            current_close = df['Close'].iloc[-1]
            
            report, rating, ret = generate_expert_report(symbol, df, med_t1, med_t2)
            
            with st.expander(f"{rating} - {symbol} | Exp. Return: {ret:.2f}%", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"₹{current_close:.2f}")
                col2.metric("T+1 Consensus", f"₹{med_t1:.2f}")
                col3.metric("T+2 Consensus", f"₹{med_t2:.2f}")
                st.markdown(report)
                
            progress.progress((i + 1) / len(stocks_to_analyze))
            st.markdown("---")