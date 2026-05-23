# Tab3b_code.py — Advanced Multi-Horizon Predictions UI
import streamlit as st
import pandas as pd
import numpy as np
from session_data import get_data_cache
from Tab4b_code import MODEL_VALIDATORS

def run_all_advanced_models(symbol: str):
    cache = get_data_cache()
    df = cache.get_feature_data(symbol)
    
    if df.empty or len(df) < 50:
        st.warning(f"Insufficient historical data for {symbol}.")
        return

    # Check if Optuna deep-tuning was run in Tab 2
    tuned_params = st.session_state.get('tuned_params', {}).get(symbol, None)
    current_close = df['Close'].iloc[-1]
    
    st.write(f"### Running Multi-Horizon Models for {symbol}")
    st.write(f"**Current LTP:** ₹{current_close:.2f}")
    if tuned_params:
        st.success("✨ Utilizing Optuna Deep-Tuned Parameters for LightGBM")

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    models_list = list(MODEL_VALIDATORS.items())
    
    for i, (name, func) in enumerate(models_list):
        status_text.text(f"Executing {name} Engine...")
        try:
            # Inject tuned params if available and applicable
            if name == 'LightGBM' and tuned_params:
                res = func(df, params=tuned_params)
            else:
                res = func(df)
                
            if res and res.get("T+1"):
                ret_t1 = ((res["T+1"] - current_close) / current_close) * 100
                ret_t2 = ((res["T+2"] - current_close) / current_close) * 100 if res.get("T+2") else None
                
                results.append({
                    "Model": name,
                    "T+1_Price": res["T+1"],
                    "T+1_Return_%": ret_t1,
                    "T+2_Price": res.get("T+2"),
                    "T+2_Return_%": ret_t2
                })
        except Exception as e:
            st.error(f"{name} Failed: {str(e)}")
            
        progress_bar.progress((i + 1) / len(models_list))
        
    status_text.text("All models completed.")
    
    if not results:
        st.error("All models failed to generate predictions.")
        return
        
    # --- UI Formatting ---
    res_df = pd.DataFrame(results).sort_values(by="T+1_Return_%", ascending=False)
    
    # Consensus Metrics
    med_t1 = res_df["T+1_Price"].median()
    med_t2 = res_df["T+2_Price"].median()
    med_ret_t1 = res_df["T+1_Return_%"].median()
    med_ret_t2 = res_df["T+2_Return_%"].median()
    
    st.subheader("🤝 Multi-Horizon Consensus")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("T+1 Consensus Price", f"₹{med_t1:.2f}", f"{med_ret_t1:.2f}%")
    col2.metric("T+2 Consensus Price", f"₹{med_t2:.2f}", f"{med_ret_t2:.2f}%")
    
    # Volatility / Agreement (Range divided by Median)
    agreement_t1 = 1 - ((res_df["T+1_Price"].max() - res_df["T+1_Price"].min()) / med_t1)
    col3.metric("T+1 Model Agreement", f"{agreement_t1*100:.1f}%")

    st.subheader("📊 Individual Model Outputs")
    styled_df = res_df.style.format({
        "T+1_Price": "₹{:.2f}", "T+1_Return_%": "{:.2f}%",
        "T+2_Price": "₹{:.2f}", "T+2_Return_%": "{:.2f}%"
    }).background_gradient(subset=["T+1_Return_%", "T+2_Return_%"], cmap="RdYlGn")
    
    st.dataframe(styled_df, use_container_width=True)
    
    st.download_button(
        label="📥 Download Model Predictions CSV",
        data=res_df.to_csv(index=False),
        file_name=f"{symbol}_MultiHorizon_Predictions.csv",
        mime="text/csv"
    )

def create_tab3b_interface():
    st.header("🧪 Advanced Models (13+) Multi-Horizon")
    
    if 'selected_symbols' not in st.session_state or not st.session_state.selected_symbols:
        st.warning("Please generate or select a qualified universe in Tab 1 first.")
        return

    selected_symbol = st.selectbox("Select Qualified Stock to Analyze", st.session_state.selected_symbols, key='tab3b_sym')
    
    if st.button(f"Run Advanced Models for {selected_symbol}", type="primary"):
        run_all_advanced_models(selected_symbol)