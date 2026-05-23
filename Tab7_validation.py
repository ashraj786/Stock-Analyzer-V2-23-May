# Tab7_validation.py — 3-Day Rolling Validation
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from Tab4b_code import MODEL_VALIDATORS
from session_data import get_data_cache

def run_rolling_validation(symbol: str, days_back=3):
    cache = get_data_cache()
    full_df = cache.get_feature_data(symbol)
    
    if len(full_df) < 100:
        st.error(f"Not enough data for rolling validation for {symbol}.")
        return pd.DataFrame()
        
    validation_records = []
    progress_bar = st.progress(0)
    
    for step, i in enumerate(range(days_back, 0, -1)):
        cut_idx = len(full_df) - i
        df_hist = full_df.iloc[:cut_idx].copy()
        
        actual_t1_idx = cut_idx 
        actual_t2_idx = cut_idx + 1
        
        if actual_t1_idx >= len(full_df): continue
        
        date_of_prediction = df_hist.index[-1]
        actual_t1_price = full_df['Close'].iloc[actual_t1_idx]
        actual_t2_price = full_df['Close'].iloc[actual_t2_idx] if actual_t2_idx < len(full_df) else None
        
        for name, func in MODEL_VALIDATORS.items():
            try:
                res = func(df_hist)
                if res and res.get("T+1"):
                    record = {
                        "Model": name,
                        "Prediction_Date": date_of_prediction.strftime("%Y-%m-%d"),
                        "Target_Horizon": "T+1",
                        "Predicted_Price": res["T+1"],
                        "Actual_Price": actual_t1_price,
                        "Error_%": ((res["T+1"] - actual_t1_price) / actual_t1_price) * 100
                    }
                    validation_records.append(record)
                    
                    if actual_t2_price and res.get("T+2"):
                        record_t2 = {
                            "Model": name,
                            "Prediction_Date": date_of_prediction.strftime("%Y-%m-%d"),
                            "Target_Horizon": "T+2",
                            "Predicted_Price": res["T+2"],
                            "Actual_Price": actual_t2_price,
                            "Error_%": ((res["T+2"] - actual_t2_price) / actual_t2_price) * 100
                        }
                        validation_records.append(record_t2)
            except Exception:
                pass
                
        progress_bar.progress((step + 1) / days_back)
        
    return pd.DataFrame(validation_records)

def create_tab7_interface():
    st.header("🔬 3-Day Rolling Multi-Horizon Validation")
    
    if 'selected_symbols' not in st.session_state or not st.session_state.selected_symbols:
        st.warning("Generate universe in Tab 1 first.")
        return

    symbol = st.selectbox("Select Stock for Validation", st.session_state.selected_symbols, key='tab7_sym')
    
    if st.button("Run Rolling Validation"):
        with st.spinner("Backtesting models without look-ahead bias..."):
            val_df = run_rolling_validation(symbol)
            
            if val_df.empty:
                st.warning("Validation failed or insufficient data.")
                return
            
            # Save properly for Tab 8
            if 'validation_store' not in st.session_state:
                st.session_state['validation_store'] = {}
            st.session_state['validation_store'][symbol] = val_df
                
            st.subheader("Model Performance Summary")
            summary = val_df.groupby(['Model', 'Target_Horizon'])['Error_%'].agg(['mean', 'std']).reset_index()
            summary.columns = ['Model', 'Horizon', 'Avg Error (%)', 'Volatility of Error']
            st.dataframe(summary.style.background_gradient(subset=['Avg Error (%)'], cmap='coolwarm'), use_container_width=True)
            
            st.subheader("Consensus vs Actual Trend")
            t1_data = val_df[val_df['Target_Horizon'] == 'T+1']
            consensus = t1_data.groupby('Prediction_Date')['Predicted_Price'].median().reset_index()
            actuals = t1_data.groupby('Prediction_Date')['Actual_Price'].first().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=consensus['Prediction_Date'], y=consensus['Predicted_Price'], mode='lines+markers', name='T+1 Consensus'))
            fig.add_trace(go.Scatter(x=actuals['Prediction_Date'], y=actuals['Actual_Price'], mode='lines+markers', name='Actual Price'))
            fig.update_layout(template='plotly_dark', xaxis_title="Date Models Were Run", yaxis_title="Price (₹)")
            st.plotly_chart(fig, use_container_width=True)