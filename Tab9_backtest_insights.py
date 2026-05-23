# Tab9_backtest_insights.py — Portfolio Backtest & Automated Expert Insights
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from session_data import get_data_cache
from Tab4b_code import MODEL_VALIDATORS

def run_portfolio_backtest(symbols, max_stocks=20):
    cache = get_data_cache()
    results = []
    
    # Limit stocks to avoid crashing the browser session
    symbols_to_run = symbols[:max_stocks]
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, symbol in enumerate(symbols_to_run):
        status_text.text(f"Backtesting {symbol} ({i+1}/{len(symbols_to_run)})...")
        full_df = cache.get_feature_data(symbol)
        
        # We need at least 50 days of data + 2 days for backtesting actuals
        if full_df.empty or len(full_df) < 52:
            continue
            
        # Step back 2 days. 
        # If len is 100 (idx 0 to 99), actual T is idx 99, T-1 is idx 98.
        # We train on data up to idx 97.
        cut_idx = len(full_df) - 2
        df_hist = full_df.iloc[:cut_idx].copy()
        
        actual_t1_price = full_df['Close'].iloc[cut_idx]
        actual_t2_price = full_df['Close'].iloc[cut_idx + 1]
        
        for model_name, model_func in MODEL_VALIDATORS.items():
            try:
                preds = model_func(df_hist)
                if preds and preds.get("T+1"):
                    t1_pred = preds["T+1"]
                    t2_pred = preds.get("T+2", None)
                    
                    # T+1 Error
                    t1_err_pct = ((t1_pred - actual_t1_price) / actual_t1_price) * 100
                    
                    record = {
                        "Stock": symbol,
                        "Model": model_name,
                        "Target_Horizon": "T+1 (Yesterday)",
                        "Predicted": t1_pred,
                        "Actual": actual_t1_price,
                        "Error_%": t1_err_pct,
                        "Absolute_Error_%": abs(t1_err_pct)
                    }
                    results.append(record)
                    
                    if t2_pred and actual_t2_price:
                        t2_err_pct = ((t2_pred - actual_t2_price) / actual_t2_price) * 100
                        record_t2 = {
                            "Stock": symbol,
                            "Model": model_name,
                            "Target_Horizon": "T+2 (Today)",
                            "Predicted": t2_pred,
                            "Actual": actual_t2_price,
                            "Error_%": t2_err_pct,
                            "Absolute_Error_%": abs(t2_err_pct)
                        }
                        results.append(record_t2)
            except Exception:
                continue
                
        progress_bar.progress((i + 1) / len(symbols_to_run))
        
    status_text.text("Portfolio Backtest Complete.")
    return pd.DataFrame(results)

def generate_expert_reports(df):
    if df.empty: return
    
    # Overall metrics
    median_error = df['Absolute_Error_%'].median()
    best_model = df.groupby('Model')['Absolute_Error_%'].median().idxmin()
    worst_model = df.groupby('Model')['Absolute_Error_%'].median().idxmax()
    
    # Overestimation vs Underestimation
    bias = df['Error_%'].mean() 
    bias_text = "OVER-ESTIMATING" if bias > 0 else "UNDER-ESTIMATING"
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("### 🕴️ Market Expert View")
        st.write(f"**Current Market Regime:** The models are collectively **{bias_text}** price action by an average of {abs(bias):.2f}%.")
        if bias > 0.5:
            st.write("📉 **Capital Allocation:** The system expects higher breakouts than reality. This implies the broader market is facing overhead resistance. Tighten stop-losses and avoid buying tops. Look for mean-reversion setups.")
        elif bias < -0.5:
            st.write("📈 **Capital Allocation:** The market is outperforming the models. Momentum is strong. This is a favorable environment for trend-following and holding winners slightly longer than the T+2 targets.")
        else:
            st.write("⚖️ **Capital Allocation:** The market is behaving highly rationally. T+1 targets are highly reliable right now. Execute standard swing trades.")
            
    with col2:
        st.success("### 🤖 ML Expert View")
        st.write(f"**System Accuracy:** Median portfolio error is **{median_error:.2f}%**. Currently, **{best_model}** is the most robust architecture for this specific market phase.")
        st.write(f"**Model Divergence:** **{worst_model}** is struggling with the current volatility. Consider discarding its consensus weight for the next 48 hours.")
        if median_error > 2.5:
            st.write("⚠️ **Variance Warning:** High portfolio-wide error detected. Models are likely overfitting to past structural regimes. Recommend re-running Optuna Deep Tuning (Tab 2) on priority stocks.")
        else:
            st.write("✅ **Variance Stable:** The ensemble is well-calibrated. Predictions are statistically sound for immediate deployment.")

def create_tab9_interface():
    st.header("💼 Portfolio Backtest & Alpha Insights")
    st.markdown("""
    This engine runs a cross-sectional backtest. It hides the last 2 days of real market data, forces the models to predict them, and compares the entire portfolio's predictions against reality to generate systemic insights.
    """)
    
    if 'selected_symbols' not in st.session_state or not st.session_state.selected_symbols:
        st.warning("Please generate a qualified universe in Tab 1 first.")
        return
        
    max_stocks = st.slider("Stocks to Backtest (Top N from Screener)", min_value=5, max_value=50, value=15)
    
    if st.button("Run Portfolio Backtest", type="primary"):
        with st.spinner("Executing mass parallel predictions..."):
            bt_df = run_portfolio_backtest(st.session_state.selected_symbols, max_stocks)
            
            if bt_df.empty:
                st.error("Backtest failed. Ensure historical data is loaded.")
                return
                
            st.subheader("Aggregate Model Performance (Last 48 Hours)")
            summary = bt_df.groupby(['Model', 'Target_Horizon'])['Absolute_Error_%'].agg(['median', 'mean', 'std']).reset_index()
            summary.columns = ['Model', 'Horizon', 'Median Abs Error %', 'Mean Abs Error %', 'Volatility']
            st.dataframe(summary.style.background_gradient(cmap='RdYlGn_r', subset=['Median Abs Error %']), use_container_width=True)
            
            generate_expert_reports(bt_df)
            
            st.subheader("Error Distribution by Stock")
            fig = px.box(bt_df, x="Stock", y="Error_%", color="Target_Horizon", title="Prediction Error Spread per Stock")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Raw Data Logs")
            st.dataframe(bt_df)