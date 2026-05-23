# Tab8_summary.py — Multi-Horizon Consensus & Validation Report
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def aggregate_all_validation_results():
    store = st.session_state.get('validation_store', {})
    if not store:
        return pd.DataFrame()

    all_rows = []
    for stock, df in store.items():
        for _, row in df.iterrows():
            r = row.copy()
            r['Stock'] = stock
            all_rows.append(r)

    return pd.DataFrame(all_rows)

def create_tab8_interface():
    st.header("📊 Multi-Horizon Validation Summary")
    
    df_validations = aggregate_all_validation_results()
    
    if df_validations.empty:
        st.info("No validation data found. Please run the Rolling Validation in Tab 7 for at least one stock.")
        return
        
    st.subheader("Aggregate Error by Model & Horizon")
    
    grouped = df_validations.groupby(['Model', 'Target_Horizon'])['Error_%'].agg(
        Avg_Absolute_Error_Pct=lambda x: np.mean(np.abs(x)),
        Volatility_of_Error='std',
        Predictions_Made='count'
    ).reset_index()
    
    grouped = grouped.sort_values(by=['Target_Horizon', 'Avg_Absolute_Error_Pct'])
    st.dataframe(grouped.style.background_gradient(cmap='RdYlGn_r', subset=['Avg_Absolute_Error_Pct']), use_container_width=True)

    st.subheader("Consensus Accuracy Trend (T+1)")
    t1_data = df_validations[df_validations['Target_Horizon'] == 'T+1']
    if not t1_data.empty:
        trend = t1_data.groupby('Prediction_Date')['Error_%'].agg(lambda x: np.mean(np.abs(x))).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=trend['Prediction_Date'], y=trend['Error_%'], marker_color='orange', name='Mean Abs Error %'))
        fig.update_layout(template='plotly_dark', xaxis_title="Prediction Date", yaxis_title="Average Error (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Performing Models by Stock (T+1)")
    stock_best_models = []
    for stock in t1_data['Stock'].unique():
        stock_df = t1_data[t1_data['Stock'] == stock]
        best_model = stock_df.groupby('Model')['Error_%'].agg(lambda x: np.mean(np.abs(x))).idxmin()
        best_error = stock_df.groupby('Model')['Error_%'].agg(lambda x: np.mean(np.abs(x))).min()
        
        stock_best_models.append({
            "Stock": stock,
            "Most Accurate Model": best_model,
            "T+1 Avg Error (%)": round(best_error, 2)
        })
        
    st.dataframe(pd.DataFrame(stock_best_models), use_container_width=True)