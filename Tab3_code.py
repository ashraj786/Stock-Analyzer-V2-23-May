# Tab3_code.py — Base Model Predictions (Single Stock)
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import r2_score

from session_data import get_data_cache

def train_base_model(df, model_type='lightgbm'):
    # Standardized features matching the new architecture
    required_features = ['RSI', 'MACD', 'BB_Position', 'SMA_10', 'EMA_10', 'Returns']
    
    df = df.copy()
    for col in required_features:
        if col not in df.columns:
            df[col] = 0

    # Create T+1 target for base models
    df['Target'] = df['Close'].shift(-1)
    train_df = df.dropna(subset=required_features + ['Target'])
    
    if len(train_df) < 30:
        return None, 0
        
    X = train_df[required_features].values
    y = train_df['Target'].values
    
    X_pred = df.iloc[-1:][required_features].values

    if model_type == 'lightgbm':
        model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1)
    elif model_type == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'gb':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'ridge':
        model = Ridge(alpha=1.0)
    elif model_type == 'lasso':
        model = Lasso(alpha=0.1)
    elif model_type == 'svr':
        model = SVR(kernel='rbf')
    else:
        return None, 0

    model.fit(X, y)
    pred = model.predict(X_pred)[0]
    
    # Calculate basic R2 on training set just for confidence metric
    train_preds = model.predict(X)
    r2 = r2_score(y, train_preds)
    
    return pred, r2

def run_base_predictions(symbol):
    cache = get_data_cache()
    df = cache.get_feature_data(symbol)
    
    if df.empty or len(df) < 50:
        st.warning(f"Insufficient historical data for {symbol}.")
        return

    current_price = df['Close'].iloc[-1]
    
    models = {
        'LightGBM': 'lightgbm',
        'Random Forest': 'rf',
        'Gradient Boosting': 'gb',
        'Ridge Regression': 'ridge',
        'Lasso Regression': 'lasso',
        'SVR': 'svr'
    }
    
    results = []
    progress = st.progress(0)
    
    for i, (name, key) in enumerate(models.items()):
        try:
            pred, r2 = train_base_model(df, key)
            if pred:
                ret = ((pred - current_price) / current_price) * 100
                results.append({
                    'Model': name,
                    'Predicted_Price': pred,
                    'Expected_Return_%': ret,
                    'Confidence_R2': r2
                })
        except Exception as e:
            pass
        progress.progress((i + 1) / len(models))
        
    if results:
        res_df = pd.DataFrame(results).sort_values(by='Expected_Return_%', ascending=False)
        st.subheader("📊 Base Models Prediction Summary (T+1)")
        
        col1, col2, col3 = st.columns(3)
        med_price = res_df['Predicted_Price'].median()
        med_ret = res_df['Expected_Return_%'].median()
        
        # Avoid division by zero if all predictions are identical
        if med_price != 0:
            agreement = 1 - ((res_df['Predicted_Price'].max() - res_df['Predicted_Price'].min()) / med_price)
        else:
            agreement = 1.0
            
        col1.metric("Consensus Target Price", f"₹{med_price:.2f}")
        col2.metric("Consensus Expected Return", f"{med_ret:.2f}%")
        col3.metric("Model Agreement", f"{agreement*100:.1f}%")
        
        st.dataframe(res_df.style.format({
            "Predicted_Price": "₹{:.2f}",
            "Expected_Return_%": "{:.2f}%",
            "Confidence_R2": "{:.2f}"
        }).background_gradient(subset=["Expected_Return_%"], cmap="RdYlGn"), use_container_width=True)

def create_tab3_interface():
    st.header("🤖 Base Model Predictions")
    st.markdown("Runs standard Scikit-Learn and LightGBM base regressors to predict the next day's (T+1) closing price.")
    
    # --- CRITICAL FIX: Checking for the correct session state variable ---
    if 'selected_symbols' not in st.session_state or not st.session_state.selected_symbols:
        st.warning("Please generate or select a qualified universe in Tab 1 first.")
        return
    # ---------------------------------------------------------------------

    selected_symbol = st.selectbox("Select Stock for Base Modeling", st.session_state.selected_symbols, key='tab3_sym')
    
    if st.button(f"Run Base Models for {selected_symbol}", type="primary"):
        with st.spinner("Training base machine learning models..."):
            run_base_predictions(selected_symbol)