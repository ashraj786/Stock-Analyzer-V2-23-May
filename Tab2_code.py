# Tab2_code.py — Performance Analysis & Deep Tuning
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna
from session_data import get_data_cache

def plot_expert_chart(df: pd.DataFrame, symbol: str):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=(f'{symbol} Price & BB', 'Volume'),
                        row_width=[0.2, 0.7])

    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'],
                                 low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
    
    if 'SMA_20' in df.columns:
        std_20 = df['Close'].rolling(window=20).std()
        upper = df['SMA_20'] + (std_20 * 2)
        lower = df['SMA_20'] - (std_20 * 2)
        fig.add_trace(go.Scatter(x=df.index, y=upper, line=dict(color='gray', width=1, dash='dot'), name='BB Upper'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=lower, line=dict(color='gray', width=1, dash='dot'), name='BB Lower'), row=1, col=1)

    colors = ['green' if row['Close'] >= row['Open'] else 'red' for _, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color=colors, name='Volume'), row=2, col=1)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template='plotly_dark')
    return fig

def run_optuna_tuning(X, y):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0)
        }
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            
            model = lgb.LGBMRegressor(**params, random_state=42, verbose=-1)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            scores.append(mean_squared_error(y_te, preds))
        return np.mean(scores)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15) 
    return study.best_params, study.best_value

def create_tab2_interface():
    st.header("📈 Expert Performance Analysis & Deep Tuning")
    
    if 'selected_symbols' not in st.session_state or not st.session_state.selected_symbols:
        st.warning("Please generate or select a qualified universe in Tab 1 first.")
        return

    selected_symbol = st.selectbox("Select Qualified Stock", st.session_state.selected_symbols)
    cache = get_data_cache()
    
    with st.spinner(f"Loading deep metrics for {selected_symbol}..."):
        df = cache.get_feature_data(selected_symbol)
        
    if df.empty:
        st.error("Insufficient data for this stock.")
        return

    last = df.iloc[-1]
    st.subheader(f"Dashboard: {selected_symbol}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current LTP", f"₹{last.get('Close', 0):.2f}", f"{last.get('Returns', 0)*100:.2f}%")
    col2.metric("RSI (14)", f"{last.get('RSI', 0):.1f}")
    col3.metric("Valuation", last.get('Valuation_Category', 'Unknown'))
    col4.metric("30D Catalysts", last.get('30D_Action_Score', 0))

    st.plotly_chart(plot_expert_chart(df, selected_symbol), use_container_width=True)

    st.markdown("---")
    st.subheader("🧪 Deep Hyperparameter Auto-Tuning (LightGBM)")
    st.write("Finds optimal model parameters for this specific stock's volatility profile using Time-Series Cross Validation.")
    
    if st.button(f"Run Deep Auto-Tune for {selected_symbol}"):
        from session_data import prepare_multi_horizon_data
        features = ['RSI', 'MACD', 'BB_Position', 'Returns', 'Log_Returns']
        for f in features:
            if f not in df.columns: df[f] = 0
            
        with st.spinner("Running Optuna Trials..."):
            try:
                X, y_t1, _, _, _ = prepare_multi_horizon_data(df, features)
                best_params, best_mse = run_optuna_tuning(X, y_t1)
                
                st.success("Tuning Complete!")
                st.json(best_params)
                
                if 'tuned_params' not in st.session_state: st.session_state['tuned_params'] = {}
                st.session_state['tuned_params'][selected_symbol] = best_params
            except Exception as e:
                st.error(f"Tuning failed: {str(e)}")