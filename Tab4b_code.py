# Tab4b_code.py — Advanced Multi-Horizon Models
import numpy as np
import pandas as pd
from session_data import prepare_multi_horizon_data
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

try:
    import catboost as cb
    CB_AVAILABLE = True
except ImportError:
    CB_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# Features expected by the models
# Features expected by the models (Updated with Macro & Depth Proxies)
CORE_FEATURES = [
    'RSI', 'MACD', 'BB_Position', 'Returns', 'SMA_10', 'EMA_10', 
    'Bid_Ask_Pressure', 'Vol_Contraction', 'NIFTY_Return', 'VIX_Close'
]

def clean_data_for_ml(df):
    """Ensures features exist and drops NaNs for ML ingestion."""
    data = df.copy()
    for f in CORE_FEATURES:
        if f not in data.columns: data[f] = 0
    return data

def train_dual_horizon_model(model_t1, model_t2, df):
    """Helper to train T+1 and T+2 simultaneously."""
    data = clean_data_for_ml(df)
    try:
        X, y_t1, y_t2, X_predict, current_close = prepare_multi_horizon_data(data, CORE_FEATURES)
        
        # Train T+1
        model_t1.fit(X, y_t1)
        pred_ret_t1 = model_t1.predict(X_predict)[0]
        
        # Train T+2
        model_t2.fit(X, y_t2)
        pred_ret_t2 = model_t2.predict(X_predict)[0]
        
        # Convert forecasted returns back to absolute prices
        pred_price_t1 = current_close * (1 + pred_ret_t1)
        pred_price_t2 = current_close * (1 + pred_ret_t2)
        
        return {"T+1": float(pred_price_t1), "T+2": float(pred_price_t2)}
    except Exception as e:
        return {"T+1": None, "T+2": None}

# --- 1. LightGBM ---
def lgb_predict(df, params=None):
    if params is None:
        params = {'n_estimators': 100, 'learning_rate': 0.05, 'random_state': 42, 'verbose': -1}
    else:
        params['verbose'] = -1
    return train_dual_horizon_model(lgb.LGBMRegressor(**params), lgb.LGBMRegressor(**params), df)

# --- 2. XGBoost ---
def xgb_predict(df):
    if not XGB_AVAILABLE: return {"T+1": None, "T+2": None}
    m1 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, objective='reg:squarederror', random_state=42)
    m2 = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, objective='reg:squarederror', random_state=42)
    return train_dual_horizon_model(m1, m2, df)

# --- 3. CatBoost ---
def cb_predict(df):
    if not CB_AVAILABLE: return {"T+1": None, "T+2": None}
    m1 = cb.CatBoostRegressor(iterations=100, learning_rate=0.05, verbose=0, random_seed=42)
    m2 = cb.CatBoostRegressor(iterations=100, learning_rate=0.05, verbose=0, random_seed=42)
    return train_dual_horizon_model(m1, m2, df)

# --- 4. Random Forest ---
def rf_predict(df):
    m1 = RandomForestRegressor(n_estimators=100, random_state=42)
    m2 = RandomForestRegressor(n_estimators=100, random_state=42)
    return train_dual_horizon_model(m1, m2, df)

# --- 5. Gradient Boosting (sklearn) ---
def gb_predict(df):
    m1 = GradientBoostingRegressor(n_estimators=100, random_state=42)
    m2 = GradientBoostingRegressor(n_estimators=100, random_state=42)
    return train_dual_horizon_model(m1, m2, df)

# --- 6. Ridge Regression (Linear Baseline) ---
def ridge_predict(df):
    m1 = Ridge(alpha=1.0)
    m2 = Ridge(alpha=1.0)
    return train_dual_horizon_model(m1, m2, df)

# --- Compose Model Validators ---
MODEL_VALIDATORS = {
    'LightGBM': lgb_predict,
    'RandomForest': rf_predict,
    'GradientBoosting': gb_predict,
    'RidgeLinear': ridge_predict
}

if XGB_AVAILABLE:
    MODEL_VALIDATORS['XGBoost'] = xgb_predict
if CB_AVAILABLE:
    MODEL_VALIDATORS['CatBoost'] = cb_predict