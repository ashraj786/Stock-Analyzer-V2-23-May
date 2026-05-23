# session_data.py — Data Cache, Macro Injection & ML Feature Engineering
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def engineer_expert_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw fundamentals and technicals into ML-optimized categorical bins."""
    data = df.copy()
    
    # Valuation: Industry Relative P/E
    if 'Industry' in data.columns and 'P/E Ratio (Trailing)' in data.columns:
        data['Industry_Median_PE'] = data.groupby('Industry')['P/E Ratio (Trailing)'].transform('median')
        data['PE_Relative'] = data['P/E Ratio (Trailing)'] / (data['Industry_Median_PE'] + 1e-8)
        data['Valuation_Category'] = np.select(
            [data['PE_Relative'] < 0.8, (data['PE_Relative'] >= 0.8) & (data['PE_Relative'] <= 1.2), data['PE_Relative'] > 1.2],
            ['Undervalued', 'Fair', 'Overvalued'], default='Unknown'
        )
        
    # Quality: ROE
    if 'ROE (%)' in data.columns:
        data['ROE_Category'] = pd.cut(data['ROE (%)'], bins=[-np.inf, 10, 20, np.inf], labels=['Low', 'Healthy', 'Exceptional'])
        
    return data

def calculate_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute standard features, plus L2 Depth Proxies and Option Proxies."""
    data = df.copy()

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    if data.empty: return data

    # Standard Features
    data['Returns'] = data['Close'].pct_change().fillna(0)
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1)).fillna(0)
    
    for window in [10, 20]:
        data[f'SMA_{window}'] = data['Close'].rolling(window).mean()
        data[f'EMA_{window}'] = data['Close'].ewm(span=window, adjust=False).mean()
    
    # 1. Level 2 Order Book Proxy: Volume Imbalance & VWAP
    data['Typical_Price'] = (data['High'] + data['Low'] + data['Close']) / 3
    data['VWAP'] = (data['Typical_Price'] * data['Volume']).cumsum() / (data['Volume'].cumsum() + 1e-8)
    data['Bid_Ask_Pressure'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'] + 1e-8)
    
    # 2. Options OI Decay Proxy: Volatility Contraction
    data['Rolling_Vol_20'] = data['Returns'].rolling(20).std() * np.sqrt(252)
    data['Vol_Contraction'] = data['Rolling_Vol_20'] / (data['Rolling_Vol_20'].rolling(50).mean() + 1e-8)
    
    # Bollinger Bands for BB_Position
    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    data['BB_Position'] = (data['Close'] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std + 1e-8)
    
    # RSI & MACD
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / (loss + 1e-8)
    data['RSI'] = 100 - (100 / (1 + rs))
    data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    
    data.bfill(inplace=True)
    return data

def inject_macro_context(df: pd.DataFrame) -> pd.DataFrame:
    """Downloads NIFTY 50 and VIX to give models market-wide context."""
    if df.empty: return df
    
    start_date = df.index.min()
    end_date = df.index.max() + timedelta(days=1)
    
    try:
        macro_data = yf.download(["^NSEI", "^INDIAVIX"], start=start_date, end=end_date, progress=False)['Close']
        if not macro_data.empty:
            macro_data.columns = ['NIFTY_Close', 'VIX_Close']
            macro_data['NIFTY_Return'] = macro_data['NIFTY_Close'].pct_change()
            df = df.join(macro_data, how='left')
            df.bfill(inplace=True)
            df.ffill(inplace=True)
    except Exception as e:
        df['NIFTY_Return'] = 0
        df['VIX_Close'] = 15.0 
        
    return df

class SessionDataCache:
    def __init__(self):
        self.STOCK_DATA_KEY = 'stock_data_cache'
        self.FEATURE_DATA_KEY = 'feature_data_cache'
        self.MASTER_UNIVERSE_KEY = 'master_universe'
        
        for key in [self.STOCK_DATA_KEY, self.FEATURE_DATA_KEY]:
            if key not in st.session_state: st.session_state[key] = {}

    def fetch_raw_stock_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        if symbol in st.session_state[self.STOCK_DATA_KEY]:
            return st.session_state[self.STOCK_DATA_KEY][symbol]
        try:
            # --- CRITICAL FIX: Ensure Yahoo Finance targets the NSE Exchange ---
            yf_symbol = symbol if symbol.endswith('.NS') else f"{symbol}.NS"
            
            df = yf.download(yf_symbol, period=period, progress=False)
            if df.empty: return pd.DataFrame()
            
            # Handle Yahoo Finance multi-index formatting changes
            if isinstance(df.columns, pd.MultiIndex): 
                df.columns = df.columns.get_level_values(0)
                
            st.session_state[self.STOCK_DATA_KEY][symbol] = df
            return df
        except: 
            return pd.DataFrame()

    def get_feature_data(self, symbol: str) -> pd.DataFrame:
        if symbol in st.session_state[self.FEATURE_DATA_KEY]:
            return st.session_state[self.FEATURE_DATA_KEY][symbol]
            
        raw_df = self.fetch_raw_stock_data(symbol)
        if raw_df.empty: return raw_df
        
        tech_df = calculate_technical_features(raw_df)
        tech_df = inject_macro_context(tech_df) 
        
        # Merge master fundamentals seamlessly
        if self.MASTER_UNIVERSE_KEY in st.session_state:
            master_df = st.session_state[self.MASTER_UNIVERSE_KEY]
            base_sym = symbol.replace('.NS', '')
            if base_sym in master_df['SYMBOL'].values:
                fund_data = master_df[master_df['SYMBOL'] == base_sym].iloc[0]
                for col in fund_data.index:
                    tech_df[col] = fund_data[col]
                
                tech_df = engineer_expert_features(tech_df)

        st.session_state[self.FEATURE_DATA_KEY][symbol] = tech_df
        return tech_df

def get_data_cache():
    if 'session_data_cache' not in st.session_state:
        st.session_state['session_data_cache'] = SessionDataCache()
    return st.session_state['session_data_cache']

def prepare_multi_horizon_data(df: pd.DataFrame, features: list):
    data = df.dropna(subset=features + ['Close']).copy()
    data['Target_T1'] = data['Close'].shift(-1)
    data['Target_T2'] = data['Close'].shift(-2)
    
    current_close = data['Close'].iloc[-1]
    data['Return_T1'] = (data['Target_T1'] - data['Close']) / data['Close']
    data['Return_T2'] = (data['Target_T2'] - data['Close']) / data['Close']
    
    train_data = data.dropna(subset=['Return_T1', 'Return_T2'])
    
    X = train_data[features]
    y_t1 = train_data['Return_T1']
    y_t2 = train_data['Return_T2']
    
    X_predict = data[features].iloc[-1:]
    
    return X, y_t1, y_t2, X_predict, current_close