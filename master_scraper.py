# master_scraper.py — Backend Exhaustive Data Generator
import pandas as pd
import requests
import io
import yfinance as yf
import time
import urllib.parse
import feedparser
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_nse_base_list() -> pd.DataFrame:
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = df.columns.str.strip()
        return df[df['SERIES'] == 'EQ'].copy()
    except Exception as e:
        print(f"Error fetching base list: {e}")
        return pd.DataFrame()

def fetch_exhaustive_fundamentals(symbol: str, max_retries: int = 3) -> dict:
    ticker_str = f"{symbol}.NS"
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker_str).info
            ltp = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if not info or ltp is None: return None
                
            prev_close = info.get('previousClose', ltp)
            chg_pct = ((ltp - prev_close) / prev_close) * 100 if prev_close else 0
            
            return {
                'SYMBOL': symbol,
                'LTP (₹)': ltp,
                '1D Change (%)': round(chg_pct, 2),
                'Market Cap (₹)': info.get('marketCap', 0),
                'P/E Ratio (Trailing)': info.get('trailingPE', None),
                'Forward P/E': info.get('forwardPE', None),
                'P/B Ratio': info.get('priceToBook', None),
                'ROE (%)': info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0,
                'Debt to Equity': info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0,
                '52W High': info.get('fiftyTwoWeekHigh', ltp),
                '52W Low': info.get('fiftyTwoWeekLow', ltp),
                'Avg Volume': info.get('averageVolume', 0),
                'Industry': info.get('industry', 'Unknown'),
                'Sector': info.get('sector', 'Unknown')
            }
        except Exception:
            time.sleep(1)
    return None

def fetch_corporate_action_news(symbol: str, company_name: str) -> dict:
    query = urllib.parse.quote(f"\"{company_name}\" OR \"{symbol}\" NSE stock")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    score = 0
    try:
        feed = feedparser.parse(url)
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        
        for entry in feed.entries:
            try:
                pub_date = parsedate_to_datetime(entry.published)
                if pub_date > thirty_days_ago:
                    score += 1
            except:
                pass
    except:
        pass
        
    return {'SYMBOL': symbol, '30D_Action_Score': min(score, 10)}

def build_exhaustive_dataset(limit: int = None) -> pd.DataFrame:
    base_df = fetch_nse_base_list()
    if limit: base_df = base_df.head(limit)
    
    symbols = base_df['SYMBOL'].tolist()
    fundamentals_list = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_sym = {executor.submit(fetch_exhaustive_fundamentals, sym): sym for sym in symbols}
        for future in as_completed(future_to_sym):
            res = future.result()
            if res: fundamentals_list.append(res)
            
    fund_df = pd.DataFrame(fundamentals_list)
    master_df = pd.merge(base_df, fund_df, on='SYMBOL', how='inner')
    
    # --- EXPERT WIDENED BAND: Allows the UI to handle the strict filtering ---
    target_df = master_df[(master_df['LTP (₹)'] >= 10) & (master_df['LTP (₹)'] <= 5000)].copy()
    if target_df.empty: return target_df
        
    tasks = list(zip(target_df['SYMBOL'], target_df['NAME OF COMPANY']))
    news_list = []
    
    with ThreadPoolExecutor(max_workers=10) as executor: 
        future_to_task = {executor.submit(fetch_corporate_action_news, sym, name): sym for sym, name in tasks}
        for future in as_completed(future_to_task):
            news_list.append(future.result())
            
    news_df = pd.DataFrame(news_list)
    final_df = pd.merge(target_df, news_df, on='SYMBOL', how='left')
    final_df['30D_Action_Score'] = final_df['30D_Action_Score'].fillna(0)
    
    return final_df