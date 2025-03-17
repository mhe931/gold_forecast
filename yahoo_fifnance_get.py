import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Constants
TICKER = "GC=F"  # Gold Futures ticker on Yahoo Finance
PARQUET_FILE = "gold_prices.parquet"

# Check if file exists and is updated
def is_file_updated():
    if not os.path.exists(PARQUET_FILE):
        return False
    df = pd.read_parquet(PARQUET_FILE)
    last_date = pd.to_datetime(df["Date"]).max()
    return last_date >= datetime.now() - timedelta(days=1)

# Fetch historical data
def fetch_gold_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)
    data = yf.download(TICKER, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    data.reset_index(inplace=True)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Save data to Parquet
def save_to_parquet(df):
    df.to_parquet(PARQUET_FILE, index=False)

# Main script
if __name__ == "__main__":
    if not is_file_updated():
        print("Fetching new data...")
        df = fetch_gold_data()
        save_to_parquet(df)
    else:
        print("Using existing data...")
        df = pd.read_parquet(PARQUET_FILE)

    print(df.head())