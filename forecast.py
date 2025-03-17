# Import necessary libraries
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.svm import SVR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import numpy as np
import joblib

# Constants
TICKER = "GC=F"  # Gold Futures ticker on Yahoo Finance
PARQUET_FILE = "gold_prices.parquet"
HISTORICAL_DAYS = 5 * 365  # Last 5 years

# Function to fetch historical gold price data
def fetch_gold_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=HISTORICAL_DAYS)
    data = yf.download(TICKER, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    data.reset_index(inplace=True)
    return data

# Function to save data to Parquet file
def save_to_parquet(data):
    data.to_parquet(PARQUET_FILE, index=False)

# Function to load data from Parquet file
def load_from_parquet():
    return pd.read_parquet(PARQUET_FILE)

# Check if the Parquet file exists and is up to date
def is_file_updated():
    if not os.path.exists(PARQUET_FILE):
        return False
    df = pd.read_parquet(PARQUET_FILE)
    # Get the column name that contains 'Date'
    date_column = 'Date'
    last_date = pd.to_datetime(df[date_column]).max()
    return last_date >= datetime.now() - timedelta(days=1)

# Fetch and save data if file does not exist or is outdated
if not is_file_updated():
    gold_data = fetch_gold_data()
    save_to_parquet(gold_data)

# Load data
data = load_from_parquet()

# Rename columns for easier access
data.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Prepare data for modeling
X = np.array(range(len(data))).reshape(-1, 1)
y = data['Close'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    "XGBoost": XGBRegressor(),
    "SVM": SVR(),
    "ARIMA": None,
    "Exponential Smoothing": None
}

# Train and predict with XGBoost and SVM
for name, model in models.items():
    if name in ["XGBoost", "SVM"]:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        models[name] = predictions

# Train and predict with ARIMA
models["ARIMA"] = ARIMA(y_train, order=(5, 1, 0)).fit()
predictions_arima = models["ARIMA"].forecast(steps=len(y_test))
models["ARIMA"] = predictions_arima

# Train and predict with Exponential Smoothing
models["Exponential Smoothing"] = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=12).fit()
predictions_exp_smooth = models["Exponential Smoothing"].forecast(steps=len(y_test))
models["Exponential Smoothing"] = predictions_exp_smooth

# Calculate accuracy
results = pd.DataFrame({
    "Real Price": y_test,
    "XGBoost": models["XGBoost"],
    "SVM": models["SVM"],
    "ARIMA": models["ARIMA"],
    "Exponential Smoothing": models["Exponential Smoothing"]
})

# Calculate Mean Squared Error (MSE)
mse_results = {
    "XGBoost": mean_squared_error(results['Real Price'], results['XGBoost']),
    "SVM": mean_squared_error(results['Real Price'], results['SVM']),
    "ARIMA": mean_squared_error(results['Real Price'], results['ARIMA']),
    "Exponential Smoothing": mean_squared_error(results['Real Price'], results['Exponential Smoothing'])
}

# Save results to a CSV file
results.to_csv("gold_price_forecast_results.csv", index=False)
mse_results_df = pd.DataFrame(list(mse_results.items()), columns=['Model', 'MSE'])
mse_results_df.to_csv("gold_price_forecast_mse_results.csv", index=False)