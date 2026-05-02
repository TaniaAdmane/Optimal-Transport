
import yfinance as yf
import numpy as np
import pandas as pd

tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", 
           "GOOGL", "META", "BRK-B", "JNJ"]

print("Downloading S&P 500 top 10...")
data = yf.download(tickers, start="2013-01-01", end="2025-12-31")

# Get Close prices
prices = data['Close'].values
# Log-returns
log_returns = np.diff(np.log(prices), axis=0)

# Create DataFrame with dates
dates = data.index[1:]  # Remove first date (no return for first day)
df = pd.DataFrame(log_returns, columns=tickers)
df.insert(0, 'Date', dates)  # Add date as first column

# Save as CSV
df.to_csv("Data/sp500_top10_returns.csv", index=False)

# Also save raw prices with dates
df_prices = pd.DataFrame(prices, columns=tickers)
df_prices.insert(0, 'Date', data.index)
df_prices.to_csv("Data/sp500_top10_prices.csv", index=False)

print("\nFirst few rows:")
print(df.head())