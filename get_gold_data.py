import yfinance as yf

# Download historical data for gold spot price
gold_data = yf.download('XAUUSD=X', start='2000-01-01', end='2025-07-30')

# Save the data to a CSV file
gold_data.to_csv('gold_prices.csv')

print("Gold price data saved to gold_prices.csv")