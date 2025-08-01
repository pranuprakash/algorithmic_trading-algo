
import yfinance as yf

# Download historical data for GLD ETF
gld_data = yf.download('GLD', start='2000-01-01', end='2025-07-30')

# Save the data to a CSV file
gld_data.to_csv('gld_prices.csv')

print("GLD price data saved to gld_prices.csv")
