#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import backtrader as bt
import quantstats as qs
import pyfolio as pf
import json
from sklearn.ensemble import RandomForestClassifier
import os

import warnings
warnings.filterwarnings('ignore')


# In[2]:


class CustomCSVData(bt.feeds.GenericCSVData):
    params = (
        ('stock_name', None),
        ('csv_directory', "Downloads/stock_dfs/stock_dfs"),
        ('data_type', None)  # Renamed from 'types' for clarity
    )

    def __init__(self):
        super().__init__()
        self.data_frame = self._load_csv()  # Load the data when the instance is created

    def _construct_file_path(self):
        """Construct the file path for the CSV file."""
        return os.path.join(self.p.csv_directory, f"{self.p.stock_name}.csv")

    def _load_csv(self):
        """Load data from a CSV file."""
        file_path = self._construct_file_path()
        data = pd.read_csv(file_path)
        data.columns = data.columns.str.lower()
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data.set_index('date', inplace=True)

        # Additional processing based on data type
        if self.p.data_type == 'hour':
            # Implement specific operations for hourly data
            pass

        return data

    def _prepare_data(self):
        """Prepare data for Backtrader feed."""
        if self.p.data_type is not None:
            return self.data_frame[['high', 'low', 'open', 'close', 'volume']]
        return self.data_frame


# In[3]:


import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class Stock:
    def __init__(self, data):
        self.data = data
        self.label = []

    def factor(self):
        """Process data to create features and labels for stock prediction."""
        # Remove 'Close' column
        del self.data["Close"]

        # Fill missing values
        self.data = self.data.fillna(method="bfill")
        self.data = self.data.fillna(method="ffill")


        # Create label based on adjusted close price
        self.data['label'] = self.data.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])['Adj Close']
        self.data['label'] = self.data['label'].shift(-1)

        # Calculate moving averages and MACD indicators
        self._calculate_moving_averages()
        self._calculate_macd()

        # Add volume and volume change over different periods
        self._add_volume_features()

        # Clean up data
        self.data = self.data.dropna(how="any")
        self.label = list(self.data["label"])
        del self.data["label"]
        del self.data["Adj Close"]

    def _calculate_macd(self):
        """Calculate MACD indicators."""
        ema_short = self.data['Adj Close'].ewm(span=12, adjust=False, min_periods=12).mean()
        ema_long = self.data['Adj Close'].ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema_short - ema_long
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        self.data['macd'] = macd
        self.data['macd_h'] = macd - signal
        self.data['macd_s'] = signal

    def _calculate_moving_averages(self):
        """Calculate moving averages."""
        self.data["ma_5"] = self.data["Adj Close"].rolling(window=5).mean()
        self.data["ma_10"] = self.data["Adj Close"].rolling(window=10).mean()
        self.data["ma_20"] = self.data["Adj Close"].rolling(window=20).mean()

    def _add_volume_features(self):
        """Add volume and its change over different periods."""
        # Logarithmic change in volume
        self.data['vol_1d'] = np.log(self.data['Volume']).diff(1)
        self.data['vol_1w'] = np.log(self.data['Volume']).diff(5)
        self.data['vol_1m'] = np.log(self.data['Volume']).diff(21)
        self.data['vol_3m'] = np.log(self.data['Volume']).diff(63)
        self.data['vol_6m'] = np.log(self.data['Volume']).diff(126)
        self.data['vol_1y'] = np.log(self.data['Volume']).diff(252)

        # Standard deviation of volume change
        self.data['vol_std_1w'] = self.data['vol_1d'].rolling(5).std()
        self.data['vol_std_1m'] = self.data['vol_1d'].rolling(21).std()
        self.data['vol_std_3m'] = self.data['vol_1d'].rolling(63).std()
        self.data['vol_std_6m'] = self.data['vol_1d'].rolling(126).std()
        self.data['vol_std_1y'] = self.data['vol_1d'].rolling(252).std()

    def standardize(self):
        """Standardize the data."""
        if self.data.empty:  # Assuming self.data is a DataFrame here
            return
        scaler = StandardScaler()
        self.data = scaler.fit_transform(self.data)

    def normalize(self):
        """Normalize the data."""
        scaler = MinMaxScaler()
        self.data = scaler.fit_transform(self.data)


# In[4]:


ticker = pd.read_csv("tickers.csv")  # Ensure this file exists in your directory
list1 = list(ticker["Ticker"])
# Stock class as provided

# Lists to store combined features and labels for all stocks
all_features_xgb = []
all_labels_xgb = []

for tic in list1:
    # Download and preprocess data for each ticker in the list
    yf_data = yf.download(tic, start="2000-01-01", end="2021-11-12")
    
    # Skip if data is empty
    if yf_data.empty:
        continue

    stock = Stock(yf_data)
    stock.factor()        # Feature creation and preprocessing
    stock.standardize()   # Standardizing the data
    stock.normalize()     # Normalizing the data

    # Append the features and labels to the respective lists
    all_features_xgb.append(pd.DataFrame(stock.data))
    all_labels_xgb.extend(stock.label)


X = pd.concat(all_features_xgb, ignore_index=True)
y = pd.Series(all_labels_xgb)

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Initialize an XGBoost classifier with adjusted parameters
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    max_depth=6,             # Adjusted max depth
    n_estimators=200,        # Increased number of trees
    learning_rate=0.05,      # Lower learning rate
    gamma=0.1,               # Adjusted gamma
    subsample=0.8,           # Adjusted subsample ratio
    colsample_bytree=0.8     # Adjusted subsample ratio of columns
)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
pred_y = xgb_model.predict(X_test)

# Calculating accuracy and precision
accuracy = accuracy_score(y_test, pred_y)
precision = precision_score(y_test, pred_y)


# In[5]:


import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
ticker = pd.read_csv("tickers.csv")  # Ensure this file exists in your directory
list1 = list(ticker["Ticker"])
# Assuming Stock class and list1 are defined as before

# Lists to store combined features and labels for all stocks
all_features_rf = []
all_labels_rf = []

for tic in list1:
    # Download and preprocess data for each ticker in the list
    yf_data = yf.download(tic, start="2000-01-01", end="2021-11-12")
    
    # Skip if data is empty
    if yf_data.empty:
        continue

    stock = Stock(yf_data)
    stock.factor()        # Feature creation and preprocessing
    stock.standardize()   # Standardizing the data
    stock.normalize()     # Normalizing the data

    # Append the features and labels to the respective lists
    all_features_rf.append(pd.DataFrame(stock.data))
    all_labels_rf.extend(stock.label)

# Concatenating all features and labels into single datasets
X = pd.concat(all_features_rf, ignore_index=True)
y = pd.Series(all_labels_rf)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Initialize a Random Forest classifier with adjusted parameters
rf_model = RandomForestClassifier(
    n_estimators=150,    # Increased number of trees
    max_depth=10,        # Maximum depth of each tree
    min_samples_split=4, # Minimum number of samples required to split an internal node
    min_samples_leaf=2,  # Minimum number of samples required to be at a leaf node
    random_state=42      # Random state for reproducibility
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
pred_y_rf = rf_model.predict(X_test)

# Calculating accuracy and precision for the combined dataset
accuracy_rf = accuracy_score(y_test, pred_y_rf)
precision_rf = precision_score(y_test, pred_y_rf)


# In[6]:


ticker = pd.read_csv("tickers.csv")  # Ensure this file exists in your directory
list1 = list(ticker["Ticker"])
# Custom data loading class for backtrader
class PredictionsData(bt.feeds.PandasData):
    lines = ('predictions',)
    params = (('predictions', -1),)

def get_stock_predictions(ticker, model):
    yf_data = yf.download(ticker, start="2000-01-01", end="2021-11-12")
    if yf_data.empty:
        return None

    # Preprocess the data
    stock = Stock(yf_data.copy())
    stock.factor()
    stock.standardize()
    stock.normalize()

    # Splitting the processed data
    X = pd.DataFrame(stock.data)
    _, X_test = train_test_split(X, test_size=0.4, random_state=42)

    # Generate predictions for the test set
    test_predictions = model.predict(X_test)

    # Store predictions in a DataFrame with the same index as the original data
    predictions_series = pd.Series(index=yf_data.index)
    predictions_series[X_test.index] = test_predictions
    yf_data['predictions'] = predictions_series

    return yf_data

# Backtrader strategy class
class RFStrategy(bt.Strategy):
    def __init__(self):
        self.predicted = self.datas[0].predictions

    def next(self):
        if not self.position:
            if self.predicted[0] == 1 and self.broker.get_cash() > 100:
                self.buy()
        elif self.predicted[0] == 0 and self.getposition().size > 0:
            self.sell()


# In[7]:


import pandas as pd

evaluation_metrics = {
    "rf_model": {
        "Accuracy": accuracy_rf, 
        "Precision": precision_rf
    },
    "xgb_model": {
        "Accuracy": accuracy, 
        "Precision": precision
    }
}

models = [rf_model, xgb_model]
model_names = ['rf_model', 'xgb_model']
top_tickers = {}  # Dictionary to store top two tickers for each model
ticker_data = pd.read_csv("tickers.csv")  # Ensure this file exists
list1 = list(ticker_data["Ticker"])

for model, model_name in zip(models, model_names):
    final_values = {}
    stock_predictions = {}
    for ticker in list1:
        processed_data = get_stock_predictions(ticker, model)
        if processed_data is not None:
            stock_predictions[ticker] = processed_data

    # Running backtest and storing final portfolio values
    for ticker, data in stock_predictions.items():
        cerebro = bt.Cerebro()
        cerebro.addstrategy(RFStrategy)
        data_feed = PredictionsData(dataname=data)
        cerebro.adddata(data_feed)
        cerebro.broker.set_cash(10000)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.run()
        final_val = cerebro.broker.getvalue()
        final_values[ticker] = final_val

    # Identify top two tickers
    top_two = sorted(final_values, key=final_values.get, reverse=True)[:2]
    top_tickers[model_name] = top_two

combined_output_df = pd.DataFrame({
    "Model": ['Random Forest', 'XGBoost'],
    "Top_Ticker_1": [top_tickers[model][0] for model in model_names],
    "Top_Ticker_2": [top_tickers[model][1] for model in model_names],
    "Accuracy_1": [evaluation_metrics[model]["Accuracy"] for model in model_names],
    "Precision_1": [evaluation_metrics[model]["Precision"] for model in model_names],
    "Accuracy_2": [evaluation_metrics[model]["Accuracy"] for model in model_names],
    "Precision_2": [evaluation_metrics[model]["Precision"] for model in model_names]
})

# # Export to CSV
combined_output_df.to_csv('small_universe_results.csv', index=False)


# In[8]:


def generate_report_for_ticker(ticker, model, strategy_class, stock_predictions):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class)
    data_feed = PredictionsData(dataname=stock_predictions[ticker])
    cerebro.adddata(data_feed)
    cerebro.broker.set_cash(10000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
    strat = cerebro.run()
    daily_returns = strat[0].analyzers.time_return.get_analysis()
    returns_series = pd.Series(daily_returns)
    returns_series.index = pd.to_datetime(returns_series.index)

    # Generate and save reports
    qs.reports.html(returns_series, output=f'quantstats_{ticker}_{model}.html')

# Generate reports for top tickers of each model
for model_name, tickers in top_tickers.items():
    for ticker in tickers:
        generate_report_for_ticker(ticker, model_name, RFStrategy, stock_predictions)


# In[9]:


def process_stock_data(data):
    if data.empty:
        return None, None

    stock = Stock(data)
    stock.factor()  # Feature creation and preprocessing
    stock.standardize()  # Standardizing the data

    # Check if data is still non-empty after standardization
    if stock.data.size == 0:  # Use .size for NumPy arrays
        return None, None

    stock.normalize()  # Normalizing the data

    X = pd.DataFrame(stock.data)
    y = pd.Series(stock.label)

    return X, y


# In[10]:


def fetch_data(ticker):
    try:
        data = yf.download(ticker,start="2000-01-01", end="2021-11-12")
        if data.empty:
            return None
        else:
            return data
    except Exception as e:
        return None


# In[ ]:


ticker_df = pd.read_csv("tickers_nasd.csv")  
tickers = list(ticker_df["Symbol"])
results_rf = {}
results_xgb = {}

# Initialize dictionaries to store results
accuracy_results_rf = {}
accuracy_results_xgb = {}

# Initialize a dictionary to store combined accuracies
combined_accuracy_results = {}

for ticker in tickers:
    fetched_data = fetch_data(ticker)
    if fetched_data is None or fetched_data.empty:
        continue

    X, y = process_stock_data(fetched_data)
    if X is None or y is None or X.empty or len(y) < 2:
        continue

    # Ensure there's enough data to split
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        # Random Forest Model
        rf_model.fit(X_train, y_train)
        accuracy_rf = accuracy_score(y_test, rf_model.predict(X_test))
        accuracy_results_rf[ticker] = accuracy_rf  # Store RF accuracy

        # XGBoost Model
        xgb_model.fit(X_train, y_train)
        accuracy_xgb = accuracy_score(y_test, xgb_model.predict(X_test))
        accuracy_results_xgb[ticker] = accuracy_xgb  # Store XGB accuracy

        # Combine accuracies (here, taking the average)
        combined_accuracy = (accuracy_rf + accuracy_xgb) / 2
        combined_accuracy_results[ticker] = combined_accuracy



# Sort and select top 10 tickers based on combined model accuracies
top_10_tickers_combined = sorted(combined_accuracy_results, key=combined_accuracy_results.get, reverse=True)[:10]


# In[12]:


# Sort and select top 10 tickers based on combined model accuracies
top_10_tickers_combined = sorted(combined_accuracy_results, key=combined_accuracy_results.get, reverse=True)[:10]


# In[13]:


# Create a DataFrame to display the tickers along with accuracies from both models
top_10_df = pd.DataFrame({
    'Ticker': top_10_tickers_combined,
    'Combined_Accuracy': [combined_accuracy_results.get(ticker, None) for ticker in top_10_tickers_combined],
    'RF_Accuracy': [accuracy_results_rf.get(ticker, None) for ticker in top_10_tickers_combined],
    'XGB_Accuracy': [accuracy_results_xgb.get(ticker, None) for ticker in top_10_tickers_combined]
})


# Convert the DataFrame to a CSV string and write to a file
csv_data = top_10_df.to_csv(index=False)
with open('top_10_combined_accuracy.csv', 'w') as file:
    file.write(csv_data)


# In[14]:


stock_predictions = {}
for ticker in top_10_tickers_combined:
    predictions = get_stock_predictions(ticker, rf_model)  # Replace 'your_model' with the actual model
    if predictions is not None:
        stock_predictions[ticker] = predictions


# In[15]:


for ticker in top_10_tickers_combined:
    cerebro = bt.Cerebro()
    cerebro.addstrategy(RFStrategy)
    data_feed = PredictionsData(dataname=stock_predictions[ticker])
    cerebro.adddata(data_feed)
    cerebro.broker.set_cash(10000)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='time_return')
    strat = cerebro.run()
    daily_returns = strat[0].analyzers.time_return.get_analysis()
    returns_series = pd.Series(daily_returns)
    returns_series.index = pd.to_datetime(returns_series.index)

    # Generate and save reports
    qs.reports.html(returns_series, output=f'quantstats_{ticker}.')

