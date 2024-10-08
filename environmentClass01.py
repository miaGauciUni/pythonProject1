import pandas as pd
import numpy as np
from gymnasium import spaces
import talib

# state, action, reward

class Environment:
    def __init__(self, pathData, tickers, ipv=100000):
        # Initialize variables
        self.ipv = ipv
        self.pv = ipv
        self.tickers = ['AAPL_Close', 'TSLA_Close', 'MSFT_Close', 'META_Close', 'GOOG_Close']
        self.volume = ['AAPL_Volume', 'TSLA_Volume', 'MSFT_Volume', 'META_Volume', 'GOOG_Volume']
        self.indicators =['CP', 'V', 'MA5']
        self.n = len(tickers)
        self.n_features = 0 # define later on
        self.pathData = pathData
        self.shares = np.zeros(self.n)
        self.current_step = 0

        # Load and process the stock data
        self.stock_data = self._load_data()
        self._initialize_portfolio()

        # Defining discrete action space - Actions: [-1 (Sell), 0 (Neutral), 1 (Hold), 2 (Buy), 3 (Strong Buy)] for each stock
        self.action_space = spaces.MultiDiscrete([5] * len(self.tickers))

    def _load_data(self):
        df = pd.read_csv(self.pathData)
        df = df[['Date'] + self.tickers + self.volume]
        return df

    def _initialize_portfolio(self):
        initial_prices = self.stock_data.iloc[0][self.tickers].values
        fund_per_stock = self.ipv / self.n
        self.shares = fund_per_stock / initial_prices
        self.pv = np.sum(self.shares * initial_prices)

# update
    def trade(self, action, date):
        self.current_step = self.stock_data[self.stock_data['Date'] == date].index[0]
        current_prices = self.stock_data.iloc[self.current_step][self.tickers].values

        for i in range(self.n):
            if action[i] == 3 or action[i] == 2:
                amount_to_invest = (self.pv / self.n) * (action[i] - 1)
                shares_to_buy = amount_to_invest / current_prices[i]
                self.shares[i] += shares_to_buy
            elif action[i] == -1:
                self.shares[i] -= self.shares[i] * 0.5

        self.pv = np.sum(self.shares * current_prices)

    def state(self, date):
        self.current_step = self.stock_data[self.stock_data['Date'] == date].index[0]
        current_prices = self.stock_data.iloc[self.current_step][self.tickers].values
        current_volumes = self.stock_data.iloc[self.current_step][self.volume].values
        return {
            'portfolio_value': self.pv,
            'shares': self.shares,
            'current_prices': current_prices,
            'current_volumes': current_volumes
        }

"""
    def technical_indicator(self, date):
        # Find the index of the specified date
        self.current_step = self.stock_data[self.stock_data['Date'] == date].index[0]

        # Extract closing prices and volumes for the specified date
        closing_prices = self.stock_data.iloc[self.current_step][self.tickers].values
        volumes = self.stock_data.iloc[self.current_step][self.volume].values

        # Calculate 5-day moving averages (MA5)
        ma5_values = {}
        for ticker in self.tickers:
            # Get the last 5 days of closing prices up to the specified date
            closing_prices_series = self.stock_data[f"{ticker}"]
            ma5 = closing_prices_series.iloc[self.current_step - 4:self.current_step + 1].mean()  # 5-day MA
            ma5_values[ticker] = ma5

        # Return the data in a structured format
        return {
            'closing_price': closing_prices,
            'volume': volumes,
            '5_day_moving_average': ma5_values
        }
"""