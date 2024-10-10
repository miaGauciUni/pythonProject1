import pandas as pd
import numpy as np
from gymnasium import spaces
import talib


class Environment:
    def __init__(self, ticker, ipv=100000):
        # Initialize variables
        self.ipv = ipv  # Initial portfolio value
        self.pv = ipv  # Portfolio value (same as ipv initially)
        self.ticker = ['AAPL', 'TSLA', 'MSFT', 'META', 'GOOG']  # Array of stock symbols
        self.ind = ['CP', 'CP1',
                    'MA5']  # List of indicators: Moving Average [MA], MA convergence divergence [MACD], RSI
        self.n = len(ticker)  # Number of stocks being traded
        self.n_features = 0  # define later on

        self.shares = np.zeros(self.n)  # Array to store shares for each stock
        self.current_step = 0  # Track the current step in the stock data

        # Defining date ranges
        self.training_data_range = ('2018-01-01', '2019-01-01')
        self.validation_data_range = ('2019-01-01', '2019-01-01')
        self.test_data_range = ('2019-01-01', '2020-01-01')

        # Load and process the stock data
        self.stock_data, self.training_data, self.validation_data, self.test_data = self._load_and_split_data()

        # Defining discrete action space - 5 possible actions for each stock: [-1, 0, 1, 2, 3]
        # Actions: [-1 (Sell), 0 (Neutral), 1 (Hold), 2 (Buy), 3 (Strong Buy)] for each stock
        self.action_space = spaces.MultiDiscrete([5] * len(self.ticker))

        # Defining action space - continuous space where each action corresponds to a value between -1 (sell) and 1 (buy)
        # shape=(len(self.ticker) ensures that there is one action for each stock (ticker)
        # self.action_space = spaces.Box(low=1, high=1, shape=(len(self.ticker),), dtype=np.float64)

        # Observation Space: price data for each stock [add shares held, net worth, max net worth, current stop]
        # self.obs_shape = self.n_features * len(self.ticker) + 2 + len(self.ticker) + 2 # NEEDS TO CHANGE
        # self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float64)

        self._initialize_portfolio()

    def _load_and_split_data(self):
        """
        Loads the CSV data, filters by the date range, and splits into training, validation, and test sets.
        """
        # Load the data from CSV
        df = pd.read_csv('stock_closing_prices_2018_2020.csv', index_col='date', parse_dates=True)

        # Remove any spaces in column names, just in case
        df.columns = df.columns.str.strip()

        # Keep only the relevant stock columns
        df = df[self.ticker]

        # Split data based on the date ranges
        training_data = df.loc[self.training_data_range[0]:self.training_data_range[1]]
        validation_data = df.loc[self.validation_data_range[0]:self.validation_data_range[1]]
        test_data = df.loc[self.test_data_range[0]:self.test_data_range[1]]

        return df, training_data, validation_data, test_data

    def _initialize_portfolio(self):
        """
        Initializes the portfolio by splitting the initial portfolio value equally among the stocks
        and calculating how many shares can be bought for each stock.
        """
        initial_prices = self.training_data.iloc[0][self.ticker].values
        fund_per_stock = self.ipv / self.n
        self.shares = fund_per_stock / initial_prices
        self.pv = np.sum(self.shares * initial_prices)

    def get_shares(self):
        """
        Outputs the current number of shares owned for each stock.
        """
        return self.shares

    def get_portfolio_value(self):
        """
        Returns the current value of the portfolio based on stock prices and shares owned.
        """
        current_prices = self.stock_data.iloc[self.current_step][self.ticker].values
        portfolio_value = np.sum(self.shares * current_prices)
        return portfolio_value

    def trade(self, action, date):
        """
        Executes trade actions for each stock based on the provided action array.

        :param action: Array of actions (size n) where each value can be:
                       3 (Strong Buy), 2 (Buy), 1 (Hold), 0 (Neutral), -1 (Sell)
        :param date: The current date (used to fetch stock prices for that day)
        """
        # Find the row corresponding to the provided date
        self.current_step = self.stock_data.index.get_loc(date)  # Use get_loc to find the row index for the given date

        # Get the stock prices for the given date
        current_prices = self.stock_data.iloc[self.current_step][self.ticker].values

        # Iterate through each stock and perform the respective action
        for i in range(self.n):
            if action[i] == 3 or action[i] == 2:  # Buy or Strong Buy
                # Calculate how much to buy based on available cash
                amount_to_invest = (self.pv / self.n) * (action[i] - 1)  # More aggressive for Strong Buy
                shares_to_buy = amount_to_invest / current_prices[i]
                self.shares[i] += shares_to_buy
            elif action[i] == -1:  # Sell
                # Sell a fraction of the shares
                self.shares[i] -= self.shares[i] * 0.5  # Sell half the holdings, for example

        # Update the portfolio value after trading
        self.pv = np.sum(self.shares * current_prices)
        print(f"Portfolio value after trading: {self.pv}")
        print(f"Updated shares: {self.shares}")

    def state(self, date):
        """
        Outputs the portfolio state at the given date, including portfolio value and shares owned.
        """
        self.current_step = self.stock_data.index.get_loc(date)  # Use get_loc to find the index position of the date
        current_prices = self.stock_data.iloc[self.current_step][self.ticker].values
        return {
            'portfolio_value': self.pv,
            'shares': self.shares,
            'current_prices': current_prices
        }

    def _calculate_indicators(self):
        """
        Calculates the closing price, yesterday's closing price, and 5-day moving average
        for each stock in the dataset.
        """
        # Yesterday's closing prices
        self.stock_data['CP1'] = self.stock_data[self.ticker].shift(1)

        # Calculate 5-day moving average using TA-Lib
        for stock in self.ticker:
            self.stock_data[f'MA5_{stock}'] = talib.SMA(self.stock_data[stock], timeperiod=5)

    def technical_indicators(self, date):
        # Returns the technical indicators for the given date, Closing price, Yesterday's closing price, 5-day moving average
        self.current_step = self.stock_data.index.get_loc(date)

        indicators = {}
        for stock in self.ticker:
            stock_indicators = {}
            for indicator in self.ind:
                if indicator == 'CP':  # Closing Price
                    stock_indicators['CP'] = self.stock_data.iloc[self.current_step][stock]
                elif indicator == 'CP1':  # Yesterday's Closing Price
                    stock_indicators['CP1'] = self.stock_data.iloc[self.current_step][f'CP1']
                elif indicator == 'MA5':  # 5-day Moving Average
                    stock_indicators['MA5'] = self.stock_data.iloc[self.current_step][f'MA5_{stock}']

            indicators[stock] = stock_indicators

        return indicators


"""
    def add_technical_indicators(df, ticker):

        #Adds technical indicators (e.g., MA, RSI) to the DataFrame for a specific ticker.

        # Add 5-day moving average
        df[f'{ticker}_5_day_MA'] = df[ticker].rolling(window=5).mean()

        # Calculate the RSI for the ticker
        delta = df[ticker].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df[f'{ticker}_RSI'] = 100 - (100 / (1 + rs))

        return df

    def apply_indicators(self):
        #Applies technical indicators to training, validation, and test datasets.

        # Apply technical indicators to training, validation, and test data
        for ticker in self.ticker:
            # Apply to training data
            self.training_data[ticker] = self.add_technical_indicators(
                self.training_data
            )

            # Apply to validation data
            self.validation_data[ticker] = self.add_technical_indicators(
                self.validation_data
            )

            # Apply to test data
            self.test_data[ticker] = self.add_technical_indicators(
                self.test_data
            )
"""


