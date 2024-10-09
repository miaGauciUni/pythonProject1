import pandas as pd
import numpy as np
from gymnasium import spaces

class Environment:
    def __init__(self, pathData, stock='Closing Price', initial_portfolio_value=100000):
        # Initialize variables for single stock
        self.initial_portfolio_value = initial_portfolio_value
        self.portfolio_value = initial_portfolio_value  # Portfolio value
        self.stock = 'Closing Price'  # Use the stock column passed as an argument (e.g., 'GOOG_Close')
        self.volume = 'Volume'  # Assuming volume data column is named 'GOOG_Volume'
        self.pathData = pathData
        self.shares = 0  # Initialize number of shares held
        self.current_step = 0

        # Load and process the stock data
        self.stock_data = self._load_data()
        self._initialize_portfolio()

        # Defining discrete action space: [-1 (Sell), 0 (Neutral), 1 (Hold), 2 (Buy), 3 (Strong Buy)]
        self.action_space = spaces.Discrete(5)

    def _load_data(self):
        """Load and prepare data from CSV file."""
        df = pd.read_csv(self.pathData)
        df = df[['Date', self.stock, self.volume]]
        return df

    def _initialize_portfolio(self):
        """Initialize the portfolio with initial shares."""
        initial_price = self.stock_data.iloc[0][self.stock]
        self.shares = self.initial_portfolio_value / initial_price  # Invest all initial value into stock

    def trade(self, action, date):
        """Execute the trading action on the given date."""
        self.current_step = self.stock_data[self.stock_data['Date'] == date].index[0]
        current_price = self.stock_data.iloc[self.current_step][self.stock]

        # Implement buy/sell logic
        if action == 3 or action == 2:  # Buy or Strong Buy
            amount_to_invest = self.portfolio_value * (action - 1)  # Determine investment amount based on action
            shares_to_buy = amount_to_invest / current_price
            self.shares += shares_to_buy
        elif action == -1:  # Sell half of the current holdings
            self.shares -= self.shares * 0.5

        # Update portfolio value
        self.portfolio_value = self.shares * current_price

    def state(self, date):
        """Get the current state of the environment on the given date."""
        self.current_step = self.stock_data[self.stock_data['Date'] == date].index[0]
        current_price = self.stock_data.iloc[self.current_step][self.stock]
        current_volume = self.stock_data.iloc[self.current_step][self.volume]
        return {
            'portfolio_value': self.portfolio_value,
            'shares': self.shares,
            'current_price': current_price,
            'current_volume': current_volume
        }
