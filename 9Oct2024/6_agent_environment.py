import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt


class AgentEnvironment:
    def __init__(self, initial_cash, stock_data_files, model_files):
        # Initial cash and total portfolio value
        self.cash = initial_cash
        self.total_value = initial_cash

        # Initialize portfolio and prices
        self.portfolio = {stock: 0 for stock in stock_data_files.keys()}  # Initial holdings of each stock set to 0
        self.current_prices = {}  # Current prices of each stock

        # Predictions for each stock for the current day
        self.predictions = {}

        # Load and process validation data for each stock
        self.validation_data = self.load_validation_data(stock_data_files)
        self.models = self.load_models(model_files)

        # Define the feature columns to use for predictions
        self.feature_columns = ['Closing_5', 'Closing_4', 'Closing_3', 'Closing_2', 'Closing_1',
                                'SMA_5_5', 'SMA_5_4', 'SMA_5_3', 'SMA_5_2', 'SMA_5_1',
                                'SMA_10_5', 'SMA_10_4', 'SMA_10_3', 'SMA_10_2', 'SMA_10_1',
                                'EMA_5_5', 'EMA_5_4', 'EMA_5_3', 'EMA_5_2', 'EMA_5_1',
                                'EMA_10_5', 'EMA_10_4', 'EMA_10_3', 'EMA_10_2', 'EMA_10_1',
                                'RSI_5', 'RSI_4', 'RSI_3', 'RSI_2', 'RSI_1',
                                'MACD_5', 'MACD_4', 'MACD_3', 'MACD_2', 'MACD_1',
                                'Signal_Line_5', 'Signal_Line_4', 'Signal_Line_3', 'Signal_Line_2', 'Signal_Line_1',
                                'BB_upper_5', 'BB_upper_4', 'BB_upper_3', 'BB_upper_2', 'BB_upper_1',
                                'BB_lower_5', 'BB_lower_4', 'BB_lower_3', 'BB_lower_2', 'BB_lower_1']

        # List to store total portfolio values and actions taken each day
        self.portfolio_values = []
        self.daily_actions = []

    def load_validation_data(self, stock_data_files):
        """
        Load validation data for each stock.
        :param stock_data_files: Dictionary of stock names and their respective file paths.
        :return: Dictionary of stock validation data.
        """
        validation_data = {}
        for stock, file_path in stock_data_files.items():
            try:
                # Load the file as a CSV
                validation_data[stock] = pd.read_csv(file_path)  # Load the CSV file
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"Error loading {file_path}: {e}")
        return validation_data

    def load_models(self, model_files):
        """
        Load trained models for each stock.
        :param model_files: Dictionary of stock names and their respective model pickle files.
        :return: Dictionary of loaded models.
        """
        models = {}
        for stock, model_file in model_files.items():
            with open(model_file, 'rb') as f:
                models[stock] = pickle.load(f)
        return models

    def update_stock_prices(self, day):
        """
        Update stock prices for the given day.
        :param day: The current day index in the validation data.
        """
        for stock, data in self.validation_data.items():
            # Access the closing price directly (assuming the first column is the closing price)
            self.current_prices[stock] = data.iloc[day].values[0]  # Get the closing price

    def get_portfolio_value(self):
        """
        Calculate and return the total portfolio value.
        :return: Total portfolio value (cash + stock holdings value).
        """
        # Calculate the value of all stocks in the portfolio
        stock_value = sum(self.portfolio[stock] * self.current_prices[stock] for stock in self.portfolio)
        # Total portfolio value is cash + stock holdings value
        self.total_value = self.cash + stock_value
        return self.total_value

    def make_decision(self, day):
        """
        Use models to make decisions based on validation data for the given day.
        :param day: The current day index in the validation data.
        """
        self.predictions = {}
        for stock, model in self.models.items():
            # Select the features for prediction
            features = self.validation_data[stock][self.feature_columns].iloc[day].values.reshape(1, -1)

            # Convert the features to a DataFrame with the original feature names
            features_df = pd.DataFrame(features, columns=self.feature_columns)

            # Make predictions for the stock
            self.predictions[stock] = model.predict(features_df)[0]

    def calculate_ratios(self):
        """
        Calculate investment ratios based on the predictions.
        :return: Dictionary of investment ratios for each stock.
        """
        positive_preds = {stock: pred for stock, pred in self.predictions.items() if pred > 0}
        # Calculate total weight of positive predictions
        total_weighted_preds = sum(2 if pred == 2 else 1 for pred in positive_preds.values())

        # Initialize all ratios to zero
        ratios = {stock: 0 for stock in self.portfolio}
        if total_weighted_preds > 0:
            for stock, pred in positive_preds.items():
                weight = 2 if pred == 2 else 1  # Assign weight for strong buy
                ratios[stock] = weight / total_weighted_preds  # Normalize the ratio

        # Print the ratios for each stock
        print("Investment Ratios:", ratios)

        return ratios

    def log_action(self, stock, action, num_shares, price):
        """
        Log the action taken by the agent.
        :param stock: The stock symbol.
        :param action: The action taken ('buy', 'sell').
        :param num_shares: Number of shares bought or sold.
        :param price: Price per share at the time of action.
        """
        action_description = f"{action.capitalize()} {num_shares} shares of {stock} at ${price:.2f}"
        self.daily_actions.append(action_description)

    def allocate_funds(self):
        """
        Allocate funds to stocks based on predictions and ratios.
        """
        # Calculate ratios for investment allocation
        ratios = self.calculate_ratios()

        for stock, pred in self.predictions.items():
            if pred == -1:
                # Sell all shares of the stock and add cash from the sale
                if self.portfolio[stock] > 0:
                    self.cash += self.portfolio[stock] * self.current_prices[stock]
                    self.log_action(stock, 'sell', self.portfolio[stock], self.current_prices[stock])
                    self.portfolio[stock] = 0  # Set holdings to 0 after selling
            elif pred > 0:
                # Allocate cash based on the ratio
                allocation = self.cash * ratios[stock]
                # Calculate number of shares that can be bought
                num_shares = allocation // self.current_prices[stock]
                if num_shares > 0:
                    self.portfolio[stock] += num_shares
                    self.cash -= num_shares * self.current_prices[stock]
                    self.log_action(stock, 'buy', num_shares, self.current_prices[stock])

    def run_simulation(self):
        """
        Run the simulation over the number of days in the validation data.
        """
        num_days = len(next(iter(self.validation_data.values())))

        for day in range(num_days):
            self.update_stock_prices(day)
            self.make_decision(day)
            self.allocate_funds()

            # Calculate and record the total portfolio value
            total_value = self.get_portfolio_value()
            self.portfolio_values.append(total_value)

            # Print the daily summary
            print(f"\nDay {day + 1}:")
            print(f"Total Portfolio Value = ${total_value:.2f}")
            print(f"Cash left to invest in more stocks = ${self.cash:.2f}")

            print("Actions:")
            for action in self.daily_actions:
                print(f" - {action}")

            print("Holdings:")
            for stock, shares in self.portfolio.items():
                print(f"  {stock}: {shares}")

            print("-" * 50)

            # Clear actions for the next day
            self.daily_actions.clear()

        # Call the plot function at the end of the simulation
        self.plot_portfolio_value()

    def plot_portfolio_value(self):
        """
        Plot the portfolio value over time with months on the x-axis.
        """
        # Create a new figure
        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio_values, label='Portfolio Value', marker='o')

        # Title and labels
        plt.title('Portfolio Value from 01-01-24 till 31-08-24')
        plt.xlabel('Days')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Show the plot
        plt.show()


if __name__ == '__main__':
    initial_cash = 100000

    stock_data_files = {
        'AAPL': '2_validation_data/AAPL_processed_validation.csv',
        'AMZN': '2_validation_data/AMZN_processed_validation.csv',
        'NVDA': '2_validation_data/NVDA_processed_validation.csv',
        'GOOG': '2_validation_data/GOOG_processed_validation.csv',
        'MSFT': '2_validation_data/MSFT_processed_validation.csv'
    }

    model_files = {
        'AAPL': '6_stock_models/AAPL_MLP_model.pkl',
        'AMZN': '6_stock_models/AMZN_MLP_model.pkl',
        'NVDA': '6_stock_models/NVDA_MLP_model.pkl',
        'GOOG': '6_stock_models/GOOG_MLP_model.pkl',
        'MSFT': '6_stock_models/MSFT_MLP_model.pkl'
    }

    agent_env = AgentEnvironment(initial_cash, stock_data_files, model_files)
    agent_env.run_simulation()
