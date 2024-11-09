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
        self.feature_columns = ['Closing_3', 'Closing_2', 'Closing_1',
                                'EMA_5_3', 'EMA_5_2', 'EMA_5_1',
                                'RSI_3', 'RSI_2', 'RSI_1',
                                'MACD_3', 'MACD_2', 'MACD_1',
                                'BB_upper_3', 'BB_upper_2', 'BB_upper_1',
                                'BB_lower_3', 'BB_lower_2', 'BB_lower_1',]

        # List to store total portfolio values and actions taken each day
        self.portfolio_values = []
        self.daily_actions = []
        self.monthly_returns = []

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

    def get_prices_on_day(self, day_index):
        """
        Get the closing prices of all stocks on a specific day.
        :param day_index: The day index (integer) in the validation data.
        :return: Dictionary of stock prices for the given day.
        """
        prices = {stock: self.validation_data[stock].iloc[day_index].values[0] for stock in self.validation_data}
        return prices

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
        Allocate funds to stocks based on predictions and ratios. If all holdings are zero, the cash is split equally.
        """
        # Check if all holdings are zero
        all_zero_holdings = all(shares == 0 for shares in self.portfolio.values())

        if all_zero_holdings:
            # Split cash equally among all stocks if all holdings are zero
            equal_investment = self.cash / len(self.portfolio)
            for stock in self.portfolio:
                # Calculate the number of shares that can be bought with the equal investment
                num_shares = equal_investment // self.current_prices[stock]
                if num_shares > 0:
                    self.portfolio[stock] += num_shares
                    self.cash -= num_shares * self.current_prices[stock]
                    self.log_action(stock, 'buy', num_shares, self.current_prices[stock])
            return

        # If not all holdings are zero, allocate based on predictions and ratios
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
        Calculate daily and monthly returns and track the portfolio performance.
        """
        num_days = len(next(iter(self.validation_data.values())))  # Total number of days in validation data
        days_per_month = 21  # Assuming 21 trading days per month for simplicity

        start_value = self.total_value  # Initial portfolio value
        current_month = 1  # Track the current month

        for day in range(num_days):
            self.update_stock_prices(day)  # Update prices for the day
            self.make_decision(day)  # Make trading decisions
            self.allocate_funds()  # Allocate funds based on decisions

            # Calculate and record the total portfolio value for the day
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

            # Check if it's the end of the month (every 21 days) or the last day of the data
            if (day + 1) % days_per_month == 0 or day == num_days - 1:
                # Calculate the monthly return
                monthly_return = (total_value - start_value) / start_value * 100
                self.monthly_returns.append(monthly_return)

                # Print monthly summary
                print(f"\nEnd of Month {current_month}:")
                print(f"Monthly Return = {monthly_return:.2f}%")
                print(f"Total Portfolio Value = ${total_value:.2f}")
                print("=" * 50)

                # Update the start value for the next month and increment the month counter
                start_value = total_value
                current_month += 1

            # Clear actions for the next day
            self.daily_actions.clear()

        # Calculate and print the average monthly return
        if self.monthly_returns:
            average_monthly_return = np.mean(self.monthly_returns)
            print(f"\nAverage Monthly Return: {average_monthly_return:.2f}%")

        # Call the plot function at the end of the simulation
        self.plot_portfolio_comparison()

    def buy_and_hold(self):
        """
        Simulate a buy-and-hold strategy by dividing the initial cash equally among the five stocks and holding them.
        Calculate the portfolio value for each day to be able to plot it dynamically.
        :return: A list of daily portfolio values using a buy-and-hold strategy.
        """
        # Calculate the amount to invest in each stock
        equal_investment = 20000
        buy_and_hold_portfolio = {}

        # Buy stocks on the first day using equal investment
        initial_prices = self.get_prices_on_day(0)
        for stock, price in initial_prices.items():
            # Calculate the number of shares bought using floating-point division
            buy_and_hold_portfolio[stock] = equal_investment / price

        # Track the portfolio value for each day
        buy_and_hold_values = []
        num_days = len(next(iter(self.validation_data.values())))
        days_per_month = 21  # Assuming 21 trading days per month for simplicity

        for day in range(num_days):
            # Get the prices for the current day
            current_prices = self.get_prices_on_day(day)
            # Calculate the total portfolio value for the day
            total_value = sum(buy_and_hold_portfolio[stock] * current_prices[stock] for stock in buy_and_hold_portfolio)
            buy_and_hold_values.append(total_value)

        # Calculate monthly returns
        buy_and_hold_monthly_returns = []
        for i in range(0, len(buy_and_hold_values), days_per_month):
            start_value = buy_and_hold_values[i]
            end_value = buy_and_hold_values[min(i + days_per_month - 1, len(buy_and_hold_values) - 1)]
            monthly_return = (end_value - start_value) / start_value * 100
            buy_and_hold_monthly_returns.append(monthly_return)

        average_monthly_return = np.mean(buy_and_hold_monthly_returns) if buy_and_hold_monthly_returns else 0
        print(f"\nBuy and Hold Strategy Final Value: ${buy_and_hold_values[-1]:.2f}")
        print(f"Buy and Hold Strategy Average Monthly Return: {average_monthly_return:.2f}%")

        return buy_and_hold_values

    def plot_portfolio_comparison(self):
        """
        Plot the portfolio value over time with days on the x-axis, along with the buy-and-hold strategy.
        """
        # Get the daily values of the buy-and-hold strategy
        buy_and_hold_values = self.buy_and_hold()

        # Create a new figure for portfolio comparison
        plt.figure(figsize=(12, 6))
        plt.plot(self.portfolio_values, label='Predicted Portfolio Value', linestyle='-', linewidth=2, color='blue')
        plt.plot(buy_and_hold_values, label='Buy and Hold Strategy', color='red', linestyle='--')

        # Title and labels
        plt.title('Portfolio Value Comparison: Predicted vs Buy and Hold')
        plt.xlabel('Days')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # Show the plot
        plt.show()

    def buy_single_stocks(self):
        """
        Buy £100,000 worth of each individual stock on the first day and track its value throughout the trading period.
        :return: A dictionary containing lists of daily values for each single-stock portfolio.
        """
        single_stock_values = {}

        # Calculate the number of shares bought for each stock on the first day
        initial_prices = self.get_prices_on_day(0)
        for stock, price in initial_prices.items():
            # Check if the initial price is valid (non-zero and positive)
            if price <= 0:
                print(f"Warning: Initial price for {stock} is not valid. Skipping.")
                continue

            # Calculate the number of shares bought for £100,000 investment
            num_shares = 100000 / price
            single_stock_values[stock] = []

            # Track the value of this single-stock portfolio over the trading period
            num_days = len(next(iter(self.validation_data.values())))
            for day in range(num_days):
                current_price = self.get_prices_on_day(day)[stock]
                total_value = num_shares * current_price
                single_stock_values[stock].append(total_value)

        return single_stock_values


    def plot_individual_stock_movements(self):
        """
        Plot the individual stock values over time on a separate plot.
        """
        # Get the individual stock values
        single_stock_values = self.buy_single_stocks()

        # Create a new figure for individual stock movements
        plt.figure(figsize=(12, 6))

        # Plot individual stock lines with different colors and thin lines
        colors = ['blue', 'green', 'purple', 'orange', 'brown']
        for idx, (stock, values) in enumerate(single_stock_values.items()):
            # Ensure that values are correctly populated and have no issues
            if not values or len(values) == 0:
                print(f"Warning: No values found for stock {stock}.")
                continue

            plt.plot(values, label=f'{stock} Single Stock', color=colors[idx % len(colors)], linestyle='-', linewidth=1)

        # Title and labels
        plt.title('Individual Stock Movements')
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
    agent_env.plot_individual_stock_movements()
