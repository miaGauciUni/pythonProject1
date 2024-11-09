import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class ProcessingData:
    def __init__(self, file_name):
        """
        Initializes the class with the CSV file name and loads the data into a DataFrame.
        """
        self.file_name = file_name
        self.df = pd.read_csv(file_name)
        print(f"Data loaded from {file_name}.")

    def calculate_price_differences(self):
        """
        Calculates the differences in closing prices from one day to another.
        """
        self.df['Price_Difference'] = self.df['Closing'].diff()  # Calculate the difference
        print("Price differences calculated.")

    def classify_differences(self, q1, q2):
        """
        Classifies the daily price differences into categories based on provided quantiles.
        """
        # Drop NaN values that result from the diff() operation
        #price_differences = self.df['Price_Difference'].dropna()

        # Print cutoff values with ranges
        print(f"Cutoff points:\nHold (0): {0}\nBuy (1): {q1}\nStrong Buy (2): {q2}")

        # Define ranges
        print(f"Classification Ranges:\nHold (0): [0, {q1})\nBuy (1): [{q1}, {q2})\nStrong Buy (2): [{q2}, ∞)")

        # Classify the price differences
        self.df['Classification'] = np.select(
            [
                self.df['Price_Difference'] < 0,
                (self.df['Price_Difference'] >= 0) & (self.df['Price_Difference'] < q1),  # between 0 and q1
                (self.df['Price_Difference'] >= q1) & (self.df['Price_Difference'] < q2),  # between q1 and q2
                (self.df['Price_Difference'] >= q2)  # greater than or equal to q2
            ],
            [-1, 0, 1, 2],
            default=-1  # This shouldn't be hit due to the conditions above
        )

        # Count occurrences of each classification
        counts = self.df['Classification'].value_counts()
        print(f"\nClassification Counts:")
        for action, count in counts.items():
            action_label = 'Loss Action (-1)' if action == -1 else f'Action {action}'
            print(f"{action_label}: {count}")

    def plot_histogram(self):
        """
        Plots a histogram of the daily price differences with classifications and cutoff points.
        """
        # Drop NaN values that result from the diff() operation
        price_differences = self.df['Price_Difference'].dropna()

        # Calculate quantiles
        # MSFT - 0.3, 0.73
        # GOOG - 0.44, 0.7
        # AAPL - 0.37, 0.72

        quantile1 = np.quantile(price_differences[price_differences > 0], 0.37)  # 45th percentile
        quantile2 = np.quantile(price_differences[price_differences > 0], 0.84)  # 80th percentile

        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(price_differences, bins=30, color='blue', alpha=0.7)
        plt.axvline(0, color='green', linestyle='dashed', linewidth=1, label='Hold Cutoff (0)')
        plt.axvline(quantile1, color='orange', linestyle='dashed', linewidth=1, label='Buy Cutoff (q1)')
        plt.axvline(quantile2, color='red', linestyle='dashed', linewidth=1, label='Strong Buy Cutoff (q2)')
        plt.title('Histogram of Daily Closing Price Differences for AAPL Stock')
        plt.xlabel('Price Difference (USD)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)


        plt.legend()
        plt.show()
    """
    def evaluate_last_5_days(self):

        # Creating a new DataFrame to hold the last 5 days of data
        actions = []

        for i in range(4, len(self.df)):
            # Extract closing prices and volumes for the last 5 days
            closes = self.df['Closing'].iloc[i - 4:i + 1].values
            volumes = self.df['Volume'].iloc[i - 4:i + 1].values

            # Calculate action based on the current closing price difference
            action = self.determine_action(closes[-1], closes[:-1], volumes)
            actions.append([*closes, *volumes, action])

        # Create a new DataFrame to save the actions
        actions_df = pd.DataFrame(actions, columns=[
            'Close_5', 'Close_4', 'Close_3', 'Close_2', 'Close_1',
            'Volume_5', 'Volume_4', 'Volume_3', 'Volume_2', 'Volume_1',
            'Action'
        ])

        # Save the actions DataFrame to a CSV file
        actions_df.to_csv("AAPL_5day_action.csv", index=False)
        print("Actions saved to AAPL_5day_action.csv.")
        
    """

    def evaluate_last_5_days(self):

        # Creating a new DataFrame to hold the last 3 days of data
        actions = []

        for i in range(2, len(self.df)):
            # Extract closing prices and volumes for the last 3 days
            closes = self.df['Closing'].iloc[i - 2:i + 1].values
            volumes = self.df['Volume'].iloc[i - 2:i + 1].values

            # Calculate action based on the current closing price difference
            action = self.determine_action(closes[-1], closes[:-1], volumes)
            actions.append([*closes, *volumes, action])

        # Create a new DataFrame to save the actions
        actions_df = pd.DataFrame(actions, columns=[
             'Close_3', 'Close_2', 'Close_1',
             'Volume_3', 'Volume_2', 'Volume_1',
            'Action'
        ])

        # Save the actions DataFrame to a CSV file
        actions_df.to_csv("AAPL_3day_action.csv", index=False)
        print("Actions saved to AAPL_3day_action.csv.")

    def determine_action(self, current_close, previous_closes, volumes):
        """
        Determine the action based on the current close price and previous closes.
        """
        price_diff = current_close - previous_closes[-1]
        #volume_avg = np.mean(volumes)

        # Determine action based on price differences and volume
        if price_diff < 0:
            return -1  # Loss action
        elif price_diff >= 0 and price_diff < 1:
            return 0  # Hold action
        elif price_diff >= 1 and price_diff < 2:
            return 1  # Buy action
        else:
            return 2  # Strong Buy action


if __name__ == '__main__':
    # Initialize the processing class with the cleaned Google CSV file
    processing = ProcessingData("cleanData/AAPL_clean.csv")

    # Calculate the daily price differences
    processing.calculate_price_differences()

    # Calculate quantiles
    price_differences = processing.df['Price_Difference'].dropna()
    quantile1 = np.quantile(price_differences[price_differences > 0], 0.37)
    quantile2 = np.quantile(price_differences[price_differences > 0], 0.84)

    # Classify the price differences into quantiles
    processing.classify_differences(quantile1, quantile2)

    # Plot the histogram of daily price differences with cutoff points
    processing.plot_histogram()

    # Evaluate the last 5 days and save the results
    processing.evaluate_last_5_days()
