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

    def classify_differences(self):
        """
        Classifies the daily price differences into categories based on quantiles.
        """
        # Drop NaN values that result from the diff() operation
        differences = self.df['Price_Difference'].dropna()

        # Calculate cutoff for loss action
        loss_cutoff = 0  # All values below 0 will be classified as -1

        # Identify the range of positive price differences
        positive_differences = differences[differences > 0]
        max_positive = positive_differences.max()

        # Divide the range of positive differences into 3 equal parts
        # Note: If max_positive is 0, the division will not work, so we handle that case
        if max_positive > 0:
            range_per_class = max_positive / 3
            buy_cutoff = range_per_class
            strong_buy_cutoff = 2 * range_per_class
        else:
            buy_cutoff = 0
            strong_buy_cutoff = 0

        # Print cutoff values
        print(f"Cutoff points:\nLoss Action (-1): {loss_cutoff}\nHold (0): {buy_cutoff}\nBuy (1): {strong_buy_cutoff}")

        # Classify the price differences
        self.df['Classification'] = np.select(
            [
                self.df['Price_Difference'] < loss_cutoff,  # Loss action
                (self.df['Price_Difference'] >= loss_cutoff) & (self.df['Price_Difference'] < buy_cutoff),  # Hold
                (self.df['Price_Difference'] >= buy_cutoff) & (self.df['Price_Difference'] < strong_buy_cutoff),  # Buy
                (self.df['Price_Difference'] >= strong_buy_cutoff)  # Strong Buy
            ],
            [-1, 0, 1, 2],
            default=-1  # This shouldn't be hit due to the conditions above
        )

    def plot_histogram(self):
        """
        Plots a histogram of the daily price differences with classifications and cutoff points.
        """
        # Drop NaN values that result from the diff() operation
        differences = self.df['Price_Difference'].dropna()

        # Calculate cutoff for loss action
        loss_cutoff = 0  # All values below 0 will be classified as -1

        # Identify the range of positive price differences
        positive_differences = differences[differences > 0]
        max_positive = positive_differences.max()

        # Divide the range of positive differences into 3 equal parts
        if max_positive > 0:
            range_per_class = max_positive / 3
            buy_cutoff = range_per_class
            strong_buy_cutoff = 2 * range_per_class
        else:
            buy_cutoff = 0
            strong_buy_cutoff = 0

        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(differences, bins=30, color='blue', alpha=0.7)
        plt.axvline(loss_cutoff, color='green', linestyle='dashed', linewidth=1, label='Loss Action Cutoff (0)')
        plt.axvline(buy_cutoff, color='orange', linestyle='dashed', linewidth=1, label='Buy Cutoff')
        plt.axvline(strong_buy_cutoff, color='red', linestyle='dashed', linewidth=1, label='Strong Buy Cutoff')
        plt.title('Histogram of Daily Closing Price Differences')
        plt.xlabel('Price Difference (USD)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)

        # Add legend
        plt.legend()

        # Show the plot
        plt.show()

if __name__ == '__main__':
    # Initialize the processing class with the cleaned Apple CSV file
    processing = ProcessingData("cleanData/TSLA_clean.csv")

    # Calculate the daily price differences
    processing.calculate_price_differences()

    # Classify the price differences into quantiles
    processing.classify_differences()

    # Plot the histogram of daily price differences with cutoff points
    processing.plot_histogram()