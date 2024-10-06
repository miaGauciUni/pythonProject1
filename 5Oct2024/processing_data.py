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
        
        #Classifies the daily price differences into categories based on quantiles.
        
        # Drop NaN values that result from the diff() operation
        differences = self.df['Price_Difference'].dropna()

        # Calculate quantiles
        q1 = np.quantile(differences[differences > 0], 0.4)  # 33rd percentile
        q2 = np.quantile(differences[differences > 0], 0.75)  # 67th percentile

        # Print cutoff values
        print(f"Cutoff points:\nHold (0): {0}\nBuy (1): {q1}\nStrong Buy (2): {q2}")

        # Classify the price differences
        self.df['Classification'] = np.select(
            [
                self.df['Price_Difference'] < 0,
                (self.df['Price_Difference'] >= 0) & (self.df['Price_Difference'] < q1), # price difference is between 0 and q1
                (self.df['Price_Difference'] >= q1) & (self.df['Price_Difference'] < q2), # price difference is between q1 and q2
                (self.df['Price_Difference'] >= q2) # price difference is greater than or equal to q2
            ],
            [-1, 0, 1, 2],
            default=-1  # This shouldn't be hit due to the conditions above
        )
    

    def plot_histogram(self):
        
        #Plots a histogram of the daily price differences with classifications and cutoff points.
        
        # Drop NaN values that result from the diff() operation
        differences = self.df['Price_Difference'].dropna()

        # Calculate quantiles
        q1 = np.quantile(differences[differences > 0], 0.4)  # 1st pos quantile - adjust accordingly
        q2 = np.quantile(differences[differences > 0], 0.75)  # 2nd pos quantile - adjust accordingly

        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(differences, bins=30, color='blue', alpha=0.7)
        plt.axvline(0, color='green', linestyle='dashed', linewidth=1, label='Hold Cutoff (0)')
        plt.axvline(q1, color='orange', linestyle='dashed', linewidth=1, label='Buy Cutoff (q1)')
        plt.axvline(q2, color='red', linestyle='dashed', linewidth=1, label='Strong Buy Cutoff (q2)')
        plt.title('Histogram of Daily Closing Price Differences for GOOG Stock')
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
