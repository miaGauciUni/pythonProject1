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
        print(f"Cutoff points:\nHold (0): {0}\nBuy (1): {q1}\nStrong Buy (2): {q2}")

        # Define ranges
        print(f"Classification Ranges:\nHold (0): [0, {q1})\nBuy (1): [{q1}, {q2})\nStrong Buy (2): [{q2}, ∞)")

        # Classify the price differences
        self.df['Classification'] = np.select(
            [
                self.df['Price_Difference'] < 0,
                (self.df['Price_Difference'] >= 0) & (self.df['Price_Difference'] < q1),
                (self.df['Price_Difference'] >= q1) & (self.df['Price_Difference'] < q2),
                (self.df['Price_Difference'] >= q2)
            ],
            [-1, 0, 1, 2],
            default=-1
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
        price_differences = self.df['Price_Difference'].dropna()

        # Calculate quantiles
        quantile1 = np.quantile(price_differences[price_differences > 0], 0.37)
        quantile2 = np.quantile(price_differences[price_differences > 0], 0.84)

        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(price_differences, bins=30, color='blue', alpha=0.7)
        plt.axvline(0, color='green', linestyle='dashed', linewidth=1, label='Hold Cutoff (0)')
        plt.axvline(quantile1, color='orange', linestyle='dashed', linewidth=1, label='Buy Cutoff (q1)')
        plt.axvline(quantile2, color='red', linestyle='dashed', linewidth=1, label='Strong Buy Cutoff (q2)')
        plt.title('Histogram of Daily Closing Price Differences for MSFT Stock')

        plt.xlabel('Price Difference (USD)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.legend()
        plt.show()

    def evaluate_last_3_days(self):
        """
        Evaluates the last 3 days of data and includes the technical indicators.
        """
        actions = []

        for i in range(2, len(self.df)):
            # Extract closing prices and technical indicators for the last 3 days
            Closings = self.df['Closing'].iloc[i - 2:i + 1].values
            sma_5 = self.df['SMA_5'].iloc[i - 2:i + 1].values
            ema_5 = self.df['EMA_5'].iloc[i - 2:i + 1].values
            rsi = self.df['RSI'].iloc[i - 2:i + 1].values
            macd = self.df['MACD'].iloc[i - 2:i + 1].values
            signal_line = self.df['Signal_Line'].iloc[i - 2:i + 1].values
            bb_upper = self.df['BB_upper'].iloc[i - 2:i + 1].values
            bb_lower = self.df['BB_lower'].iloc[i - 2:i + 1].values

            # Calculate action based on the current closing price difference
            action = self.determine_action(Closings[-1], Closings[:-1])
            actions.append([
                *Closings, *sma_5, *ema_5, *rsi, *macd, *signal_line, *bb_upper, *bb_lower, action
            ])

        # Create a new DataFrame with the technical indicators and actions
        actions_df = pd.DataFrame(actions, columns=[
            'Closing_3', 'Closing_2', 'Closing_1',
            'SMA_5_3', 'SMA_5_2', 'SMA_5_1',
            'EMA_5_3', 'EMA_5_2', 'EMA_5_1',
            'RSI_3', 'RSI_2', 'RSI_1',
            'MACD_3', 'MACD_2', 'MACD_1',
            'Signal_Line_3', 'Signal_Line_2', 'Signal_Line_1',
            'BB_upper_3', 'BB_upper_2', 'BB_upper_1',
            'BB_lower_3', 'BB_lower_2', 'BB_lower_1',
            'Action'
        ])

        # Save the actions DataFrame to a CSV file
        actions_df.to_csv("processedWithActions/MSFT_3day_tech_action.csv", index=False)
        print("Actions with technical indicators saved to MSFT_3day_tech_action.csv.")

    def determine_action(self, current_Closing, previous_Closings):
        """
        Determine the action based on the current Closing price and previous Closings.
        """
        price_diff = current_Closing - previous_Closings[-1]

        if price_diff < 0:
            return -1  # Loss action
        elif price_diff >= 0 and price_diff < 1:
            return 0  # Hold action
        elif price_diff >= 1 and price_diff < 2:
            return 1  # Buy action
        else:
            return 2  # Strong Buy action


if __name__ == '__main__':
    # Initialize the processing class with the technical indicator CSV file
    processing = ProcessingData("techIndicators/MSFT_tech_ind.csv")

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

    # Evaluate the last 3 days with technical indicators and save the results
    processing.evaluate_last_3_days()
