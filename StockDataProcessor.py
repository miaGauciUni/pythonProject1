import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# The user decides where to set the cutoff points by analysing the normal distribution curve

class DataProcessor:
    def __init__(self, df):
        self.df = df

    def split_data(self, test_size=0.2):
        """Splitting the data into training and test sets."""
        train_df, test_df = train_test_split(self.df, test_size=test_size, shuffle=False)
        return train_df, test_df

    def calculate_technical_indicators(self):
        """Calculate 5-day Simple Moving Average (SMA), MACD, and RSI."""
        # 5-day Simple Moving Average (SMA)
        self.df['SMA_5'] = self.df['Closing Price'].rolling(window=5).mean()
        self.df['SMA_20'] = self.df['Closing Price'].rolling(window=20).mean()  # 20-day SMA

        # MACD Calculation
        short_ema = self.df['Closing Price'].ewm(span=12, adjust=False).mean()  # 12-day EMA
        long_ema = self.df['Closing Price'].ewm(span=26, adjust=False).mean()  # 26-day EMA
        self.df['MACD'] = short_ema - long_ema  # MACD Line
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()  # Signal Line

        # Percentage returns (daily returns)
        self.df['Returns'] = self.df['Closing Price'].pct_change()

        # RSI Calculation (using 14-day window)
        delta = self.df['Closing Price'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

    def save_to_csv(self, file_name):
        """Save the DataFrame with calculated indicators to a CSV file."""
        self.df.to_csv(file_name, index=False)

    def classify_actions(self):
        # Manually define return ranges for each action
        cutoffs = [-0.4, -0.1, 0.0, 0.1, 0.3, 0.4]  # Example cutoffs for each action
        labels = [-1, 0, 1, 2, 3]  # Actions corresponding to the ranges

        # Output the action cutoffs for reference
        print(f"Cutoff Points: {cutoffs}")

        # Plot the histogram of returns with action cutoffs
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['Returns'].dropna(), bins=50, alpha=0.75, color='blue', edgecolor='black', range=(-0.5, 0.5))
        plt.title('Histogram of Stock Returns with Action Cutoffs')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.grid(True)

        # Mark the cutoff points on the plot
        for i in range(len(cutoffs)):  # Loop through each cutoff
            plt.axvline(x=cutoffs[i], color='red', linestyle='--')
            if i < len(labels):  # Check if there's a corresponding label
                plt.text(cutoffs[i], plt.gca().get_ylim()[1] * 0.8, f'Action {labels[i]}', color='red', ha='center')

        # Show the plot
        plt.show()

        # Classify actions based on the defined cutoff ranges
        self.df['Action'] = pd.cut(self.df['Returns'], bins=cutoffs, labels=labels, include_lowest=True)

    def preprocess_data(self):
        """Preprocess the data by scaling, calculating indicators, and classifying actions."""
        scaler = StandardScaler()
        self.df[['Open Price', 'Closing Price', 'Volume']] = scaler.fit_transform(
            self.df[['Open Price', 'Closing Price', 'Volume']])

        self.calculate_technical_indicators()
        self.classify_actions()

        return self.df


# Example usage
if __name__ == "__main__":
    # Load the CSV file
    df = pd.read_csv("AAPL_2019_to_2023.csv")

    # Create a DataProcessor instance
    processor = DataProcessor(df)

    # Preprocess the data (calculate indicators and classify actions)
    processed_df = processor.preprocess_data()

    # Split the data into training and test sets
    train_df, test_df = processor.split_data()

    # Output the tail of the processed DataFrame
    print(processed_df.tail())
