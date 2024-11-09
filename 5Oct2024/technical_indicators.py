import pandas as pd
import numpy as np


class TechnicalIndicators:
    def __init__(self, file_name):
        """
        Initialize the class by loading the data from the given CSV file.
        """
        self.file_name = file_name
        self.data = pd.read_csv(file_name)
        print(f"Data loaded from {file_name}.")

    def calculate_sma(self, window=5):
        """
        Calculate the 5-day Simple Moving Average (SMA).
        """
        self.data['SMA_5'] = self.data['Closing'].rolling(window=window).mean()
        print("5-Day SMA calculated.")

    def calculate_ema(self, span=5):
        """
        Calculate the 5-day Exponential Moving Average (EMA).
        """
        self.data['EMA_5'] = self.data['Closing'].ewm(span=span, adjust=False).mean()
        print("5-Day EMA calculated.")

    def calculate_rsi(self, window=14):
        """
        Calculate the Relative Strength Index (RSI).
        """
        delta = self.data['Closing'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        print("RSI calculated.")

    def calculate_macd(self):
        """
        Calculate the Moving Average Convergence Divergence (MACD).
        """
        short_ema = self.data['Closing'].ewm(span=12, adjust=False).mean()
        long_ema = self.data['Closing'].ewm(span=26, adjust=False).mean()
        self.data['MACD'] = short_ema - long_ema
        self.data['Signal_Line'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        print("MACD and Signal Line calculated.")

    def calculate_bollinger_bands(self, window=20):
        """
        Calculate the Bollinger Bands.
        """
        self.data['SMA_20'] = self.data['Closing'].rolling(window=window).mean()
        self.data['BB_upper'] = self.data['SMA_20'] + 2 * self.data['Closing'].rolling(window=window).std()
        self.data['BB_lower'] = self.data['SMA_20'] - 2 * self.data['Closing'].rolling(window=window).std()
        print("Bollinger Bands calculated.")

    def drop_nan_values(self):
        """
        Drop rows with any NaN values.
        """
        self.data.dropna(inplace=True)
        print("NaN values dropped.")

    def save_to_csv(self, output_file_name):
        """
        Save the data with the technical indicators to a new CSV file.
        """
        self.data.to_csv(output_file_name, index=False)
        print(f"Data with technical indicators saved to {output_file_name}.")

    def run(self, output_file_name):
        """
        Execute the full workflow: calculate indicators, drop NaN values, and save to CSV.
        """
        self.calculate_sma()
        self.calculate_ema()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.drop_nan_values()
        self.save_to_csv(output_file_name)


# Example usage
if __name__ == '__main__':
    # Initialize the technical indicator class with the CSV file
    indicator_calculator = TechnicalIndicators("cleanData/GOOG_clean.csv")

    # Run the full process and save the data to a new CSV file
    indicator_calculator.run("techIndicators/GOOG_tech_ind.csv")
