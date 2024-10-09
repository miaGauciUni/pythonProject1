import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os


class TechnicalIndicators:
    def __init__(self, file_name):
        """
        Initialize the class by loading the data from the given CSV file.
        """
        self.file_name = file_name
        self.data = pd.read_csv(file_name)
        print(f"Data loaded from {file_name}.")

    def calculate_sma5(self, window=5):
        """
        Calculate the 5-day Simple Moving Average (SMA).
        """
        self.data['SMA_5'] = self.data['Closing'].rolling(window=window).mean()
        print("5-Day SMA calculated.")

    def calculate_sma10(self, window=10):
        """
        Calculate the 5-day Simple Moving Average (SMA).
        """
        self.data['SMA_10'] = self.data['Closing'].rolling(window=window).mean()
        print("10-Day SMA calculated.")

    def calculate_ema5(self, span=5):
        """
        Calculate the 5-day Exponential Moving Average (EMA).
        """
        self.data['EMA_5'] = self.data['Closing'].ewm(span=span, adjust=False).mean()
        print("5-Day EMA calculated.")

    def calculate_ema10(self, span=10):
        """
        Calculate the 5-day Exponential Moving Average (EMA).
        """
        self.data['EMA_10'] = self.data['Closing'].ewm(span=span, adjust=False).mean()
        print("10-Day EMA calculated.")

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

    def calculate_pca(self):
        """
        Calculate PCA on the 'Closing' price data.
        """
        # Prepare data for PCA
        closing_prices = self.data[['Closing']].values
        pca = PCA(n_components=1)  # We're interested in the first principal component
        principal_components = pca.fit_transform(closing_prices)
        self.data['PCA'] = principal_components
        print("PCA calculated.")

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
        self.calculate_sma5()
        self.calculate_sma10()
        self.calculate_ema5()
        self.calculate_ema10()
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_pca()
        self.drop_nan_values()
        self.save_to_csv(output_file_name)


# Example usage
if __name__ == '__main__':
    # Define the list of ticker symbols
    tickers = ['AAPL', 'AMZN', 'MSFT', 'NVDA', 'GOOG']

    # Iterate through each ticker and process the corresponding cleaned data file
    for ticker in tickers:
        # Construct the file name for the cleaned data
        file_name = f"3_clean_raw_data/{ticker}_clean.csv"

        if not os.path.exists(file_name):
            print(f"File {file_name} not found, skipping.")
            continue

        # Initialize the technical indicator class with the cleaned CSV file
        indicator_calculator = TechnicalIndicators(file_name)

        # Run the full process and save the data with indicators to a new CSV file
        output_file_name = f"4_data_w_indicators/{ticker}_tech_ind.csv"
        indicator_calculator.run(output_file_name)
