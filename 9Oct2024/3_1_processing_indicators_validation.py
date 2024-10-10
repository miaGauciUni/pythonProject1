import pandas as pd
import numpy as np
import os
import pickle  # Import the pickle module


class ProcessingIndicators:
    def __init__(self, stock_data_files):
        """
        Initialize the ProcessingIndicators class.
        :param stock_data_files: Dictionary of stock names and their respective file paths.
        """
        self.stock_data_files = stock_data_files

    def load_data(self, file_path):
        """Load the CSV data."""
        return pd.read_csv(file_path)

    def calculate_closing_prices(self, df):
        """Calculate shifted closing prices."""
        for i in range(5, 0, -1):
            df[f'Closing_{i}'] = df['Closing'].shift(i)
        print("Closing prices calculated.")

    def calculate_sma(self, df, window=5):
        """Calculate Simple Moving Averages (SMA) for the specified window."""
        for i in range(5, 0, -1):
            df[f'SMA_{window}_{i}'] = df['Closing'].rolling(window=window).mean().shift(i)
        print(f"SMA_{window} calculated.")

    def calculate_ema(self, df, span=5):
        """Calculate Exponential Moving Averages (EMA) for the specified span."""
        for i in range(5, 0, -1):
            df[f'EMA_{span}_{i}'] = df['Closing'].ewm(span=span, adjust=False).mean().shift(i)
        print(f"EMA_{span} calculated.")

    def calculate_rsi(self, df, window=14):
        """Calculate Relative Strength Index (RSI) for the specified window."""
        delta = df['Closing'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / (loss + 1e-10)  # Avoid division by zero

        # Calculate RSI for the specified lag periods
        for i in range(5, 0, -1):
            df[f'RSI_{i}'] = (100 - (100 / (1 + rs))).shift(i)

        print("RSI calculated.")

    def calculate_macd(self, df):
        """Calculate the Moving Average Convergence Divergence (MACD)."""
        short_ema = df['Closing'].ewm(span=12, adjust=False).mean()
        long_ema = df['Closing'].ewm(span=26, adjust=False).mean()
        df['MACD'] = short_ema - long_ema
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # Shift MACD and Signal Line for the last 5 days
        for i in range(5, 0, -1):
            df[f'MACD_{i}'] = df['MACD'].shift(i)
            df[f'Signal_Line_{i}'] = df['Signal_Line'].shift(i)

        print("MACD and Signal Line calculated.")

    def calculate_bollinger_bands(self, df, window=20):
        """Calculate the Bollinger Bands."""
        rolling_mean = df['Closing'].rolling(window=window).mean()
        rolling_std = df['Closing'].rolling(window=window).std()

        # Calculate upper and lower bands for lagged values
        for i in range(5, 0, -1):
            df[f'BB_upper_{i}'] = (rolling_mean + (rolling_std * 2)).shift(i)
            df[f'BB_lower_{i}'] = (rolling_mean - (rolling_std * 2)).shift(i)

        print("Bollinger Bands calculated.")

    def drop_nan_values(self, df):
        """Drop rows with any NaN values."""
        df.dropna(inplace=True)
        print("NaN values dropped.")

    def save_to_csv(self, df, output_file_name):
        """Save the DataFrame with technical indicators to a new CSV file."""
        df.to_csv(output_file_name, index=False)
        print(f"Processed data saved to {output_file_name}.")

    def save_to_pickle(self, df, output_file_name):
        """Save the DataFrame as a NumPy array in a pickle file."""
        np_array = df.to_numpy()  # Convert DataFrame to NumPy array
        with open(output_file_name, 'wb') as file:
            pickle.dump(np_array, file)  # Save the array to a pickle file
        print(f"Processed data saved as NumPy array in {output_file_name}.")

    def process_validation_data(self):
        """Process validation data for each stock."""
        for stock, file_path in self.stock_data_files.items():
            # Load data
            df = self.load_data(file_path)

            # Calculate indicators
            self.calculate_closing_prices(df)
            self.calculate_sma(df, window=5)
            self.calculate_sma(df, window=10)
            self.calculate_ema(df, span=5)
            self.calculate_ema(df, span=10)
            self.calculate_rsi(df, window=14)  # Use default RSI window of 14
            self.calculate_macd(df)
            self.calculate_bollinger_bands(df)

            # Drop unnecessary columns (Date and Closing)
            df.drop(columns=['Date', 'Closing'], inplace=True)

            # Drop rows with NaN values after adding indicators
            self.drop_nan_values(df)

            # Define the exact order of columns to save
            columns_order = [
                'Closing_5', 'Closing_4', 'Closing_3', 'Closing_2', 'Closing_1',
                'SMA_5_5', 'SMA_5_4', 'SMA_5_3', 'SMA_5_2', 'SMA_5_1',
                'SMA_10_5', 'SMA_10_4', 'SMA_10_3', 'SMA_10_2', 'SMA_10_1',
                'EMA_5_5', 'EMA_5_4', 'EMA_5_3', 'EMA_5_2', 'EMA_5_1',
                'EMA_10_5', 'EMA_10_4', 'EMA_10_3', 'EMA_10_2', 'EMA_10_1',
                'RSI_5', 'RSI_4', 'RSI_3', 'RSI_2', 'RSI_1',
                'MACD_5', 'MACD_4', 'MACD_3', 'MACD_2', 'MACD_1',
                'Signal_Line_5', 'Signal_Line_4', 'Signal_Line_3', 'Signal_Line_2', 'Signal_Line_1',
                'BB_upper_5', 'BB_upper_4', 'BB_upper_3', 'BB_upper_2', 'BB_upper_1',
                'BB_lower_5', 'BB_lower_4', 'BB_lower_3', 'BB_lower_2', 'BB_lower_1'
            ]

            # Save the processed data in the specified column order
            df = df[columns_order]
            processed_file_path_csv = f'2_validation_data/{stock}_processed_validation.csv'
            self.save_to_csv(df, processed_file_path_csv)

            # Save as a NumPy array in a pickle file
            processed_file_path_pickle = f'2_validation_data/{stock}_processed_validation.pkl'
            self.save_to_pickle(df, processed_file_path_pickle)


if __name__ == '__main__':
    stock_data_files = {
        'AAPL': '2_validation_data/AAPL_validation.csv',
        'AMZN': '2_validation_data/AMZN_validation.csv',
        'NVDA': '2_validation_data/NVDA_validation.csv',
        'GOOG': '2_validation_data/GOOG_validation.csv',
        'MSFT': '2_validation_data/MSFT_validation.csv'
    }

    processor = ProcessingIndicators(stock_data_files)
    processor.process_validation_data()
