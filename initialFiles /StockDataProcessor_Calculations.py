import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

# Responsible for loading, cleaning, transforming, and preparing your stock data for analysis or modeling

class DataProcessor:
    def __init__(self, df):
        self.df = df

    def split_data(self, train_ratio=0.6):
        """Splitting the data into training and test sets based on temporal order."""
        split_index = int(len(self.df) * train_ratio)
        train_df = self.df.iloc[:split_index]  # First 60% for training
        test_df = self.df.iloc[split_index:]    # Last 40% for testing
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
        # Calculate the mean and standard deviation of returns
        mean_return = self.df['Returns'].mean()
        std_return = self.df['Returns'].std()

        # Output the mean and standard deviation
        print(f"Mean Return: {mean_return}")
        print(f"Standard Deviation: {std_return}")

        # Determine the fixed range for the histogram and normal curve
        fixed_range = (mean_return - 3 * std_return, mean_return + 3 * std_return)  # Example: +/- 3 SD

        # Plot the histogram of returns
        plt.figure(figsize=(12, 6))
        counts, bins, patches = plt.hist(self.df['Returns'].dropna(), bins=50, alpha=0.75,
                                          color='blue', edgecolor='black', density=True,
                                          range=fixed_range)

        # Define cutoff points based on the mean and standard deviations
        cutoffs = {
            "loss_cutoff": mean_return - std_return,  # -1 SD
            "small_gain_cutoff": mean_return,  # 0 SD
            "moderate_gain_cutoff": mean_return + std_return,  # +1 SD
            "significant_gain_cutoff": mean_return + 2 * std_return,  # +2 SD
        }

        # Mark the cutoff points on the plot
        plt.axvline(x=cutoffs['loss_cutoff'], color='red', linestyle='--', label='Loss Cutoff (-1 SD)')
        plt.axvline(x=cutoffs['small_gain_cutoff'], color='green', linestyle='--', label='Small Gain Cutoff (0 SD)')
        plt.axvline(x=cutoffs['moderate_gain_cutoff'], color='purple', linestyle='--', label='Moderate Gain Cutoff (+1 SD)')
        plt.axvline(x=cutoffs['significant_gain_cutoff'], color='orange', linestyle='--', label='Significant Gain Cutoff (+2 SD)')

        # Overlay normal distribution curve
        x = np.linspace(fixed_range[0], fixed_range[1], 1000)
        y = norm.pdf(x, mean_return, std_return)
        plt.plot(x, y, color='black', linewidth=2, label='Normal Distribution Curve')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.title('Histogram of Stock Returns with Normal Distribution Curve')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()

        # Define the conditions based on the new cutoffs
        conditions = [
            (self.df['Returns'] < cutoffs['loss_cutoff']),  # Loss
            (self.df['Returns'] >= cutoffs['loss_cutoff']) & (self.df['Returns'] < cutoffs['small_gain_cutoff']),  # Small gain
            (self.df['Returns'] >= cutoffs['small_gain_cutoff']) & (self.df['Returns'] < cutoffs['moderate_gain_cutoff']),  # Moderate gain
            (self.df['Returns'] >= cutoffs['moderate_gain_cutoff']) & (self.df['Returns'] < cutoffs['significant_gain_cutoff']),  # Significant gain
            (self.df['Returns'] >= cutoffs['significant_gain_cutoff']),  # Strong gain
        ]

        # Define action choices based on the conditions
        choices = [-1, 0, 1, 2, 3]

        # Apply conditions to classify actions
        self.df['Action'] = np.select(conditions, choices, default=1)  # Default is Hold (1)

    def count_actions(self):
        """Count occurrences of each action."""
        action_counts = self.df['Action'].value_counts().sort_index()
        print("\nAction Counts:")
        for action, count in action_counts.items():
            print(f"Action {action}: {count} occurrences")

    def preprocess_data(self):
        """Preprocess the data by scaling, calculating indicators, and classifying actions."""
        scaler = StandardScaler()
        self.df[['Open Price', 'Closing Price', 'Volume']] = scaler.fit_transform(
            self.df[['Open Price', 'Closing Price', 'Volume']])

        self.calculate_technical_indicators()
        self.classify_actions()
        self.count_actions()  # Count actions after classifying them

        return self.df

    def set_random_seed(self, seed_value):
        """Set random seed for reproducibility."""
        np.random.seed(seed_value)

# Example usage
if __name__ == "__main__":
    # Load the CSV file
    df = pd.read_csv("MSFT_2019_to_2023.csv")

    # Create a DataProcessor instance
    processor = DataProcessor(df)

    # Preprocess the data (calculate indicators and classify actions)
    processed_df = processor.preprocess_data()

    # Split the data into training and test sets
    train_df, test_df = processor.split_data()

    # Output the tail of the processed DataFrame
    print(processed_df.tail())
