import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# using the calculations discussed in the past meeting, not working properly

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
        # Calculate the average gain and standard deviation of returns (daily percentage change)
        avg_gain = self.df['Returns'].mean()
        std_gain = self.df['Returns'].std()  # θ can be set to std_gain or a fraction of it
        theta = std_gain  # Use standard deviation as θ, or modify based on your needs

        # Output the average gain and standard deviation
        print(f"Average Gain: {avg_gain}")
        print(f"Standard Deviation: {std_gain}")

        # Plot the histogram of returns
        plt.figure(figsize=(10, 6))
        plt.hist(self.df['Returns'].dropna(), bins=50, alpha=0.75, color='blue', edgecolor='black', range=(-0.5, 0.5))
        plt.title('Histogram of Stock Returns')
        plt.xlabel('Returns')
        plt.ylabel('Frequency')
        plt.grid(True)

        plt.plot(self.df['Returns'].dropna())
        plt.show()

        # Define the cutoff points
        cutoffs = {
            "avg_gain": avg_gain,
            "moderate_gain": avg_gain + 0.5 * theta,
            "significant_gain": avg_gain + theta
        }

        # Mark the cutoff points on the plot
        plt.axvline(x=0, color='red', linestyle='--', label='Loss/Small Gain Cutoff (0)')
        plt.axvline(x=cutoffs['avg_gain'], color='green', linestyle='--', label=f'Small Gain Cutoff ({avg_gain:.4f})')
        plt.axvline(x=cutoffs['moderate_gain'], color='purple', linestyle='--', label=f'Moderate Gain Cutoff ({cutoffs["moderate_gain"]:.4f})')
        plt.axvline(x=cutoffs['significant_gain'], color='orange', linestyle='--', label=f'Significant Gain Cutoff ({cutoffs["significant_gain"]:.4f})')

        # Add a legend
        plt.legend()

        # Show the plot
        plt.show()

        # Define the conditions based on the rules you provided
        conditions = [
            (self.df['Returns'] < 0),  # Loss
            (self.df['Returns'] >= 0) & (self.df['Returns'] < avg_gain),  # Small gain
            (self.df['Returns'] >= avg_gain) & (self.df['Returns'] < avg_gain + 0.5 * theta),  # Moderate gain
            (self.df['Returns'] >= avg_gain + 0.5 * theta) & (self.df['Returns'] < avg_gain + theta),  # Significant gain
            (self.df['Returns'] >= avg_gain + theta),  # Strong gain
        ]

        # Define action choices based on the conditions
        choices = [-1, 0, 1, 2, 3]

        # Apply conditions to classify actions
        self.df['Action'] = np.select(conditions, choices, default=1)  # Default is Hold (1)

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
    df = pd.read_csv("../MLP/GOOG_2019_to_2023.csv")

    # Create a DataProcessor instance
    processor = DataProcessor(df)

    # Preprocess the data (calculate indicators and classify actions)
    processed_df = processor.preprocess_data()

    # Split the data into training and test sets
    train_df, test_df = processor.split_data()

    # Output the tail of the processed DataFrame
    print(processed_df.tail())
