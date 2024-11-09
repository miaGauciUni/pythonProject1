import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class ProcessingData:
    def __init__(self, file_name):
        self.file_name = file_name
        self.df = pd.read_csv(file_name)
        print(f"Data loaded from {file_name}.")

    def calculate_price_differences(self):
        self.df['Price_Difference'] = self.df['Closing'].diff()
        print("Price differences calculated.")
        self.df.to_csv(self.file_name, index=False)
        print(f"Price differences saved to {self.file_name}.")

    def calculate_quantiles(self):
        price_differences = self.df['Price_Difference'].dropna()
        quantile1 = np.quantile(price_differences[price_differences > 0], 0.37)
        quantile2 = np.quantile(price_differences[price_differences > 0], 0.7)
        print(f"Quantiles calculated:\nBuy (q1): {quantile1}\nStrong Buy (q2): {quantile2}")
        return quantile1, quantile2

    def classify_differences(self, q1, q2):
        print(f"Cutoff points for classification:\nHold (0): {0}\nBuy (1): {q1}\nStrong Buy (2): {q2}")

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

        counts = self.df['Classification'].value_counts()
        print("\nClassification Counts:")
        for action, count in counts.items():
            action_label = 'Loss Action (-1)' if action == -1 else f'Action {action}'
            print(f"{action_label}: {count}")

    def plot_histogram(self, q1, q2):
        price_differences = self.df['Price_Difference'].dropna()
        plt.figure(figsize=(10, 6))
        plt.hist(price_differences, bins=30, color='blue', alpha=0.7)
        plt.axvline(0, color='green', linestyle='dashed', linewidth=1, label='Hold Cutoff (0)')
        plt.axvline(q1, color='orange', linestyle='dashed', linewidth=1, label='Buy Cutoff (q1)')
        plt.axvline(q2, color='red', linestyle='dashed', linewidth=1, label='Strong Buy Cutoff (q2)')
        plt.title('Histogram of Daily Closing Price Differences for AAPL Stock')
        plt.xlabel('Price Difference (USD)')
        plt.ylabel('Frequency')
        plt.grid(axis='y', alpha=0.75)
        plt.legend()
        plt.show()

    def evaluate_last_3_days(self):
        actions = []

        for i in range(3, len(self.df)):
            Closings = self.df['Closing'].iloc[i - 3:i].values
            ema_5 = self.df['EMA_5'].iloc[i - 3:i].values
            rsi = self.df['RSI'].iloc[i - 3:i].values
            macd = self.df['MACD'].iloc[i - 3:i].values
            bb_upper = self.df['BB_upper'].iloc[i - 3:i].values
            bb_lower = self.df['BB_lower'].iloc[i - 3:i].values

            action = self.determine_action(Closings[-1], Closings[:-1])
            actions.append([
                *Closings, *ema_5, *rsi, *macd, *bb_upper, *bb_lower, action
            ])

        actions_df = pd.DataFrame(actions, columns=[
            'Closing_3', 'Closing_2', 'Closing_1',
            'EMA_5_3', 'EMA_5_2', 'EMA_5_1',
            'RSI_3', 'RSI_2', 'RSI_1',
            'MACD_3', 'MACD_2', 'MACD_1',
            'BB_upper_3', 'BB_upper_2', 'BB_upper_1',
            'BB_lower_3', 'BB_lower_2', 'BB_lower_1',
            'Action'
        ])

        # Update the Action column directly from the Classification
        actions_df['Action'] = self.df['Classification'].iloc[3:].values  # Skip the first 3 entries

        actions_df.to_csv("5_processed_with_actions/MSFT_3day_tech_action.csv", index=False)
        print("Actions with technical indicators saved to MSFT_3day_tech_action.csv.")

    def determine_action(self, current_Closing, previous_Closings):
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
    processing = ProcessingData("4_data_w_indicators/MSFT_tech_ind.csv")
    processing.calculate_price_differences()
    quantile1, quantile2 = processing.calculate_quantiles()
    processing.classify_differences(quantile1, quantile2)
    processing.plot_histogram(quantile1, quantile2)
    processing.evaluate_last_3_days()
