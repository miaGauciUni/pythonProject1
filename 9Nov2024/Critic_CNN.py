import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras.optimizers import Adam




class MultiStockMarketOptimizerCNN:
    def __init__(self, num_days=3, num_stocks=5, num_features=6, learning_rate=0.001, gamma=0.95):
        self.num_days = num_days
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.input_shape = (num_days, num_stocks, num_features)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.model = self.build_model()

    def load_data(self, file_paths):
        stock_data = []
        for file_path in file_paths:
            stock_name = file_path.split('_')[0]
            data = pd.read_csv(file_path)
            data['Stock'] = stock_name
            data = self.reformat_data(data)
            stock_data.append(data)
        all_data = pd.concat(stock_data)
        return all_data

    def reformat_data(self, data):
        reshaped_data = []
        for i in range(1, 4):
            day_data = data[[f'Closing_{i}', f'EMA_5_{i}', f'RSI_{i}', f'MACD_{i}', f'BB_upper_{i}', f'BB_lower_{i}']]
            day_data.columns = ['Closing', 'EMA_5', 'RSI', 'MACD', 'BB_upper', 'BB_lower']
            reshaped_data.append(day_data)
        return pd.concat(reshaped_data, keys=[2, 1, 0])

    def feature_engineering(self, data):
        return data.groupby(level=0).transform(lambda x: (x - x.mean()) / x.std())

    def prepare_data(self, data, window_size=3):
        X = []
        for i in range(len(data) - window_size):
            X_window = []
            for stock in data.index.get_level_values(0).unique():
                stock_data = data.loc[stock].iloc[i:i + window_size].values
                # Only add data if it matches the required shape
                if stock_data.shape == (window_size, self.num_features):
                    X_window.append(stock_data)
            if len(X_window) == self.num_stocks:  # Only add if we have data for all stocks
                X.append(np.stack(X_window, axis=1))
        X = np.array(X)
        return X

    def build_model(self):
        model = Sequential([
            Conv2D(64, (2, 2), activation='relu', input_shape=self.input_shape),
            Conv2D(32, (2, 2), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def calculate_reward(self, current_price, previous_price):
        return np.mean((current_price - previous_price) / previous_price)

    def train_step(self, X, y_real):
        self.model.fit(X, y_real, epochs=1, verbose=0)

    def update(self, X, next_X, previous_price, current_price):
        value_current = self.model.predict(X)
        value_next = self.model.predict(next_X)
        reward = self.calculate_reward(current_price, previous_price)
        y_real = reward + self.gamma * value_next
        self.train_step(X, y_real)

    def predict(self, X):
        return self.model.predict(X)

def main():
    # Paths to your stock files
    file_paths = [
        '3-Day_Window/AAPL_3day_tech_action.csv',
        '3-Day_Window/AMZN_3day_tech_action.csv',
        '3-Day_Window/GOOG_3day_tech_action.csv',
        '3-Day_Window/MSFT_3day_tech_action.csv',
        '3-Day_Window/NVDA_3day_tech_action.csv'
    ]

    # Initialize and prepare data
    optimizer = MultiStockMarketOptimizerCNN(num_days=3, num_stocks=len(file_paths), num_features=6)
    data = optimizer.load_data(file_paths)
    data_normalized = optimizer.feature_engineering(data)
    X_data = optimizer.prepare_data(data_normalized)

    # Training loop example
    for i in range(len(X_data) - 1):
        X = X_data[i]
        next_X = X_data[i + 1]
        previous_price = data.groupby(level=0)['Closing'].shift(1).values[i]
        current_price = data.groupby(level=0)['Closing'].values[i + 1]
        optimizer.update(X, next_X, previous_price, current_price)

    # Predict future rewards
    predictions = optimizer.predict(X_data[-1:])

    # Plotting the actual closing prices and predicted prices for comparison
    closing_prices = data.groupby(level=0)['Closing'].apply(list)

    # Real Stock Performance
    plt.figure(figsize=(12, 5))
    for stock_name, prices in closing_prices.items():
        plt.plot(prices, label=f"{stock_name} - Actual")
    plt.title("Real Stock Performance")
    plt.xlabel("Time")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.show()

    # Predicted Price Performance
    plt.figure(figsize=(12, 5))
    for idx, stock_name in enumerate(closing_prices.index):
        plt.plot([pred[0] for pred in predictions], label=f"{stock_name} - Predicted")
    plt.title("Predicted Stock Performance by Neural Network")
    plt.xlabel("Time")
    plt.ylabel("Predicted Reward")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
