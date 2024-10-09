import pandas as pd
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from gymnasium import spaces

class DataProcessor:
    def __init__(self, df):
        self.df = df

    def split_data(self, train_ratio=0.6):
        split_index = int(len(self.df) * train_ratio)
        train_df = self.df.iloc[:split_index]
        test_df = self.df.iloc[split_index:]
        return train_df, test_df

    def calculate_technical_indicators(self):
        self.df['SMA_5'] = self.df['Closing Price'].rolling(window=5).mean()
        self.df['SMA_20'] = self.df['Closing Price'].rolling(window=20).mean()
        short_ema = self.df['Closing Price'].ewm(span=12, adjust=False).mean()
        long_ema = self.df['Closing Price'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = short_ema - long_ema
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['Returns'] = self.df['Closing Price'].pct_change()

        delta = self.df['Closing Price'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))

        # Fill NaN values generated during indicator calculations
        self.df.fillna(0, inplace=True)

    def classify_actions(self):
        mean_return = self.df['Returns'].mean()
        std_return = self.df['Returns'].std()

        loss_cutoff = mean_return - std_return
        small_gain_cutoff = mean_return
        moderate_gain_cutoff = mean_return + std_return
        significant_gain_cutoff = mean_return + 2 * std_return

        # Define a function to classify actions based on returns
        def classify(row):
            if row['Returns'] < loss_cutoff:
                return -1  # Loss
            elif row['Returns'] < small_gain_cutoff:
                return 0  # No action
            elif row['Returns'] < moderate_gain_cutoff:
                return 1  # Small gain
            elif row['Returns'] < significant_gain_cutoff:
                return 2  # Moderate gain
            else:
                return 3  # Significant gain

        self.df['Action'] = self.df.apply(classify, axis=1)

    def preprocess_data(self):
        self.calculate_technical_indicators()
        self.classify_actions()  # Ensure this is called before preprocessing

        # Drop 'Date' column before scaling
        self.df = self.df.drop(columns=['Date'])

        # Scale only numerical columns
        scaler = StandardScaler()
        numeric_cols = ['Open Price', 'Closing Price', 'Volume', 'SMA_5', 'SMA_20', 'MACD', 'Signal_Line', 'Returns', 'RSI']
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])

        return self.df

class MLPAgent:
    def __init__(self, n_features, n_actions, epsilon=0.1):
        self.n_features = n_features
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        self.experience_replay = []
        self._initialize_model()

    def _initialize_model(self):
        dummy_states = pd.DataFrame([[random.random() for _ in range(self.n_features)] for _ in range(10)])
        dummy_actions = pd.Series(random.randint(0, self.n_actions - 1) for _ in range(10))
        self.model.fit(dummy_states, dummy_actions)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            action_probabilities = self.model.predict_proba(state)
            return action_probabilities.argmax()

    def store_experience(self, state, action, reward, next_state):
        self.experience_replay.append((state, action, reward, next_state))

    def train(self, batch_size=32):
        if len(self.experience_replay) < batch_size:
            return

        batch = random.sample(self.experience_replay, batch_size)
        states = pd.DataFrame([experience[0] for experience in batch])
        actions = pd.Series([experience[1] for experience in batch])
        self.model.fit(states, actions)

    def fit(self, X_train, y_train):
        """Fit the MLP model to the training data."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        try:
            y_pred = self.model.predict(X_test)
            print(classification_report(y_test, y_pred, zero_division=1))
        except Exception as e:
            print("Error during evaluation:", e)

class Environment:
    def __init__(self, pathData):
        self.initial_portfolio_value = 100
        self.portfolio_value = self.initial_portfolio_value
        self.stock = 'Closing Price'
        self.pathData = pathData
        self.shares = 0
        self.current_step = 0

        self.stock_data = self._load_data()
        self._initialize_portfolio()

        self.action_space = spaces.Discrete(5)

    def _load_data(self):
        df = pd.read_csv(self.pathData)
        print("Loaded features:", df.columns.tolist())  # This will help verify the feature names
        return df

    def _initialize_portfolio(self):
        initial_price = self.stock_data.iloc[0][self.stock]
        self.shares = self.initial_portfolio_value / initial_price

    def trade(self, action, date):
        self.current_step = self.stock_data[self.stock_data['Date'] == date].index[0]
        current_price = self.stock_data.iloc[self.current_step][self.stock]

        if action in (2, 3):  # Buying action
            amount_to_invest = self.portfolio_value * (action - 1) / 2  # Invest half for action 2, all for action 3
            shares_to_buy = amount_to_invest / current_price
            self.shares += shares_to_buy
            self.portfolio_value -= amount_to_invest
        elif action == -1:  # Selling action
            self.portfolio_value += (self.shares * 0.5) * current_price
            self.shares *= 0.5

        self.portfolio_value += self.shares * current_price

    def state(self, date):
        self.current_step = self.stock_data[self.stock_data['Date'] == date].index[0]
        current_price = self.stock_data.iloc[self.current_step]['Closing Price']

        state_features = self.stock_data.iloc[self.current_step][[
            'Open Price', 'Closing Price', 'Volume', 'SMA_5', 'SMA_20', 'MACD', 'Signal_Line', 'Returns', 'RSI'
        ]]

        return {
            'portfolio_value': self.portfolio_value,
            'shares': self.shares,
            'current_price': current_price,
            'current_volume': self.stock_data.iloc[self.current_step]['Volume'],
            **state_features.to_dict()
        }

import matplotlib.pyplot as plt
import numpy as np

class Main:
    def __init__(self, pathData):
        self.df = pd.read_csv(pathData)

        # Initialize DataProcessor with the data
        self.processor = DataProcessor(self.df)
        processed_df = self.processor.preprocess_data()

        # Split the processed DataFrame into training and testing datasets
        self.train_df, self.test_df = self.processor.split_data()

        # Update feature count to match
        n_features = processed_df.shape[1] - 1  # Exclude 'Action' column
        n_actions = 5  # Since you have 5 possible actions

        self.agent = MLPAgent(n_features, n_actions)

        self.environment = Environment(pathData)

        # Save the processed data without 'Date' column
        self.df.to_csv(pathData, index=False)

        # Store data for plotting
        self.dates = []
        self.prices = []
        self.actions = []

    def train_agent(self):
        print("Train DataFrame before processing:", self.train_df.columns.tolist())  # Check columns before processing

        # Ensure 'Action' exists before dropping
        if 'Action' in self.train_df.columns:
            y_train = self.train_df['Action']
        else:
            raise ValueError("'Action' column not found in training DataFrame.")

        # Select numerical features and drop 'Action'
        X_train = self.train_df.drop(columns=['Action'])  # 'Date' has already been dropped
        print("Training features:", X_train.columns.tolist())  # Verify features used for training

        self.agent.fit(X_train, y_train)

    def simulate_trading(self):
        action_counts = {0: 0, -1: 0, 1: 0, 2: 0, 3: 0}  # Initialize action count dictionary

        for index, row in self.environment.stock_data.iterrows():
            state = self.environment.state(row['Date'])

            # Create state vector with consistent feature names
            state_vector = pd.DataFrame([[
                state['Open Price'],
                state['Closing Price'],
                state['Volume'],
                state['SMA_5'],
                state['SMA_20'],
                state['MACD'],
                state['Signal_Line'],
                state['Returns'],
                state['RSI'],
            ]], columns=[
                'Open Price',
                'Closing Price',
                'Volume',
                'SMA_5',
                'SMA_20',
                'MACD',
                'Signal_Line',
                'Returns',
                'RSI'
            ])

            action = self.agent.select_action(state_vector)
            self.environment.trade(action, row['Date'])

            # Store the date, current price, and action for plotting
            self.dates.append(row['Date'])
            self.prices.append(self.environment.portfolio_value)  # or state['current_price'] for closing price
            self.actions.append(action)

            print(f"Date: {row['Date']}, Action: {action}, Portfolio Value: {self.environment.portfolio_value:.2f}")

            # Update the count of the action taken
            if action in action_counts:
                action_counts[action] += 1

        # Print the action counts after simulation
        print("Action counts during simulation:", action_counts)

        # Call the plotting function after the simulation
        self.plot_results()

    def plot_results(self):
        # Convert dates to a suitable format for plotting
        x = np.arange(len(self.dates))

        # Plot the closing prices
        plt.figure(figsize=(12, 6))
        plt.plot(x, self.prices, label='Portfolio Value', color='blue', alpha=0.6)

        # Add arrows for buy/sell actions
        for i, action in enumerate(self.actions):
            if action == 2:  # Small gain (buy)
                plt.annotate('↑', xy=(x[i], self.prices[i]), xytext=(x[i], self.prices[i] + 5000),
                             arrowprops=dict(facecolor='green', shrink=0.05), fontsize=12)
            elif action == 3:  # Moderate gain (buy)
                plt.annotate('↑', xy=(x[i], self.prices[i]), xytext=(x[i], self.prices[i] + 10000),
                             arrowprops=dict(facecolor='green', shrink=0.05), fontsize=12)
            elif action == -1:  # Loss (sell)
                plt.annotate('↓', xy=(x[i], self.prices[i]), xytext=(x[i], self.prices[i] - 5000),
                             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12)

        plt.title('Trading Simulation Results')
        plt.xlabel('Days')
        plt.ylabel('Portfolio Value')
        plt.xticks(ticks=x, labels=self.dates, rotation=45)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    path_to_data = "GOOG_with_techInd.csv"
    main = Main(path_to_data)
    main.train_agent()
    main.simulate_trading()