import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
from scipy.stats import norm
import random
from gymnasium import spaces

from sklearn.exceptions import NotFittedError

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

        cutoffs = {
            "loss_cutoff": mean_return - std_return,
            "small_gain_cutoff": mean_return,
            "moderate_gain_cutoff": mean_return + std_return,
            "significant_gain_cutoff": mean_return + 2 * std_return,
        }

        conditions = [
            (self.df['Returns'] < cutoffs['loss_cutoff']),
            (self.df['Returns'] >= cutoffs['loss_cutoff']) & (self.df['Returns'] < cutoffs['small_gain_cutoff']),
            (self.df['Returns'] >= cutoffs['small_gain_cutoff']) & (self.df['Returns'] < cutoffs['moderate_gain_cutoff']),
            (self.df['Returns'] >= cutoffs['moderate_gain_cutoff']) & (self.df['Returns'] < cutoffs['significant_gain_cutoff']),
            (self.df['Returns'] >= cutoffs['significant_gain_cutoff']),
        ]
        choices = [-1, 0, 1, 2, 3]
        self.df['Action'] = np.select(conditions, choices, default=1)

    def preprocess_data(self):
        self.calculate_technical_indicators()
        self.classify_actions()  # Ensure this is called before preprocessing

        # Drop 'Date' column before scaling
        self.df = self.df.drop(columns=['Date'])

        # Drop 'Action' column if it exists for scaling
        # We will keep 'Action' for the model training later
        if 'Action' in self.df.columns:
            # We keep 'Action' for training
            action_col = self.df['Action']
        else:
            raise ValueError("'Action' column not found for preprocessing.")

        scaler = StandardScaler()
        # Scale only numerical columns
        numeric_cols = ['Open Price', 'Closing Price', 'Volume', 'SMA_5', 'SMA_20', 'MACD', 'Signal_Line', 'Returns', 'RSI']
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])

        # Add 'Action' column back to the DataFrame
        self.df['Action'] = action_col

        return self.df

class MLPAgent:
    def __init__(self, n_features, n_actions, epsilon=0.1):
        self.n_features = n_features
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
        self.scaler = StandardScaler()
        self.experience_replay = []
        self._initialize_model()

    def _initialize_model(self):
        dummy_states = np.random.rand(10, self.n_features)
        dummy_actions = np.random.randint(0, self.n_actions, size=(10,))
        self.model.fit(dummy_states, dummy_actions)

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            action_probabilities = self.model.predict_proba(state.reshape(1, -1))
            return np.argmax(action_probabilities)

    def store_experience(self, state, action, reward, next_state):
        self.experience_replay.append((state, action, reward, next_state))

    def train(self, batch_size=32):
        if len(self.experience_replay) < batch_size:
            return

        batch = random.sample(self.experience_replay, batch_size)
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])
        self.model.fit(states, actions)

    def fit_scaler(self, X):
        self.scaler.fit(X)

    def fit(self, X_train, y_train):
        """Fit the MLP model to the training data."""
        self.model.fit(X_train, y_train)
        self.feature_names = X_train.columns.tolist()  # Store feature names

    def evaluate(self, X_test, y_test):
        try:
            y_pred = self.model.predict(X_test)
            print(classification_report(y_test, y_pred, zero_division=1))
        except NotFittedError as e:
            print("Model is not fitted yet. Error:", e)


class Environment:
    def __init__(self, pathData):
        self.initial_portfolio_value = 100000
        self.portfolio_value = self.initial_portfolio_value
        self.stock = 'Closing Price'
        self.volume = 'Volume'
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

        if action == 3 or action == 2:
            amount_to_invest = self.portfolio_value * (action - 1)
            shares_to_buy = amount_to_invest / current_price
            self.shares += shares_to_buy
            self.portfolio_value -= amount_to_invest
        elif action == -1:
            self.portfolio_value += (self.shares * 0.5) * current_price
            self.shares *= 0.5

        self.portfolio_value += self.shares * current_price

    def state(self, date):
        self.current_step = self.stock_data[self.stock_data['Date'] == date].index[0]
        current_price = self.stock_data.iloc[self.current_step]['Closing Price']

        state_features = self.stock_data.iloc[self.current_step][[
            'Open Price','Closing Price', 'Volume', 'SMA_5', 'SMA_20', 'MACD', 'Signal_Line', 'Returns', 'RSI'
        ]]

        return {
            'portfolio_value': self.portfolio_value,
            'shares': self.shares,
            'current_price': current_price,
            'current_volume': self.stock_data.iloc[self.current_step]['Volume'],
            **state_features.to_dict()
        }


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

    def train_agent(self, num_epochs=10):
        print("Train DataFrame before processing:", self.train_df.columns.tolist())  # Check columns before processing

        # Ensure 'Action' exists before dropping
        if 'Action' in self.train_df.columns:
            y_train = self.train_df['Action'].values
        else:
            raise ValueError("'Action' column not found in training DataFrame.")

        # Select numerical features and drop 'Action'
        X_train = self.train_df.drop(columns=['Action'])  # 'Date' has already been dropped
        print("Training features:", X_train.columns.tolist())  # Print the training feature names
        print("Training data shape:", X_train.shape)  # Check the shape of the training data

        self.agent.fit_scaler(X_train)
        self.agent.fit(X_train, y_train)

    def evaluate_agent(self):
        print("Test DataFrame before processing:", self.test_df.columns.tolist())  # Check columns before processing

        if 'Action' in self.test_df.columns:
            y_test = self.test_df['Action'].values
        else:
            raise ValueError("'Action' column not found in test DataFrame.")

        X_test = self.test_df.drop(columns=['Action', 'Date'], errors='ignore')
        print("Testing features:", X_test.columns.tolist())
        print("Testing data shape:", X_test.shape)

        X_test_scaled = self.agent.scaler.transform(X_test)

        # Create a DataFrame with the right feature names
        X_test_scaled_df = pd.DataFrame(X_test_scaled,
                                        columns=X_test.columns.tolist())  # Make sure to retain column names

        # Check for feature mismatch
        if X_test_scaled_df.shape[1] != self.agent.model.n_features_in_:
            print(
                f"Feature mismatch detected. Expected {self.agent.model.n_features_in_} features, but got {X_test_scaled_df.shape[1]}.")
            return

        self.agent.evaluate(X_test_scaled_df, y_test)

    def simulate_trading(self):
        for index, row in self.environment.stock_data.iterrows():
            state = self.environment.state(row['Date'])
            state_vector = np.array([
                state['Open Price'],
                state['Closing Price'],
                state['Volume'],
                state['SMA_5'],
                state['SMA_20'],
                state['MACD'],
                state['Signal_Line'],
                state['Returns'],
                state['RSI'],
            ])

            action = self.agent.select_action(state_vector)
            self.environment.trade(action, row['Date'])
            print(f"Date: {row['Date']}, Action: {action}, Portfolio Value: {self.environment.portfolio_value:.2f}")


if __name__ == "__main__":
    path_to_data = "GOOG_with_techInd.csv"
    main = Main(path_to_data)
    main.train_agent(num_epochs=10)
    main.evaluate_agent()
    main.simulate_trading()