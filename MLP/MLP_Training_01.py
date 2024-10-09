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

        # Fix: Fill NaN values generated during indicator calculations
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
        scaler = StandardScaler()
        # Fix: Ensure only numerical columns are scaled
        numeric_cols = ['Open Price', 'Closing Price', 'Volume']
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        self.calculate_technical_indicators()
        self.classify_actions()
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

        # Save feature names after training
        self.feature_names = X_train.columns.tolist()  # Assuming X_train is a DataFrame

    def evaluate(self, X_test, y_test):
        try:
            y_pred = self.model.predict(X_test)
            print(classification_report(y_test, y_pred))
        except NotFittedError as e:
            print("Model is not fitted yet. Error:", e)


class Environment:
    def __init__(self, pathData):
        # Initialize variables for single stock
        self.initial_portfolio_value = 100000
        self.portfolio_value = self.initial_portfolio_value  # Portfolio value
        self.stock = 'Closing Price'  # Use the stock column passed as an argument
        self.volume = 'Volume'  # Assuming volume data column is named 'Volume'
        self.pathData = pathData
        self.shares = 0  # Initialize number of shares held
        self.current_step = 0

        # Load and process the stock data
        self.stock_data = self._load_data()
        self._initialize_portfolio()

        # Defining discrete action space: [-1 (Sell), 0 (Neutral), 1 (Hold), 2 (Buy), 3 (Strong Buy)]
        self.action_space = spaces.Discrete(5)

    def _load_data(self):
        """Load and prepare data from CSV file."""
        df = pd.read_csv(self.pathData)
        df = df[['Date', self.stock, self.volume]]
        return df

    def _initialize_portfolio(self):
        """Initialize the portfolio with initial shares."""
        initial_price = self.stock_data.iloc[0][self.stock]
        self.shares = self.initial_portfolio_value / initial_price  # Invest all initial value into stock

    def trade(self, action, date):
        """Execute the trading action on the given date."""
        self.current_step = self.stock_data[self.stock_data['Date'] == date].index[0]
        current_price = self.stock_data.iloc[self.current_step][self.stock]

        # Implement buy/sell logic
        if action == 3 or action == 2:  # Buy or Strong Buy
            amount_to_invest = self.portfolio_value * (action - 1)  # Determine investment amount based on action
            shares_to_buy = amount_to_invest / current_price
            self.shares += shares_to_buy
            self.portfolio_value -= amount_to_invest  # Deduct the investment from portfolio
        elif action == -1:  # Sell half of the current holdings
            self.portfolio_value += (self.shares * 0.5) * current_price  # Sell half of the shares
            self.shares *= 0.5  # Keep half of the shares

        # Update portfolio value
        self.portfolio_value += self.shares * current_price

    def state(self, date):
        """Get the current state of the environment on the given date."""
        self.current_step = self.stock_data[self.stock_data['Date'] == date].index[0]
        current_price = self.stock_data.iloc[self.current_step]['Closing Price']  # Assuming you want Closing Price

        # Print column names for debugging
        print("Available columns:", self.stock_data.columns)

        # Attempt to access features
        try:
            state_features = self.stock_data.iloc[self.current_step][['Open Price', 'Closing Price', 'Volume',
                                                                      'SMA_5', 'SMA_20', 'MACD',
                                                                      'Signal_Line', 'Returns', 'RSI']]
        except KeyError as e:
            print(f"KeyError: {e}. Check the column names in the DataFrame.")
            raise  # Re-raise the error after logging it

        return {
            'portfolio_value': self.portfolio_value,
            'shares': self.shares,
            'current_price': current_price,
            'current_volume': self.stock_data.iloc[self.current_step]['Volume'],  # Add current volume
            **state_features.to_dict()  # Unpack the state features
        }


class Main:
    def __init__(self, pathData):
        # Load the data directly, including the Date column
        self.df = pd.read_csv(pathData)

        # Drop the 'Date' column right after loading it
        self.df = self.df.drop(columns=['Date'])

        # Process the data
        self.environment = Environment(pathData)  # If Environment still needs Date, keep it here
        self.processor = DataProcessor(self.df)
        processed_df = self.processor.preprocess_data()

        # Split data after preprocessing
        self.train_df, self.test_df = self.processor.split_data()

        # Update number of features based on processed data (excluding 'Action')
        n_features = processed_df.shape[1] - 1  # Exclude the Action column
        n_actions = len(np.unique(processed_df['Action']))

        # Initialize MLPAgent
        self.agent = MLPAgent(n_features, n_actions)

        # Save processed data (if required)
        self.save_processed_data(processed_df)

        # Print the last few rows of processed data for debugging
        print(processed_df.tail())

    def train_agent(self, num_epochs=10):
        # Prepare training data (features and target)
        X_train = self.train_df.drop(columns=['Action']).select_dtypes(include=[np.number])
        y_train = self.train_df['Action'].values

        # Fit the scaler to the training data
        self.agent.fit_scaler(X_train)

        # Training loop
        for epoch in range(num_epochs):
            self.agent.train(batch_size=32)
            print(f'Epoch {epoch + 1}/{num_epochs} completed.')

    def evaluate_agent(self):
        """Evaluate the trained MLP agent using the test data."""
        # Prepare test data by selecting only numeric columns (excluding 'Action')
        X_test = self.test_df.drop(columns=['Action']).select_dtypes(include=[np.number])
        y_test = self.test_df['Action'].values  # Actions

        # Scale the test data using the pre-fitted scaler in MLPAgent
        X_test_scaled = self.agent.scaler.transform(X_test)

        # Ensure the number of features match the model's expected input
        print(f'Number of features in X_test_scaled: {X_test_scaled.shape[1]}')
        print(f'Number of features expected by model: {self.agent.model.n_features_in_}')

        # Check for mismatch in number of features
        if X_test_scaled.shape[1] != self.agent.model.n_features_in_:
            print(
                f"Feature mismatch detected. Expected {self.agent.model.n_features_in_} features, but got {X_test_scaled.shape[1]}.")
            return  # Stop evaluation due to feature mismatch

        # Perform evaluation
        y_pred = self.agent.model.predict(X_test_scaled)
        print(classification_report(y_test, y_pred))

    def save_processed_data(self, processed_df):
        processed_df.to_csv('processed_data_with_actions.csv', index=False)
        print("Processed data saved to 'processed_data_with_actions.csv'.")

    def simulate_trading(self):
        for index, row in self.environment.stock_data.iterrows():
            state = self.environment.state(row['Date'])

            # Include all necessary features in the state vector
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
                state['portfolio_value'],  # Optional, can be included if needed
                state['shares'],  # Optional, can be included if needed
                state['current_price'],  # Optional, can be included if needed
                state['current_volume'],  # Optional, can be included if needed
            ])

            action = self.agent.select_action(state_vector)
            self.environment.trade(action, row['Date'])
            print(f"Date: {row['Date']}, Action: {action}, Portfolio Value: {self.environment.portfolio_value:.2f}")


if __name__ == "__main__":
    path_to_data = "GOOG_2019_to_2023.csv"

    main = Main(path_to_data)
    main.train_agent(num_epochs=10)
    main.evaluate_agent()
    main.simulate_trading()
    main.processor.plot_histogram()  # Optional if the method exists in DataProcessor
