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

# Class responsible for loading, cleaning, transforming, and preparing stock data for analysis or modeling
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

    def classify_actions(self):
        """Classify actions based on returns using defined thresholds."""
        # Calculate the mean and standard deviation of returns
        mean_return = self.df['Returns'].mean()
        std_return = self.df['Returns'].std()

        # Define cutoff points based on the mean and standard deviations
        cutoffs = {
            "loss_cutoff": mean_return - std_return,  # -1 SD
            "small_gain_cutoff": mean_return,  # 0 SD
            "moderate_gain_cutoff": mean_return + std_return,  # +1 SD
            "significant_gain_cutoff": mean_return + 2 * std_return,  # +2 SD
        }

        # Define the conditions for classification
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

    def preprocess_data(self):
        """Preprocess the data by scaling, calculating indicators, and classifying actions."""
        scaler = StandardScaler()
        self.df[['Open Price', 'Closing Price', 'Volume']] = scaler.fit_transform(
            self.df[['Open Price', 'Closing Price', 'Volume']])

        self.calculate_technical_indicators()
        self.classify_actions()

        return self.df

    def set_random_seed(self, seed_value):
        """Set random seed for reproducibility."""
        np.random.seed(seed_value)


# MLPAgent Class for training and predicting actions using MLPClassifier
class MLPAgent:
    def __init__(self, n_features, n_actions, epsilon=0.1):
        self.n_features = n_features
        self.n_actions = n_actions
        self.epsilon = epsilon  # Exploration rate

        # Initialize model for predicting actions (MLP Classifier)
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32),
                                   max_iter=1000,
                                   random_state=42)

        # Initialize scaler
        self.scaler = StandardScaler()  # Add this line

        # Experience replay for training
        self.experience_replay = []  # Buffer for storing experiences

        # To fit the model initially, we need a dummy fit with random data
        self._initialize_model()



    def _initialize_model(self):
        # Initial fit with dummy data to set up the model structure
        dummy_states = np.random.rand(10, self.n_features)
        dummy_actions = np.random.randint(0, self.n_actions, size=(10,))
        self.model.fit(dummy_states, dummy_actions)

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.n_actions - 1)  # Explore
        else:
            action_probabilities = self.model.predict_proba(state.reshape(1, -1))
            return np.argmax(action_probabilities)  # Exploit (best action)

    def store_experience(self, state, action, reward, next_state):
        """Store experience in memory buffer."""
        self.experience_replay.append((state, action, reward, next_state))

    def train(self, batch_size=32):
        """Train the model using experience replay."""
        if len(self.experience_replay) < batch_size:
            return  # Not enough samples to train yet

        # Randomly sample a batch from experience replay
        batch = random.sample(self.experience_replay, batch_size)

        # Prepare training data
        states = np.array([experience[0] for experience in batch])
        actions = np.array([experience[1] for experience in batch])

        # Train the model
        self.model.fit(states, actions)

    def fit_scaler(self, X):
        """Fit the scaler to the training data."""
        self.scaler.fit(X)
        self.feature_names = X.columns.tolist()  # Store the feature names

    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data."""
        # Convert X_test back to a DataFrame to access column names
        # Here, X_test is the original DataFrame before scaling, so we get the column names from it
        feature_columns = X_test.columns.tolist()

        # Convert X_test_scaled back to a DataFrame
        X_test_scaled_df = pd.DataFrame(X_test, columns=feature_columns)

        print("X_test shape after scaling:", X_test_scaled_df.shape)
        print("X_test columns:", X_test_scaled_df.columns)  # Now this will work

        # Drop any unnecessary columns if needed, e.g., 'dummy'
        if 'dummy' in X_test_scaled_df.columns:
            X_test_scaled_df = X_test_scaled_df.drop(columns=['dummy'])

        # Make predictions using the model
        y_pred = self.model.predict(X_test_scaled_df)  # Use the DataFrame for prediction

        # Print classification report
        print(classification_report(y_test, y_pred))

    def update_epsilon(self):
        """Decay epsilon for exploration over time."""
        self.epsilon = max(0.01, self.epsilon * 0.995)


# Environment Class for stock trading simulation
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
        current_price = self.stock_data.iloc[self.current_step][self.stock]
        current_volume = self.stock_data.iloc[self.current_step][self.volume]
        return {
            'portfolio_value': self.portfolio_value,
            'shares': self.shares,
            'current_price': current_price,
            'current_volume': current_volume
        }


class Main:
    def __init__(self, pathData):
        # Load the CSV file
        self.df = pd.read_csv(pathData)

        self.environment = Environment(pathData)

        # Create a DataProcessor instance
        self.processor = DataProcessor(self.df)

        # Preprocess the data (calculate indicators and classify actions)
        processed_df = self.processor.preprocess_data()

        # Split the data into training and test sets
        self.train_df, self.test_df = self.processor.split_data()

        # Initialize MLPAgent
        n_features = processed_df.shape[1] - 1  # Number of features (excluding 'Action')
        n_actions = len(np.unique(processed_df['Action']))  # Number of unique actions/classes
        self.agent = MLPAgent(n_features, n_actions)

        # Save processed data with actions to a new CSV file
        self.save_processed_data(processed_df)

        # Print out the last few rows of the processed DataFrame
        print(processed_df.tail())

    def train_agent(self, num_epochs=10):
        """Train the MLP agent using the training data."""
        # Convert date column to numeric if needed
        if 'Date' in self.train_df.columns:
            self.train_df.loc[:, 'Date'] = pd.to_datetime(self.train_df['Date'])  # Convert to datetime
            self.train_df.loc[:, 'Date'] = self.train_df['Date'].apply(lambda x: x.timestamp())  # Convert to numeric

        # Prepare training data
        X_train = self.train_df.drop(columns=['Action']).select_dtypes(include=[np.number])  # Keep only numeric columns
        y_train = self.train_df['Action'].values  # Actions

        # Fit the scaler in the agent
        self.agent.fit_scaler(X_train)  # Fit the scaler with training data

        for epoch in range(num_epochs):
            self.agent.train(batch_size=32)
            print(f'Epoch {epoch + 1}/{num_epochs} completed.')

    def evaluate_agent(self):
        """Evaluate the trained MLP agent using the test data."""
        # Ensure 'Date' column is not used as a feature
        if 'Date' in self.test_df.columns:
            self.test_df = self.test_df.drop(columns=['Date'])  # Drop 'Date' column entirely

        # Prepare test data by selecting only numeric columns (excluding 'Action')
        X_test = self.test_df.drop(columns=['Action']).select_dtypes(include=[np.number])
        y_test = self.test_df['Action'].values  # Actions

        print(f'X_test shape before scaling: {X_test.shape}')  # Debugging line

        # Scale the test data using the pre-fitted scaler in MLPAgent
        X_test_scaled = self.agent.scaler.transform(X_test)  # Use only the feature columns for scaling

        # Perform evaluation (passing X_test for column names)
        self.agent.evaluate(X_test, y_test)

    def save_processed_data(self, processed_df):
        """Save the processed data along with actions to a new CSV file."""
        processed_df.to_csv('processed_data_with_actions.csv', index=False)
        print("Processed data saved to 'processed_data_with_actions.csv'.")

    def plot_histogram(self):
        """Plot the histogram of returns."""
        returns = self.processor.df['Returns'].dropna()
        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=50, alpha=0.75, color='blue', edgecolor='black', density=True)
        plt.title('Histogram of Stock Returns')
        plt.xlabel('Returns')
        plt.ylabel('Density')
        plt.grid(True)
        plt.show()

    def simulate_trading(self):
        """Simulate trading using the environment and the trained agent."""
        for index, row in self.environment.stock_data.iterrows():
            state = self.environment.state(row['Date'])
            state_vector = np.array([state['portfolio_value'], state['shares'], state['current_price'], state['current_volume']])
            action = self.agent.select_action(state_vector)

            # Perform trading action
            self.environment.trade(action, row['Date'])
            print(f"Date: {row['Date']}, Action: {action}, Portfolio Value: {self.environment.portfolio_value:.2f}")


if __name__ == "__main__":
    path_to_data = "MLP/GOOG_2019_to_2023.csv"

    main = Main(path_to_data)

    # Train the MLP agent
    main.train_agent(num_epochs=10)

    # Evaluate the MLP agent
    main.evaluate_agent()

    # Simulate trading
    main.simulate_trading()

    # Plot histogram of returns
    main.plot_histogram()
