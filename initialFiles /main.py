from environment_1_Stock import Environment
from StockDataProcessor_Calculations import DataProcessor
from MLP.MLPAgent import MLPAgent
import pandas as pd
import numpy as np


def main():
    # Load the CSV file
    df = pd.read_csv("../MLP/GOOG_2019_to_2023.csv")

    # Create the data processor and preprocess the data
    processor = DataProcessor(df)
    processed_df = processor.preprocess_data()

    # Split the data into training and test sets (optional, depending on use case)
    train_df, test_df = processor.split_data()

    # Create the environment
    env = Environment(pathData="GOOG_2019_to_2023.csv", stock='Closing Price')

    # Define the agent
    n_features = 4  # Assuming state includes price, volume, shares, and portfolio value
    n_actions = env.action_space.n
    agent = MLPAgent(n_features=n_features, n_actions=n_actions, epsilon=0.1, learning_rate=0.01, gamma=0.99)

    # Set initial state
    initial_date = "2020-01-02"  # Use a starting date from your dataset
    initial_state = env.state(initial_date)

    # Extract state variables to pass to the agent
    state = np.array([
        initial_state['current_price'],  # Current stock price
        initial_state['current_volume'],  # Volume of stock traded
        initial_state['shares'],         # Number of shares held
        initial_state['portfolio_value'] # Current portfolio value
    ])

    # Training loop (Example)
    for _ in range(100):  # Set to the number of episodes you want to run
        action = agent.select_action(state)  # Agent chooses an action

        # Execute the action in the environment
        env.trade(action, initial_date)  # Execute the action in the environment

        # Get the next state
        next_state_data = env.state(initial_date)

        # Extract the next state variables
        next_state = np.array([
            next_state_data['current_price'],  # Current stock price
            next_state_data['current_volume'],  # Volume of stock traded
            next_state_data['shares'],         # Number of shares held
            next_state_data['portfolio_value'] # Current portfolio value
        ])

        # Calculate reward (Example: change in portfolio value)
        reward = next_state_data['portfolio_value'] - initial_state['portfolio_value']

        # Store experience in memory
        agent.store_experience(state, action, reward, next_state)

        # Train the agent
        agent.train(batch_size=32)

        # Update epsilon for exploration
        agent.update_epsilon()

        # Move to the next state
        state = next_state

        # Update the initial state for reward calculation in the next iteration
        initial_state = next_state_data


if __name__ == "__main__":
    main()
