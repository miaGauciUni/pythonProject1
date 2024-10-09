from sklearn.neural_network import MLPRegressor
import numpy as np
import random

"""
Implementation of the machine learning model 
"""

class MLPAgent:
    def __init__(self, n_features, n_actions, epsilon=0.1):
        self.n_features = n_features
        self.n_actions = n_actions
        self.epsilon = epsilon  # Exploration rate

        # Initialize model for predicting actions (MLP Classifier)
        self.model = MLPClassifier(hidden_layer_sizes=(64, 32),
                                   max_iter=1000,
                                   random_state=42)

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

    def evaluate(self, X_test, y_test):
        """Evaluate the model on the test data and print the accuracy and classification report."""
        # Make predictions on the test data
        y_pred = self.model.predict(X_test)

        # Calculate the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

        # Generate a classification report
        class_report = classification_report(y_test, y_pred)
        print("Classification Report:\n", class_report)

    def update_epsilon(self):
        """Decay epsilon for exploration over time."""
        self.epsilon = max(0.01, self.epsilon * 0.995)

