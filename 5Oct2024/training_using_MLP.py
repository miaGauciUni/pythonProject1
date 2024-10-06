import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.model_selection import GridSearchCV


class training_using_MLP:
    def __init__(self, file_name):
        """
        Initialize the class by loading the data from the given CSV file.
        """
        self.file_name = file_name
        self.data = pd.read_csv(file_name)
        print(f"Data loaded from {file_name}.")

    def save_data_as_pickle(self, pickle_file_name):
        """
        Save the data as a NumPy array in a pickle file.
        """
        data_array = self.data.to_numpy()
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(data_array, f)
        print(f"Data saved to {pickle_file_name} as a pickle.")

    def split_data(self):
        """
        Split the data into train and test sets (70%/30%).
        The features are the closing prices and volumes, the target is the 'Action' column.
        """
        X = self.data[['Close_5', 'Close_4', 'Close_3', 'Close_2', 'Close_1',
                       'Volume_5', 'Volume_4', 'Volume_3', 'Volume_2', 'Volume_1']]
        y = self.data['Action']

        # Splitting the data into 70% train and 30% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(f"Data split into train and test sets. Train size: {len(X_train)}, Test size: {len(X_test)}.")

        return X_train, X_test, y_train, y_test

    def train_mlp(self, X_train, y_train):
        """
        Train the MLP model using the training data.
        """
        # Define the MLPClassifier model
        #activation='tanh',

        mlp = MLPClassifier(hidden_layer_sizes=(100,),  max_iter=500, random_state=42)

        # Train the MLP model'
        mlp.fit(X_train, y_train)
        print("MLP model trained.")

        return mlp

    def evaluate_model(self, mlp, X_test, y_test):
        """
        Evaluate the model on the test data.
        Predict the actions and compare them to the true labels.
        Output the confusion matrix, recall, accuracy, and precision.
        """
        # Predict the actions for the test data
        y_pred = mlp.predict(X_test)

        # Calculate the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Convert confusion matrix to a pandas DataFrame for better visualization
        labels = [-1, 0, 1, 2]
        conf_matrix_df = pd.DataFrame(conf_matrix, index=[f'Actual {label}' for label in labels],
                                      columns=[f'Predicted {label}' for label in labels])

        # Print confusion matrix with labeled rows and columns
        print("\nConfusion Matrix:")
        print(conf_matrix_df)

        # Calculate recall, accuracy, and precision
        recall = recall_score(y_test, y_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')

        # Print evaluation metrics
        print(f"\nRecall: {recall:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")

    def run(self, pickle_file_name):
        """
        Full workflow: Load data, save as pickle, split, train MLP, and evaluate.
        """
        # Save the data as pickle
        self.save_data_as_pickle(pickle_file_name)

        # Split the data
        X_train, X_test, y_train, y_test = self.split_data()

        # Train the MLP model
        mlp = self.train_mlp(X_train, y_train)

        # Evaluate the model
        self.evaluate_model(mlp, X_test, y_test)


if __name__ == '__main__':
    # Initialize the training class with the CSV file
    mlp_training = training_using_MLP("processedWithActions/AAPL_5day_action.csv")

    # Run the full process and save the data to a pickle file
    mlp_training.run("processedWithActions/AAPL_5day_action.pkl")
