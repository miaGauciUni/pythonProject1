import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

class TrainingUsingMLP:
    def __init__(self, file_name):
        """
        Initializes the class with the CSV file name and loads the data into a DataFrame.
        """
        self.file_name = file_name
        self.data = pd.read_csv(file_name)
        print(f"Data loaded from {file_name}.")

    def save_data_as_pickle(self, pickle_file_name):
        """
        Saves the dataset as a pickle file for later use.
        """
        data_array = self.data.to_numpy()
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(data_array, f)
        print(f"Data saved to {pickle_file_name} as a pickle.")

    def split_data(self):
        """
        Splits the data into features (X) and target (y) and then splits into training and test sets.
        X now includes technical indicators as well as the past 5 days' closing prices.
        """
        # Features with new column names from your .csv
        X = self.data[['Closing_5', 'Closing_4', 'Closing_3', 'Closing_2', 'Closing_1',
            'SMA_5_5', 'SMA_5_4', 'SMA_5_3', 'SMA_5_2', 'SMA_5_1',
            'EMA_5_5', 'EMA_5_4', 'EMA_5_3', 'EMA_5_2', 'EMA_5_1',
            'RSI_5', 'RSI_4', 'RSI_3', 'RSI_2', 'RSI_1',
            'MACD_5', 'MACD_4', 'MACD_3', 'MACD_2', 'MACD_1',
            'Signal_Line_5', 'Signal_Line_4', 'Signal_Line_3', 'Signal_Line_2', 'Signal_Line_1',
            'BB_upper_5', 'BB_upper_4', 'BB_upper_3', 'BB_upper_2', 'BB_upper_1',
            'BB_lower_5', 'BB_lower_4', 'BB_lower_3', 'BB_lower_2', 'BB_lower_1']]

        y = self.data['Action']  # Target variable remains the action labels

        # Splitting the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        print(f"Data split into train and test sets. Train size: {len(X_train)}, Test size: {len(X_test)}.")
        return X_train, X_test, y_train, y_test

    def train_mlp(self, X_train, y_train):
        """
        Trains an MLP classifier using the training data.
        """
        #MSFT: (200, 200, 200, 200, 200,), max_iter=700, random_state=42, alpha=0.00001, learning_rate_init=0.0001, activation='tanh'
        mlp = MLPClassifier(hidden_layer_sizes=(200, 200, 200, ), max_iter=500, random_state=42, alpha=0.0001, learning_rate_init=0.0001, activation='logistic')
        mlp.fit(X_train, y_train)
        print("MLP model trained.")
        return mlp

    def evaluate_model(self, mlp, X_train, X_test, y_train, y_test):
        """
        Evaluates the MLP model on both the training and test sets.
        """
        y_pred_test = mlp.predict(X_test)
        y_pred_train = mlp.predict(X_train)

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_test)
        labels = [-1, 0, 1, 2]  # Corresponding to loss, hold, buy, strong buy
        conf_matrix_df = pd.DataFrame(conf_matrix, index=[f'Actual {label}' for label in labels],
                                      columns=[f'Predicted {label}' for label in labels])

        print("\nConfusion Matrix for MSFT: (200, 200, 200, 200, 200,), max_iter=700, random_state=42, alpha=0.00001, learning_rate_init=0.0001, activation='tanh'")
        print(conf_matrix_df)

        # Metrics
        recall_test = recall_score(y_test, y_pred_test, average='weighted')
        accuracy_test = accuracy_score(y_test, y_pred_test)
        from sklearn.metrics import precision_score
        precision_test = precision_score(y_test, y_pred_test, average='weighted', zero_division=1)

        accuracy_train = accuracy_score(y_train, y_pred_train)

        print(f"\nTest Recall: {recall_test:.4f}")
        print(f"Test Accuracy: {accuracy_test:.4f}")
        print(f"Test Precision: {precision_test:.4f}")
        print(f"Training Accuracy: {accuracy_train:.4f}")

        # Overfitting or underfitting warning
        if accuracy_train > accuracy_test:
            print("Potential overfitting detected: High training accuracy but lower test accuracy.")
        elif accuracy_train < accuracy_test:
            print("Potential underfitting detected: Low training accuracy compared to test accuracy.")
        else:
            print("Training and test accuracies are similar. Model seems well-balanced.")

    def run(self, pickle_file_name):
        """
        Executes the full pipeline: save data, split data, train the model, and evaluate performance.
        """
        self.save_data_as_pickle(pickle_file_name)
        X_train, X_test, y_train, y_test = self.split_data()
        mlp = self.train_mlp(X_train, y_train)
        self.evaluate_model(mlp, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    # Load the dataset with technical indicators and run the training pipeline
    mlp_training = TrainingUsingMLP("processedWithActions/MSFT_5day_tech_action.csv")
    mlp_training.run("processedWithActions/MSFT_5day_tech_action.pkl")
