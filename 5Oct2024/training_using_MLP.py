import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

class training_using_MLP:
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = pd.read_csv(file_name)
        print(f"Data loaded from {file_name}.")

    def save_data_as_pickle(self, pickle_file_name):
        data_array = self.data.to_numpy()
        with open(pickle_file_name, 'wb') as f:
            pickle.dump(data_array, f)
        print(f"Data saved to {pickle_file_name} as a pickle.")

    def split_data(self):
        X = self.data[['Close_3', 'Close_2', 'Close_1',
                       'Volume_3', 'Volume_2', 'Volume_1']]
        y = self.data['Action']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(f"Data split into train and test sets. Train size: {len(X_train)}, Test size: {len(X_test)}.")
        return X_train, X_test, y_train, y_test

    def train_mlp(self, X_train, y_train):
        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        print("MLP model trained.")
        return mlp

    def evaluate_model(self, mlp, X_train, X_test, y_train, y_test):
        y_pred_test = mlp.predict(X_test)
        y_pred_train = mlp.predict(X_train)

        conf_matrix = confusion_matrix(y_test, y_pred_test)
        labels = [-1, 0, 1, 2]
        conf_matrix_df = pd.DataFrame(conf_matrix, index=[f'Actual {label}' for label in labels],
                                      columns=[f'Predicted {label}' for label in labels])

        print("\nConfusion Matrix (Test Set):")
        print(conf_matrix_df)

        recall_test = recall_score(y_test, y_pred_test, average='weighted')
        accuracy_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(y_test, y_pred_test, average='weighted')

        accuracy_train = accuracy_score(y_train, y_pred_train)

        print(f"\nTest Recall: {recall_test:.4f}")
        print(f"Test Accuracy: {accuracy_test:.4f}")
        print(f"Test Precision: {precision_test:.4f}")
        print(f"Training Accuracy: {accuracy_train:.4f}")

        if accuracy_train > accuracy_test:
            print("Potential overfitting detected: High training accuracy but lower test accuracy.")
        elif accuracy_train < accuracy_test:
            print("Potential underfitting detected: Low training accuracy compared to test accuracy.")
        else:
            print("Training and test accuracies are similar. Model seems well-balanced.")

    def run(self, pickle_file_name):
        self.save_data_as_pickle(pickle_file_name)
        X_train, X_test, y_train, y_test = self.split_data()
        mlp = self.train_mlp(X_train, y_train)
        self.evaluate_model(mlp, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    mlp_training = training_using_MLP("processedWithActions/AAPL_3day_action.csv")
    mlp_training.run("processedWithActions/AAPL_3day_action.pkl")
