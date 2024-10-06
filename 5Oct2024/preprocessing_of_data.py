import pandas as pd
import matplotlib.pyplot as plt


class PreprocessingOfData:
    def __init__(self, file_name):
        """
        Initializes the class with the CSV file name and loads the data into a DataFrame.
        """
        self.file_name = file_name
        self.df = pd.read_csv(file_name)
        print(f"Data loaded from {file_name}.")

    def clean_data(self):
        """
        Cleans the data by removing rows with any NaN values.
        """
        initial_rows = self.df.shape[0]

        # Drop rows with NaN values
        self.df.dropna(inplace=True)

        # Report how many rows were removed
        removed_rows = initial_rows - self.df.shape[0]
        if removed_rows > 0:
            print(f"Removed {removed_rows} rows containing NaN values.")
        else:
            print("No NaN values found in the dataset.")

    def save_clean_data(self, output_file_name="cleanData/GOOG_clean.csv"):
        """
        Saves the cleaned data to a new CSV file, defaulting to 'meta_clean.csv'.
        """
        self.df.to_csv(output_file_name, index=False)
        print(f"Cleaned data saved as {output_file_name}.")

    def plot_stock_data(self):
        """
        Plots a solid line of the stock's 'Closing' price against 'Date'.
        """
        # Ensure the 'Date' column is in datetime format
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Line plot of the 'Closing' price against 'Date'
        plt.figure(figsize=(10, 6))
        plt.plot(self.df['Date'], self.df['Closing'], label='Closing Price', color='blue')

        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.title('GOOGLE Stock Price Over Time (Line Plot)')
        plt.legend()

        # Show the plot
        plt.grid(True)
        plt.show()

        # I tried with z-score and IQR and went through the data myself

    def detect_outliers_modified_zscore(self):
        """
        Detects outliers in the 'Closing' price column using the Modified Z-score method.
        """
        median = self.df['Closing'].median()
        mad = (self.df['Closing'] - median).abs().median()

        # Calculate Modified Z-scores
        modified_z_scores = 0.6745 * (self.df['Closing'] - median) / mad

        # Define a threshold for identifying outliers
        threshold = 3.5  # Commonly used threshold
        outliers = self.df[modified_z_scores > threshold]

        print(f"Found {outliers.shape[0]} outliers using Modified Z-score method.")
        return outliers


    def plot_outliers(self):
        """
        Plots a solid line of the stock's 'Closing' price with outliers highlighted as red dots.
        """
        # Ensure the 'Date' column is in datetime format
        self.df['Date'] = pd.to_datetime(self.df['Date'])

        # Detect outliers using the z-score method
        outliers = self.detect_outliers_modified_zscore()

        # Line plot of the 'Closing' price against 'Date'
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['Date'], self.df['Closing'], label='Closing Price', color='blue', s=5)

        # Highlight the outliers as red dots
        plt.scatter(outliers['Date'], outliers['Closing'], label='Outliers', color='red', s=40)

        # Add labels and title
        plt.xlabel('Date')
        plt.ylabel('Closing Price (USD)')
        plt.title('GOOGLE Stock Price with Outliers Highlighted (Line Plot)')
        plt.legend()

        # Show the plot
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == '__main__':
    # Initialize the preprocessing class with the raw Apple CSV file
    preprocessing = PreprocessingOfData("rawData/GOOG_raw.csv")

    # Clean the data by removing rows with NaN values
    preprocessing.clean_data()

    # Save the cleaned data to 'meta_clean.csv'
    preprocessing.save_clean_data()  # Default file name is 'meta_clean.csv'

    # Plot the stock data (line plot)
    preprocessing.plot_stock_data()

    # Plot the stock data with outliers highlighted
    preprocessing.plot_outliers()
