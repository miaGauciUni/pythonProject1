import yfinance as yf
import pandas as pd

def download_and_save_stock_data():
    # Define the list of ticker symbols and time period
    tickers = ['AAPL', 'TSLA', 'MSFT', 'META', 'GOOG']
    start_date = '2019-01-01'
    end_date = '2023-01-01'  # End date is non-inclusive, so using start of 2023

    for ticker in tickers:
        # Download historical data including 'Open', 'Close' price, and 'Volume'
        df = yf.download(ticker, start=start_date, end=end_date, interval='1d')

        if df.empty:
            print(f"No data found for {ticker}.")
            continue  # Skip this ticker if no data is found

        # Keep only the 'Date', 'Open', 'Close', and 'Volume' columns
        df = df[['Open', 'Close', 'Volume']]

        # Reset index to move 'Date' from index to a column
        df.reset_index(inplace=True)

        # Rename columns to be more explicit
        df.columns = ['Date', 'Open', 'Closing', 'Volume']

        # Define the CSV file name
        csv_filename = f"{ticker}_2019_to_2023.csv"

        # Save the DataFrame to a CSV file
        df.to_csv(csv_filename, index=False)

        print(f"CSV file saved for {ticker} as {csv_filename}")


def load_csvs_to_list(tickers):
    data_frames = []

    for ticker in tickers:
        # Define the CSV file name based on the ticker symbol
        csv_filename = f"{ticker}_2019_to_2023.csv"

        try:
            # Load the CSV file into a DataFrame
            df = pd.read_csv(csv_filename)

            # Append the DataFrame to the list
            data_frames.append(df)

            print(f"Loaded {csv_filename} into the list.")
        except FileNotFoundError:
            print(f"CSV file for {ticker} not found!")

    return data_frames


if __name__ == '__main__':
    #download_and_save_stock_data()

    # Define the list of ticker symbols for which CSV files have been saved
    tickers = ['AAPL', 'TSLA', 'MSFT', 'META', 'GOOG']

    # Load all the CSV files into a list of DataFrames
    stock_data_list = load_csvs_to_list(tickers)

    # Now stock_data_list contains all the data in list form
    # Example: Access the DataFrame for AAPL (first element in the list)
    aapl_df = stock_data_list[0]
    print(aapl_df.head())  # Print first few rows of AAPL data
