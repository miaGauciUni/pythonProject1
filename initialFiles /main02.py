from environmentClass01 import Environment
from tabulate import tabulate

# Initialize the environment with the path to your CSV and stock symbols
path_to_csv = 'stock_closing_prices_2018_2020.csv'  # Ensure the CSV file is in the same directory
stock_symbols = ['AAPL', 'TSLA', 'MSFT', 'META', 'GOOG']
env = Environment(pathData=path_to_csv, stockAbbr=stock_symbols)

# Check initial portfolio state
initial_state = env.state(date='2018-01-02')
print("Initial State:")
initial_table = [["Portfolio Value", initial_state['portfolio_value']]]
for stock, shares, price in zip(stock_symbols, initial_state['shares'], initial_state['current_prices']):
    initial_table.append([f"{stock} Shares", shares])
    initial_table.append([f"{stock} Current Price", price])
print(tabulate(initial_table, headers=["Description", "Value"], tablefmt="grid"))

# Test trading: buying more of the first stock, holding others
actions = [3, 0, 0, 0, 0]  # Strong Buy AAPL, Hold others
env.trade(action=actions, date='2018-01-02')

# Check state after trading
post_trade_state = env.state(date='2018-01-02')
print("\nState after trading:")
post_trade_table = [["Portfolio Value", post_trade_state['portfolio_value']]]
for stock, shares, price in zip(stock_symbols, post_trade_state['shares'], post_trade_state['current_prices']):
    post_trade_table.append([f"{stock} Shares", shares])
    post_trade_table.append([f"{stock} Current Price", price])
print(tabulate(post_trade_table, headers=["Description", "Value"], tablefmt="grid"))

# Check technical indicators
indicators = env.indicator(date='2018-01-02')
print("\nIndicators:")
print(tabulate([["Closing Prices", indicators['closing_price']]], headers=["Description", "Value"], tablefmt="grid"))

# Check updated portfolio value
portfolio_value = env.get_portfolio_value()
print("\nUpdated Portfolio Value:")
print(tabulate([["Portfolio Value", portfolio_value]], headers=["Description", "Value"], tablefmt="grid"))
