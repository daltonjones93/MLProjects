import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def calculate_yearly_growth(stock_data):
    # Calculate yearly growth for each year
    stock_data['Year'] = stock_data.index.year
    yearly_growth = stock_data.groupby('Year')['Close'].pct_change().add(1).prod() - 1

    # Calculate average yearly growth
    average_yearly_growth = yearly_growth.mean()

    return average_yearly_growth

def get_stocks_with_high_growth():
    # Get the list of S&P 500 tickers
    sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

    # Define the date range for the last 10 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 10)

    # Fetch historical stock data for each S&P 500 company
    high_growth_stocks = []
    for ticker in sp500_tickers:
        try:
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            average_growth = calculate_yearly_growth(stock_data)

            if average_growth > 0.15:
                high_growth_stocks.append((ticker, average_growth))

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    return high_growth_stocks

# Example usage
high_growth_stocks = get_stocks_with_high_growth()

if high_growth_stocks:
    print("Stocks with Average Yearly Growth > 15% in the Last 10 Years:")
    for stock in high_growth_stocks:
        print(f"{stock[0]}: {stock[1]:.2%}")
else:
    print("No stocks found.")
