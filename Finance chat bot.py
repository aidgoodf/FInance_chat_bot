#!/usr/bin/env python
# coding: utf-8

# # Stock Advisor Chat Bot
# ## Aidan Goodfellow

# In[5]:


#import necessary packages and libraries
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import requests
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pandas import Timestamp



# In[6]:



def fetch_finnhub_news(api_key, ticker):
    # Finnhub requires dates in YYYY-MM-DD format
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Adjust based time range

    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&token={api_key}"

    response = requests.get(url)
    if response.status_code == 200:
        news_items = response.json()
        return news_items  # This returns a list of news articles
    else:
        print("Failed to fetch news")
        return []

def analyze_finbert_sentiment(text):
    finbert = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    results = finbert(text)
    return results


def create_sentiment_df(news_items):
    sentiments = []
    for item in news_items:
        if 'headline' in item:
            sentiment_result = analyze_finbert_sentiment(item['headline'])
            # sentiment_result structure: [{'label': 'LABEL', 'score': SCORE}]
            # Adjust according to your actual result format
            if sentiment_result:
                sentiment_label = sentiment_result[0]['label']
                sentiment_score = sentiment_result[0]['score']
                sentiment_date = datetime.fromtimestamp(item['datetime']).strftime('%Y-%m-%d')
                sentiments.append({'date': sentiment_date, 'label': sentiment_label, 'score': sentiment_score})
    
    sentiment_df = pd.DataFrame(sentiments)
    # Convert 'date' to datetime and aggregate scores by date
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    return sentiment_df



# In[7]:



def fetch_stock_prices(ticker):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)  # Fetching more days for SMA calculation
    df = yf.download(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    if df.empty:
        raise ValueError("Failed to fetch stock prices")
    return df


def calculate_sentiment_scores(sentiment_df, current_date):
    # Convert strings to datetime for comparison
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    current_date = pd.to_datetime(current_date)

    # Define the short term and medium term periods
    short_term_period = current_date - Timestamp.resolution * 7  # Last 7 days
    medium_term_period = current_date - Timestamp.resolution * 30  # Last 30 days

    # Filter the DataFrame for the respective periods
    short_term_df = sentiment_df[(sentiment_df['date'] > short_term_period) & (sentiment_df['date'] <= current_date)]
    medium_term_df = sentiment_df[(sentiment_df['date'] > medium_term_period) & (sentiment_df['date'] <= current_date)]

    # Calculate average scores for short term and medium term
    short_term_avg_score = short_term_df['score'].mean()
    medium_term_avg_score = medium_term_df['score'].mean()

    return short_term_avg_score, medium_term_avg_score


def calculate_sma(prices_df, window=30):
    return prices_df['Close'].rolling(window=window).mean()

def trading_decision(prices_df, sentiment_df, current_date):
    
    # Convert string to pandas Timestamp for consistent handling
    query_date = pd.to_datetime(current_date)

    if query_date in prices_df.index:
    # Date exists, proceed with analysis
        current_price = prices_df.loc[query_date]['Close']
    else:
    # Date does not exist, find the closest previous date
        current_date = prices_df.index[prices_df.index < query_date].max()
        current_price = prices_df.loc[current_date]['Close']
        
    short_term_sentiment, medium_term_sentiment = calculate_sentiment_scores(sentiment_df, current_date)    

    prices_df['30d_sma'] = calculate_sma(prices_df)
    current_sentiment = sentiment_df[sentiment_df['date'] == current_date]
    current_price = prices_df.loc[current_date]['Close']
    current_sma = prices_df.loc[current_date]['30d_sma']
    
    sentiment_threshold = 0.5
    if short_term_sentiment > sentiment_threshold and medium_term_sentiment > sentiment_threshold and current_price > current_sma:
        return "Buy"
    elif short_term_sentiment < -sentiment_threshold and medium_term_sentiment < -sentiment_threshold and current_price < current_sma:
        return "Sell"
    else:
        return "Hold"



# In[10]:


model_name = "yiyanghkust/finbert-tone"  #FinBERT model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example usage
api_key = "cnkfihhr01qiclq85a0gcnkfihhr01qiclq85a10"
ticker = "AAPL"  # Example stock ticker
news_items = fetch_finnhub_news(api_key, ticker)
for item in news_items[:5]:  # Display first 5 news items
    print(item['headline'])

sentiment_df = create_sentiment_df(news_items)

# Sample usage:
ticker = 'AAPL'
try:
    prices_df = fetch_stock_prices(ticker)
    # Assume sentiment_df is already defined with 'date', 'short_term_sentiment', 'medium_term_sentiment'
    current_date = datetime.now().strftime('%Y-%m-%d')  # Adjust as needed
    prices_df.index = pd.to_datetime(prices_df.index)
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
    decision = trading_decision(prices_df, sentiment_df, current_date)
    print(f"Trading decision for {current_date}: {decision}")
except ValueError as e:
    print(e)


# In[11]:


def main_chatbot_function():
    print("Welcome to the Stock Advice Chatbot!")
    while True:
        ticker = input("Enter a stock ticker for advice or 'exit' to quit: ").upper()
        if ticker == 'EXIT':
            print("Thank you for using the Stock Advice Chatbot. Goodbye!")
            break

        try:
            print(f"Fetching news and analyzing sentiment for {ticker}...")
            news_items = fetch_finnhub_news(api_key, ticker)
            sentiment_df = create_sentiment_df(news_items)

            print(f"Fetching stock prices for {ticker}...")
            prices_df = fetch_stock_prices(ticker)
            prices_df.index = pd.to_datetime(prices_df.index)

            print(f"Making trading decision for {ticker}...")
            current_date = datetime.now().strftime('%Y-%m-%d')  # Use the last available date in your data
            decision = trading_decision(prices_df, sentiment_df, current_date)
            print(f"Trading decision for {ticker} on {current_date}: {decision}\n")
        except Exception as e:
            print(f"An error occurred: {e}\n")

if __name__ == "__main__":
    main_chatbot_function()


# In[ ]:




