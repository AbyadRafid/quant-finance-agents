import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random

class MarketDataAgent:
    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self.alt_data_sources = {
            "satellite": random.uniform(-0.5, 0.5),
            "credit_card_spend": random.uniform(-0.5, 0.5),
            "app_downloads": random.uniform(-0.5, 0.5)
        }

    def fetch_stock_data(self, ticker, period="6mo", interval="1d"):
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        return df.dropna()

    def engineer_features(self, df):
        df["Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Return"].rolling(window=10).std()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["Momentum"] = df["Close"] - df["Close"].shift(10)
        df.dropna(inplace=True)
        return df

    def cross_asset_correlation(self, tickers, period="3mo", interval="1d"):
        df = pd.DataFrame()
        for ticker in tickers:
            df[ticker] = yf.Ticker(ticker).history(period=period, interval=interval)['Close']
        return df.corr()

    def sentiment_from_news(self, ticker):
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            r = requests.get(url, headers=headers)
            soup = BeautifulSoup(r.text, 'html.parser')
            rows = soup.find("table", class_="fullview-news-outer").findAll("tr")
            return [row.a.text for row in rows[:5]]
        except:
            return []

    def analyze_sentiment_scores(self, headlines):
        results = []
        for title in headlines:
            score = self.vader.polarity_scores(title)["compound"]
            results.append({"headline": title, "sentiment": score})
        return pd.DataFrame(results)

    def macro_sentiment_score(self, transcript_text):
        keywords = ['inflation', 'growth', 'recession', 'hawkish', 'dovish']
        score = sum([transcript_text.lower().count(k) for k in keywords])
        return score / len(keywords)

    def earnings_call_sentiment(self, transcript_text):
        score = self.vader.polarity_scores(transcript_text)["compound"]
        return score

    def get_alternative_data_signals(self):
        return self.alt_data_sources
