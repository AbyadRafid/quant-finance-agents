# agents/market_data_agent.py

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from bs4 import BeautifulSoup
import requests
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class MarketDataAgent:
    def __init__(self):
        self.alt_data_sources = {
            "satellite": "https://www.orbitalinsight.com/",
            "app_downloads": "https://www.data.ai/",
            "credit_card": "https://www.yodlee.com/"
        }

    def fetch_stock_data(self, ticker: str, period="6mo", interval="1d"):
        df = yf.Ticker(ticker).history(period=period, interval=interval)
        df = df.dropna()
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Return"] = df["Close"].pct_change()
        df["Volatility"] = df["Return"].rolling(window=10).std()
        df["SMA_20"] = df["Close"].rolling(window=20).mean()
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["Momentum"] = df["Close"] - df["Close"].shift(10)
        df.dropna(inplace=True)
        return df

    def cross_asset_correlation(self, tickers: list, period="3mo", interval="1d"):
        df = pd.DataFrame()
        for ticker in tickers:
            hist = yf.Ticker(ticker).history(period=period, interval=interval)
            df[ticker] = hist['Close']
        return df.corr()

    def sentiment_from_news(self, ticker):
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers)
        soup = BeautifulSoup(r.text, 'html.parser')
    
        try:
            table = soup.find("table", class_="fullview-news-outer")
            rows = table.findAll("tr")
            headlines = [row.a.text for row in rows[:5]]  # Top 5 headlines
            return headlines
        except Exception as e:
            print("❌ Error fetching headlines:", e)
            return []


    def analyze_sentiment_scores(self, headlines):
        sid = SentimentIntensityAnalyzer()
        results = []

        for title in headlines:
            score = sid.polarity_scores(title)
            results.append({
                "headline": title,
                "sentiment": score["compound"]
            })

        return pd.DataFrame(results)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df.fillna(0, inplace=True)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df)
        return pd.DataFrame(scaled, columns=df.columns, index=df.index)
