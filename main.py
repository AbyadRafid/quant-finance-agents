
# main.py

from agents.market_data_agent import MarketDataAgent

def main():
    agent = MarketDataAgent()
    
    # Fetch and display stock data
    df = agent.fetch_stock_data("AAPL", period="3mo", interval="1d")
    print("📈 Raw Stock Data (AAPL):")
    print(df.tail())

    # Feature Engineering
    features = agent.engineer_features(df)
    print("\n🧠 Engineered Features:")
    print(features.tail())

    # Cross Asset Correlation
    correlation = agent.cross_asset_correlation(["AAPL", "MSFT", "TLT", "GLD"])
    print("\n📊 Cross Asset Correlation:")
    print(correlation)

    # News Headlines
    headlines = agent.sentiment_from_news("AAPL")
    print("\n📰 Latest News Headlines:")
    for h in headlines:
        print("-", h)

    # Analyze sentiment of headlines
    sentiment_scores = agent.analyze_sentiment_scores(headlines)
    print("\n📈 Sentiment Analysis of Headlines:")
    print(sentiment_scores)

if __name__ == "__main__":
    main()
