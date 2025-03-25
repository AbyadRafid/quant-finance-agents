from agents.market_data_agent import MarketDataAgent
from agents.signal_generation_agent import SignalGenerationAgent

def main():
    market = MarketDataAgent()
    signal = SignalGenerationAgent()

    # Load and process data
    df = market.fetch_stock_data("AAPL", period="6mo", interval="1d")
    df = market.engineer_features(df)
    alt_signals = market.get_alternative_data_signals()

    # Cross-asset correlation
    print("\n📊 Cross-Asset Correlation:")
    print(market.cross_asset_correlation(["AAPL", "MSFT", "TLT", "GLD"]))

    # News & sentiment
    headlines = market.sentiment_from_news("AAPL")
    print("\n📰 Headlines:")
    for h in headlines:
        print("-", h)
    print("\n📈 Sentiment Scores:")
    print(market.analyze_sentiment_scores(headlines))

    # Macro & earnings (simulated)
    macro_sent = market.macro_sentiment_score("Recession risk due to inflation spike. Hawkish policy ahead.")
    earnings_sent = market.earnings_call_sentiment("We beat expectations. Strong iPhone demand. New product launch.")
    print(f"\n🌐 Macro Sentiment Score: {macro_sent}")
    print(f"💼 Earnings Call Sentiment Score: {earnings_sent}")
    print(f"📡 Alt-Data Signals: {alt_signals}")

    # Train and predict
    signal.train_model(df)
    final_signal = signal.predict_signal(df.iloc[-1])
    print(f"\n🚦 Final Trading Signal: {final_signal}")

if __name__ == "__main__":
    main()
