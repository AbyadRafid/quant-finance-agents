from agents.market_data_agent import MarketDataAgent
from agents.signal_generation_agent import SignalGenerationAgent
from agents.portfolio_optimization_agent import PortfolioOptimizationAgent
import pandas as pd

def main():
    tickers = ["AAPL", "MSFT", "TLT", "GLD"]
    market_agent = MarketDataAgent()
    signal_agent = SignalGenerationAgent()
    portfolio_agent = PortfolioOptimizationAgent()

    price_data = pd.DataFrame()
    signals = {}
    liquidity_scores = {
        "AAPL": 0.9,
        "MSFT": 0.9,
        "TLT": 0.7,
        "GLD": 0.6
    }
    tax_penalties = {
        "AAPL": 0.01,
        "MSFT": 0.00,
        "TLT": 0.015,
        "GLD": 0.02
    }

    print("\n📥 Fetching & processing data for multiple assets...\n")

    for ticker in tickers:
        print(f"➡️  Processing {ticker}")
        df = market_agent.fetch_stock_data(ticker, period="6mo", interval="1d")
        df = market_agent.engineer_features(df)

        if len(df) > 0:
            signal_agent.train_model(df)
            last_row = df.iloc[-1]
            sig = signal_agent.predict_signal(last_row)
            signals[ticker] = sig
            price_data[ticker] = df["Close"]
            print(f"   🚦 Signal: {sig}")
        else:
            print(f"   ⚠️ Not enough data for {ticker}, skipping...")

    print("\n📊 Signals:")
    for t, s in signals.items():
        print(f"{t}: {s}")

    print("\n📈 Optimizing Portfolio with Liquidity & Tax Awareness...")
    weights = portfolio_agent.build_portfolio(
        price_df=price_data,
        signals=signals,
        liquidity_scores=liquidity_scores,
        tax_penalties=tax_penalties
    )

    if weights.empty:
        print("⚠️ No valid portfolio could be constructed. Try relaxing constraints or check signals.")
        return

    print("\n📊 Final Portfolio Weights:")
    print(weights)

    # Stress Test: COVID crash (Feb 2020 to Apr 2020)
    print("\n🧪 Running Stress Test (2020 Crash)...")
    test = portfolio_agent.stress_test(
        price_df=price_data,
        weights=weights,
        crisis_start="2020-02-01",
        crisis_end="2020-04-30"
    )

    print("\n📉 Stress Test Results (Feb–Apr 2020):")
    for k, v in test.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()
