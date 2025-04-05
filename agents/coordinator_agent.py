# agents/coordinator_agent.py

import pandas as pd

class CoordinatorAgent:
    def __init__(self, agents):
        self.market_agent = agents["market"]
        self.signal_agent = agents["signal"]
        self.portfolio_agent = agents["portfolio"]
        self.risk_agent = agents["risk"]
        self.macro_agent = agents["macro"]
        self.eval_agent = agents["eval"]

    def run(self, tickers):
        print("\n🧠 [Coordinator] Starting multi-agent execution...\n")

        price_data = pd.DataFrame()
        signals = {}

        for ticker in tickers:
            print(f"➡️  Processing {ticker}")
            df = self.market_agent.fetch_stock_data(ticker, period="6mo", interval="1d")
            df = self.market_agent.engineer_features(df)

            if len(df) > 0:
                self.signal_agent.train_model(df)
                last_row = df.iloc[-1]
                signal = self.signal_agent.predict_signal(last_row)
                signals[ticker] = signal
                price_data[ticker] = df["Close"]
                print(f"   🚦 Signal: {signal}")
            else:
                print(f"   ⚠️ Not enough data for {ticker}, skipping...")

        print("\n📊 Signals:")
        for t, s in signals.items():
            print(f"{t}: {s}")

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

        print("\n📈 Optimizing Portfolio...")
        weights = self.portfolio_agent.build_portfolio(
            price_df=price_data,
            signals=signals,
            liquidity_scores=liquidity_scores,
            tax_penalties=tax_penalties
        )

        if weights.empty:
            print("⚠️ [Coordinator] Portfolio construction failed — not enough assets or infeasible inputs.")
            return

        print("\n📊 Final Portfolio Weights:")
        print(weights)

        # Stress test
        print("\n🧪 Running Stress Test (2020 Crash)...")
        test = self.portfolio_agent.stress_test(
            price_df=price_data,
            weights=weights,
            crisis_start="2020-02-01",
            crisis_end="2020-04-30"
        )

        print("\n📉 Stress Test Results:")
        for k, v in test.items():
            print(f"{k}: {v:.4f}")

        # Risk monitoring
        returns_df = price_data.pct_change().dropna()
        baseline_corr = returns_df.corr()

        print("\n🛡 Monitoring Portfolio Risk...")
        risk_report = self.risk_agent.monitor(
            price_df=price_data,
            weights=weights,
            baseline_corr=baseline_corr
        )

        print(f"\n📉 VaR (95%): {risk_report['VaR_95']:.4f}")
        print(f"📊 Volatility: {risk_report['Volatility']:.4f}")
        print(f"⚠️ Correlation Breakdown: {risk_report['Corr_Breakdown']}")

        # Macro environment
        print("\n🌐 Analyzing Macroeconomic Conditions...")
        macro = self.macro_agent.summarize()
        for k, v in macro["Indicators"].items():
            print(f"{k}: {v}")
        if macro["Warnings"]:
            print("\n🚨 Macroeconomic Warnings:")
            for w in macro["Warnings"]:
                print(w)

        # Self-evaluation
        print("\n📈 Performing Strategy Self-Evaluation...")
        eval_report = self.eval_agent.evaluate(returns_df, weights)

        print(f"\n🧠 20-Day Sharpe Ratio: {eval_report['Sharpe Ratio (20d)']:.4f}")
        if eval_report["Alpha Decay Report"]:
            for k, v in eval_report["Alpha Decay Report"].items():
                print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
        print("\n🔍 Attribution:")
        print(eval_report["Attribution Report"])

        # Escalation Criteria
        if risk_report["Corr_Breakdown"] or macro["Warnings"]:
            print("\n🚨 [Coordinator] Escalating to human oversight: unusual conditions detected.")

        print("\n✅ [Coordinator] Execution complete.")

