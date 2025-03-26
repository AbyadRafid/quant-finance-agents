# agents/portfolio_optimization_agent.py

import numpy as np
import pandas as pd
import cvxpy as cp
import riskfolio as rp

class PortfolioOptimizationAgent:
    def __init__(self):
        self.optimized_weights = None

    def build_portfolio(self, price_df, signals, liquidity_scores=None, tax_penalties=None):
        """
        price_df: DataFrame of historical prices (columns = tickers)
        signals: dict of signals {'AAPL': 'BUY', 'MSFT': 'SELL', ...}
        liquidity_scores: dict of values from 0 (illiquid) to 1 (high liquidity)
        tax_penalties: dict of values representing tax cost to exit asset (0 = no cost)
        """
        returns = price_df.pct_change().dropna()
        tickers = list(signals.keys())

        # Only consider assets to BUY or HOLD
        selected_assets = [t for t in tickers if signals[t] in ["BUY", "HOLD"]]

        if len(selected_assets) < 3:
            print("⚠️ Not enough BUY or HOLD assets to build a diversified portfolio (min = 3).")
            return pd.Series(dtype=float)

        selected_returns = returns[selected_assets]

        port = rp.Portfolio(returns=selected_returns)

        # Estimate mean return (mu) and covariance (cov)
        port.assets_stats(method_mu='hist', method_cov='ledoit')
        mu = port.mu.copy()

        # Apply tax penalties: subtract tax cost from expected return
        if tax_penalties:
            tax_penalty = pd.Series(tax_penalties).reindex(mu.index).fillna(0)
            mu -= tax_penalty
            port.mu = mu

        # Apply liquidity penalty as position weight adjustment
        liquidity_penalty = pd.Series(0, index=mu.index)
        if liquidity_scores:
            liquidity_penalty = pd.Series({t: 1 - liquidity_scores.get(t, 1.0) for t in mu.index})
            liquidity_penalty = liquidity_penalty.clip(lower=0, upper=0.5)

        model = "Classic"
        rm = "MV"
        obj = "Sharpe"

        port.lowerret = 0.0  # minimum asset weight
        port.upperret = 1.0   # maximum asset weight

        w = port.optimization(model=model, rm=rm, obj=obj)

        if w is None:
            print("❌ Optimization failed: Problem non-convex or infeasible with current inputs.")
            return pd.Series(dtype=float)

        # Apply liquidity penalty
        w = w * (1 - liquidity_penalty)


        self.optimized_weights = w
        return w.sort_values(ascending=False)

    def stress_test(self, price_df, weights, crisis_start, crisis_end):
        """
        Simulate portfolio performance during a crisis period.
        """
        crisis_prices = price_df.loc[crisis_start:crisis_end]
        crisis_returns = crisis_prices.pct_change().dropna()

        # Match columns with weights
        matching_returns = crisis_returns[weights.index]
        portfolio_returns = matching_returns @ weights

        cumulative = (1 + portfolio_returns).cumprod()
        drawdown = (cumulative / cumulative.cummax()) - 1
        max_dd = drawdown.min()

        return {
            "cumulative_return": cumulative[-1] - 1,
            "max_drawdown": max_dd,
            "volatility": portfolio_returns.std(),
            "sharpe": portfolio_returns.mean() / portfolio_returns.std()
        }
