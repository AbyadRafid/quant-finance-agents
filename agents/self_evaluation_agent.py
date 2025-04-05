# agents/self_evaluation_agent.py

import numpy as np
import pandas as pd

class SelfEvaluationAgent:
    def __init__(self):
        pass

    def rolling_sharpe(self, returns, window=20, rf=0.01):
        excess = returns - rf / 252
        return excess.rolling(window).mean() / excess.rolling(window).std()

    def detect_alpha_decay(self, portfolio_returns):
        recent = portfolio_returns[-20:]
        past = portfolio_returns[-60:-20]
        if len(recent) < 10 or len(past) < 10:
            return None

        decay = recent.mean() < past.mean()
        return {
            "Recent Alpha": recent.mean(),
            "Past Alpha": past.mean(),
            "Alpha Decay Detected": decay
        }

    def attribution(self, returns_df, weights):
        """
        Calculates contribution to portfolio return by each asset.
        """
        weighted_returns = returns_df * weights
        contribution = weighted_returns.mean()
        return contribution.sort_values(ascending=False)

    def evaluate(self, returns_df, weights):
        portfolio_returns = returns_df @ weights
        sharpe_series = self.rolling_sharpe(portfolio_returns)
        sharpe_ratio = sharpe_series.iloc[-1] if not sharpe_series.empty else np.nan

        decay_report = self.detect_alpha_decay(portfolio_returns)
        attribution_report = self.attribution(returns_df, weights)

        return {
            "Sharpe Ratio (20d)": sharpe_ratio,
            "Alpha Decay Report": decay_report,
            "Attribution Report": attribution_report
        }
