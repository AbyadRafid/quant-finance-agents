# agents/risk_management_agent.py

import pandas as pd
import numpy as np

class RiskManagementAgent:
    def __init__(self):
        pass

    def compute_var(self, returns, confidence_level=0.95):
        """
        Compute historical Value-at-Risk (VaR)
        """
        if len(returns) < 1:
            return None
        return np.percentile(returns, (1 - confidence_level) * 100)

    def compute_volatility(self, returns, window=20):
        return returns.rolling(window=window).std().iloc[-1]

    def correlation_breakdown(self, asset_returns, baseline_corr):
        """
        asset_returns: df of % changes
        baseline_corr: historical baseline correlation matrix (DataFrame)
        """
        live_corr = asset_returns.corr()
        diff = (live_corr - baseline_corr).abs()

        # Trigger alert if any pair breaks down by more than 0.5
        triggered = (diff > 0.5).any().any()
        return triggered, diff

    def simulate_macro_scenario(self, price_df, weights, shock_pct):
        """
        Simulate impact of a macro shock reducing all prices by `shock_pct`
        """
        shocked_prices = price_df * (1 - shock_pct)
        shocked_returns = shocked_prices.pct_change().dropna()
        portfolio_returns = shocked_returns @ weights
        cumulative = (1 + portfolio_returns).cumprod()
        return {
            "cumulative_return": cumulative[-1] - 1,
            "volatility": portfolio_returns.std(),
            "max_drawdown": (cumulative / cumulative.cummax() - 1).min()
        }

    def monitor(self, price_df, weights, baseline_corr):
        returns = price_df.pct_change().dropna()
        portfolio_returns = returns @ weights

        var_95 = self.compute_var(portfolio_returns, 0.95)
        vol = self.compute_volatility(portfolio_returns)
        corr_break, diff_matrix = self.correlation_breakdown(returns, baseline_corr)
        macro_impact = self.simulate_macro_scenario(price_df, weights, 0.10)

        return {
            "VaR_95": var_95,
            "Volatility": vol,
            "Corr_Breakdown": corr_break,
            "Corr_Change": diff_matrix,
            "Macro_Shock_Impact": macro_impact
        }
