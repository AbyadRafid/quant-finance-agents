# agents/macroeconomic_agent.py

import numpy as np
import pandas as pd

class MacroeconomicAgent:
    def __init__(self):
        pass

    def fetch_mock_macro_data(self):
        """
        Simulates macroeconomic indicators for demo purposes.
        In real-world use, pull from FRED, World Bank, or economic APIs.
        """
        data = {
            "GDP_growth": 1.8,       # Percent annualized
            "Unemployment_rate": 4.1,  # Percent
            "Inflation_rate": 3.9,     # Percent YoY
            "Fed_sentiment": -0.3,     # Negative = hawkish tone
            "Yield_10yr": 3.8,         # 10Y bond yield
            "Yield_2yr": 4.2           # 2Y bond yield
        }
        return data

    def check_yield_curve_inversion(self, yield_10yr, yield_2yr):
        """
        Detects yield curve inversion — typical recession signal.
        """
        return yield_2yr > yield_10yr

    def detect_macro_risks(self, indicators):
        warnings = []

        if indicators["GDP_growth"] < 0.5:
            warnings.append("⚠️ GDP growth dangerously low")

        if indicators["Inflation_rate"] > 4:
            warnings.append("⚠️ Inflation running hot")

        if indicators["Unemployment_rate"] > 5:
            warnings.append("⚠️ High unemployment risk")

        if indicators["Fed_sentiment"] < -0.2:
            warnings.append("⚠️ Fed sounding hawkish")

        if self.check_yield_curve_inversion(indicators["Yield_10yr"], indicators["Yield_2yr"]):
            warnings.append("⚠️ Yield curve inversion detected")

        return warnings

    def summarize(self):
        indicators = self.fetch_mock_macro_data()
        warnings = self.detect_macro_risks(indicators)

        return {
            "Indicators": indicators,
            "Warnings": warnings
        }
