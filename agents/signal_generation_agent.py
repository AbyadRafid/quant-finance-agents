# agents/signal_generation_agent.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

class SignalGenerationAgent:
    def __init__(self):
        self.model = None

    def label_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Target"] = np.where(df["Return"].shift(-1) > 0, 1, 0)
        return df.dropna()

    def train_model(self, df: pd.DataFrame):
        df = self.label_data(df)
        features = ["Return", "Volatility", "SMA_20", "SMA_50", "Momentum"]
        X = df[features]
        y = df["Target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ XGBoost model trained — Accuracy: {acc:.2f}")

    def predict_signal(self, latest_row: pd.DataFrame) -> str:
        features = ["Return", "Volatility", "SMA_20", "SMA_50", "Momentum"]
        row_scaled = StandardScaler().fit_transform([latest_row[features].values])
        prediction = self.model.predict(row_scaled)[0]

        # Ensemble logic: include SMA cross as rule
        sma_cross = latest_row["SMA_20"] > latest_row["SMA_50"]
        if prediction == 1 and sma_cross:
            return "BUY"
        elif prediction == 0 and not sma_cross:
            return "SELL"
        else:
            return "HOLD"
