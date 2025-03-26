import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

class SignalGenerationAgent:
    def __init__(self):
        self.model = None
        self.last_model_accuracy = 0.5

    def label_data(self, df):
        df["Target"] = np.where(df["Return"].shift(-1) > 0, 1, 0)
        return df.dropna()

    def train_model(self, df):
        df = self.label_data(df)
        X = df[["Return", "Volatility", "SMA_20", "SMA_50", "Momentum"]]
        y = df["Target"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        self.last_model_accuracy = acc
        print(f"✅ Model trained - Accuracy: {acc:.2f}")

    def detect_regime(self, df):
        volatility = df["Return"].rolling(window=10).std().iloc[-1]
        if volatility > 0.025:
            return "volatile"
        elif df["SMA_20"].iloc[-1] > df["SMA_50"].iloc[-1]:
            return "bull"
        else:
            return "bear"

    def detect_anomaly(self, df):
        z = (df["Return"] - df["Return"].rolling(20).mean()) / df["Return"].rolling(20).std()
        latest_z = z.iloc[-1]
        return abs(latest_z) > 2

    def predict_signal(self, latest_row):
        features = latest_row[["Return", "Volatility", "SMA_20", "SMA_50", "Momentum"]].values.reshape(1, -1)
        model_pred = self.model.predict(features)[0]

        sma_crossover = latest_row["SMA_20"] > latest_row["SMA_50"]

        vote_score = (
            0.5 * model_pred +
            0.3 * int(sma_crossover) +
            0.2 * int(self.last_model_accuracy > 0.55)
        )

        if vote_score >= 0.6:
            return "BUY"
        elif vote_score <= 0.3:
            return "SELL"
        else:
            return "HOLD"
