import pandas as pd

from config import SIGNAL_CONFIDENCE_THRESHOLD


def generate_signals(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate buy/sell/hold signals from model predictions.

    Rules:
        BUY:  predicted_direction == 1 AND confidence > THRESHOLD
        SELL: predicted_direction == 0 AND confidence > THRESHOLD
        HOLD: confidence <= THRESHOLD (uncertain prediction)

    Adds a 'signal' column ('buy', 'sell', 'hold') to the DataFrame.
    Also adds 'signal_numeric' (1=buy, -1=sell, 0=hold) for charting.
    """
    df = prediction_df.copy()

    conditions = []
    for _, row in df.iterrows():
        direction = row.get("predicted_direction", -1)
        confidence = row.get("prediction_confidence", 0.0)

        if confidence > SIGNAL_CONFIDENCE_THRESHOLD:
            if direction == 1:
                conditions.append("buy")
            else:
                conditions.append("sell")
        else:
            conditions.append("hold")

    df["signal"] = conditions
    df["signal_numeric"] = df["signal"].map({"buy": 1, "sell": -1, "hold": 0})

    return df
