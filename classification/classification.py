# -*- coding: utf-8 -*-
"""
Created on Tue May 13 16:57:04 2025

@author: Urko López Sácristan
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import time
import os


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=2,
                 output_size=3):
        super(LSTMClassifier, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def classify_anomaly(model, df):
    features = df[["Solenoid1", "Solenoid2"]].values.astype(np.float32)
    sequence = torch.tensor(features,
                            dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(sequence)
        predicted = torch.argmax(output, dim=1)
    return predicted.item()


def save_anomaly(df, label):
    os.makedirs("anomaly_data", exist_ok=True)
    df["label"] = label
    df.to_csv(f"anomaly_data/anomaly_{int(time.time())}.csv", index=False)


model = LSTMClassifier()
# model.load_state_dict(torch.load("model.pt"))
model.eval()

while True:
    try:
        trigger = pd.read_csv("anomalies.csv")
        if not (trigger.empty and "Solenoid1" in trigger.columns
                and "Solenoid2" in trigger.columns):
            label = classify_anomaly(model, trigger)
            save_anomaly(trigger, label)
            open("anomalies.csv", "w").close()
    except FileNotFoundError:
        pass
    except Exception as e:
        print("Error:", e)
    time.sleep(5)
