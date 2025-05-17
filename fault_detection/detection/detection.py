# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:01:00 2025

@author: Urko López Sácristan
"""

import numpy as np
from influxdb_client import InfluxDBClient
from shared_config import get_redis_client, REDIS_QUEUE
import pandas as pd
import time
import os
import json
from scipy.signal import find_peaks


ewma_alpha = 0.1


def get_data():
    # Aldatu behar, Adaptazioa eta Zarataren arabera
    # Host: neurtzen duen dispositiboa.
    # Topic: neurtutakoen aldagaien tag-a.
    query = f'''
    from(bucket: "{bucket}")
        |> range(start: -30s)
        |> filter(fn: (r) => r["_measurement"] == "mqtt_consumer")
        |> filter(fn: (r) => r["_field"] == "ForwardPower" or r["_field"] == "ReflectionCoefficientMagnitude" or r["_field"] == "IncidentPowerReference")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    # pivot: Zutabeak timestamp-aren arabera antolatzeko, lerroka.

    with InfluxDBClient(url=influx_url, token=influx_token,
                        org=organization) as client:
        # Datuak lortzeko objetua hasiarazi.
        query_api = client.query_api()
        try:
            # Lehen definitutako query-arentzat datuak lortu dataframe moduan.
            df = query_api.query_data_frame(query)
        except Exception as e:
            print(f'[ERROREA]: Akatsa InfluxDB kontsultatzean: {e}')
            return pd.DataFrame(columns=["_time", "ForwardPower",
                                         "ReflectionCoefficientMagnitude",
                                         "IncidentPowerReference"])

        if df.empty or "_time" not in df.columns:
            print("[INFO] InfluxDB hutsa edo '_time' zutaberik gabe.")
            return pd.DataFrame(columns=["_time", "ForwardPower",
                                         "ReflectionCoefficientMagnitude",
                                         "IncidentPowerReference"])
        # Time zutabea datetime moduko objetu-etara bihurtu, gero ordenatzeko.
        df["_time"] = pd.to_datetime(df["_time"])
        # Soilik bi zutabe bueltatu: time, value.
        return df[["_time", "ForwardPower", "ReflectionCoefficientMagnitude",
                   "IncidentPowerReference"]]


def calculate_adaptation(df):
    if ("ForwardPower" in df.columns
            and "ReflectionCoefficientMagnitude" in df.columns
            and "IncidentPowerReference" in df.columns):
        # Kalkulatu islatutako potentzia koefizientea erabiliz
        df["ReflectedPower"] = (df["ReflectionCoefficientMagnitude"] ** 2
                                * df["IncidentPowerReference"])
        # Adaptazioa kalkulatu (ForwardPower - ReflectedPower)
        df["Adaptation"] = df["ForwardPower"] - df["ReflectedPower"]
    else:
        df["Adaptation"] = np.nan
    return df


def calculate_noisefwd(df):
    if ("ForwardPower" in df.columns
            and "ReflectionCoefficientMagnitude" in df.columns
            and "IncidentPowerReference" in df.columns):
        # Kalkulatu zarata neurtutako eta erreferentziazko potentziak erabiliz
        df["NoiseForward"] = (df["IncidentPowerReference"]-df["ForwardPower"])
    else:
        df["Adaptation"] = np.nan
    return df


def ewma_filter(df, column, alpha=0.1):
    if column in df.columns:
        return df[column].ewm(alpha=alpha).mean()
    else:
        print(f'[ERROR]: {column} magnitudea ez da datuetan aurkitu.')
        return df[column]


def detect_anomaly(df, d_threshold=0.5, prominence_noise=0.1):
    if ("Adaptation_filtered" in df.columns
            and "NoiseForward_filtered" in df.columns):
        # Datuak denboraren arabera ordenatu, arazorik balego.
        df = df.sort_values("_time")
        # Lortu zarata maximoa duten puntuak
        peak_noise_indices, _ = find_peaks(df["NoiseForward_filtered"],
                                           prominence=prominence_noise)
        noise_times = (df["_time"].iloc[peak_noise_indices].values
                       if peak_noise_indices.size > 0 else [])
        noise_condition = df["_time"].isin(noise_times)
        # Datuentzako deribatua kalkulatu np funtzioarekin.
        df["d_Adaptation"] = np.gradient(df["Adaptation_filtered"],
                                         df["_time"].values.astype(float))
        adaptation_condition = np.abs(df["d_Adaptation"]) > d_threshold
        # Bueltatu deribatua ataria baino handiago duten lerroak.
        anomalies = df[noise_condition & adaptation_condition]
        if not anomalies.empty:
            return anomalies[["_time", "Adaptation_Filtered",
                              "NoiseForward_filtered"]]
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()


# def save_anomaly(anomalies):
#     # Anomaliarik badago, artxibo batean gorde, beste programak klasifikatzeko.
#     if not anomalies.empty:
#         data = anomalies[["Solenoid1", "Solenoid2"]].values.tolist()
#         r.lpush(REDIS_QUEUE, json.dumps(data))
#         print("[INFO] Anomalia klasifikatzaileari bidalita.")


if __name__ == "__main__":
    influx_url = os.getenv("INFLUX_URL")
    influx_token = os.getenv("INFLUX_TOKEN")
    organization = os.getenv("ORGANIZATION")
    bucket = os.getenv("BUCKET")
    r = get_redis_client()
    while True:
        df = get_data()
        if not df.empty:
            df = calculate_adaptation(df)
            df = calculate_noisefwd(df)
            if "Adaptation" in df.columns and "NoiseForward" in df.columns:
                df["Adaptation_filtered"] = ewma_filter(df, "Adaptation")
                df["NoiseForward_filtered"] = ewma_filter(df, "NoiseForward")
            anomalies = detect_anomaly(df)
            # save_anomaly(anomalies)
        else:
            print("[INFO] Ez dago daturik. Itxaroten...")
            time.sleep(5)
            continue
        time.sleep(5)
