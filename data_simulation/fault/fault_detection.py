# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 09:03:46 2025

@author: lopez
"""

import numpy as np
from influxdb_client import InfluxDBClient
import pandas as pd
import time
import os
import json
from scipy.signal import find_peaks
from scipy.stats import linregress
from influxdb_client import Point
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# DBSCAN parametroak
eps_dict = {
    "rf_taldea": 0.4,
    "huts_taldea": 0.4,
    "CameraLuminosity": 0.4
}
min_samples = 100

# ALDAGAI TALDEAK
rf_taldea = ["ForwardPower", "ReflectionCoefficientMagnitude",
             "IncidentPowerReference"]
huts_taldea = ["GasFlow", "PressureLEBT"]
isolatu_taldea = ["CameraLuminosity"]


def get_data(client):
    query = f'''
    from(bucket: "{bucket}")
        |> range(start: -50s)
        |> filter(fn: (r) => r["_measurement"] == "mqtt_consumer")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    # pivot: Zutabeak timestamp-aren arabera antolatzeko, lerroka.

    # Datuak lortzeko objetua hasiarazi.
    query_api = client.query_api()
    try:
        # Lehen definitutako query-arentzat datuak lortu dataframe moduan.
        df = query_api.query_data_frame(query)
    except Exception as e:
        print(f'[ERROREA]: Akatsa InfluxDB kontsultatzean: {e}')
        return pd.DataFrame()

    if df.empty or "_time" not in df.columns:
        print("[INFO] InfluxDB hutsa edo '_time' zutaberik gabe.")
        return pd.DataFrame()
    # Time zutabea datetime moduko objetu-etara bihurtu, gero ordenatzeko.
    df["_time"] = pd.to_datetime(df["_time"])
    return df.sort_values("_time").reset_index(drop=True)


def dbscan_multi(df, columns, talde, eps):
    if not all(col in df.columns for col in columns):
        print(f'[ERROR] DBSCAN egiteko zutabeak falta dira: {columns}')
        return pd.DataFrame()
    X = df.dropna(subset=columns)
    if len(X) < min_samples:
        print("[INFO] Ez dago DBSCAN aplikatzeko datu nahikorik.")
        return pd.DataFrame()
    try:
        X_scaled = StandardScaler().fit_transform(X)
        dbs = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
        X["anomaly"] = dbs.labels_ == -1
        anomalies = X[X["anomaly"]]
        if not anomalies.empty:
            anomalies["source"] = f"{talde}_multi"
            return anomalies[["_time"] + columns + ["source"]]
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f'[ERROR] DBSCAN exekutatzean errorea: {e}.')
        return pd.DataFrame()


def dbscan_uni(df, column, eps):
    if column not in df.columns:
        print(f'[ERROR] Ez da zutabea existitzen: {column}')
        return pd.DataFrame()
    X = df.dropna(subset=[column])
    if len(X) < min_samples:
        print("[INFO] Ez dago DBSCAN aplikatzeko datu nahikorik.")
        return pd.DataFrame()
    try:
        X_scaled = StandardScaler().fit_transform(X)
        dbs = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
        X["anomaly"] = dbs.labels_ == -1
        anomalies = X[X["anomaly"]]
        if not anomalies.empty:
            return anomalies[["_time"] + column]
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f'[ERROR] DBSCAN exekutatzean errorea: {e}.')
        return pd.DataFrame()


def save_anomaly(anomalies, write_api):
    for _, row in anomalies.iterrows():
        point = (
            Point("anomalies")
            .time(row["_time"])
            .tag("source", row["source"])
            .field("anomaly", True)
            )
        for c in row.index:
            if c not in ["_time", "source"]:
                point = point.field(c, float(row[c]))
        write_api.write(bucket=bucket, record=point)


if __name__ == "__main__":
    influx_url = os.getenv("INFLUX_URL")
    influx_token = os.getenv("INFLUX_TOKEN")
    organization = os.getenv("ORGANIZATION")
    bucket = os.getenv("BUCKET")
    with InfluxDBClient(url=influx_url, token=influx_token,
                        org=organization) as client:
        write_api = client.write_api()
        while True:
            start_time = time.time()
            df = get_data(client)
            if not df.empty:
                anomalies = []
                for columns, name in [(rf_taldea, "rf"),
                                      (huts_taldea, "huts")]:
                    eps = eps_dict.get(f"{name}_taldea")
                    a = dbscan_multi(df, columns, name)
                    if not a.empty:
                        anomalies.append(a)
                for col in isolatu_taldea:
                    eps = eps_dict.get(col)
                    b = dbscan_uni(df, col)
                    if not b.empty:
                        anomalies.append(b)
                if not anomalies.empty:
                    anomaly = pd.concat(anomalies).reset_index(drop=True)
                    save_anomaly(anomaly, write_api)
            else:
                print("[INFO] Ez dago daturik. Itxaroten...")
                time.sleep(5)
                prev_adaptation = None
                prev_noise = None
                continue
        elapsed = time.time()-start_time
        time.sleep(max(0, 5 - elapsed))
