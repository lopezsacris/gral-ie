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
from influxdb_client import Point
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)


# DBSCAN parametroak
eps_dict = {
    "rf_taldea": 14,
    "huts_taldea": 0.32,
    "CameraLuminosity": 0.6
}
min_samples = 30
window = 800

# ALDAGAI TALDEAK
rf_taldea = ["ForwardPower", "ReflectionCoefficientMagnitude",
             "IncidentPowerReference"]
huts_taldea = ["GasFlow", "PressureLEBT"]
isolatu_taldea = ["CameraLuminosity"]


def get_data(bucket, client, window):
    # Datuak lortzeko objetua hasiarazi.
    query_api = client.query_api()
    fields = ["ForwardPower", "ReflectionCoefficientMagnitude",
              "IncidentPowerReference", "GasFlow", "PressureLEBT",
              "CameraLuminosity"]
    measurement = "mqtt_consumer"
    dataframes = {}
    for field in fields:
        query = f'''
        from(bucket: "{bucket}")
            |> range(start: -{window}s)
            |> filter(fn: (r) => r["_measurement"] == "{measurement}")
            |> filter(fn: (r) => r["_field"] == "{field}")
            |> sort(columns: ["_time"])
        '''

        try:
            # Lehen definitutako query-arentzat datuak lortu dataframe moduan.
            result = query_api.query_data_frame(query)
            if isinstance(result, list):
                df = pd.concat(result, ignore_index=True)
            else:
                df = result

            if (not df.empty and "_time" in df.columns
                    and "_value" in df.columns):
                df = df[["_time", "_value"]].rename(columns={"_value": field})

                df["_time"] = pd.to_datetime(df["_time"])
                df = df.dropna(subset=["_time"])

                df[field] = pd.to_numeric(df[field], errors='coerce')
                df = df.dropna(subset=[field])
                df = df.sort_values("_time").reset_index(drop=True)

                dataframes[field] = df
            else:
                print(f"[INFO] Ez dira datuak aurkitu {field} eremurako.")
                dataframes[field] = pd.DataFrame(columns=["_time", field])
        except Exception as e:
            print(f'[ERROREA]: Akatsa InfluxDB kontsultatzean: {e}')
            dataframes[field] = pd.DataFrame(columns=["_time", field])

    merged_df = None
    for field, df in dataframes.items():
        if df.empty:
            continue
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge_asof(
                merged_df.sort_values("_time"),
                df.sort_values("_time"),
                on="_time",
                direction="nearest"
            )

    return (merged_df if merged_df is not None
            and not merged_df.empty else pd.DataFrame())


def dbscan_multi(df, columns, talde, eps):
    if not all(col in df.columns for col in columns):
        print(f'[ERROR] DBSCAN egiteko zutabeak falta dira: {columns}')
        return pd.DataFrame()
    Xf = df.dropna(subset=columns)
    X = Xf[columns]
    if len(X) < min_samples:
        print("[INFO] Ez dago DBSCAN aplikatzeko datu nahikorik.")
        return pd.DataFrame()
    try:
        X_scaled = StandardScaler().fit_transform(X)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
        Xf.loc[:, "anomaly"] = labels == -1
        anomalies = Xf.loc[Xf["anomaly"]].copy()
        if not anomalies.empty:
            anomalies.loc[:, "source"] = f"{talde}_multi"
            print(f'[INFO] Anomaliak aurkitu dira {talde} sentsoreetan.')
            return anomalies[["_time"] + columns + ["source"]]
        else:
            print('[INFO] Ez da anomaliarik aurkitu.')
            return pd.DataFrame()
    except Exception as e:
        print(f'[ERROR] DBSCAN exekutatzean errorea: {e}.')
        return pd.DataFrame()


def dbscan_uni(df, column, eps):
    if column not in df.columns:
        print(f'[ERROR] Ez da zutabea existitzen: {column}')
        return pd.DataFrame()
    Xf = df.dropna(subset=[column])
    X = Xf[[column]]
    if len(X) < min_samples:
        print("[INFO] Ez dago DBSCAN aplikatzeko datu nahikorik.")
        return pd.DataFrame()
    try:
        X_scaled = StandardScaler().fit_transform(X)
        labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
        Xf.loc[:, "anomaly"] = labels == -1
        anomalies = Xf.loc[Xf["anomaly"]].copy()
        if not anomalies.empty:
            print(f'[INFO] Anomaliak aurkitu dira {column} sentsoreetan.')
            anomalies.loc[:, "source"] = f"{column}_uni"
            return anomalies[["_time", column, "source"]]
        else:
            print('[INFO] Ez da anomaliarik aurkitu.')
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
            df = get_data(bucket, client, window)
            if not df.empty:
                anomalies = []
                for columns, name in [(rf_taldea, "rf"),
                                      (huts_taldea, "huts")]:
                    eps = eps_dict.get(f"{name}_taldea")
                    a = dbscan_multi(df, columns, name, eps)
                    if not a.empty:
                        anomalies.append(a)
                for col in isolatu_taldea:
                    eps = eps_dict.get(col)
                    b = dbscan_uni(df, col, eps)
                    if not b.empty:
                        anomalies.append(b)
                if anomalies:
                    print("[INFO] Anomalia detektatuta! Leihoa gorde...")
                    anomaly = pd.concat(anomalies).reset_index(drop=True)
                    save_anomaly(anomaly, write_api)
                else:
                    print("[INFO] Ez dago anomaliarik...")
            else:
                print("[INFO] Ez dago daturik. Itxaroten...")
                time.sleep(15)
                prev_adaptation = None
                prev_noise = None
                continue
            elapsed = time.time()-start_time
            time.sleep(max(0, 15 - elapsed))
