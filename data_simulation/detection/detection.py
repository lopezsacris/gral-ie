# -*- coding: utf-8 -*-
"""
Created on Tue May 13 15:01:00 2025

@author: Urko López Sácristan
"""

import numpy as np
from influxdb_client import InfluxDBClient
import pandas as pd
import time
import os
from scipy.signal import find_peaks
from scipy.stats import linregress
from influxdb_client import Point
import warnings
from influxdb_client.client.warnings import MissingPivotFunction

warnings.simplefilter("ignore", MissingPivotFunction)


def get_data(client):
    # Datuak lortzeko objetua hasiarazi.
    query_api = client.query_api()
    fields = ["ForwardPower", "ReflectionCoefficientMagnitude",
              "IncidentPowerReference"]
    dataframes = {}
    for field in fields:
        query = f'''
        from(bucket: "{bucket}")
            |> range(start: -50s)
            |> filter(fn: (r) => r["_measurement"] == "mqtt_consumer")
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
        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge_asof(
                merged_df.sort_values("_time"),
                df.sort_values("_time"),
                on="_time",
                direction="nearest",
                tolerance=pd.Timedelta("1s")
            )

    if merged_df is not None and not merged_df.empty:
        return merged_df

    else:
        return pd.DataFrame(columns=["_time"] + fields)


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
        df["NoiseForward"] = np.nan
    return df


def ewma_filter(df, column, alpha=0.01, prev_filtered_last_value=None):
    if column not in df.columns:
        print(f"[ERROR] '{column}' ez dago datuetan.")
        return df[column]

    # Aurreko baliorik balego, erabili iragazketa zentzuzkoa izateko
    if prev_filtered_last_value is not None:
        temp_series = pd.concat([
            pd.Series([prev_filtered_last_value]),
            df[column]
        ], ignore_index=True)
        filtered = temp_series.ewm(alpha=alpha, adjust=False).mean()
        # Kendu lehen balio artifiziala
        return filtered.iloc[1:].reset_index(drop=True)
    else:
        return df[column].ewm(alpha=alpha,
                              adjust=False).mean().reset_index(drop=True)


def detect_anomaly(df, threshold=0.007, prominence_noise=0.7):
    if ("Adaptation_filtered" in df.columns
            and "NoiseForward_filtered" in df.columns):
        # Datuak denboraren arabera ordenatu, arazorik balego.
        df = df.sort_values("_time")
        # Lortu zarata maximoa duten puntuak
        peak_noise_indices, _ = find_peaks(df["NoiseForward_filtered"],
                                           prominence=prominence_noise)
        df["is_noise_peak"] = False
        df.loc[peak_noise_indices, "is_noise_peak"] = True
        # Datuentzako deribatua kalkulatu np funtzioarekin.
        x = df["_time"].astype(np.int64) / 1e9  # convertir a segundos
        y = df["Adaptation_filtered"]
        slope, _, _, _, _ = linregress(x, y)
        df["is_adaptation_jump"] = abs(slope) > threshold
        # Bueltatu deribatua ataria baino handiago duten lerroak.
        anomalies = df[df["is_noise_peak"] | df["is_adaptation_jump"]]  # ****
        if not anomalies.empty:
            return anomalies[["_time", "Adaptation_filtered",
                              "NoiseForward_filtered"]]
        else:
            return pd.DataFrame()
    else:
        print('[ERROR] Beharrezko zutabeak ez dira aurkitu.')
        return pd.DataFrame()


def save_anomaly(anomalies, write_api):
    for _, row in anomalies.iterrows():
        point = (
            Point("anomalies")
            .time(row["_time"])
            .field("adaptation_filtered", row["Adaptation_filtered"])
            .field("noiseforward_filtered", row["NoiseForward_filtered"])
            .field("anomaly", True)
            )
        write_api.write(bucket=bucket, record=point)


if __name__ == "__main__":
    influx_url = os.getenv("INFLUX_URL")
    influx_token = os.getenv("INFLUX_TOKEN")
    organization = os.getenv("ORGANIZATION")
    bucket = os.getenv("BUCKET")
    with InfluxDBClient(url=influx_url, token=influx_token,
                        org=organization) as client:
        write_api = client.write_api()
        prev_adaptation = None
        prev_noise = None
        while True:
            start_time = time.time()
            df = get_data(client)
            if not df.empty:
                df = calculate_adaptation(df)
                df = calculate_noisefwd(df)
                if "Adaptation" in df.columns and "NoiseForward" in df.columns:
                    df["Adaptation_filtered"] = ewma_filter(df, "Adaptation",
                                                            0.001,
                                                            prev_adaptation)
                    df["NoiseForward_filtered"] = ewma_filter(df,
                                                              "NoiseForward",
                                                              0.02, prev_noise)
                anomalies = detect_anomaly(df)
                print(anomalies.empty)
                if not df["Adaptation_filtered"].empty:
                    prev_adaptation = df["Adaptation_filtered"].iloc[-1]
                if not df["NoiseForward_filtered"].empty:
                    prev_noise = df["NoiseForward_filtered"].iloc[-1]
                if not anomalies.empty:
                    save_anomaly(anomalies, write_api)

            else:
                print("[INFO] Ez dago daturik. Itxaroten...")
                time.sleep(5)
                prev_adaptation = None
                prev_noise = None
                continue
            elapsed = time.time() - start_time
            time.sleep(max(0, 5 - elapsed))
