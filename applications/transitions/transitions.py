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
# Query-ak ematen duen abisua ez agertzeko (pivot komando falta)
import warnings
from influxdb_client.client.warnings import MissingPivotFunction
warnings.simplefilter("ignore", MissingPivotFunction)


def get_data(bucket, client):
    # Datuak lortzeko objetua hasiarazi.
    query_api = client.query_api()
    fields = ["ForwardPower", "ReflectionCoefficientMagnitude",
              "IncidentPowerReference"]
    measurement = "mqtt_consumer"
    dataframes = {}
    for field in fields:
        query = f'''
        from(bucket: "{bucket}")
            |> range(start: -50s)
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


def calculate_adaptation(df):
    if ("ForwardPower" in df.columns
            and "ReflectionCoefficientMagnitude" in df.columns
            and "IncidentPowerReference" in df.columns):
        # Kalkulatu islatutako potentzia koefizientea erabiliz
        df["ReflectedPower"] = (df["ReflectionCoefficientMagnitude"]
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
        df["NoiseForward"] = (df["ForwardPower"] -
                              df["IncidentPowerReference"])
    else:
        df["NoiseForward"] = np.nan
    return df


def ewma_filter(df, column, alpha=0.01):
    if column in df.columns:
        return df[column].ewm(alpha=alpha).mean()
    else:
        print(f'[ERROR]: {column} magnitudea ez da datuetan aurkitu.')
        return df[column]


def detect_transition(df, threshold=0.05, prominence_noise=0.5):
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
        anomalies = df[df["is_noise_peak"] & df["is_adaptation_jump"]]
        if not anomalies.empty:
            return True
        else:
            return False
    else:
        print('[ERROR] Beharrezko zutabeak ez dira aurkitu.')
        return False


def save_transition(df, bucket, write_api):
    for _, row in df.iterrows():
        try:
            point = (
                Point("transitions")
                .time(row["_time"])
                .field("Adaptation_Filtered", row["Adaptation_filtered"])
                .field("Noiseforward_Filtered", row["NoiseForward_filtered"])
                )
            write_api.write(bucket=bucket, record=point)
        except Exception as e:
            print(f'[ERROR] Ezin izan da puntua InfluxDB-n idatzi: {e}')


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
            df = get_data(bucket, client)
            if not df.empty:
                df = calculate_adaptation(df)
                df = calculate_noisefwd(df)
                if "Adaptation" in df.columns and "NoiseForward" in df.columns:
                    df["Adaptation_filtered"] = ewma_filter(
                        df, "Adaptation", 0.001)
                    df["NoiseForward_filtered"] = ewma_filter(
                        df, "NoiseForward", 0.02)
                if detect_transition(df):
                    print("[INFO] Trantsizioa detektatuta! Leihoa gorde...")
                    save_transition(df, bucket, write_api)
                else:
                    print("[INFO] Ez dago trantsiziorik...")
            else:
                print("[INFO] Ez dago daturik. Itxaroten...")
                time.sleep(5)
                prev_adaptation = None
                prev_noise = None
                continue
            elapsed = time.time() - start_time
            time.sleep(max(0, 5 - elapsed))
