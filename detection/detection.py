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


def get_data():
    # Aldatu behar, Adaptazioa eta Zarataren arabera
    # Host: neurtzen duen dispositiboa.
    # Topic: neurtutakoen aldagaien tag-a.
    query = f'''
    from(bucket: "{bucket}")
        |> range(start: -30s)
        |> filter(fn: (r) => r["_measurement"] == "data")
        |> filter(fn: (r) => r["_field"] == "value")
        |> filter(fn: (r) => r["host"] == "2544d8fef1fd")
        |> filter(fn: (r) => r["topic"] == "data/Solenoid1Temperature" or r["topic"] == "data/Solenoid2Temperature")
        |> pivot(rowKey:["_time"], columnKey: ["topic"], valueColumn: "_value")
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
            return pd.DataFrame(columns=["_time", "Solenoid1", "Solenoid2"])

        if df.empty or "_time" not in df.columns:
            print("[INFO] InfluxDB hutsa edo '_time' zutaberik gabe.")
            return pd.DataFrame(columns=["_time", "Solenoid1", "Solenoid2"])
        # Time zutabea datetime moduko objetu-etara bihurtu, gero ordenatzeko.
        df["_time"] = pd.to_datetime(df["_time"])
        # Zutabeen izena aldatu, argiagoa izateko.
        df = df.rename(columns={
            "data/Solenoid1Temperature": "Solenoid1",
            "data/Solenoid2Temperature": "Solenoid2"
            })
        # Soilik bi zutabe bueltatu: time, value.
        return df[["_time", "Solenoid1", "Solenoid2"]]


def detect_anomaly(df, th1=0.5, th2=0.5):
    # Datuak denboraren arabera ordenatu, arazorik balego.
    df = df.sort_values("_time")
    # Datuentzako deribatua kalkulatu np funtzioarekin: ondoz ondoko balioak.
    df["d_zar"] = np.gradient(df["Solenoid1"])
    df["d_adap"] = np.gradient(df["Solenoid2"])
    # Bueltatu deribatua ataria baino handiago duten lerroak.
    condition = (np.abs(df["d_zar"]) > th1) | (np.abs(df["d_adap"]) > th2)
    anomalies = df[condition]
    return anomalies


def save_anomaly(anomalies):
    # Anomaliarik badago, artxibo batean gorde, beste programak klasifikatzeko.
    if not anomalies.empty:
        data = anomalies[["Solenoid1", "Solenoid2"]].values.tolist()
        r.lpush(REDIS_QUEUE, json.dumps(data))
        print("[INFO] Anomalia klasifikatzaileari bidalita.")


if __name__ == "__main__":
    influx_url = os.getenv("INFLUX_URL")
    influx_token = os.getenv("INFLUX_TOKEN")
    organization = os.getenv("ORGANIZATION")
    bucket = os.getenv("BUCKET")
    r = get_redis_client()
    while True:
        df = get_data()
        if df.empty:
            print("[INFO] Ez dago daturik. Itxaroten...")
            time.sleep(5)
            continue
        anomalies = detect_anomaly(df)
        save_anomaly(anomalies)
        time.sleep(5)
