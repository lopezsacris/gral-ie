# -*- coding: utf-8 -*-
"""
Created on Thu May 15 09:34:56 2025

@author: Urko López Sácristan
"""

import pandas as pd
import paho.mqtt.client as mqtt
import time
import os
import json
import sys

CSV_FILES = {
    'CameraLuminosity': '/app/data/CameraLuminosity_df.csv',
    'ForwardPower': '/app/data/ForwardPower_df.csv',
    'GasFlow': '/app/data/GasFlow_df.csv',
    'IncidentPowerReference': '/app/data/IncidentPowerReference_df.csv',
    'PressureLEBT': '/app/data/PressureLEBT_df.csv',
    'ReflectionCoefficientMagnitude': '/app/data/ReflectionCoefficientMagnitude_df.csv',
    'ReflectionCoefficientPhase': '/app/data/ReflectionCoefficientPhase_df.csv',
    'RfPower': '/app/data/RfPower_df.csv'
}

MQTT_BROKER = os.environ.get('MQTT_BROKER_HOST', 'mqtt_broker')
MQTT_PORT = 1883
MQTT_TOPIC_PREFIX = os.environ.get('MQTT_TOPIC_PREFIX', 'sensor')
INDEX_COLUMN = 0
TIMESTAMP_COLUMN = 'time'


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("[INFO] MQTT Broker-era konektatuta.")
    else:
        print(f"[ERROR] MQTT Broker-era konektatzean arazoa, kodea: {rc}")


def publish_data(client, topic, payload):
    result = client.publish(topic, payload)
    if result[0] == 0:
        print(f"[INFO]{topic}-en publikatuta: {payload}")
    else:
        print(f"[ERROR] {topic}-en publikatzerakoan: {result}")


if __name__ == "__main__":
    try:
        # Fitxategi guztiak irakurri eta hiztegi batean gorde df-ak.
        dataframes = {}
        for sensor_name, csv_file in CSV_FILES.items():
            try:
                df = pd.read_csv(csv_file, parse_dates=[TIMESTAMP_COLUMN],
                                 index_col=INDEX_COLUMN)
                df = df.sort_values(by=TIMESTAMP_COLUMN)
                df = df.reset_index(drop=True)
                dataframes[sensor_name] = df
                print(f"[INFO] '{csv_file}' artxiboa irakurrita.")
            except FileNotFoundError:
                print(f"[ERROR]: Ez da '{csv_file}' fitxategia aurkitu.")
                dataframes = {}
                break
            except KeyError as e:
                print(f"[ERROR]: '{e}' zutabea faltan '{csv_file}' artxiboan.")
                dataframes = {}
                break

        if not dataframes:
            print("[ERROR]: Ez dira artxiboak kargatu. Irteten...")
            sys.exit(1)

        time.sleep(5)
        # MQTT zerbitzaria hasiarazi
        client = mqtt.Client()
        client.on_connect = on_connect

        # EMQX broker-ak denbora behar du hasteko...
        time.sleep(10)
        client.connect(MQTT_BROKER, MQTT_PORT)
        client.loop_start()

        all_indices = {sensor: 0 for sensor in dataframes}
        all_done = False
        previous_timestamp = None

        while not all_done:
            current_timestamps = {}
            next_data = {}
            earliest_timestamp = None

            # DataFrame guztien artean timestamp txikiena aurkitu
            for sensor_name, df in dataframes.items():
                current_index = all_indices[sensor_name]
                if current_index < len(df):
                    current_timestamps[sensor_name] = df.loc[current_index,
                                                             TIMESTAMP_COLUMN]
                    if (earliest_timestamp is None
                        or current_timestamps[sensor_name] <
                            earliest_timestamp):
                        earliest_timestamp = current_timestamps[sensor_name]

            if earliest_timestamp is None:
                all_done = True
                continue

            # Timestamp horri dagozkion datuak zerbitzarian publikatu.
            payload = {}
            for sensor_name, df in dataframes.items():
                current_index = all_indices[sensor_name]
                if (current_index < len(df)
                    and df.loc[current_index,
                               TIMESTAMP_COLUMN] == earliest_timestamp):
                    data_row = df.drop(columns=[TIMESTAMP_COLUMN]).iloc[
                        current_index].to_dict()
                    current_time = pd.Timestamp.utcnow()
                    data_row["time"] = current_time.isoformat(
                        timespec="nanoseconds") + "Z"
                    payload[sensor_name] = data_row
                    # Sensore bakoitzarentzat topic batean publikatu.
                    topic = f"{MQTT_TOPIC_PREFIX}/{sensor_name}"
                    publish_data(client, topic, json.dumps(data_row))
                    all_indices[sensor_name] += 1

            # Itxarote denbora kalkulatu
            if previous_timestamp is not None:
                time_difference = (earliest_timestamp -
                                   previous_timestamp).total_seconds()
                if time_difference > 0:
                    print(f"[INFO] {time_difference:.2f} segundo itxaroten...")
                    time.sleep(time_difference)

            previous_timestamp = earliest_timestamp

        # MQTT broker-etik deskonektatu
        client.loop_stop()
        client.disconnect()
        print("[INFO] Artxibo guztietarako simulazioak bukatuta.")

    except Exception as e:
        print(f"[ERROR]: {e}. Irteten...")
        sys.exit(1)
