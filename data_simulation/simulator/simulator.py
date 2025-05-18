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

csv_files_list = {
    'CameraLuminosity': 'CameraLuminosity_df.csv',
    'ForwardPower': 'ForwardPower_df.csv',
    'GasFlow': 'GasFlow_df.csv',
    'IncidentPowerReference': 'IncidentPowerReference_df.csv',
    'PressureLEBT': 'PressureLEBT_df.csv',
    'ReflectionCoefficientMagnitude':
        'ReflectionCoefficientMagnitude_df.csv',
    'ReflectionCoefficientPhase':
        'ReflectionCoefficientPhase_df.csv',
    'RfPower': 'RfPower_df.csv'
}

MQTT_BROKER = os.environ.get('MQTT_BROKER_HOST', 'mqtt_broker')
MQTT_PORT = 1883
MQTT_TOPIC_PREFIX = os.environ.get('MQTT_TOPIC_PREFIX', 'sensor')
INDEX_COLUMN = 0
TIMESTAMP_COLUMN = 'time'
DATA_DIR = '/app/data/'
DAY_SLEEP = 60


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
        subdir = [d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d))]
        if not subdir:
            print(f"[INFO]: Ez da azpikarpetarik aurkitu '{DATA_DIR}'-en.")
            sys.exit(0)

        # MQTT zerbitzaria hasiarazi
        client = mqtt.Client()
        client.on_connect = on_connect

        # EMQX broker-ak denbora behar du hasteko...
        client.connect(MQTT_BROKER, MQTT_PORT)
        client.loop_start()

        for folder in subdir:
            folder_path = os.path.join(DATA_DIR, folder)
            print(f"[INFO] '{folder}' karpeta irakurtzen.")
            dataframes = {}
            # Fitxategi guztiak irakurri eta hiztegi batean gorde df-ak.
            for sensor_name, csv_file in csv_files_list.items():
                file = os.path.join(folder_path, csv_file)
                try:
                    df = pd.read_csv(file,
                                     parse_dates=[TIMESTAMP_COLUMN],
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
                    print(f"[ERROR]: '{e}' faltan '{csv_file}' artxiboan.")
                    dataframes = {}
                    break

            if not dataframes:
                print("[ERROR]: Ez dira artxiboak kargatu. Irteten...")
                sys.exit(1)

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
                        current_timestamps[sensor_name] = df.loc[
                            current_index, TIMESTAMP_COLUMN]
                        if (earliest_timestamp is None
                            or current_timestamps[sensor_name] <
                                earliest_timestamp):
                            earliest_timestamp = current_timestamps[
                                sensor_name]

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
                        print(f"[INFO] {time_difference:.2f}s itxaroten...")
                        time.sleep(time_difference)

                previous_timestamp = earliest_timestamp

            print(f"[INFO]: {folder}-eko datuen simulaizoa bukatuta.")
            time.sleep(DAY_SLEEP)

        # MQTT broker-etik deskonektatu
        client.loop_stop()
        client.disconnect()
        print("[INFO] Artxibo guztietarako simulazioak bukatuta.")

    except Exception as e:
        print(f"[ERROR]: {e}. Irteten...")
        sys.exit(1)
