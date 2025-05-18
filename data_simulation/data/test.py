# -*- coding: utf-8 -*-
"""
Created on Sat May 17 14:58:01 2025

@author: lopez
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import linregress
import sys
import os


def calculate_adaptation(df):
    if ("ForwardPower" in df.columns
            and "ReflectionCoefficientMagnitude" in df.columns
            and "IncidentPowerReference" in df.columns):
        df["ReflectedPower"] = (df["ReflectionCoefficientMagnitude"] ** 2
                                * df["IncidentPowerReference"])
        df["Adaptation"] = df["ForwardPower"] - df["ReflectedPower"]
    else:
        df["Adaptation"] = np.nan
    return df


def calculate_noisefwd(df):
    if ("ForwardPower" in df.columns
            and "IncidentPowerReference" in df.columns):
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


def detect_anomaly(df, threshold=0.007, prominence_noise=0.7, window_size=500):
    if ("Adaptation_filtered" in df.columns
            and "NoiseForward_filtered" in df.columns):
        # Datuak denboraren arabera ordenatu, arazorik balego.
        df = df.sort_values("time")
        # Lortu zarata maximoa duten puntuak
        peak_noise_indices, _ = find_peaks(df["NoiseForward_filtered"],
                                           prominence=prominence_noise)
        df["is_noise_peak"] = False
        df.loc[peak_noise_indices, "is_noise_peak"] = True
        # Datuentzako deribatua kalkulatu np funtzioarekin.
        df["is_adaptation_jump"] = False
        for start in range(0, len(df)-window_size):
            end = start + window_size
            x = df["time"].iloc[start:end].astype(np.int64) / 1e9  # Segundutan
            y = df["Adaptation_filtered"].iloc[start:end]
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            if abs(slope) > threshold:
                df.loc[start:end, "is_adaptation_jump"] = True
        # Bueltatu deribatua ataria baino handiago duten lerroak.
        anomalies = df[df["is_noise_peak"] & df["is_adaptation_jump"]]
        if not anomalies.empty:
            return anomalies[["time", "Adaptation_filtered",
                              "NoiseForward_filtered"]]
        else:
            return pd.DataFrame()
    else:
        print('[ERROR] Beharrezko zutabeak ez dira aurkitu.')
        return pd.DataFrame()


# --- Leer el archivo CSV ---
DATA_DIR = '.'
TIMESTAMP_COLUMN = 'time'
INDEX_COLUMN = 0
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

if __name__ == "__main__":
    try:
        subdir = [d for d in os.listdir(DATA_DIR)
                  if os.path.isdir(os.path.join(DATA_DIR, d))]
        if not subdir:
            print(f"[INFO]: Ez da azpikarpetarik aurkitu '{DATA_DIR}'-en.")
            sys.exit(0)
        for folder in subdir:
            folder_path = os.path.join(DATA_DIR, folder)
            print(f"[INFO] '{folder}' karpeta irakurtzen.")
            dataframes = {}
            # Fitxategi guztiak irakurri eta hiztegi batean gorde df-ak.
            for sensor_name, csv_file in csv_files_list.items():
                file = os.path.join(folder_path, csv_file)
                try:
                    df = pd.read_csv(file, parse_dates=[TIMESTAMP_COLUMN],
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
                print("[ERROR] Ez dira artxiboak kargatu. Irteten...")
                sys.exit(1)

            if all(key in dataframes for key
                   in ['ForwardPower', 'ReflectionCoefficientMagnitude',
                       'IncidentPowerReference']):
                # Sortu df berri bat beharrezkoak ditugun magnitudeak batuz
                df_fp = dataframes['ForwardPower'][
                    ['time', 'ForwardPower']].sort_values('time')
                df_rcm = dataframes['ReflectionCoefficientMagnitude'][
                    ['time',
                     'ReflectionCoefficientMagnitude']].sort_values('time')
                df_ipr = dataframes['IncidentPowerReference'][
                    ['time', 'IncidentPowerReference']].sort_values('time')

                # Hacer merges aproximados (por tiempo más cercano)
                df_merged = pd.merge_asof(df_fp, df_rcm, on='time',
                                          direction='nearest')
                df_merged = pd.merge_asof(df_merged, df_ipr, on='time',
                                          direction='nearest')
                # Kalkulatu adaptazioa eta aurreranzko zarata
                df_merged = calculate_adaptation(df_merged.copy())
                df_merged = calculate_noisefwd(df_merged.copy())
                # Aplikatu EWMA iragazketa
                df_merged["Adaptation_filtered"] = ewma_filter(df_merged,
                                                               "Adaptation",
                                                               0.001)
                df_merged[
                    "NoiseForward_filtered"] = ewma_filter(df_merged,
                                                           "NoiseForward",
                                                           0.02)
                # Anomaliak detektatu
                anomalies_df = detect_anomaly(df_merged.copy())
                # (Ikustarazi 1)
                fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

                # Lehenengo grafikoa: Adaptazioa eta Zarata
                axs[0].plot(df_merged['time'],
                            df_merged['Adaptation_filtered'],
                            label='Adaptazioa iragazita',
                            color='red', alpha=0.7)
                axs[0].plot(df_merged['time'],
                            df_merged['NoiseForward_filtered'],
                            label='Aurreranzko zarata iragazita',
                            color='green', alpha=0.7)
                if not anomalies_df.empty:
                    axs[0].scatter(anomalies_df['time'],
                                   anomalies_df['Adaptation_filtered'],
                                   color='magenta', marker='o',
                                   label='Anomalia (Adaptazioa)')
                    axs[0].scatter(anomalies_df['time'],
                                   anomalies_df['NoiseForward_filtered'],
                                   color='cyan', marker='x',
                                   label='Anomalia (Zarata)')
                axs[0].set_ylabel("Iragazitako magnitudeak")
                axs[0].set_title("Adaptazioa eta Zarata iragazita anomaliekin")
                axs[0].legend()
                axs[0].grid(True)

                # Segundo gráfico: Luminosidad de la cámara
                df_lum = dataframes['CameraLuminosity'][['time',
                                                         'CameraLuminosity']
                                                        ].sort_values('time')
                axs[1].plot(df_lum["time"], df_lum["CameraLuminosity"],
                            label="Luminositatea", color='blue')
                axs[1].set_xlabel("Denbora")
                axs[1].set_ylabel("Luminositatea")
                axs[1].set_title("Luminositatea denboran zehar")
                axs[1].legend()
                axs[1].grid(True)
                # Ajustar espacio
                plt.tight_layout()
                plt.show()
            else:
                print("[ERROR]: Datuak falta dira.")
            if not dataframes:
                print("[ERROR]: Ez dira artxiboak kargatu. Irteten...")
                sys.exit(1)
    except Exception as e:
        print(f"[ERROR]: {e}. Irteten...")
        sys.exit(1)
