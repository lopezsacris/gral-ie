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
        df["ReflectedPower"] = (df["ReflectionCoefficientMagnitude"]
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


def detect_anomaly(df, threshold=0.06, prominence_noise=0.5, window_size=500):
    if "Adaptation" in df.columns and "NoiseForward" in df.columns:
        df = df.sort_values("time")
        df["is_adaptation_jump"] = False
        df["is_noise_peak"] = False

        for start in range(0, len(df) - window_size):
            end = start + window_size
            window_df = df.iloc[start:end].copy()

            # EWMA en la ventana (Adaptation y NoiseForward)
            adaptation_filtered = ewma_filter(window_df,
                                              "Adaptation", alpha=0.001)
            noise_filtered = ewma_filter(window_df, "NoiseForward", alpha=0.02)
            # Detectar picos de ruido filtrado en la ventana
            peak_indices, _ = find_peaks(noise_filtered,
                                         prominence=prominence_noise)
            if len(peak_indices) == 0:
                continue  # No hay picos, pasar a siguiente ventana

            time_seconds = window_df["time"].astype(np.int64) / 1e9
            if adaptation_filtered.isna().any() or time_seconds.isna().any():
                continue

            slope, _, _, _, _ = linregress(time_seconds, adaptation_filtered)
            if abs(slope) > threshold:
                # Marcar los puntos de la ventana donde hay pico y pendiente
                df.loc[start:end, "is_adaptation_jump"] = True
                for peak in peak_indices:
                    peak_global_idx = start + peak
                    df.loc[peak_global_idx, "is_noise_peak"] = True

        anomalies = df[df["is_noise_peak"] & df["is_adaptation_jump"]]
        if not anomalies.empty:
            return anomalies[["time", "Adaptation",
                              "NoiseForward"]]
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
                # Anomaliak detektatu
                anomalies_df = detect_anomaly(
                    df_merged.copy())
                # (Ikustarazi 1)
                fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

                # Lehenengo grafikoa: Adaptazioa eta Zarata
                axs[0].plot(df_merged['time'],
                            df_merged['Adaptation'],
                            label='Adaptazioa',
                            color='red', alpha=0.7)
                axs[0].plot(df_merged['time'],
                            df_merged['NoiseForward'],
                            label='Aurreranzko zarata',
                            color='green', alpha=0.7)
                # for window_df in filtered_windows:
                #     axs[0].plot(window_df["time"],
                #                 window_df["Adaptation_filtered"],
                #                 color='orange', alpha=0.4, linewidth=1)
                #     axs[0].plot(window_df["time"],
                #                 window_df["NoiseForward_filtered"],
                #                 color='blue', alpha=0.3, linewidth=1)
                if not anomalies_df.empty:
                    axs[0].scatter(anomalies_df['time'],
                                   anomalies_df['Adaptation'],
                                   color='magenta', marker='o',
                                   label='Trantsizioa (Adaptazioa)')
                    axs[0].scatter(anomalies_df['time'],
                                   anomalies_df['NoiseForward'],
                                   color='cyan', marker='x',
                                   label='Trantsizioa (Zarata)')
                axs[0].set_ylabel("Magnitudeak")
                axs[0].set_title("Adaptazioa eta Zarata iragazita trantsizioekin")
                axs[0].legend()
                axs[0].grid(True)

                # Bigarren grafikoa: Argitasuna
                df_lum = dataframes['CameraLuminosity'][['time',
                                                         'CameraLuminosity']
                                                        ].sort_values('time')
                if not anomalies_df.empty:
                    luminosity_anomalies = pd.merge_asof(anomalies_df[['time']],
                                         df_lum[['time', 'CameraLuminosity']],
                                         on='time', direction='nearest')

                    axs[1].scatter(luminosity_anomalies['time'],
                                   luminosity_anomalies['CameraLuminosity'],
                                   color='orange', marker='o',
                                   label='Trantsizioa (Luminositatea)')
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
