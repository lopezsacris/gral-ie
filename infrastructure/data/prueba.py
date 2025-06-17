# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 00:28:10 2025

@author: lopez
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import sys
import os
from datetime import timedelta
import seaborn as sns

# DBSCAN PARAMETERS
eps_dict = {
    "rf_taldea": 14,
    "huts_taldea": 0.32,
    "CameraLuminosity": 0.7
}
min_samples = 30


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
            return anomalies[["time"] + columns + ["source"]]
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
            return anomalies[["time", column, "source"]]
        else:
            print('[INFO] Ez da anomaliarik aurkitu.')
            return pd.DataFrame()
    except Exception as e:
        print(f'[ERROR] DBSCAN exekutatzean errorea: {e}.')
        return pd.DataFrame()


def aplicar_dbscan_por_ventanas(df, columnas, talde, eps, ventana_segundos=30):
    if 'time' not in df.columns:
        print('[ERROR] DataFrame-ak ez du "time" zutabea.')
        return pd.DataFrame()

    df = df.dropna(subset=columnas).sort_values('time').reset_index(drop=True)
    t_inicio = df['time'].iloc[0]
    t_final = df['time'].iloc[-1]
    anomalies_total = []

    while t_inicio < t_final:
        t_fin_ventana = t_inicio + timedelta(seconds=ventana_segundos)
        df_ventana = df[(df['time'] >= t_inicio) & (df['time']
                                                    < t_fin_ventana)]

        if not df_ventana.empty:
            anomalies = dbscan_multi(df_ventana, columnas, talde, eps)
            if not anomalies.empty:
                anomalies_total.append(anomalies)

        t_inicio = t_fin_ventana

    if anomalies_total:
        return pd.concat(anomalies_total).reset_index(drop=True)
    else:
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

            rf_taldea_cols = ["ForwardPower", "IncidentPowerReference",
                              "RfPower"]
            huts_taldea_cols = ["PressureLEBT", "GasFlow"]
            if all(col in dataframes for col in rf_taldea_cols):
                # Combinar columnas necesarias
                df_rf = pd.concat(
                    [dataframes[col][["time", col]] for col in rf_taldea_cols],
                    axis=1
                ).loc[:, ~pd.concat(
                    [dataframes[col][["time", col]] for col in rf_taldea_cols],
                    axis=1
                ).columns.duplicated()]

                df_huts = pd.concat(
                    [dataframes[col][["time", col]] for col in huts_taldea_cols],
                    axis=1
                ).loc[:, ~pd.concat(
                    [dataframes[col][["time", col]] for col in huts_taldea_cols],
                    axis=1
                ).columns.duplicated()]

                # Aplicar DBSCAN por ventanas
                anomalies_rf = aplicar_dbscan_por_ventanas(
                    df_rf,
                    columnas=rf_taldea_cols,
                    talde="rf_taldea",
                    eps=eps_dict["rf_taldea"],
                    ventana_segundos=800
                )
                
                anomalies_huts = aplicar_dbscan_por_ventanas(
                    df_huts,
                    columnas=huts_taldea_cols,
                    talde="huts_taldea",
                    eps=eps_dict["huts_taldea"],
                    ventana_segundos=800
                )

                # Imprimir anomalías si las hay
                df_rf_full = df_rf.dropna(subset=rf_taldea_cols).copy()
                df_rf_full["anomaly"] = False  # Inicializar
                if not anomalies_rf.empty:
                    # Convertir tiempos a índice para búsqueda eficiente
                    anomalies_index = anomalies_rf["time"]
                    df_rf_full["anomaly"] = df_rf_full["time"].isin(anomalies_index)

                    # Graficar usando la columna 'anomaly' generada desde resultados por ventanas
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(data=df_rf_full, x="ForwardPower", y="RfPower",
                                    hue="anomaly", palette={False: "blue", True: "red"})
                    plt.title("DBSCAN bidezko anomalia detekzioa")
                    plt.legend(title="Anomalía")
                    plt.show()
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(data=df_rf_full, x="ForwardPower", y="IncidentPowerReference",
                                    hue="anomaly", palette={False: "blue", True: "red"})
                    plt.title("DBSCAN bidezko anomalia detekzioa")
                    plt.legend(title="Anomalía")
                    plt.show()
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(data=df_rf_full, x="IncidentPowerReference", y="RfPower",
                                    hue="anomaly", palette={False: "blue", True: "red"})
                    plt.title("DBSCAN bidezko anomalia detekzioa")
                    plt.legend(title="Anomalía")
                    plt.show()
                else:
                    print("[INFO] Ez da anomaliarik aurkitu RF taldean.")
                df_huts_full = df_huts.dropna(subset=huts_taldea_cols).copy()
                df_huts_full["anomaly"] = False
                if not anomalies_huts.empty:
                    df_huts_full["anomaly"] = df_huts_full["time"].isin(anomalies_huts["time"])
            
                    # Graficar combinaciones de variables
                    plt.figure(figsize=(10, 6))
                    sns.scatterplot(data=df_huts_full, x="PressureLEBT", y="GasFlow",
                                    hue="anomaly", palette={False: "blue", True: "red"})
                    plt.title("Huts taldeko anomaliak (PressureLEBT vs GasFlow)")
                    plt.legend(title="Anomalía")
                    plt.show()
                else:
                    print("[INFO] Ez da anomaliarik aurkitu huts taldean.")
                if "CameraLuminosity" in dataframes:
                    df_cam = dataframes["CameraLuminosity"].copy()
                
                    # Aplicar DBSCAN univariado por ventanas
                    anomalies_camera = []
                    ventana_segundos = 800
                    df_cam = df_cam.dropna(subset=["CameraLuminosity"]).sort_values("time")
                    t_inicio = df_cam["time"].iloc[0]
                    t_final = df_cam["time"].iloc[-1]
                
                    while t_inicio < t_final:
                        t_fin_ventana = t_inicio + timedelta(seconds=ventana_segundos)
                        df_ventana = df_cam[(df_cam["time"] >= t_inicio) & (df_cam["time"] < t_fin_ventana)]
                
                        if not df_ventana.empty:
                            anomalies = dbscan_uni(df_ventana, "CameraLuminosity", eps=eps_dict["CameraLuminosity"])
                            if not anomalies.empty:
                                anomalies_camera.append(anomalies)
                
                        t_inicio = t_fin_ventana
                
                    df_cam["anomaly"] = False
                    if anomalies_camera:
                        anomalies_total = pd.concat(anomalies_camera).reset_index(drop=True)
                        df_cam["anomaly"] = df_cam["time"].isin(anomalies_total["time"])
                
                        # Graficar resultado
                        plt.figure(figsize=(10, 6))
                        sns.scatterplot(data=df_cam, x="time", y="CameraLuminosity",
                                        hue="anomaly", palette={False: "blue", True: "red"})
                        plt.title("Anomalías en CameraLuminosity (univariado)")
                        plt.legend(title="Anomalía")
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.show()
                    else:
                        print("[INFO] Ez da anomaliarik aurkitu CameraLuminosity datuetan.")
            else:
                print("[ERROR] Ez da 'CameraLuminosity' aurkitu.")


    except Exception as e:
        print(f"[ERROR]: {e}. Irteten...")
        sys.exit(1)
