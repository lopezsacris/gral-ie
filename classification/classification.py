# -*- coding: utf-8 -*-
"""
Created on Tue May 13 16:57:04 2025

@author: Urko López Sácristan
"""

import numpy as np
import tensorflow as tf
from shared_config import get_redis_client, REDIS_QUEUE
import time
import json


def load_model(path="model/model.h5"):
    return tf.keras.models.load_model(path)


if __name__ == "__main__":
    # model = load_model()
    r = get_redis_client()
    while True:
        try:
            item = r.brpop(REDIS_QUEUE, timeout=5)
            if item is None:
                print("[INFO] Ez dago daturik Redis-en. Itxaroten...")
                continue
            _, data = item
            sequence = np.array(json.loads(data), dtype=np.float32)
            # Normalizazioa:
            sequence = (sequence - np.mean(sequence,
                                           axis=0)) / (np.std(sequence, axis=0)
                                                       + 1e-6)
            sequence = np.expand_dims(sequence, axis=0)
            prediction = model.predict(sequence)
            label = np.argmax(prediction)
            confidence = np.max(prediction)
            print(f'[INFO] Saltoa: {label}, Konfiantza: {confidence:.2f}')
        except Exception as e:
            print("[ERROR]:", e)
        time.sleep(5)
