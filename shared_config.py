# -*- coding: utf-8 -*-
"""
Created on Wed May 14 16:43:41 2025

@author: Urko López Sacristán
"""

import redis
import os


def get_redis_client():
    return redis.Redis(
        host=os.getenv("REDIS_HOST", "redis"),
        port=6379,
        decode_responses=True
    )


REDIS_QUEUE = "anomaly_queue"
