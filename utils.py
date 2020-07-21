import torch
import numpy as np
import time


class Counter:
    def __init__(self):
        self.data = dict()

    def append(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.data:
                self.data[key] = []

            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy()
            elif isinstance(value, Timer):
                value = float(value)

            self.data[key].append(value)

    def __getattr__(self, key):
        if key not in self.data:
            return 0
        return np.mean(self.data[key])


class Timer:
    def __init__(self):
        self.start_time = -1
        self.stop_time = -1

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_time = time.time()

    def __str__(self):
        return str(self.stop_time - self.start_time)

    def __float__(self):
        return float(self.stop_time - self.start_time)
