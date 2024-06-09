import numpy as np
from scipy.signal import butter


class HighPassFilter:
    def __init__(self, cutoff, sr):
        self.cutoff = cutoff
        self.sr = sr

        self.sos = butter(4, cutoff / (sr / 2.0), btype="high", output="sos")
        self.zi = np.zeros([2, 2])

    def process(self, input_data):
        ...
