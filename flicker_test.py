import numpy as np
import matplotlib.pyplot as plt
from utils import flicker_noise, estimate_signal_and_noise

fs = 20e6

x = np.zeros(int(10e4))
x += flicker_noise(x, fs, 1)

a, b = estimate_signal_and_noise(x, fs=fs, f1=fs / 6, f2=fs / 3, PLOT_PSD=True)
