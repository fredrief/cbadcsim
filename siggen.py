import numpy as np
import matplotlib.pyplot as plt
from utils import flicker_noise


class sine_generator(object):
    def __init__(self, length, thermal_noise_density, fs, flicker_noise_coef):
        self.length = length
        self.thermal_noise_density = thermal_noise_density
        self.fs = fs
        self.C = flicker_noise_coef

    def generate(self, freq, amp, offset):
        """ Generate sine wave """
        time = np.arange(self.length) / self.fs
        x = amp * np.sin(2 * np.pi * freq * time) + offset / 2
        # Add thermal noise
        npwr = np.power(self.thermal_noise_density, 2) * self.fs / 2
        x += np.random.normal(scale=np.sqrt(npwr),
                              size=time.shape)
        x += flicker_noise(x, self.fs, C=self.C)

        return x
