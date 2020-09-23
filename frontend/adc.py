import numpy as np


class adc(object):
    """Generic ADC Parent class"""

    def __init__(self, N, VREF, fs):
        self.N = N
        self.VREF = VREF
        self.fs = fs

    def convert(self, val):
        """Converts a single analog value to digital word of
        length N"""
        vda = self.VREF / 2 - self.VREF / np.power(2, self.N + 1)
        bout = np.zeros((self.N,), dtype=int)
        for i in range(1, self.N + 1):
            if val > vda:
                bout[i - 1] = 1
                vda = vda + (self.VREF / np.power(2, i + 1))
            else:
                bout[i - 1] = 0
                vda = vda - (self.VREF / np.power(2, i + 1))
        return bout


class ctrlbnd_adc(object):
    """Control Bounded ADC"""

    def __init__(self):
        pass

    def __str__(self):
        return "Control Bounded ADC Object"
