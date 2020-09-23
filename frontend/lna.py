import numpy as np


class lna(object):
    """Generic ADC Parent class"""

    def __init__(self, gain):
        self.gain = gain

    def amplify(self, val):
        return val * self.gain
