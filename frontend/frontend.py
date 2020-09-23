import numpy as np


class frontend(object):
    """Frontend Parent class"""

    def __init__(self, lna, adc):
        self.lna = lna
        self.adc = adc

    def convert(self, val):
        return self.adc.convert(self.lna.amplify(val))
