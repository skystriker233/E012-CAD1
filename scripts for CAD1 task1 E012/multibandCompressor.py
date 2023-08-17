import numpy as np
from scipy.signal import butter, lfilter
#from clarity.enhancer.compressor import Compressor

class MultibandCompressor:
    def __init__(self, bands):
        self.bands = bands


    def butter_bandpass(self, lowcut, highcut, sample_rate, order=5):
        nyq = 0.5 * sample_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def butter_bandpass_filter(self, data, lowcut, highcut, sample_rate, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, sample_rate, order=order)
        y = lfilter(b, a, data)
        return y
               
    def process(self, signal, sample_rate, compressor):
        #compressed_signal = np.zeros_like(signal)
        compressed_signal = np.copy(signal)
        for band in self.bands:
            band_signal = self.butter_bandpass_filter(signal, band[0], band[1], sample_rate)
            compressed_signal -= band_signal
            band_signal_new, _, _ = compressor.process(band_signal)
            # Add the compressed band signal to the final signal
            compressed_signal += band_signal_new
        return compressed_signal
