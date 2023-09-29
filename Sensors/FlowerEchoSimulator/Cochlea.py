"""
Cochlear Transformation using Gammatone Filtering
"""

from typing import Any
import numpy as np
from scipy.signal import butter, lfilter, gammatone
from .Setting import CochleaSetting as Setting


class CochleaFilter:
    def __init__(self):
        self.emission_freq = Setting.EMISSION_FREQ
        self.sampling_freq = Setting.SAMPLING_FREQ
        self.broadband_spec = {'order':4, 'low':2e4, 'high':8e4}
        self.gammatone_banksize = Setting.GAMMATONE_BANKSIZE
        self.exp_compression_power = Setting.EXP_COMPRESSION_POWER
        self.lowpass_freq = Setting.LOWPASS_FREQ
        self.bidirectional = Setting.BIDIRECTIONAL_FILTERING
        self.cache = {}

    def transform(self, data: np.ndarray):
        # Broad bandpass filter
        if 'broadband' not in self.cache.keys():
            b, a = butter(self.broadband_spec['order'],[self.broadband_spec['low'],self.broadband_spec['high']],
                          'band', fs = self.sampling_freq )
            self.cache['broadband'] = (b, a)
        else: b, a = self.cache['broadband']
        if self.bidirectional:
            y = lfilter(b,a,np.flip(data,axis=0), axis=0)
            y = lfilter(b,a,np.flip(y,axis=0), axis=0)
        else:
            y = lfilter(b,a,data, axis=0)
        # Gammatone Filter
        if 'gammatone' not in self.cache.keys():
            b, a = gammatone(self.emission_freq, 'fir', fs=self.sampling_freq)
            self.cache['gammatone'] = (b, a)
        else: b, a = self.cache['gammatone']
        if self.bidirectional: 
            for _ in range(2): y = lfilter(b,a,np.flip(y,axis=0), axis=0)
        else: y = lfilter(b,a,y, axis=0)
        # halfwave rectifier
        y[y<0] = 0
        # exponential compression
        y = np.power(y,0.4)
        # lowpass filter
        if 'lowpass' not in self.cache.keys():
            b,a = butter(2, self.lowpass_freq, 'low', fs=self.sampling_freq)
            self.cache['lowpass'] = (b,a)
        else: b, a = self.cache['lowpass']
        if self.bidirectional: 
            for _ in range(2): y = lfilter(b,a,np.flip(y,axis=0), axis=0)
        else: y = lfilter(b,a,y, axis=0)
        return y

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.transform(*args, **kwds)

    def _reset(self):
        self.cache.clear()
