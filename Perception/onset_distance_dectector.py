import sys
import os

import numpy as np

from .base import PerceptionBase
from .setting import LogDecayProfile, TransformConfig

from scipy.signal import find_peaks

class ClosestPeakDistance(PerceptionBase):
    def __init__(self, transform_input=True):
        super(ClosestPeakDistance, self).__init__(to_tensor=False)
        self.profile = LogDecayProfile.PROFILE
        self.compressed_distance = TransformConfig.DISTANCE_ENCODING
        self.transform_input = transform_input
        
    def run(self, x):
        self._check_input(x)
        if self.transform_input: inputs = self.transform(x)
        left_peaks, _ = find_peaks(self.transform(x[0, 0, :]).numpy(), distance= int(TransformConfig.ENVELOPE_LENGTH/16), prominence=0.01, height=self.profile)
        right_peaks, _ = find_peaks(self.transform(x[0,1,:]).numpy(), distance= int(TransformConfig.ENVELOPE_LENGTH/16), prominence=0.01, height=self.profile)
        self.closest_peak_idx = min([left_peaks[np.argmax(self.transform(inputs[0,0,:]).numpy()[ left_peaks])],
                      left_peaks[np.argmax(self.transform(inputs[0,1,:]).numpy()[ right_peaks ])]])
        
        return self.compressed_distance[self.closest_peak_idx]


    def _check_input(sefl, x):
        if len(x) !=1:
            raise ValueError('Input must be a single batch but got {}'.format(len(x)))
        if x.shape[1] != 2 or len(x.shape) != 3:
            raise ValueError('Input shape must be (batch, 2, time) but got {}'.format(x.shape))