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
        
    def run(self, x, ouput_index=False):
        self._check_single_input(x)
        if self.transform_input: inputs = self.transform(x)
        left_peaks, _ = find_peaks(self.transform(x[0, 0, :]), distance= int(TransformConfig.ENVELOPE_LENGTH/16), prominence=0.01, height=self.profile)
        right_peaks, _ = find_peaks(self.transform(x[0,1,:]), distance= int(TransformConfig.ENVELOPE_LENGTH/16), prominence=0.01, height=self.profile)
        candidates = []
        if len(left_peaks) == 0 and len(right_peaks) == 0: return None
        if len(left_peaks) > 0 : candidates.append(left_peaks[np.argmax(self.transform(inputs[0,0,:])[ left_peaks])])
        if len(right_peaks) > 0 : candidates.append(right_peaks[np.argmax(self.transform(inputs[0,1,:])[ right_peaks ])])
        self.closest_peak_idx = min(candidates)
        if ouput_index: return self.closest_peak_idx
        return self.compressed_distance[self.closest_peak_idx]