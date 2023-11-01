import sys
import os

import numpy as np

from .base import PerceptionBase
from .setting import LogDecayProfile, PresenceDetectorConfig


class LogDecayProfilePresenceDector(PerceptionBase):
    def __init__(self, transform_input=True):
        super(LogDecayProfilePresenceDector, self).__init__(to_tensor=False)
        self.profile = LogDecayProfile.PROFILE
        self.detection_threshold = PresenceDetectorConfig.DETECTION_THRESHOLD
        self.gain = PresenceDetectorConfig.GAIN
        self.score_mode = PresenceDetectorConfig.SCORE_MOD
        self.transform_input = transform_input

    def run(self, x):
        self._check_input(x)
        if self.transform_input: inputs = self.transform(x)
        left_score = np.sum(np.maximum(self.profile, self.transform(inputs[:,0,:])) - self.profile) / len(inputs)
        right_score = np.sum(np.maximum(self.profile, self.transform(inputs[:,1,:])) - self.profile) / len(inputs)
        if self.score_mode == 'sum':
            score = left_score + right_score
        elif self.score_mode == 'max':
            score = max(left_score, right_score)
        else:
            raise ValueError('score_mode must be either "sum" or "max" but got {}'.format(self.score_mode))
        
        self.score = score*self.gain
        if self.score > self.detection_threshold:
            return True
        else:
            return False    

    def get_presence_score(self):
        return self.score
    
    def _check_input(sefl, x):
        if x.shape[1] != 2 or len(x.shape) != 3:
            raise ValueError('Input shape must be (batch, 2, time) but got {}'.format(x.shape))