import sys
import os

from .tranforms import get_transform
from .setting import TransformConfig

QUIET_THRESHOLD = 0.15

class PerceptionBase:
    def __init__(self, to_tensor=False):
        self.transform = get_transform(quiet_threshold=QUIET_THRESHOLD, 
                                       normalize_min_value=TransformConfig.NORMALIZE_MIN_VALUE, 
                                       normalize_max_value=TransformConfig.NORMALIZE_MAX_VALUE,
                                       to_tensor=to_tensor)

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        raise NotImplementedError