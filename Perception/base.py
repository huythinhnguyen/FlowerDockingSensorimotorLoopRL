import sys
import os
import torch
from .tranforms import get_transform
from .setting import TransformConfig

QUIET_THRESHOLD = 0.15


class PerceptionBase:
    def __init__(self, to_tensor:bool=False):
        self.transform = get_transform(quiet_threshold=QUIET_THRESHOLD, 
                                       normalize_min_value=TransformConfig.NORMALIZE_MIN_VALUE, 
                                       normalize_max_value=TransformConfig.NORMALIZE_MAX_VALUE,
                                       to_tensor=to_tensor)
        
        self.device = self._get_torch_device() # where to run the pytorch model.

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    def run(self, *args, **kwargs):
        raise NotImplementedError
    
    def _check_input_shape(self, x):
        if x.shape[1] != 2 or len(x.shape) != 3:
            raise ValueError('Input shape must be (batch, 2, time) but got {}'.format(x.shape))
    
    def _check_single_input(sefl, x):
        if len(x) !=1:
            raise ValueError('Input must be a single batch but got {}'.format(len(x)))
        if x.shape[1] != 2 or len(x.shape) != 3:
            raise ValueError('Input shape must be (batch, 2, time) but got {}'.format(x.shape))
        
    def _get_torch_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        if torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')