import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

class ZeroUnderThreshold(nn.Module):
    def __init__(self, threshold = 0.15, *args, **kwargs):
        super(ZeroUnderThreshold, self).__init__()
        self.threshold = threshold
    def forward(self, x):
        out = x.copy()
        out[out<self.threshold] = 0
        return out


class ZeroUnderRandomThreshold(nn.Module):
    def __init__(self, lo_lim = 0.05, hi_lim = 0.2, *args, **kwargs):
        super(ZeroUnderRandomThreshold, self).__init__()
        self.lo_lim = lo_lim
        self.hi_lim = hi_lim

    def forward(self, x):
        if len(x.shape) !=3: raise ValueError('Input shape must be (batch, channels, time) but got {}'.format(x.shape))
        return np.where(x < np.random.uniform(low=self.lo_lim, high=self.hi_lim, size=(x.shape[0], 1, 1)), 0, x)


class FixValMinMaxNormalizer(nn.Module):
    def __init__(self, min_value=0, max_value=5, *args, **kwargs):
        super(FixValMinMaxNormalizer, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        out = x.copy()
        out = (out-self.min_value)/(self.max_value-self.min_value)
        return out


class NumpyEchoToTensor(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NumpyEchoToTensor, self).__init__()
    
    def forward(self, x):
        return torch.from_numpy(x).float()


def get_transform(**kwargs):
    transformations = [
        ZeroUnderThreshold(threshold=kwargs['quiet_threshold']),
        FixValMinMaxNormalizer(min_value=kwargs['normalize_min_value'], max_value=kwargs['normalize_max_value']),
    ]
    if kwargs['to_tensor']:
        transformations.append(NumpyEchoToTensor())
    return transforms.Compose(transformations)