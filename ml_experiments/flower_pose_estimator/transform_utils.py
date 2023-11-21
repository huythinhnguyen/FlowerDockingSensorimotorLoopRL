import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

class ZeroUnderThreshold(nn.Module):
    def __init__(self, threshold = 0.15):
        super(ZeroUnderThreshold, self).__init__()
        self.threshold = threshold
    def forward(self, x):
        out = x.copy()
        out[out<self.threshold] = 0
        return out

class ZeroUnderSmoothThreshold(nn.Module):
    def __init__(self, threshold = 0.15, strict_threshold = 0.05, gamma=2):
        super(ZeroUnderSmoothThreshold, self).__init__()
        if strict_threshold>threshold: raise ValueError('strict_threshold must be smaller than threshold but got {}>{}'.format(strict_threshold, threshold))
        self.threshold = threshold
        self.strict_threshold = strict_threshold
        self.gamma = gamma

    def forward(self, x):
        out = x.copy()
        out[out<self.strict_threshold] = 0
        out[out<self.threshold]*=((out[out<self.threshold]-self.strict_threshold)/  \
                                  (self.threshold-self.strict_threshold))\
                                    **(self.gamma - 1)
        return out
    
class ZeroUnderRandomThreshold(nn.Module):
    def __init__(self, lo_lim = 0.05, hi_lim = 0.2):
        super(ZeroUnderRandomThreshold, self).__init__()
        self.lo_lim = lo_lim
        self.hi_lim = hi_lim

    def forward(self, x):
        if len(x.shape) !=3: raise ValueError('Input shape must be (batch, channels, time) but got {}'.format(x.shape))
        return np.where(x < np.random.uniform(low=self.lo_lim, high=self.hi_lim, size=(x.shape[0], 1, 1)), 0, x)


class PercentileNormalizer(nn.Module):
    def __init__(self, percentile_=98):
        super(PercentileNormalizer, self).__init__()
        self.percentile_ = percentile_

    def forward(self, x):
        if len(x.shape) !=3: raise ValueError('Input shape must be (batch, channels, time) but got {}'.format(x.shape))
        out = x.copy()
        for i in range(out.shape[1]):
            upper_percentile_ = np.percentile(out[:,i,:], self.percentile_, axis=1).reshape(-1,1)
            lower_percentile_ = np.percentile(out[:,i,:], 100-self.percentile_, axis=1).reshape(-1,1)
            out[:,i,:] = (out[:,i,:]-lower_percentile_)/(upper_percentile_-lower_percentile_)
        return out

class FixValMinMaxNormalizer(nn.Module):
    def __init__(self, min_value=0, max_value=5):
        super(FixValMinMaxNormalizer, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        out = x.copy()
        out = (out-self.min_value)/(self.max_value-self.min_value)
        return out

class AdjsutGamma(nn.Module):
    def __init__(self, gamma=0.5, gain=1.):
        super(AdjsutGamma, self).__init__()
        self.gamma = gamma
        self.gain = gain

    def forward(self, x):
        if len(x.shape) !=3: raise ValueError('Input shape must be (batch, channels, time) but got {}'.format(x.shape))
        out = x.copy()
        for i in range(out.shape[1]):
            out[:,i,:] = self.gain*out[:,i,:]**(self.gamma)
        return out
    
class NumpyEchoToTensor(nn.Module):
    def __init__(self):
        super(NumpyEchoToTensor, self).__init__()
    
    def forward(self, x):
        return torch.from_numpy(x).float()


def getTrainTransfrom():
    return transforms.Compose([
    ZeroUnderRandomThreshold(lo_lim=0.05, hi_lim=0.25),
    FixValMinMaxNormalizer(min_value=0., max_value=5.),
    NumpyEchoToTensor()
])

def getEvalTransfrom(quiet_threshold = 0.):
    return transforms.Compose([
    ZeroUnderThreshold(threshold=quiet_threshold),
    FixValMinMaxNormalizer(min_value=0., max_value=5.),
    NumpyEchoToTensor()
])
