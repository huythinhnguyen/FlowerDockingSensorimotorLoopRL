import numpy as np
from numpy.typing import ArrayLike

from dataclasses import dataclass, field


@dataclass
class AcousticSetting:
    SAMPLE_FREQ: int = 3e5
    SPEED_OF_SOUND: int = 340
    RAW_DATA_LENGTH: int = 7000
    OUTWARD_SPREAD = 1
    INWARD_SPREAD = 0.5
    AIR_ABSORPTION = 1.31

@dataclass
class CompressionSetting:
    QUIET_THRESHOLD: float = 0.2
    N_SAMPLE: int = 125 # number of samples to be averaged with Uniform Subsampling
    METHOD: str = 'Uniform Subsampling'
    NORMALIZED: bool = False
    QUIET_NORMALIZER: float = 5e-2

@dataclass
class SeriesEncoding:
    DISTANCE_ENCODING: ArrayLike = np.arange(AcousticSetting.RAW_DATA_LENGTH) \
        * 0.5 * (1/AcousticSetting.SAMPLE_FREQ) * AcousticSetting.SPEED_OF_SOUND
    if CompressionSetting.METHOD == 'Uniform Subsampling':
        COMPRESSED_DISTANCE_ENCODING = np.mean(DISTANCE_ENCODING.reshape(-1,CompressionSetting.N_SAMPLE), axis=1)

@dataclass
class ObjectID:
    pole: int = 1
    plant: int = 2
    flower: int = 3

@dataclass
class FlowerDataConfig:
    DISTANCE_REFERENCE = [0.2, 0.6, 1.0]
    ORIENTATION_REFERENCE: ArrayLike = np.arange(-180, 185, 5).astype(int)
    NECK_ANGLE_REFERENCE: ArrayLike = np.concatenate((
        np.arange(-90., -70. + 10., 10.),
        np.arange(-60., -35. + 5., 5.),
        np.arange(-30., 30. + 2., 2.),
        np.arange(35., 60. + 5., 5.),
        np.arange(70., 90. + 10., 10.)
    )).astype(int)
    PERIPHERAL_LIMIT: int = 110
    

@dataclass
class ViewerSetting:
    #OLD VERSION  ||| FOV_LINEAR: float = 3. ||| FOV_ANGULAR: float = np.pi*(7/9)
    FOVS = [(1., np.radians(110)*2), (4., np.radians(90)*2)] #[(1., np.radians(110)*2), (4., np.radians(90)*2)])
    COLLISION_THRESHOLD: float = 0.15
    FOV_OBSCURING: bool = False # NOT IMPLEMENTED YET IN THIS VERSION
    

@dataclass
class GainCurveProperty: #MODELING OF ROSE CURVE
    NUMBER_OF_PEDALS: float = 2.2
    ROSE_CURVE_B = 1.
    LEFT_EAR_FIT_ANGLE = (3/18)*np.pi
    RIGHT_EAR_FIT_ANGLE=-(4/18)*np.pi
    COLLECTION_ANGLE_LIMIT = 45 # must be in degree!!!
    MIN_GAIN_DB = -40

### COCHLEA SETTING:
@dataclass
class CochleaSetting:
    EMISSION_FREQ: int = 4.2e4
    SAMPLING_FREQ: int = 3e5
    BROADBAND_SPEC = {'order':4, 'low':2e4, 'high':8e4}
    GAMMATONE_BANKSIZE : int = 1
    EXP_COMPRESSION_POWER: float = 0.4
    LOWPASS_FREQ: int = 1e3
    BIDIRECTIONAL_FILTERING: bool = True
