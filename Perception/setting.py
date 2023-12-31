import os
import sys

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike

import Perception

from Sensors.FlowerEchoSimulator.Spatializer import Compressing, SeriesEncoding

@dataclass
class TransformConfig:
    QUIET_THRESHOLD: float = 0.15
    ENVELOPE_LENGTH: int = 512
    NORMALIZE_MIN_VALUE: float = 0.0
    NORMALIZE_MAX_VALUE: float = 5.0
    DISTANCE_ENCODING: ArrayLike = Compressing.Downsample512()(SeriesEncoding.DISTANCE_ENCODING)


@dataclass
class LogDecayProfile:
    EMISSION_PEAK_ENCODED_INDEX: float = 0.026
    EMISSION_WIDTH_ENCODED_INDEX: float = 0.12
    EMISSION_PEAK_INDEX: int = int( EMISSION_PEAK_ENCODED_INDEX * TransformConfig.ENVELOPE_LENGTH )
    EMISSION_WIDTH_INDEX: int = int( EMISSION_WIDTH_ENCODED_INDEX * TransformConfig.ENVELOPE_LENGTH )
    PLATEAU_LENGTH: int = EMISSION_PEAK_INDEX + int(TransformConfig.ENVELOPE_LENGTH / 256) + 1
    NORMALIZED_NOISE_LEVEL: float = 0.03 # from 0 to 1
    DECAY_RATE: int = 30 # Higher to hug the envelope stricter (faster decay)
    PROFILE: ArrayLike = np.concatenate(( np.ones(PLATEAU_LENGTH), 
                                         (1-NORMALIZED_NOISE_LEVEL)* \
                                            np.power(np.e, -np.linspace(0, 30, TransformConfig.ENVELOPE_LENGTH - PLATEAU_LENGTH ))+NORMALIZED_NOISE_LEVEL ))

@dataclass
class PresenceDetectorConfig:
    DETECTION_THRESHOLD: float = 2.
    GAIN: float = 1. # Multiplier for presence score prior to applying detection threshold
    SCORE_MODE = 'sum' # 'sum' or 'max'


@dataclass
class ImportantPath:
    REPO_PATH = Perception.REPO_PATH
    POSE_ESTIMATOR_MODEL = os.path.join(REPO_PATH, 'saved_models', 'flower_pose_estimator_v1.pt')
