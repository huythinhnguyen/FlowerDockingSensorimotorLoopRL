import sys
import os
import torch
import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, List

from .setting import *
from .base import PerceptionBase
from .model_utils import UniEchoVGG_PoseEstimator
from .onset_distance_dectector import ClosestPeakDistance
from .presence_detector import LogDecayProfilePresenceDector

SAVED_MODEL_PATH =  ImportantPath.POSE_ESTIMATOR_MODEL


def find_wave_package_of_given_index(x: ArrayLike, index:int , min_width: int =None) -> Tuple[int, int]:
    if not index: return 0,0
    # if there's not zeros in x, raise error
    if np.all(x != 0):
        raise ValueError('Input must be quieted but no zero was found')
    # if index is out of range, raise error
    if len(x.shape) > 1:
        raise ValueError('Input must be a rank-1 array but got shape {}'.format(x.shape))
    if index < 0 or index >= len(x):
        raise ValueError('Index must be within the range of input but got {}'.format(index))
    if min_width is None: min_width = int(len(x)/64)
    # if x[index] != 0: find the left and right boundaries
    # else: find return the index as left and right boundaries --> this is the case where no wave package is found at given index.
    # you can conveniently check whether a wave package is found by checking the difference in left and right boundaries.
    # compute the width of the wave package
    # if the width is smaller than min_width, return the index as left and right boundaries --> No wave package is found.
    if x[index] == 0: return index, index
    # left = index
    # right = index
    # while left >= 0 and x[left] == 0: left -= 1
    # while right < len(x) and x[right] == 0: right += 1
    # if right - left < min_width: return index, index
    # return left, right
    left = np.where(x[:index] == 0)[0]
    right = np.where(x[index+1:] == 0)[0] + index + 1

    left_boundary = left[-1] if left.size > 0 else index
    right_boundary = right[0] if right.size > 0 else index

    if right_boundary - left_boundary < min_width:
        return index, index

    return left_boundary, right_boundary

def zero_outside_of_wave_package(x: ArrayLike, left_wave_package_boundaries: Tuple[int, int],
                    right_wave_package_boundaries: Tuple[int, int],
                    emission_width_index: int,):
    x_cleanup = x.copy()
    # return left and right boundaries as union of all boundaries except for emission wave package
    left = min(left_wave_package_boundaries[0], right_wave_package_boundaries[0])
    right = max(left_wave_package_boundaries[1], right_wave_package_boundaries[1])
    x_cleanup[:,:, right:] = 0
    if left > emission_width_index:
        x_cleanup[:,:,emission_width_index:left] = 0
    return x_cleanup

def find_first_zero_from_index(x: ArrayLike, index: int) -> int:
    if np.all(x != 0):
        raise ValueError('Input must be quieted but no zero was found')
    if len(x.shape) > 1:
        raise ValueError('Input must be a rank-1 array but got shape {}'.format(x.shape))
    if index < 0 or index >= len(x):
        raise ValueError('Index must be within the range of input but got {}'.format(index))
    if x[index] == 0: return index
    right = np.where(x[index+1:] == 0)[0] + index + 1
    right_index = right[0] if right.size > 0 else index
    return right_index
        

def find_idx_from_distance_array(distance: float, compressed_distance: ArrayLike) -> int:
    return np.argmin(np.abs(compressed_distance - distance))


# NaiveOneShotFlowerPoseEstimator
# Main Method: run()
# Input: x: ArrayLike
# Output: Tuple[distance: float (meter), azimuth: float (radian), orientation: float (radian)]
# Description:
#   - Check whether something is presence prior to pose estimation
#   - If something is presence, run the pose estimation
#   - If nothing is presence, return None, None, None
#   - The pose estimation is done by a pre-trained model
#   - Very accurate when there is only one single flower present in the input
#   - Inaccurate when there are multiple flowers present in the input (that's why it's naive)
class NaiveOneShotFlowerPoseEstimator(PerceptionBase):
    def __init__(self, to_tensor:bool=True, check_presence:bool=True, cache_inputs:bool=False):
        super(NaiveOneShotFlowerPoseEstimator, self).__init__(to_tensor)
        self.presence_detector = LogDecayProfilePresenceDector()
        self.model = UniEchoVGG_PoseEstimator(input_echo_length=TransformConfig.ENVELOPE_LENGTH, dropout=False)
        self.model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.check_presence = check_presence
        self.cache = {}
        self.cache_inputs = cache_inputs

    def run(self, x: ArrayLike) -> Tuple[float, float, float]:
        self._check_single_input(x)
        # check whether something is presence prior to pose estimation
        if self.check_presence:
            if not self.presence_detector(x):
                if self.cache_inputs: self.cache['inputs'] = np.zeros((1,2,TransformConfig.ENVELOPE_LENGTH))
                return None, None, None
        inputs = self.transform(x)
        d_pred, a_pred, o_pred = self.model(inputs.to(self.device))
        distance = d_pred.item()
        azimuth = a_pred.item()
        orientation = o_pred.item()
        if self.cache_inputs:
            self.cache['inputs'] = inputs.numpy(force=True)
        return distance, azimuth, orientation


# OnsetOneShotFlowerPoseEstimator
# Main Method: run()
# Input: x: ArrayLike
# Output: Tuple[distance: float (meter), azimuth: float (radian), orientation: float (radian)]
# Description:
#   - Check whether something is presence prior to pose estimation
#   - If something is presence, run the pose estimation
#   - If nothing is presence, return None, None, None
#   - oneset distance is used to select main wave package of the input, zero out all wave-package except for the emission
#   - The pose estimation is done by a pre-trained model
#   - This potentially can solve the issue of multiple flowers present in the input by focusing on the onset wave package.
#   - However, since the onset_distance_detector will find the the strongest peaks first prior to ranking them by distance,
#     it is possible that the strongest peak belongs to farther objects when the farther object has smaller azimuth while closer object has large azimuth.
#     Pose estimation will still be accurate, however, it will focus on the strongest signal instead of the closest signal.

class OnsetOneShotFlowerPoseEstimator(PerceptionBase):
    def __init__(self, to_tensor:bool=False, check_presence:bool=True, cache_inputs:bool=False):
        super(OnsetOneShotFlowerPoseEstimator, self).__init__(to_tensor)
        self.presence_detector = LogDecayProfilePresenceDector()
        self.onset_distance_detector = ClosestPeakDistance()
        self.model = UniEchoVGG_PoseEstimator(input_echo_length=TransformConfig.ENVELOPE_LENGTH, dropout=False)
        self.model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.check_presence = check_presence
        self.emission_width_index = LogDecayProfile.EMISSION_WIDTH_INDEX
        self.compressed_distance = TransformConfig.DISTANCE_ENCODING
        self.cache = {}
        self.cache_inputs = cache_inputs

    def run(self, x: ArrayLike) -> Tuple[float, float, float]:
        self._check_single_input(x)
        # check whether something is presence prior to pose estimation
        if self.check_presence: 
            if not self.presence_detector(x):
                if self.cache_inputs: self.cache['inputs'] = np.zeros((1,2,TransformConfig.ENVELOPE_LENGTH))
                return None, None, None
        onset_distance_idx = self.onset_distance_detector(x, ouput_index=True)
        left_wave_package_boundaries = find_wave_package_of_given_index(self.transform(x[0,0,:]), onset_distance_idx)
        right_wave_package_boundaries = find_wave_package_of_given_index(self.transform(x[0,1,:]), onset_distance_idx)
        x_clean = zero_outside_of_wave_package(self.transform(x), left_wave_package_boundaries, right_wave_package_boundaries, self.emission_width_index)
        inputs = torch.from_numpy(x_clean).float()
        d_pred, a_pred, o_pred = self.model(inputs.to(self.device))
        distance = d_pred.item()
        azimuth = a_pred.item()
        orientation = o_pred.item()
        if self.cache_inputs:
            self.cache['onset_distance_idx'] = onset_distance_idx
            self.cache['left_wave_package_boundaries'] = left_wave_package_boundaries
            self.cache['right_wave_package_boundaries'] = right_wave_package_boundaries
            self.cache['inputs'] = inputs.numpy(force=True)
        return distance, azimuth, orientation


# OnsetOneShotFlowerPoseEstimator
# Main Method: run()
# Input: x: ArrayLike
# Output: Tuple[distance: float (meter), azimuth: float (radian), orientation: float (radian)]
# Description:
#   - Check whether something is presence prior to pose estimation
#   - If something is presence, run the pose estimation
#   - If nothing is presence, return None, None, None
#   - oneset distance is used to select main wave package of the input, zero out all wave-package except for the emission
#   - The pose estimation is done by a pre-trained model
#   - This is a modified version of OnsetOneShotFlowerPoseEstimator. Here is the key difference:
#      + First, NaiveOneShotFlowerPoseEstimator is used to estimate the distance.
#      + Second, predicted distance and onset distance will be ranked.
#         
class TwoShotFlowerPoseEstimator(PerceptionBase):
    def __init__(self, to_tensor=False, check_presence=True, cache_inputs=False):
        super(TwoShotFlowerPoseEstimator, self).__init__(to_tensor)
        self.onset_distance_detector = ClosestPeakDistance()
        self.presence_detector = LogDecayProfilePresenceDector()
        self.model = UniEchoVGG_PoseEstimator(input_echo_length=TransformConfig.ENVELOPE_LENGTH, dropout=False)
        self.model.load_state_dict(torch.load(SAVED_MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.check_presence = check_presence
        self.emission_width_index = LogDecayProfile.EMISSION_WIDTH_INDEX
        self.compressed_distances = TransformConfig.DISTANCE_ENCODING
        self.cache = {}
        self.cache_inputs = cache_inputs

    
    def clean_up_input_for_second_shot(self, x, pred_distance, onset_distance, compressed_distances):
        new_x = self.transform(x)
        left_wpb_onset = find_wave_package_of_given_index(new_x[0,0,:], self.onset_distance_detector.closest_peak_idx)
        right_wpb_onset = find_wave_package_of_given_index(new_x[0,1,:], self.onset_distance_detector.closest_peak_idx)
        pred_distance_idx =  find_idx_from_distance_array(pred_distance, compressed_distances)
        if self.cache_inputs:
            self.cache['pred_distance_idx'] = pred_distance_idx
            self.cache['final_pred_distance_idx'] = pred_distance_idx
            self.cache['onset_distance_idx'] = self.onset_distance_detector.closest_peak_idx
            self.cache['final_onset_distance_idx'] = self.onset_distance_detector.closest_peak_idx
        left_wpb_predicted = find_wave_package_of_given_index(new_x[0,0,:], pred_distance_idx)
        right_wpb_predicted = find_wave_package_of_given_index(new_x[0,1,:], pred_distance_idx)
        # scenario 1: onset distance is approximately equal to predicted distance
        # scenario 3: onset distance is smaller than predicted distance
        if ( left_wpb_onset == left_wpb_predicted ) or ( right_wpb_onset == right_wpb_predicted ) or (onset_distance < pred_distance):
            return zero_outside_of_wave_package(self.transform(x), left_wpb_onset, right_wpb_onset, self.emission_width_index)
        # scenario 2: onset distance is larger than predicted distance
        # zeros out all the wave package from the predicted distance. 
        # --> Need to make sure that Im not cutting in the current wave package that the predicted distance is in.
        # then repeat the steps in OnsetOneshotFlowerPoseEstimator (calc onset_distance, find left and right wave package boundaries, zero out all wave package except for emission)
        ##################################################
        pred_distance_idx =  max([find_first_zero_from_index(new_x[0,0,:], pred_distance_idx), find_first_zero_from_index(new_x[0,1,:], pred_distance_idx)] )
        new_x = x.copy()
        new_x[:,:,pred_distance_idx:] = 0
        onset_distance_idx = self.onset_distance_detector(new_x, ouput_index=True)
        left_wpb = find_wave_package_of_given_index(self.transform(new_x[0,0,:]), onset_distance_idx)
        right_wpb = find_wave_package_of_given_index(self.transform(new_x[0,1,:]), onset_distance_idx)
        return zero_outside_of_wave_package(self.transform(x), left_wpb, right_wpb, self.emission_width_index)


    def run(self, x):
        self._check_single_input(x)
        # check whether something is presence prior to pose estimation
        if self.check_presence: 
            if not self.presence_detector(x):
                if self.cache_inputs: self.cache['inputs'] = np.zeros((1,2,TransformConfig.ENVELOPE_LENGTH))
                return None, None, None
        ############################################################
        # First shot
        ############################################################
        inputs = torch.from_numpy(self.transform(x)).float()
        pred = self.model(inputs.to(self.device))
        distance = pred[0].item()
        ###### SOME CODE HERE ######
        # Checking the results of the first shot
        # Clean up the input x to remove interfering objects.
        # Do the second shot.
        onset_distance = self.onset_distance_detector(x)
        x_clean = self.clean_up_input_for_second_shot(x, pred_distance=distance, onset_distance=onset_distance, compressed_distances=self.compressed_distances)
        ############################################################
        # Second shot
        ############################################################
        inputs = torch.from_numpy(x_clean).float()
        d_pred, a_pred, o_pred = self.model(inputs.to(self.device))
        distance = d_pred.item()
        azimuth = a_pred.item()
        orientation = o_pred.item()

        if self.cache_inputs:
            self.cache['inputs'] = inputs.numpy(force=True)
            self.cache['final_pred_distance_idx'] = find_idx_from_distance_array(distance, self.compressed_distances)
            self.cache['final_onset_distance_idx'] = self.onset_distance_detector.closest_peak_idx
        return distance, azimuth, orientation