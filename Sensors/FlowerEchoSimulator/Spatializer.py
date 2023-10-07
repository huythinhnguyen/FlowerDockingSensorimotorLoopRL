from typing import Dict, List, Tuple, Any, AnyStr
from numpy.typing import ArrayLike
import sys
import os
import numpy as np


REPO_NAME = 'FlowerDockingSensorimotorLoopRL'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)

import multiprocessing as mp
from .Setting import FlowerDataConfig, AcousticSetting, SeriesEncoding
from .Utils import get_snippet, get_emission_snippet, get_noise_data_array
from . import Viewer
from . import Cochlea
from . import Compressing

FLOWER_DISTANCE_REFERENCE = FlowerDataConfig.DISTANCE_REFERENCE
FLOWER_ORIENTATION_REFERENCE = FlowerDataConfig.ORIENTATION_REFERENCE
NECK_ANGLE_REFERENCE = FlowerDataConfig.NECK_ANGLE_REFERENCE
PERIPHERAL_NECK_ANGLE_LIMIT = FlowerDataConfig.PERIPHERAL_LIMIT

DATA_LENGTH = AcousticSetting.RAW_DATA_LENGTH
DISTANCE_ENCODING = SeriesEncoding.DISTANCE_ENCODING

def wrapToPi(x):
    if type(x) == list: x = np.asarray(x).astype(np.float32)
    return np.mod(x + np.pi, 2*np.pi) - np.pi
def wrapTo180(x):
    if type(x) == list: x = np.asarray(x).astype(np.float32)
    return np.mod(x + 180, 360) - 180

class OrientationalRetriever:
    def __init__(self):
        self.random_retreiving = True
        self.distance_interpolation_marks = [0.4, 0.8]
        self.distance_interpolation_power = 1.2
        self.noise_level = 0.5 # float from 0 to 1

    def get_emission_snippet(self):
        data = get_emission_snippet()
        if self.random_retreiving:
            return data['left'][:,np.random.randint(0, data['left'].shape[1])].reshape(-1,DATA_LENGTH),\
                    data['right'][:,np.random.randint(0, data['right'].shape[1])].reshape(-1,DATA_LENGTH)
        return np.mean(data['left'], axis=1).reshape(-1,DATA_LENGTH),\
                np.mean(data['right'], axis=1).reshape(-1,DATA_LENGTH)
    

    def get_noise_sample(self):
        data = get_noise_data_array()
        if self.random_retreiving:
            return self.noise_level*data['left'][:,np.random.randint(0, data['left'].shape[1])].reshape(-1,DATA_LENGTH),\
                    self.noise_level*data['right'][:,np.random.randint(0, data['right'].shape[1])].reshape(-1,DATA_LENGTH)
        return self.noise_level*np.mean(data['left'], axis=1).reshape(-1,DATA_LENGTH),\
                self.noise_level*np.mean(data['right'], axis=1).reshape(-1,DATA_LENGTH)



    def get_echo_snippets(self, objects:np.ndarray):
    # objects needs too be a numpy array of shape (n, 3) where each row n is an objects.
    # the three columns are distance (r) [distance between bat and object],
    #                       azimuth (theta) [relative angles from object to bat],
    #                       orientation (phi) [relative orientation of object to bat] 
        echo_snippets_left = np.asarray([], dtype=np.float32).reshape(0, DATA_LENGTH)
        echo_snippets_right = np.asarray([], dtype=np.float32).reshape(0, DATA_LENGTH)
        for obj in objects:
            distance, neck_angle, orientation = obj[:3]
            snip_left, snip_right = self._get_propagated_echo_snippet(distance, neck_angle, orientation)
            echo_snippets_left = np.vstack((echo_snippets_left, snip_left))
            echo_snippets_right = np.vstack((echo_snippets_right, snip_right))
        return echo_snippets_left, echo_snippets_right
    
    def _get_propagated_echo_snippet(self, distance:float, neck_angle:float, orientation:float):
        # 1. Solve for which references we need to use
        #   - Reference distances will be round down to the nearest smaller value in FLOWER_DISTANCE_REFERENCE
        #   - Orientation will be referenced from an upper_reference_orient and lower_reference_orient
        #   - The upper_reference_orient is rounded up to the 5 degrees
        #   - The lower_reference_orient is rounded down to the 5 degrees
        #   - neck_angle referenced from the nearest upper and lower from NECK_ANGLE_REFERENCE
        #   - if neck_angle is outside of -90 to 90 degrees, reference from the closes 90.
        #       - we can just linear extrapolate so that echoes will be zeros when neck_angle = PERIPHERAL_NECK_ANGLE_LIMIT
        # 2. Interpolate neck_angles:
        #   - For each combination, (reference_distance, upper_reference_orient) 
        #                       and (reference_distance, lower_reference_orient)
        #   - Linear interpolate echo_snippets between neck_angles
        # 3. Interpolate orientation:
        #   - Linear interpolate echo_snippets between orientation
        # 4. Propagate echo_snippets
        references = self._get_references(distance, neck_angle, orientation)
        snip_lower_orient, snip_upper_orient = self._get_interpolated_snippet_from_ref_neck_angles(
            references, neck_angle)
        snip_left, snip_right = self._interpolate_snippet_between_orientations(
            snip_upper_orient, snip_lower_orient, references, orientation)
        snip_left, snip_right = self._propagate_snippet(
            snip_left, snip_right, references, distance, neck_angle, orientation)

        return snip_left, snip_right
    
    def get_snippet_attenuation_mask(self, snippet:np.ndarray, original_distance:float, target_distance:float):
        # find index of the first non-zero and the last non-zero of the snippet
        start_index = np.argmax(snippet != 0)
        end_index = snippet.shape[0] - np.argmax(snippet[::-1] != 0) - 1
        from_dist_array = np.arange(start_index, end_index+1) \
            * 0.5 * (1/AcousticSetting.SAMPLE_FREQ) * AcousticSetting.SPEED_OF_SOUND

        orignal_distance_index = np.searchsorted(DISTANCE_ENCODING, original_distance, side='right') - 1
        target_distance_index = np.searchsorted(DISTANCE_ENCODING, target_distance, side='right') - 1
        diff_index = target_distance_index - orignal_distance_index
        to_dist_array = np.arange(start_index+diff_index,end_index+diff_index+1) \
            * 0.5 * (1/AcousticSetting.SAMPLE_FREQ) * AcousticSetting.SPEED_OF_SOUND

        atmospheric = np.power(10, AcousticSetting.AIR_ABSORPTION*2*(original_distance - target_distance)/20)
        spreading = np.divide(from_dist_array , to_dist_array) \
            ** (AcousticSetting.OUTWARD_SPREAD + AcousticSetting.INWARD_SPREAD)
        attenuation = atmospheric * spreading
        return np.concatenate((np.zeros(start_index), attenuation, np.zeros(DATA_LENGTH - (end_index+1)) ))


    def _propagate_snippet_base(self, snip_left:np.ndarray, snip_right:np.ndarray, original_distance:float, target_distance:float):
        if not np.all(snip_left==0):
            attenuation = self.get_snippet_attenuation_mask(snippet=snip_left, original_distance=original_distance, target_distance=target_distance)
            snip_left = snip_left * attenuation
        if not np.all(snip_right==0):
            attenuation = self.get_snippet_attenuation_mask(snippet=snip_right, original_distance=original_distance, target_distance=target_distance)
            snip_right = snip_right * attenuation
        return self._move_snippets(snippet_left=snip_left, snippet_right=snip_right,
                                   original_distance=original_distance, target_distance=target_distance)

    def _propagate_snippet(self, snip_left:np.ndarray, snip_right:np.ndarray, references:dict,
                            target_distance:float, target_neck_angle:float, target_orientation:float):
        original_distance = references['distance']
        need_interpolation = 0
        ### Maybe need to block more elegant
        if target_distance < np.min(FLOWER_DISTANCE_REFERENCE): need_interpolation = -1
        elif target_distance > self.distance_interpolation_marks[0] and target_distance < FLOWER_DISTANCE_REFERENCE[1]: need_interpolation = 1
        elif target_distance > self.distance_interpolation_marks[1] and target_distance < FLOWER_DISTANCE_REFERENCE[2]: need_interpolation = 2
        if need_interpolation==0: return self._propagate_snippet_base(snip_left, snip_right, original_distance, target_distance)

        if need_interpolation==-1:
            return self._propagate_snippet_base(snip_left, snip_right, np.min(FLOWER_DISTANCE_REFERENCE), target_distance)

        head_snip_left, head_snip_right = self._propagate_snippet_base(snip_left, snip_right, original_distance, target_distance)
        tail_ref_distance = FLOWER_DISTANCE_REFERENCE[FLOWER_DISTANCE_REFERENCE.index(original_distance)+1]
        tail_references = references.copy()
        tail_references['distance'] = tail_ref_distance
        tail_snip_lower_orient, tail_snip_upper_orient = self._get_interpolated_snippet_from_ref_neck_angles(
            tail_references, target_neck_angle)
        tail_snip_left, tail_snip_right = self._interpolate_snippet_between_orientations(
            tail_snip_upper_orient, tail_snip_lower_orient, tail_references, target_orientation)
        
        tail_snip_left, tail_snip_right = self._propagate_snippet_base(tail_snip_left, tail_snip_right, tail_ref_distance, target_distance)

        lower_ref_dist = self.distance_interpolation_marks[need_interpolation-1]
        lower_multiplier, upper_multiplier = self._generate_power_interpolation_factor(
            lower_ref=lower_ref_dist, upper_ref=tail_ref_distance, target=target_distance)
        snip_left = head_snip_left * lower_multiplier + tail_snip_left * upper_multiplier
        snip_right = head_snip_right * lower_multiplier + tail_snip_right * upper_multiplier
        return snip_left, snip_right

    def _move_snippets(self, snippet_left: np.ndarray, snippet_right: np.ndarray, original_distance: float, target_distance: float):
        original_distance_index = np.searchsorted(DISTANCE_ENCODING, original_distance, side='right') - 1
        target_distance_index = np.searchsorted(DISTANCE_ENCODING, target_distance, side='right') - 1
        move_forward = target_distance >= original_distance
        diff_index = np.abs(target_distance_index - original_distance_index)
        result_left, result_right = np.zeros(DATA_LENGTH,), np.zeros(DATA_LENGTH,)
        if move_forward:
            result_left[diff_index:] = snippet_left[:DATA_LENGTH-diff_index]
            result_right[diff_index:] = snippet_right[:DATA_LENGTH-diff_index]
        else:
            result_left[:DATA_LENGTH-diff_index] = snippet_left[diff_index:]
            result_right[:DATA_LENGTH-diff_index] = snippet_right[diff_index:]
        return result_left, result_right

    def _interpolate_snippet_between_orientations(self, snip_upper_orient:dict, snip_lower_orient:dict,
                                                  references:dict, target_orientation:float):
        # generate linear interpolation factor
        if references['lower_orient'] > references['upper_orient'] and references['lower_orient'] < -1*references['upper_orient']:
            lower_multiplier, upper_multiplier = self._generate_linear_interpolation_factor(
                lower_ref=references['lower_orient'], upper_ref=-1*references['upper_orient'], target=target_orientation)
        else:
            lower_multiplier, upper_multiplier = self._generate_linear_interpolation_factor(
                lower_ref=references['lower_orient'], upper_ref=references['upper_orient'], target=target_orientation)
        # interpolate
        #print('lower_orient={:.1f}, upper_orient={:.1f}, lower_multiplier={:.2f}, upper_multiplier={:.2f}'.format(references['lower_orient'], references['upper_orient'], lower_multiplier, upper_multiplier))
        snip_left = snip_lower_orient['left'] * lower_multiplier \
                    + snip_upper_orient['left'] * upper_multiplier
        snip_right = snip_lower_orient['right'] * lower_multiplier \
                    + snip_upper_orient['right'] * upper_multiplier
        return snip_left, snip_right
    
    def _get_interpolated_snippet_from_ref_neck_angles(self, references:dict, neck_angle:float):
        if np.abs(neck_angle) >= PERIPHERAL_NECK_ANGLE_LIMIT:
            return {'left':np.zeros(DATA_LENGTH,), 'right':np.zeros(DATA_LENGTH,)},\
                    {'left':np.zeros(DATA_LENGTH,), 'right':np.zeros(DATA_LENGTH,)}
        if references['lower_neck_angle'] is None:
            snip_lower_orient_lower_neck_angle= {'left':np.zeros(DATA_LENGTH,),
                                                'right':np.zeros(DATA_LENGTH,)}
            snip_upper_orient_lower_neck_angle = {'left':np.zeros(DATA_LENGTH,),
                                                'right':np.zeros(DATA_LENGTH,)}
            references['lower_neck_angle'] = -1*PERIPHERAL_NECK_ANGLE_LIMIT
        else:
            snip_lower_orient_lower_neck_angle = self._snippet_retriever(distance=references['distance'],
                                                orientation=references['lower_orient'],
                                                neck_angle=references['lower_neck_angle'])
            snip_upper_orient_lower_neck_angle = self._snippet_retriever(distance=references['distance'],
                                                orientation=references['upper_orient'],
                                                neck_angle=references['lower_neck_angle'])
        if references['upper_neck_angle'] is None:
            snip_lower_orient_upper_neck_angle = {'left':np.zeros(DATA_LENGTH,),
                                                'right':np.zeros(DATA_LENGTH,)}
            snip_upper_orient_upper_neck_angle = {'left':np.zeros(DATA_LENGTH,),
                                                'right':np.zeros(DATA_LENGTH,)}
            references['upper_neck_angle'] = PERIPHERAL_NECK_ANGLE_LIMIT
        else:
            snip_lower_orient_upper_neck_angle = self._snippet_retriever(distance=references['distance'],
                                                            orientation=references['lower_orient'],
                                                            neck_angle=references['upper_neck_angle'])
            snip_upper_orient_upper_neck_angle = self._snippet_retriever(distance=references['distance'],
                                                            orientation=references['upper_orient'],
                                                            neck_angle=references['upper_neck_angle'])
        # generate linear interpolation factor
        lower_multiplier, upper_multiplier = self._generate_linear_interpolation_factor(
            lower_ref=references['lower_neck_angle'], upper_ref=references['upper_neck_angle'], target=neck_angle)
        # interpolate
        snip_lower_orient, snip_upper_orient = {},{}

        snip_lower_orient['left'] = snip_lower_orient_lower_neck_angle['left'] * lower_multiplier \
                                + snip_lower_orient_upper_neck_angle['left'] * upper_multiplier
        snip_lower_orient['right'] = snip_lower_orient_lower_neck_angle['right'] * lower_multiplier \
                                + snip_lower_orient_upper_neck_angle['right'] * upper_multiplier
        snip_upper_orient['left'] = snip_upper_orient_lower_neck_angle['left'] * lower_multiplier \
                                + snip_upper_orient_upper_neck_angle['left'] * upper_multiplier
        snip_upper_orient['right'] = snip_upper_orient_lower_neck_angle['right'] * lower_multiplier \
                                + snip_upper_orient_upper_neck_angle['right'] * upper_multiplier
        return snip_lower_orient, snip_upper_orient

    def _generate_linear_interpolation_factor(self, lower_ref, upper_ref, target):
        upper_multiplier = (target - lower_ref) / (upper_ref - lower_ref)
        lower_multiplier = 1-upper_multiplier
        return lower_multiplier, upper_multiplier

    def _generate_power_interpolation_factor(self, lower_ref, upper_ref, target):
        upper_multiplier = (target - lower_ref) / (upper_ref - lower_ref)
        upper_multiplier = upper_multiplier**self.distance_interpolation_power
        lower_multiplier = 1-upper_multiplier
        return lower_multiplier, upper_multiplier

    def _snippet_retriever(self, distance:float, neck_angle:float, orientation:float):
        if not distance: distance = np.min(FLOWER_DISTANCE_REFERENCE)
        if np.abs(neck_angle) >= PERIPHERAL_NECK_ANGLE_LIMIT:
            return {'left':np.zeros(DATA_LENGTH,), 'right':np.zeros(DATA_LENGTH,)}
        raw = get_snippet(distance=distance, neck_angle=neck_angle, orientation=orientation)
        data = {}
        if self.random_retreiving:
            random_index = np.random.randint(0, raw['left'].shape[1])
            data['left'] = raw['left'][:, random_index].reshape(DATA_LENGTH,)
            data['right'] = raw['right'][:, random_index].reshape(DATA_LENGTH,)
        else:
            data['left'] = np.mean(raw['left'], axis=1)
            data['right'] = np.mean(raw['right'], axis=1)
        return data

    def _get_references(self, target_distance:float, target_neck_angle:float, target_orientation:float):
        references = {}
        references['distance'] = self._get_reference_distance(target_distance)
        references['lower_orient'], references['upper_orient'] = self._get_reference_orient(target_orientation)
        references['lower_neck_angle'], references['upper_neck_angle'] = self._get_reference_neck_angle(target_neck_angle)
        return references

    # --> passed test. correctness. 20% speed up.
    def _get_reference_distance(self, target_distance:float):
        min_dist = np.min(FLOWER_DISTANCE_REFERENCE)
        max_dist = np.max(FLOWER_DISTANCE_REFERENCE)
        if target_distance < min_dist: return None
        if target_distance >= max_dist: return max_dist
        index = np.searchsorted(FLOWER_DISTANCE_REFERENCE, target_distance, side='right') - 1
        return FLOWER_DISTANCE_REFERENCE[index]
    # --> passed test. correct and fast.
    def _get_reference_orient(self, target_orientation:float):
        index = np.searchsorted(FLOWER_ORIENTATION_REFERENCE, target_orientation, side='right') - 1
        lower_ref_orient, upper_ref_orient =  wrapTo180([FLOWER_ORIENTATION_REFERENCE[index], 
                                                         FLOWER_ORIENTATION_REFERENCE[index+1]])
        return lower_ref_orient, upper_ref_orient
    
    def _get_reference_neck_angle(self, target_neck_angle:float):
        min_neck_angle = np.min(NECK_ANGLE_REFERENCE)
        max_neck_angle = np.max(NECK_ANGLE_REFERENCE)
        if target_neck_angle < min_neck_angle: 
            return None, min_neck_angle
        if target_neck_angle >= max_neck_angle: 
            return max_neck_angle, None
        index = np.searchsorted(NECK_ANGLE_REFERENCE, target_neck_angle, side='right') - 1
        return NECK_ANGLE_REFERENCE[index], NECK_ANGLE_REFERENCE[index+1]
    

class RenderBase:
    def __init__(self, cache_dict:Dict=dict()):
        self.cache_dict = cache_dict

    def run(self, *args: Any, **kwds: Any) -> List[np.ndarray]:
        raise NotImplementedError('RenderBase.run() is not implemented.')

    def get_waveform(self):
        if 'waveform' in self.cache_dict.keys(): 
            return self.cache_dict['waveform']
        return {'left': [], 'right': []}
    
    def get_envelope(self) -> Dict[str, np.ndarray]:
        if 'envelope' in self.cache_dict.keys(): 
            return self.cache_dict['envelope']
        return {'left': [], 'right': []}
    
    def get_compress(self) -> Dict[str, np.ndarray]:
        if 'compress' in self.cache_dict.keys(): 
            return self.cache_dict['compress']
        return {'left': [], 'right': []}
    
    def __call__(self, *args: Any, **kwds: Any) -> List[np.ndarray]:
        return self.run(*args, **kwds)

    @property
    def waveform(self) -> Dict[str, np.ndarray]:
        return self.get_waveform()
    
    @property
    def waveform_left(self) -> np.ndarray:
        return self.get_waveform()['left']
    
    @property
    def waveform_right(self) -> np.ndarray:
        return self.get_waveform()['right']
    
    @property
    def envelope(self) -> Dict[str, np.ndarray]:
        return self.get_envelope()
    
    @property
    def envelope_left(self) -> np.ndarray:
        return self.get_envelope()['left']
    
    @property
    def envelope_right(self) -> np.ndarray:
        return self.get_envelope()['right']
    
    @property
    def compress(self) -> Dict[str, np.ndarray]:
        return self.get_compress()
    
    @property
    def compress_left(self) -> np.ndarray:
        return self.get_compress()['left']
    
    @property
    def compress_right(self) -> np.ndarray:
        return self.get_compress()['right']
    
    @property
    def snippet_left(self) -> np.ndarray:
        return self.cache_dict['snippet']['left']
    
    @property
    def snippet_right(self) -> np.ndarray:
        return self.cache_dict['snippet']['right']


class Render(RenderBase):
    def __init__(self, mode:AnyStr='compress', cache_dict:Dict=dict(),):
        super().__init__(cache_dict=cache_dict)
        if mode not in ['compress', 'envelope', 'waveform']: 
            raise ValueError('mode must be one of compress, envelope, or waveform')
        self.render_mode = mode
        self.fetch = OrientationalRetriever()
        self.viewer = Viewer.FieldOfView()
        self.compression_filter = Compressing.Subsample()
        self.cochlea_filter = Cochlea.CochleaFilter()

    def run(self, bat_pose: ArrayLike, cartesian_objects_matrix: ArrayLike,):
        if type(cartesian_objects_matrix)==list: cartesian_objects_matrix = np.asarray(cartesian_objects_matrix).reshape(-1,4)
        polar_objects_matrix = self.viewer(bat_pose, cartesian_objects_matrix)
        if self.viewer.output_angular_unit in ['radians', 'rad', 'radian']:
            polar_objects_matrix[:,1:3] = np.degrees(polar_objects_matrix[:,1:3])
        echo_snippets_left, echo_snippets_right = self.fetch.get_echo_snippets(polar_objects_matrix)
        emission_left, emission_right = self.fetch.get_emission_snippet()
        noise_left, noise_right = self.fetch.get_noise_sample()
        self.cache_dict['snippet'] = {'left': echo_snippets_left, 'right': echo_snippets_right}
        self.cache_dict['waveform'] = {'left': (np.sum(echo_snippets_left, axis=0) + emission_left + noise_left).reshape(DATA_LENGTH,),
                                       'right':(np.sum(echo_snippets_right, axis=0) + emission_right + noise_right).reshape(DATA_LENGTH,)
        }
        if self.render_mode == 'waveform':
            return {'left': self.cache_dict['waveform']['left'], 'right': self.cache_dict['waveform']['right']}
        self.cache_dict['envelope'] = {'left': self.cochlea_filter(self.cache_dict['waveform']['left']),
                                           'right': self.cochlea_filter(self.cache_dict['waveform']['right'])}
        if self.render_mode == 'envelope': return self.cache_dict['envelope']
        self.cache_dict['compress'] = {'left': self.compression_filter(self.cache_dict['envelope']['left']),
                                       'right':self.compression_filter(self.cache_dict['envelope']['right'])}
        if self.render_mode == 'compress': return self.cache_dict['compress']
        return Warning(f'No render mode {self.render_mode} found. Return the waveform instead.\n \
                       Access Render.cache_dict[`waveform`] to get the waveform.')


class UniformRetrierver:
    def __init__(self):
        pass
