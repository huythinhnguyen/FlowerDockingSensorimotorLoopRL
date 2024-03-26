from SensorimotorLoops import *
from .settings import *
import numpy as np
from numpy.typing import ArrayLike
from typing import List, Tuple, Optional, Any, Dict
from collections import deque

from Sensors.FlowerEchoSimulator.Spatializer import wrapToPi

def select_pose_estimator(estimator_type:str = 'Onset'):
    if estimator_type not in ['Naive', 'Onset', 'TwoShot']:
        raise ValueError('estimator_type must be either Naive, Onset or TwoShot. but got {}'.format(estimator_type))
    if estimator_type == 'Naive':
        return NaiveOneShotFlowerPoseEstimator(cache_inputs=True)
    if estimator_type == 'Onset':
        return OnsetOneShotFlowerPoseEstimator(cache_inputs=True)
    if estimator_type == 'TwoShot':
        return TwoShotFlowerPoseEstimator(cache_inputs=True)
    
def convert_polar_to_cartesian(bat_pose: ArrayLike,
                               flower_distance: float, flower_azimuth: float, flower_orientation: float,) -> ArrayLike:
    # return flower cartesian pose (x,y,theta)
    # bat_pose: (x,y,theta)
    # flower_distance: float
    # flower_azimuth: float
    # flower_orientation: float
    # return: (x,y,theta)
    flower_pose = np.zeros(3)
    flower_pose[0] = bat_pose[0] + flower_distance*np.cos(bat_pose[2] + flower_azimuth)
    flower_pose[1] = bat_pose[1] + flower_distance*np.sin(bat_pose[2] + flower_azimuth)
    flower_pose[2] = wrapToPi(bat_pose[2] + flower_azimuth + np.pi - flower_orientation)
    return flower_pose

class HomeInFlower:
    def __init__(self, *args, **kwargs):
        self.path_planner = DubinsDockZonePathPlanner()
        self.kinematic_converter = DubinsToKinematics()
        self.pose_estimator = select_pose_estimator(kwargs['estimator_type']) if 'estimator_type' in kwargs\
            else kwargs['pose_estimator'] if 'pose_estimator' in kwargs else select_pose_estimator() # convenient when need to feed in the pose_estimator.
        self.init_v = kwargs['init_v'] if 'init_v' in kwargs else 0.
        self.caching = kwargs['caching'] if 'caching' in kwargs else False
        self.distance_memory = deque(maxlen=DISTANCE_MEMORY_SIZE)
        self.v_course = np.asarray([]).astype(np.float32)
        self.w_course = np.asarray([]).astype(np.float32)
        self.is_course_from_estimation = False
        self.random_walk_quantities = {'L': RANDOM_WALK_QUANTITY, 'R': -RANDOM_WALK_QUANTITY}
        self.random_walk_turning_radius = kwargs['random_walk_turning_radius'] if 'random_walk_turning_radius' in kwargs \
            else BatKinematicParams.MIN_TURNING_RADIUS*2
        self.cache = {}
        
    def __call__(self, *args, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
        return self.step(*args, **kwargs)
    
    # TODO: This works OK but not perfect.
    # def get_execution_steps(self, prediction: Tuple[float], segments_len: List[int], path: DubinsParams) -> int:
    #     # This is a catch all for no found path.
    #     if len(segments_len) < 2: return 1
    #     # If prediction is far away, just a portion of the path so that I can get closer towards the flower.
    #     if prediction[0] > 1.5:
    #         tt = 0.5 if prediction[0] > 2.5 else (.5-1.)/(2.5-1.5)*(prediction[0]-1.5) + 1.
    #         return segments_len[0] + int(segments_len[1] * tt)
    #     # when the prediction is closer, I will try to orient myself in front of the flower without finish.
    #     if prediction[0] > DockZoneParams.COLLISION_DISTANCE*1.5: #or np.abs(prediction[1]) > np.radians(45):
    #         if np.abs(path.quantities[-1]) < np.pi/2:
    #             tt = np.abs(path.quantities[-1])/np.pi + 0.5
    #             return sum(segments_len[:-2]) + int(segments_len[-2]*tt) 
    #         else:
    #             tt = 0.5 if path.quantities[-1] > np.pi else np.abs(path.quantities[-1])/np.pi - 0.5
    #             return sum(segments_len[:-1]) + int(segments_len[-1] * tt)
    #     # when the prediction is really close, and the azimuth of the prediction is small, continuously re-estimate pose.
    #     return 2
    
    def get_execution_steps(self, prediction: Tuple[float], segments_len: List[int], path: DubinsParams,
                            distance_threshold_1: float = 1.2, distance_threshold_2: float = 0.5) -> int:
        # This is a catch all for no found path.
        if len(segments_len) < 2: return 1
        # If the flower is far away, execute the initial portion to get closer to the flower.
        if prediction[0] > distance_threshold_1:
            return segments_len[0] + int(segments_len[1] * 0.5)
        # If the flower is close, execute all except of portions of the last segment --> bring bat to the front of the flower
        if prediction[0] > distance_threshold_2:
            if np.abs(path.quantities[-1]) < np.pi/2:
                return sum(segments_len[:-1]) + 1
            else:
                return sum(segments_len[:-1]) + int(segments_len[-1] * 0.5) + 1
        # If the flower is really close, continuously re-estimate pose.
        return 1


    def random_walk_course_plan(self, turn_sharpness: float = 1., *args, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
        direction = np.random.choice(['L', 'R'])
        path = DubinsParams(modes=[direction], radii=[self.random_walk_turning_radius/turn_sharpness],
                            quantities=[self.random_walk_quantities[direction]], cost=0.)
        v, w = self.kinematic_converter(path, self.init_v)
        self.is_course_from_estimation = False
        self.cache['path'] = path
        return v, w
    
    def from_prediction_course_plan(self, prediction: Tuple[float], *args, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
        dockzone = get_dockzone_notched_circle_from_flower_pose(convert_polar_to_cartesian(np.zeros(3).astype(np.float32), *prediction))
        path = self.path_planner(np.zeros(3).astype(np.float32), dockzone)
        self.cache['path'] = path
        if path.cost<np.inf:
            v, w = self.kinematic_converter(path, self.init_v)
            self.is_course_from_estimation = True
        else:
            v, w = self.random_walk_course_plan()
            # Base on estimate, I'm close to a collision, I better get out.
            # I will need to reset distance memory, since I need to get away from the flower. If I can get attracted to a better position, I will move on.
            self.distance_memory.clear()
        return v, w

    
    def step(self, compressed_envelop_left: ArrayLike, compressed_envelop_right: ArrayLike, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
        if self.caching: self.cache['use_random_walk'] = False
        # format inputs for pose_estimator
        model_inputs = np.concatenate([compressed_envelop_left, compressed_envelop_right]).reshape(1, 2, -1)
        prediction = self.pose_estimator(model_inputs)
        use_prediction =  False
        # screen the prediction to decide whether to use it or not
        if prediction[0]:
            # if this is the very first step, fill the memory with first distance prediction
            if len(self.distance_memory) == 0:
                use_prediction = True
                for _ in range(DISTANCE_MEMORY_SIZE): self.distance_memory.append(prediction[0])
            elif prediction[0] < np.mean(self.distance_memory) + USE_PREDICTION_INCREASING_DISTANCE_DEVIATION*np.std(self.distance_memory) \
                or prediction[0] < USE_PREDICTION_DISTANCE_THRESHOLD:
                use_prediction = True
                self.distance_memory.append(prediction[0])
            else:
                use_prediction = False

        if use_prediction:
            self.v_course, self.w_course = self.from_prediction_course_plan(prediction)
            if self.is_course_from_estimation: execution_steps = self.get_execution_steps(prediction, self.kinematic_converter.segments_len, self.cache['path'])
            else: execution_steps = 2
        else:
            # I have not run out of previous course, and the previous course was gererated from estimation
            # execute a single step of previous course
            if len(self.v_course)>0 and self.is_course_from_estimation:
                execution_steps = 1
            # I have run out of previous course, or the previous course was generated from random walk
            # generate new random walk course
            else:
                self.v_course, self.w_course = self.random_walk_course_plan(turn_sharpness=1.)
                execution_steps = len(self.v_course)
                if self.caching: self.cache['use_random_walk'] = True
        
        if self.caching:
            self.cache['prediction'] = prediction
            self.cache['use_prediction'] = use_prediction

        v, w = self.v_course[:execution_steps], self.w_course[:execution_steps]
        self.v_course, self.w_course = self.v_course[execution_steps:], self.w_course[execution_steps:]
        self.init_v = v[-1]
        return v, w

    def change_flower_pose_estimator(self, estimator_type:str):
        self.pose_estimator = select_pose_estimator(estimator_type)

    def reset(self, *args, **kwargs):
        self.distance_memory.clear()
        self.cache.clear()
        self.v_course.clear()
        self.w_course.clear()
        self.is_course_from_estimation = False
        self.init_v = kwargs['init_v'] if 'init_v' in kwargs else 0.