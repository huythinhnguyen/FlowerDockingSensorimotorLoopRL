from typing import Any
import numpy as np


class DiscreteActions1:
    # Rotate left, rotate right, move forward
    # -1 turn left, 0 go straight, 1 turn right
    def __init__(self, v=0.5, w=np.pi/2):
        self._v = v
        self._w = w
        self._cmd_dict = {-1: (0, w), 0: (v, 0), 1: (0, -w)}

    def __call__(self, action):
        return self.run(action)
    
    def run(self, action):
        return self._cmd_dict[action]
    
class DiscreteActions2:
    def __init__(self, max_v=1, max_w = np.pi):
        self._max_v = max_v
        self._max_w = max_w
        self._cmd_dict = {-2: (0, max_w), -1: (0.5*max_v, 0.5*max_w),
                          0: (max_v, 0), 1: -(0.5*max_v, -0.5*max_w), 2:(0, max_w)}
        
    def __call__(self, action):
        return self.run(action)
    
    def run(self, action):
        return self._cmd_dict[action]
    

class ContinuousActions1:
    def __init__(self, max_v:float, max_w:float, v_action_range:list, w_action_range:list):
        self._max_v = max_v
        self._max_w = max_w
        self._v_action_range = v_action_range
        self._w_action_range = w_action_range

    def __call__(self, action):
        return self.run(action)
    
    def run(self, action:list):
        v_action, w_action = action
        v = self._max_v * (v_action - self._v_action_range[0]) / (self._v_action_range[1] - self._v_action_range[0])
        w = self._max_w * (w_action - self._w_action_range[0]) / (self._w_action_range[1] - self._w_action_range[0])
        return (v, w) 