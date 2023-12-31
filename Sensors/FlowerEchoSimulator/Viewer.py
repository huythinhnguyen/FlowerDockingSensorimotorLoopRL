import os
import sys
import numpy as np
from typing import Tuple
from numpy.typing import ArrayLike
from typing import List

from .Setting import ViewerSetting

# Field of view class
# Given a set of objects in the world coordinates and bats pose.
# The class will filter out which object is in the field of view of the bat. (predefined field)
# The class will also transform the pose of each object in world coordinates into relative polar coordinates to bats pose.

# objects format:
#   - cartesian: shape(N, 4), each row = [x, y, theta, k]
#   - polar: shape(N, 3), each row = [r, rho, phi, k]
# where:
#   - x, y: position of the object in world coordinates
#   - theta: orientation of the object in the xy-plane of world coordinates
#   - k: object class id (int)
#   - r: distance of the object to the bat
#   - rho: angle between the object and the bat in the xy-plane of world coordinates
#   - phi: relative orientation of the object to the bat. 
#           Which mean this is the orientaiton of the object in the flower-bat's coordinate system.
# Field of view can be combined from multiple FOVs.
# Each FOV is defined by a tuple of (linear, angular) FOV. 
#       where: linear is the linear range of the FOV, 
#               angular is the angular range of the FOV (left right total)

def wrapToPi(x):
    if type(x) == list: x = np.asarray(x).astype(np.float32)
    return np.mod(x + np.pi, 2*np.pi) - np.pi
def wrapTo180(x):
    if type(x) == list: x = np.asarray(x).astype(np.float32)
    return np.mod(x + 180, 360) - 180


class FieldOfViewBase:
    def __init__(self):
        self.intput_angular_unit = 'radian'
        self.output_angular_unit = 'radian'

    @property
    def filtered_objects_inview_polar(self) -> ArrayLike:
        return self._filtered_objects_inview_polar
    
    @filtered_objects_inview_polar.setter
    def filtered_objects_inview_polar(self, value: ArrayLike):
        self._filtered_objects_inview_polar = value

    @property
    def filtered_objects_inview_cartesian(self) -> ArrayLike:
        return self._filtered_objects_inview_cartesian
    
    @filtered_objects_inview_cartesian.setter
    def filtered_objects_inview_cartesian(self, value: ArrayLike):
        self._filtered_objects_inview_cartesian = value

    @property
    def collision_status(self) -> bool:
        return self._collision_status
    
    @collision_status.setter
    def collision_status(self, value: bool):
        self._collision_status = value

    @property
    def collision_object_ID(self) -> int:
        return self._collision_object_ID
    
    @collision_object_ID.setter
    def collision_object_ID(self, value: int):
        self._collision_object_ID = value

    @property
    def bat_pose(self) -> ArrayLike:
        return self._bat_pose
    
    @bat_pose.setter
    def bat_pose(self, value: ArrayLike):
        self._bat_pose = value

    def run(self, bat_pose: ArrayLike, cartesian_objects_matrix: ArrayLike):
        raise NotImplementedError
    
    def __call__(self, bat_pose: ArrayLike, cartesian_objects_matrix: ArrayLike):
        return self.run(bat_pose, cartesian_objects_matrix)


class FieldOfView(FieldOfViewBase):
    def __init__(self,FoVs: list[list]=ViewerSetting.FOVS, collision_distance:float = ViewerSetting.COLLISION_THRESHOLD):
        super().__init__()
        self.FoVs = FoVs
        self.collision_distance = collision_distance

    def run(self, bat_pose: ArrayLike, cartesian_objects_matrix: ArrayLike):
        self.bat_pose = np.copy(bat_pose)
        if self.intput_angular_unit=='degree':
            bat_pose[2] = np.radians(bat_pose[2])
            cartesian_objects_matrix[:,2] = np.radians(cartesian_objects_matrix[:,2])
        polar_objects_matrix = self._convert_objects_from_cartesian_to_polar_format(bat_pose, cartesian_objects_matrix)
        if self._collision_check(polar_objects_matrix):
            self._bounce_bat_pose_back_to_collision_distance(bat_pose, cartesian_objects_matrix)
            polar_objects_matrix = self._convert_objects_from_cartesian_to_polar_format(self.bat_pose, cartesian_objects_matrix)
        polar_objects_matrix = self._filter_objects_in_fov(polar_objects_matrix)
        if self.output_angular_unit=='degree':
            polar_objects_matrix[:,1:3] = np.degrees(polar_objects_matrix[:,1:3])
        self.filtered_objects_inview_polar = polar_objects_matrix
        return np.copy(polar_objects_matrix)

    def _collision_check(self, polar_objects_matrix):
        if np.any(polar_objects_matrix[:,0] <= self.collision_distance):
            self.collision_status = True
            self.collision_object_ID = np.argmin(polar_objects_matrix[:,0])
        else:
            self.collision_status = False
            self.collision_object_ID = -1
        return self.collision_status

    def _bounce_bat_pose_back_to_collision_distance(self, bat_pose, cartesian_objects_matrix):
        x_collide_object, y_collide_object = cartesian_objects_matrix[self.collision_object_ID,:2]
        x_bat, y_bat = bat_pose[:2]
        angle_object_to_bat = np.arctan2(y_bat-y_collide_object, x_bat-x_collide_object)
        self.bat_pose[0] = x_collide_object + self.collision_distance * np.cos(angle_object_to_bat)
        self.bat_pose[1] = y_collide_object + self.collision_distance * np.sin(angle_object_to_bat)

    def _filter_objects_in_fov(self, polar_objects_matrix):
        # result = np.empty((0, polar_objects_matrix.shape[1]))
        # for fov in self.FoVs:
        #    for object in polar_objects_matrix:
        #        if object[0] <= fov[0] and np.abs(object[1]) <= fov[1]/2:
        #            result = np.vstack((result, object))
        # return result
        if type(self.FoVs)==list: FoVs = np.asarray(self.FoVs)
        fov_max_distances = np.asarray(FoVs)[:,0]
        fov_max_angles = np.asarray(FoVs)[:,1]

        object_angles = np.abs(polar_objects_matrix[:,1])
        distance_mask = polar_objects_matrix[:,0] <= fov_max_distances[:, np.newaxis]
        angle_mask = object_angles <= fov_max_angles[:, np.newaxis]/2

        combined_mask = np.logical_or(*np.logical_and(distance_mask,
                                       angle_mask))
        return polar_objects_matrix[combined_mask]


    def _convert_objects_from_cartesian_to_polar_format(self, 
                                                        bat_pose: ArrayLike, 
                                                        cartesian_objects_matrix: ArrayLike):
        
        """
        Convert objects from cartesian format to polar format.
        """
        direction_vectors = cartesian_objects_matrix[:,:2] - bat_pose[:2]
        polar_objects_matrix = np.empty((direction_vectors.shape[0],direction_vectors.shape[1]+1))
        # solve for r
        polar_objects_matrix[:,0] = np.linalg.norm(direction_vectors, axis=1)
        # solve for rho
        polar_objects_matrix[:,1] = wrapToPi(np.arctan2(direction_vectors[:,1],direction_vectors[:,0]) \
                          - bat_pose[2])
        # solve for phi
        polar_objects_matrix[:,2] = wrapToPi(np.arctan2(-1*direction_vectors[:,1],-1*direction_vectors[:,0])\
                          -cartesian_objects_matrix[:,2])
        polar_objects_matrix = np.hstack((polar_objects_matrix,
                                          cartesian_objects_matrix[:,3:]))
        return polar_objects_matrix
    
    def _convert_objects_from_polar_to_cartesian_format(self,
                                                        bat_pose: ArrayLike,
                                                        polar_objects_matrix: ArrayLike):
        """
        Convert objects from polar format to cartesian format.
        """
        # solve for x
        x = bat_pose[0] + polar_objects_matrix[:,0] * np.cos(polar_objects_matrix[:,1] + bat_pose[2])
        # solve for y
        y = bat_pose[1] + polar_objects_matrix[:,0] * np.sin(polar_objects_matrix[:,1] + bat_pose[2])
        # solve for theta
        direction_vectors = np.hstack((x.reshape(-1,1), y.reshape(-1,1))) - bat_pose[:2]
        theta = wrapToPi(np.arctan2(-1*direction_vectors[:,1],-1*direction_vectors[:,0]) \
                         -polar_objects_matrix[:,2])
        cartesian_objects_matrix = np.hstack((x.reshape(-1,1), y.reshape(-1,1), 
                                              theta.reshape(-1,1), 
                                              polar_objects_matrix[:,3:]))
        return cartesian_objects_matrix
