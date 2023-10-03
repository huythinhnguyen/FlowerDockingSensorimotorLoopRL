import sys
import os

REPO_NAME = 'FlowerDockingSensorimotorLoopRL'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)

import numpy as np

DATA_ROOT = os.path.join(REPO_PATH, 'Dataset')
SNIPPET_PATH = os.path.join(DATA_ROOT, 'flower3x_snippet')
EMISSION_SNIPPET_PATH = os.path.join(SNIPPET_PATH, 'emission_snippet.npz')
NOISE_PATH = os.path.join(SNIPPET_PATH, 'noise.npz')


def get_noise_data_array():
    return np.load(NOISE_PATH)

def get_emission_snippet():
    return np.load(EMISSION_SNIPPET_PATH)

def get_snippet(distance, orientation, neck_angle):
    #print('referencing from dist={}, neck={}, orient={}'.format(distance, neck_angle, orientation))
    return np.load(os.path.join(SNIPPET_PATH, 'dist'+str(np.round(distance, 1)),  'orient'+str(int(orientation)), 'neck'+str(int(neck_angle))+'.npz'))

