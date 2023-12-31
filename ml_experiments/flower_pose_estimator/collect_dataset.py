import sys
import os
import multiprocessing as mp
import numpy as np
import time
import pandas as pd
import logging
REPO_NAME = 'FlowerDockingSensorimotorLoopRL'
REPO_PATH = os.path.abspath(__file__)
while os.path.basename(REPO_PATH) != REPO_NAME: REPO_PATH = os.path.dirname(REPO_PATH)
if REPO_PATH not in sys.path: sys.path.append(REPO_PATH)

from Sensors.FlowerEchoSimulator import Spatializer
from Sensors.FlowerEchoSimulator.Setting import *

DATASET_DIR = os.path.join(REPO_PATH, 'ml_experiments/datasets/flower_pose_estimator')

###
# Dataset generation
# Distances will be ranging from 0.2 to 3.2 meters
# Azimuth will be ranging from -100 to 100 degrees
# Orientation will be ranging from -180 to 180 degrees

# collection will be broken into 16 processes
# Distance will also be browken into 16 ranges

# Data will be collected from NUMBER_OF_DATAPOINTS / 16
###
DISTANCE_RANGE = (0.2, 3.2)
AZIMUTH_RANGE = (-110., 110.)
ORIENTATION_RANGE = (-180., 180.)

NUMBER_OF_SKIP_CPU = 0

N_SAMPLES = 10_000*(os.cpu_count() - NUMBER_OF_SKIP_CPU)

NUMBER_OF_STAIRS = 4

COARSE_N = int((os.cpu_count() - NUMBER_OF_SKIP_CPU) / NUMBER_OF_STAIRS)

def generate_dataset(process_id, number_of_datapoints, distance_range, outputs_dict):
    init_bat_theta = np.radians(90)
    init_flower_theta = np.radians(-90)
    bat_pose = np.asarray([0., 0., init_bat_theta]).astype(np.float32)
    objects = np.asarray([[0., 0., init_flower_theta, 3.]]).astype(np.float32)
    render = Spatializer.Render(
        viewer_kwargs={'FoVs': [(1., np.radians(110)*2), (4., np.radians(110)*2)]})
    for i in range(number_of_datapoints):
        distance = np.random.uniform(distance_range[0], distance_range[1])
        azimuth = np.radians(np.random.uniform(AZIMUTH_RANGE[0], AZIMUTH_RANGE[1]))
        orientation = np.radians(np.random.uniform(ORIENTATION_RANGE[0], ORIENTATION_RANGE[1]))
        objects[0, 1] = distance
        bat_pose[2] = Spatializer.wrapToPi(init_bat_theta - azimuth)
        objects[0, 2] = Spatializer.wrapToPi(init_flower_theta - orientation)
        render(bat_pose, objects)
        outputs_dict['distance'].append(distance)
        outputs_dict['azimuth'].append(azimuth)
        outputs_dict['orientation'].append(orientation)
        outputs_dict['envelope_left'].append(render.envelope_left)
        outputs_dict['envelope_right'].append(render.envelope_right)
        outputs_dict['compress_left'].append(render.compress_left)
        outputs_dict['compress_right'].append(render.compress_right)
        if i%1000 == 0:
            logging.debug('Process {} finished {}/{} points'.format(process_id, i, number_of_datapoints))
    return logging.debug('Process {} finished.'.format(process_id))

def generate_dataset_multiprocess(number_of_datapoints, number_of_processes):
    outputs_dict_ls = []
    manager = mp.Manager()
    processes = []
    for i in range(number_of_processes):
        outputs_dict_ls.append(manager.dict(distance = manager.list(),
                                            azimuth = manager.list(),
                                            orientation = manager.list(),
                                            envelope_left = manager.list(),
                                            envelope_right = manager.list(),
                                            compress_left = manager.list(),
                                            compress_right = manager.list()))

        distance_range = (DISTANCE_RANGE[0], DISTANCE_RANGE[0] + (int(i/COARSE_N)+1) \
                          * (DISTANCE_RANGE[1] - DISTANCE_RANGE[0]) / (number_of_processes/COARSE_N) )
        p = mp.Process(target=generate_dataset, args=(i, number_of_datapoints, distance_range, outputs_dict_ls[i]))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

    return outputs_dict_ls

def collect_and_save():
    logging.info('===============================\nCollecting dataset...')
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    tic = time.time()
    number_of_processes = os.cpu_count() - NUMBER_OF_SKIP_CPU
    number_of_datapoints = int(N_SAMPLES / number_of_processes)
    outputs_dict_ls = generate_dataset_multiprocess(number_of_datapoints, number_of_processes)
    
    final_outputs = dict(distance = [],
                         azimuth = [],
                         orientation = [],
                         envelope_left = [],
                         envelope_right = [],
                         compress_left = [],
                         compress_right = [])
    for i, outputs in enumerate(outputs_dict_ls):
        for key in outputs.keys():
            final_outputs[key] += outputs[key]

    df = pd.DataFrame.from_dict(final_outputs)

    # save to pickle with date MMDDYY format
    date = time.strftime('%m%d%y')
    
    df.to_pickle(os.path.join(DATASET_DIR, 'dataset_{}.pkl'.format(date)))
    
    logging.info('Elapsed time: {} hours'.format((time.time() - tic) / 3600))


def main():
    #return print(DATASET_DIR)
    return collect_and_save()

if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', encoding='utf-8', level=logging.DEBUG)
    main()