import numpy as np
import tensorflow as tf

LEARNING_RATE = 1e-4
MLP_LAYERS_PARAMS = (128, 128, 128, 64, 32)

STARTING_EPSILON = 1.0
ENDING_EPSILON = 0.001
EXPLORE_STEPS = 1_000_000
DISCOUNT_FACTOR = 0.9

REPLAY_BUFFER_CAPACITY = 1_000_000

# For replay buffer
# Number of timestep each sample consist of.
# Most performant when equal to `num_steps` passed to `as_dataset`.
REPLAY_BUFFER_SEQUENCE_LENGTH = 2 #default was 2.

NUMBER_OF_EVAL_EPISODES = 5
