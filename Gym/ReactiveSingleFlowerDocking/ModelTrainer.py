import os
import sys

import numpy as np

import reverb
import tensorflow as tf

tf.autograph.set_verbosity(1)

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from .Environment import SimpleFlowerDocking

from .LearningConfig import *

class MyLinearEpsilonDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_epsilon, final_epsilon, decay_steps):
        super().__init__()
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps

    def __call__(self, step):
        if step > self.decay_steps:
            return self.final_epsilon
        else: # decay linearly
            return self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * step / self.decay_steps


def dense_layer(num_units):
    return tf.keras.layers.Dense(
        units=num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_in', distribution='truncated_normal'))

def create_mlp_network(num_actions, layers_params=(100,)):
    dense_layers = [dense_layer(num_units) for num_units in layers_params]
    q_values_layer = tf.keras.layers.Dense(
        num_actions,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    return sequential.Sequential(dense_layers + [q_values_layer])

def sumon_agent(train_env, create_q_network=create_mlp_network, 
                td_errors_loss_fn=common.element_wise_squared_loss,
                initialize=True):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    epsilon_schedule = MyLinearEpsilonDecay(STARTING_EPSILON, ENDING_EPSILON, EXPLORE_STEPS)
    train_step_counter = tf.Variable(0)
    num_actions = train_env.action_spec().maximum - train_env.action_spec().minimum + 1
    q_net = create_q_network(num_actions, MLP_LAYERS_PARAMS)
    agent = dqn_agent.DqnAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        epsilon_greedy=epsilon_schedule,
        td_errors_loss_fn=td_errors_loss_fn,
        train_step_counter=train_step_counter,
    )
    if initialize: agent.initialize()
    return agent

def create_replay_buffer(agent):
    table_name = 'uniform_table'
    replay_buffer_signature = tensor_spec.from_spec(agent.collect_data_spec)
    replay_buffer_signature = tensor_spec.add_outer_dim(replay_buffer_signature)
    table = reverb.Table(
        table_name,
        max_size=REPLAY_BUFFER_CAPACITY,
        sampler=reverb.selectors.Uniform(),
        remover=reverb.selectors.Fifo(),
        rate_limiter=reverb.rate_limiters.MinSize(1))
    
    reverb_server = reverb.Server([table])

    replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
        agent.collect_data_spec,
        table_name=table_name,
        sequence_length=REPLAY_BUFFER_SEQUENCE_LENGTH,
        local_server=reverb_server)
    
    rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
        replay_buffer.py_client,
        table_name,
        sequence_length=REPLAY_BUFFER_SEQUENCE_LENGTH,)

    return replay_buffer, rb_observer, replay_buffer



def create_q_networks(num_actions, layers_params):
    q_net = create_mlp_network(num_actions, layers_params)
    return q_net

def compute_average_return(environment, policy, num_episodes):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward.numpy()
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]



def train_from_random():
    train_py_env = SimpleFlowerDocking()
    eval_py_env = SimpleFlowerDocking()
    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    agent = sumon_agent(train_env)
    replay_buffer, rb_observer, replay_buffer = create_replay_buffer(agent)
    

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_average_return(eval_env, agent.policy, NUMBER_OF_EVAL_EPISODES)
    returns = [avg_return]

    # Reset the environment.
    time_step = train_py_env.reset()

    
