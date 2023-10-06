import os
import sys

import numpy as np

import reverb
import tensorflow as tf

tf.autograph.set_verbosity(1)

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver, dynamic_episode_driver, dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents import networks
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer, tf_uniform_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.networks import q_network

# enable eager execution
tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

from .Environment import SimpleFlowerDocking

from .LearningConfig import *

class MyLinearEpsilonDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_epsilon, final_epsilon, decay_steps, global_step):
        super().__init__()
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps
        self.global_step = global_step

    def __call__(self, step=None):
        if step is None:
            step = self.global_step.numpy()
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
        units=3,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.03, maxval=0.03),
        bias_initializer=tf.keras.initializers.Constant(-0.2))
    return networks.Sequential(dense_layers + [q_values_layer])


def sumon_agent2(train_env, global_step, create_q_network=create_mlp_network, 
                td_errors_loss_fn=common.element_wise_squared_loss,
                initialize=True):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    epsilon_schedule = MyLinearEpsilonDecay(STARTING_EPSILON, ENDING_EPSILON, EXPLORE_STEPS, global_step=global_step)
    train_step_counter = tf.Variable(0)
    num_actions = train_env.action_spec().maximum - train_env.action_spec().minimum + 1
    q_net = create_q_network(num_actions, MLP_LAYERS_PARAMS)
    #q_net = q_network.QNetwork(input_tensor_spec=train_env.observation_spec(), 
    #                            action_spec=train_env.action_spec(), 
    #                            fc_layer_params=MLP_LAYERS_PARAMS,
    #                            #activation_fn=tf.keras.activations.relu,
    #                            #kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),)
    #                             )
    agent = dqn_agent.DqnAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        #epsilon_greedy=0.5,
        epsilon_greedy=epsilon_schedule,
        td_errors_loss_fn=td_errors_loss_fn,
        train_step_counter=train_step_counter,
    )
    if initialize: agent.initialize()
    return agent


def sumon_agent(train_env, global_step, create_q_network=create_mlp_network, 
                td_errors_loss_fn=common.element_wise_squared_loss,
                initialize=True):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    epsilon_schedule = MyLinearEpsilonDecay(STARTING_EPSILON, ENDING_EPSILON, EXPLORE_STEPS, global_step=global_step)
    train_step_counter = tf.Variable(0)
    num_actions = train_env.action_spec().maximum - train_env.action_spec().minimum + 1
    q_net = create_q_network(num_actions, MLP_LAYERS_PARAMS)
    #q_net = q_network.QNetwork(input_tensor_spec=train_env.observation_spec(), 
    #                            action_spec=train_env.action_spec(), 
    #                            fc_layer_params=MLP_LAYERS_PARAMS,
    #                            #activation_fn=tf.keras.activations.relu,
    #                            #kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.03, maxval=0.03),)
    #                             )
    agent = dqn_agent.DqnAgent(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        #epsilon_greedy=0.5,
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
    for ep in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            sys.stdout.write('\r')
            sys.stdout.write('episode: {} return: {} '.format(ep, episode_return.numpy()[0]))
        sys.stdout.write('\n')
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def save_agent_checkpoints(agent, dir, replay_buffer, train_checkpointer=None):
    if train_checkpointer is None:
        if not os.path.exists(dir): os.makedirs(dir)
        checkpoint_dir = os.path.join(dir, 'checkpoint')
        train_checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,)
    train_checkpointer.save(global_step=agent.train_step_counter)

def save_agent_policy(agent, dir, episode):
    if not os.path.exists(dir): os.makedirs(dir)
    policy_dir = os.path.join(dir, 'policy_{:.1f}K_eps'.format(episode/1000))
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    tf_policy_saver.save(policy_dir)

def load_agent_checkpoints(agent, dir, replay_buffer, train_step_counter=None):
    checkpoint_dir = os.path.join(dir, 'checkpoint')
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=1,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,)
    if train_checkpointer.latest_checkpoint:
        train_checkpointer.initialize_or_restore()
        loaded_step = agent.train_step_counter.numpy()
        print('Loaded checkpoint from step {}'.format(loaded_step))
        if train_step_counter is not None:
            train_step_counter.assign(train_step_counter)
    return agent, loaded_step

def load_agent_policy(dir, episode):
    policy_dir = os.path.join(dir, 'policy_{:.1f}K_eps'.format(episode/1000))
    return tf.saved_model.load(policy_dir)


def train_from_random(dir):
    collect_py_env = SimpleFlowerDocking()
    eval_py_env = SimpleFlowerDocking()
    collect_env = tf_py_environment.TFPyEnvironment(collect_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
    global_step = tf.compat.v1.train.get_or_create_global_step()
    agent = sumon_agent(collect_env, global_step=global_step)
    
    

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step.
    agent.train_step_counter.assign(0)
    #
    #global_step = tf.compat.v1.train.get_or_create_global_step()
    # Evaluate the agent's policy once before training.
    avg_return = compute_average_return(eval_env, agent.policy, NUMBER_OF_EVAL_EPISODES)
    returns = [avg_return]
    print('step = {}: Average Return = {}'.format(0, avg_return))
    # Reset the environment.
    time_step = collect_py_env.reset()
    
    # replay_buffer, rb_observer, replay_buffer = create_replay_buffer(agent)
    # collect_driver = py_driver.PyDriver(
    #     tf_env,
    #     py_tf_eager_policy.PyTFEagerPolicy(
    #     agent.collect_policy, use_tf_function=True),
    #     [rb_observer],
    #     max_steps=COLLECT_STEPS_PER_ITERATION,
    #     #max_episodes=COLLECT_EPISODES_PER_ITERATION,
    #     )

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=collect_env.batch_size,
        max_length=REPLAY_BUFFER_CAPACITY)

    # collect_driver = dynamic_step_driver.DynamicStepDriver(
    #     collect_env,
    #     agent.collect_policy,
    #     observers=[replay_buffer.add_batch],
    #     num_steps=COLLECT_STEPS_PER_ITERATION)    

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        collect_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_episodes=COLLECT_EPISODES_PER_ITERATION)
    

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=NUMBER_OF_PARALLEL_CALL,
        sample_batch_size=BATCH_SIZE,
        num_steps=REPLAY_BUFFER_SEQUENCE_LENGTH,
        single_deterministic_pass=False).prefetch(PREFETCH_NUMBER_OF_BATCH)

    iterator = iter(dataset)
    
    # The training loop
    for _ in range(NUMBER_OF_ITERATIONS):
        # collect a few steps and save to replay buffer
        collect_driver.run()
        
        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience)

        step = agent.train_step_counter.numpy()

        if step % LOG_INTERVAL==0:
            print('step = {}: loss = {}'.format(step, train_loss.loss))
        if step % EVAL_ITEATION_INTERVAL == 0:
            avg_return = compute_average_return(eval_env, agent.policy, NUMBER_OF_EVAL_EPISODES)
            print('step = {}: Average Return = {} Level ={}'.format(step, avg_return, collect_py_env.level))
            if avg_return > LEVEL_UP_RETURN_THRESHOLD:
                print('Level up!')
                collect_py_env.level += 1
                eval_py_env.level += 1
            returns.append(avg_return)
            #save_agent_checkpoints(agent, dir, replay_buffer)
            save_agent_policy(agent, dir, episode=step*COLLECT_EPISODES_PER_ITERATION)
            np.save(os.path.join(dir, 'returns.npy'), returns)

    return agent, returns



    
    
