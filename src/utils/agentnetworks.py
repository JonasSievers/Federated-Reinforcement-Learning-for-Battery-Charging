# Packages to support model training
import numpy as np
import tensorflow as tf
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

# Packages to support custom networks
import abc
from tf_agents.environments import random_py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import encoding_network
from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.specs import array_spec
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils

class ActorNetworkCustom(network.Network):

    def __init__(self,
                observation_spec,
                action_spec,
                name='ActorNetworkCustom'):
        super(ActorNetworkCustom, self).__init__(
            input_tensor_spec=observation_spec, state_spec=(), name=name)

        # For simplicity we will only support a single action float output.
        self._action_spec = action_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        self._single_action_spec = flat_action_spec[0]

        # Initialize the custom tf layers here:
        self._dense1 = tf.keras.layers.Dense(400, name='Dense1')
        self._dense2 = tf.keras.layers.Dense(300, name='Dense2')
        initializer = tf.keras.initializers.RandomUniform(
            minval=-0.003, maxval=0.003)
        self._action_projection_layer = tf.keras.layers.Dense(
            flat_action_spec[0].shape.num_elements(),
            activation=tf.keras.activations.tanh,
            kernel_initializer=initializer,
            name='action')

    def call(self, observations, step_type=(), network_state=()):
        # We use batch_squash here in case the observations have a time sequence component.
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)

        # Forward pass through the custom tf layers here (defined above):
        state = self._dense1(observations)
        state = self._dense2(state)
        actions = self._action_projection_layer(state)

        actions = common_utils.scale_to_spec(actions, self._single_action_spec)
        actions = batch_squash.unflatten(actions)
        return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state
    
class CriticNetworkCustom(network.Network):

    def __init__(self,
                observation_spec,
                action_spec,
                name='CriticNetworkCustom'):
        # Invoke constructor of network.Network
        super(CriticNetworkCustom, self).__init__(
              input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)

        self._obs_spec = observation_spec
        self._action_spec = action_spec

        # Encoding layer concatenates state and action inputs, adds dense layer:
        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1./3., mode='fan_in', distribution='uniform')
        combiner = tf.keras.layers.Concatenate(axis=-1)
        self._encoder = encoding_network.EncodingNetwork(
            (observation_spec, action_spec),
            fc_layer_params=(400,),
            preprocessing_combiner = combiner,
            activation_fn = tf.keras.activations.relu,
            kernel_initializer = kernel_initializer,
            batch_squash=True)

        # Initialize the custom tf layers here:
        self._dense1 = tf.keras.layers.Dense(400, name='Dense1')
        self._value_layer = tf.keras.layers.Dense(1,
                                                  activation=tf.keras.activations.linear,
                                                  name='Value') # Q-function output

    def call(self, observations, step_type=(), network_state=()):
        # Forward pass through the custom tf layers here (defined above):
        state, network_state = self._encoder(observations, 
                                             step_type=step_type, 
                                             network_state=network_state)
        state = self._dense1(state)
        value = self._value_layer(state)

        return tf.reshape(value,[-1]), network_state