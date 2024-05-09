import tensorflow as tf
from tf_agents.networks import network
from tf_agents.networks import lstm_encoding_network
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils
from tf_agents.networks import encoding_network
from tf_agents.utils import common as common_utils
from tf_agents.networks import utils

import tensorflow as tf

class CustomActorRNN(network.Network):
    def __init__(self, input_tensor_spec, output_tensor_spec, fc_layer_params, activation_fn, name='CustomActorRNN'):
        super(CustomActorRNN, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)
        
        self._output_tensor_spec = output_tensor_spec

        self._encoder_layer = tf.keras.layers.Reshape((input_tensor_spec.shape[0], -1))
        self._rnn_layer = [tf.keras.layers.SimpleRNN(units=units,activation=activation_fn) for units in fc_layer_params] 
        self._action_layer = tf.keras.layers.Dense(units=output_tensor_spec.shape[0])

        
    def call(self, observations, step_type=(), network_state=()):
        #Preprocess Input
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)
        observations_flat = tf.nest.flatten(observations)
    
        state = tf.concat(observations_flat, axis=-1)


        state = self._encoder_layer(state)



        for layer in self._rnn_layer:
            state = layer(state)
        
        actions = self._action_layer(state)
        actions = common_utils.scale_to_spec(actions, self._output_tensor_spec)
        actions = batch_squash.unflatten(actions)

        return tf.nest.pack_sequence_as(self._output_tensor_spec, [actions]), network_state
    
class CustomCriticRNN(network.Network):
    def __init__(self, input_tensor_spec, observation_fc_layer_params, joint_fc_layer_params, activation_fn, name='CustomActorRNN'):
        super(CustomCriticRNN, self).__init__(
            input_tensor_spec=input_tensor_spec,
            state_spec=(),
            name=name)
        print(tensor_spec)
        self._observation_spec, self._action_spec = input_tensor_spec
        
        self._encoder_layer_observation = tf.keras.layers.Reshape((self._observation_spec.shape[0], -1))

        self._encoder_layer_joint = tf.keras.layers.Reshape((observation_fc_layer_params[-1]+self._action_spec.shape[0], -1))

        self._rnn_layer_observation = [tf.keras.layers.SimpleRNN(units=units, activation=activation_fn) for units in observation_fc_layer_params] 
        self._rnn_layer_joint = [tf.keras.layers.SimpleRNN(units=units, activation=activation_fn) for units in joint_fc_layer_params]

        self._action_layer = tf.keras.layers.Dense(units=1)

    def call(self, inputs, step_type=(), network_state=()):
        observations, actions = inputs
        observations = tf.nest.flatten(observations)

        observations = self._encoder_layer_observation(observations)

        for layer in self._rnn_layer_observation:
            observations = layer(observations)

        joint = tf.concat([observations, actions], 1)
        joint = self._encoder_layer_joint(joint)
        for layer in self._rnn_layer_joint:
            joint = layer(joint)
        
        joint = self._action_layer(joint)

        return tf.reshape(joint, [-1]), network_state