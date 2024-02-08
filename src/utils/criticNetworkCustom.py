
import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils
from tf_agents.networks import encoding_network

class CriticNetwork(network.Network):

    def __init__(self, 
                 observation_spec, 
                 action_spec, 
                 custom_layers=[tf.keras.layers.Dense(units=16, activation='relu'),tf.keras.layers.Dense(units=16, activation='relu')], 
                 name='CriticNetwork',
                 use_ensemble=False
                 ):
        # Invoke constructor of network.Network
        super(CriticNetwork, self).__init__(input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)

        self._obs_spec = observation_spec
        self._action_spec = action_spec

        #flat_action_spec = tf.nest.flatten(action_spec)
        #self._single_action_spec = flat_action_spec[0]
                
        # Encoding layer concatenates state and action inputs, adds dense layer:
        kernel_initializer = tf.keras.initializers.VarianceScaling(scale=1./3., mode='fan_in', distribution='uniform')
        combiner = tf.keras.layers.Concatenate(axis=-1)
        self._encoder = encoding_network.EncodingNetwork(
            (observation_spec, action_spec),
            fc_layer_params=(64,),
            preprocessing_combiner = combiner,
            activation_fn = tf.keras.activations.relu,
            kernel_initializer = kernel_initializer,
            batch_squash=True)

        # Initialize the custom tf layers here:
        self._custom_layers = custom_layers

        # Optional - TO BE IMPLEMENTED CORRECTLY: Use ensemble mode
        self._use_ensemble = use_ensemble
        if self._use_ensemble:
            self._ensemble_layer = tf.keras.layers.Concatenate(axis=-1, name='ensemble_layer')
        
          # Initialize the value layer -> output_dim = 1 (Q-Value)
        self._value_layer = tf.keras.layers.Dense(
            units= 1,
            activation=tf.keras.activations.linear,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003),
            name='Value')  # Q-function output


    def call(self, observations, step_type=(), network_state=()):
        # Forward pass through the custom tf layers here (defined above):
        state, network_state = self._encoder(observations, step_type=step_type, network_state=network_state)
                          
        # Apply custom layers
        layer_outputs = []
        for layer in self._custom_layers:
            layer_outputs.append(state)
        # Additional logic for ensemble mode, if needed
        if self._use_ensemble:
            state = self._ensemble_layer(layer_outputs)

        value = self._value_layer(state)
    
        return tf.reshape(value, [-1]), network_state