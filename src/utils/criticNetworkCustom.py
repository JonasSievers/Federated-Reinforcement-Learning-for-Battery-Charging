
import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils
from tf_agents.networks import encoding_network

class CriticNetworkCustom(network.Network):

    def __init__(self,
                observation_spec,
                action_spec,
                custom_layers=None,
                name='CriticNetworkCustom'):
        # Invoke constructor of network.Network
        super(CriticNetworkCustom, self).__init__(
              input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)

        self._obs_spec = observation_spec
        self._action_spec = action_spec

        # Initialize the custom Keras layers provided:
        self._custom_layers = custom_layers
        if custom_layers is not None:
            # When custom layers are provided, use Concatenate as preprocessing_combiner
            self._encoder = encoding_network.EncodingNetwork(
                (observation_spec, action_spec),
                fc_layer_params=(400,),
                preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
                activation_fn=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1./3., mode='fan_in', distribution='uniform'),
                batch_squash=True)
        else:
            self._encoder = encoding_network.EncodingNetwork(
                (observation_spec, action_spec),
                fc_layer_params=(400,),
                activation_fn=tf.keras.activations.relu,
                kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1./3., mode='fan_in', distribution='uniform'),
                batch_squash=True)

        # Initialize the custom tf layers here:
        self._dense1 = tf.keras.layers.Dense(400, name='Dense1')
        self._value_layer = tf.keras.layers.Dense(1,
                                                  activation=tf.keras.activations.linear,
                                                  name='Value')  # Q-function output

    def call(self, observations, step_type=(), network_state=()):
        # Forward pass through the custom Keras layers provided:
        state, network_state = self._encoder(observations,
                                             step_type=step_type,
                                             network_state=network_state)

        if self._custom_layers is not None:
            # Additional logic for custom layers
            layer_outputs = [state]
            for layer in self._custom_layers:
                state = layer(state)
                layer_outputs.append(state)

            value = self._value_layer(state)

        else:
            # Forward pass without custom layers
            state = self._dense1(state)
            value = self._value_layer(state)

        return tf.reshape(value, [-1]), network_state



"""class CriticNetworkCustom(network.Network):

    def __init__(self,
                observation_spec,
                action_spec,
                custom_layers=[tf.keras.layers.Dense(units=16, activation='relu'),tf.keras.layers.Dense(units=16, activation='relu')],
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

        return tf.reshape(value,[-1]), network_state"""