
import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils
from tf_agents.networks import encoding_network

class CriticNetworkCustom(network.Network):

    def __init__(self, observation_spec, action_spec, custom_layers, name='CriticNetworkCustom'):
        # Invoke constructor of network.Network
        super(CriticNetworkCustom, self).__init__(
              input_tensor_spec=(observation_spec, action_spec), state_spec=(), name=name)

        self._obs_spec = observation_spec
        self._action_spec = action_spec
        
        print("observation_spec: ", observation_spec) #shape=(4,)
        print("action_spec: ", action_spec) #shape=(1,)
        
        # Encoding layer concatenates state and action inputs, adds dense layer:
        kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1./3., mode='fan_in', distribution='uniform')
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
        
        # Add custom layers dynamically
        for layer in self._custom_layers:
            setattr(self, f"_custom_{layer.name}", layer)

        # Initialize the value layer
        self._value_layer = tf.keras.layers.Dense(1,
                                                  activation=tf.keras.activations.linear,
                                                  name='Value')  # Q-function output


    def call(self, observations, step_type=(), network_state=()):
        # Forward pass through the custom tf layers here (defined above):
        state, network_state = self._encoder(observations, 
                                             step_type=step_type, 
                                             network_state=network_state)
        
        print("state: ", state.shape) #shape=(1, 400)
            
        # Apply custom layers
        for layer in self._custom_layers:
            print(f"Custom Layer {layer.name} Input Shape:", state.shape)
            state = getattr(self, f"_custom_{layer.name}")(state)
            print(f"Custom Layer {layer.name} Output Shape:", state.shape)

        print("Own layers done")
        value = self._value_layer(state)
        print("value: ", value) #shape=(1, 1)

        return tf.reshape(value, [-1]), network_state