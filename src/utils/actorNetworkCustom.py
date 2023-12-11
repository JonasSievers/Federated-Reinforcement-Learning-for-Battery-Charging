
import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils


class ActorNetworkCustom(network.Network):

    def __init__(self, 
                 observation_spec, 
                 action_spec, 
                 custom_layers=[tf.keras.layers.Dense(units=16, activation='relu'),tf.keras.layers.Dense(units=16, activation='relu')],
                 name='AgentNetworkCustomLayers', 
                 use_ensemble=False
                 ):
        super(ActorNetworkCustom, self).__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)

        # Preprocess Action: Flatten
        self._action_spec = action_spec
        flat_action_spec = tf.nest.flatten(action_spec)
        self._single_action_spec = flat_action_spec[0]
    
        # Initialize the custom Keras layers
        self._custom_layers = custom_layers

        # Initialize Output layer -> output_dim )= action_spec
        self._action_projection_layer = tf.keras.layers.Dense(
            units= flat_action_spec[0].shape.num_elements(),
            activation=tf.keras.activations.tanh,
            kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.003, maxval=0.003),
            name='action'
        )

        # Optional - TO BE IMPLEMENTED CORRECTLY: Use ensemble mode
        self._use_ensemble = use_ensemble
        if self._use_ensemble:
            self._ensemble_layer = tf.keras.layers.Concatenate(axis=-1, name='ensemble_layer')

    def call(self, observations, step_type=(), network_state=()):
        
        #Preprocess Input
        # We use batch_squash here in case the observations have a time sequence component.
        outer_rank = nest_utils.get_outer_rank(observations, self.input_tensor_spec)
        batch_squash = utils.BatchSquash(outer_rank)
        observations = tf.nest.map_structure(batch_squash.flatten, observations)
        observations_flat = tf.nest.flatten(observations)

        #Custom Layers
        state = tf.concat(observations_flat, axis=-1)  
        layer_outputs = []
        for layer in self._custom_layers:
            layer_outputs.append(state)
        # Additional logic for ensemble mode, if needed
        if self._use_ensemble:
            state = self._ensemble_layer(layer_outputs)

        #Output layer
        actions = self._action_projection_layer(state)
        actions = common_utils.scale_to_spec(actions, self._single_action_spec)
        actions = batch_squash.unflatten(actions)
        return tf.nest.pack_sequence_as(self._action_spec, [actions]), network_state
