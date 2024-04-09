
import tensorflow as tf

from tf_agents.networks import network
from tf_agents.networks import utils
from tf_agents.utils import common as common_utils
from tf_agents.utils import nest_utils
from tf_agents.networks import encoding_network


class ActorNetwork(network.Network):

    def __init__(self, 
                 observation_spec, 
                 action_spec, 
                 custom_layers=[tf.keras.layers.Dense(units=16, activation='relu'),tf.keras.layers.Dense(units=16, activation='relu')],
                 name='AgentNetwork', 
                 use_ensemble=False
                 ):
        super(ActorNetwork, self).__init__(input_tensor_spec=observation_spec, state_spec=(), name=name)

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

class CustomLayers():

  def get_dense_layers(layers=3, units=256, dropout=0.2, activation="relu"):
    """
    Generate a list of dense layers followed by a dropout layer.

    Parameters:
    - layers (int): Number of dense layers.
    - units (int): Number of units in each dense layer.
    - dropout (float): Dropout rate.
    - activation (str): Activation function for dense layers.

    Returns:
    - List of tf.keras.layers.Layer: List of dense layers followed by a dropout layer.
    """
    dense_layers = [tf.keras.layers.Dense(units=units, activation=activation) for _ in range(layers)]
    dense_layers.append(tf.keras.layers.Dropout(dropout))
    return dense_layers

  def get_cnn_layers(
      layers=2, filters=24, kernel_size=3, padding="causal", strides=1, dilation_rate=1, groups=1,
      pool_size=3, pool_strides=3, pool_padding="same", dropout=0.2,
      ):
    """
    Generate a list of 1D convolutional layers followed by max pooling, flattening, and dropout.

    Parameters:
    - layers (int): Number of convolutional layers.
    - filters (int): Number of filters in each convolutional layer.
    - kernel_size (int): Size of the convolutional kernel.
    - padding (str): Padding type for convolutional layers.
    - strides (int): Stride for convolutional layers.
    - dilation_rate (int): Dilation rate for convolutional layers.
    - groups (int): Number of groups for grouped convolutional layers.
    - pool_size (int): Size of the max pooling window.
    - pool_strides (int): Stride for max pooling.
    - pool_padding (str): Padding type for max pooling.
    - dropout (float): Dropout rate.

    Returns:
    - List of tf.keras.layers.Layer: List of convolutional layers, max pooling, flattening, and dropout.
    """
    cnn_layers = [
      tf.keras.layers.Conv1D(
        filters=filters, kernel_size=kernel_size, padding=padding, 
        strides=strides, dilation_rate=dilation_rate, groups=groups
        ) for _ in range(layers)
      ]
    cnn_layers.append(tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides=pool_strides, padding=pool_padding,))
    cnn_layers.append(tf.keras.layers.Flatten())
    cnn_layers.append(tf.keras.layers.Dropout(dropout))
    return cnn_layers
  
  def get_lstm_layers(layers=2, units=20, dropout=0.2):
    """
    Generate a list of LSTM layers followed by global average pooling and a dropout layer.

    Parameters:
    - layers (int): Number of LSTM layers.
    - units (int): Number of units in each LSTM layer.
    - dropout (float): Dropout rate.

    Returns:
    - List of tf.keras.layers.Layer: List of LSTM layers followed by global average pooling and a dropout layer.
    """
    lstm_layers = [tf.keras.layers.LSTM(units=units, return_sequences=True) for _ in range(layers)]
    lstm_layers.append(tf.keras.layers.GlobalAveragePooling1D())
    lstm_layers.append(tf.keras.layers.Dropout(dropout))
    return lstm_layers
  
  def get_bilstm_layers(layers=2, units=20, dropout=0.2):
    """
    Generate a list of Bidirectional LSTM layers followed by global average pooling, dropout, and a dense layer.

    Parameters:
    - layers (int): Number of Bidirectional LSTM layers.
    - units (int): Number of units in each LSTM layer.
    - dropout (float): Dropout rate.

    Returns:
    - List of tf.keras.layers.Layer: List of Bidirectional LSTM layers followed by global average pooling, dropout, and a dense layer.
    """
    bilstm_layers = [tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=units, return_sequences=True)) for _ in range(layers)]
    bilstm_layers.append(tf.keras.layers.GlobalAveragePooling1D())
    bilstm_layers.append(tf.keras.layers.Dropout(dropout))
    bilstm_layers.append(tf.keras.layers.Dense(units=16))
    return bilstm_layers
       
  def get_cnn_lstm_layers(
      cov_layers=1, filters=24, kernel_size=3, padding="causal", strides=1, dilation_rate=1, groups=1,
      lstm_layers=2, lstm_units=128, dropout=0.2):
    """
    Generate a model with a combination of Conv1D and LSTM layers.

    Parameters:
    - cov_layers (int): Number of Conv1D layers.
    - filters (int): Number of filters in each Conv1D layer.
    - kernel_size (int): Size of the convolutional kernel.
    - padding (str): Padding type for Conv1D layers.
    - strides (int): Stride size for Conv1D layers.
    - dilation_rate (int): Dilation rate for Conv1D layers.
    - groups (int): Number of groups for grouped convolution.
    - lstm_layers (int): Number of LSTM layers.
    - lstm_units (int): Number of units in each LSTM layer.
    - dropout (float): Dropout rate.

    Returns:
    - List of tf.keras.layers.Layer: List of Conv1D and LSTM layers followed by Dropout, GlobalAveragePooling1D, and Dense layers.
    """
    cnn_lstm_layers = [
      tf.keras.layers.Conv1D(
          filters=filters, kernel_size=kernel_size, padding=padding,
          strides=strides, dilation_rate=dilation_rate, groups=groups
      ) for _ in range(cov_layers)
    ]
    # Convert the generator expression to a list comprehension
    lstm_layers_list = [tf.keras.layers.LSTM(lstm_units, return_sequences=True) for _ in range(lstm_layers)]
    cnn_lstm_layers.extend(lstm_layers_list)
    cnn_lstm_layers.extend([
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(units=16)
    ])
    return cnn_lstm_layers

  def get_resnet_layers(blocks=1, filters=24, kernel_size=3, padding="causal", strides=1, dilation_rate=1, groups=1,):
    """
    Generate a ResNet model with a specified number of blocks.

    Parameters:
    - blocks (int): Number of ResNet blocks to create.
    - filters (int): Number of filters in Conv1D layers.
    - kernel_size (int): Size of the convolutional kernel.
    - padding (str): Type of padding used in Conv1D layers.
    - strides (int): Stride used in Conv1D layers.
    - dilation_rate (int): Dilation rate for Conv1D layers.
    - groups (int): Number of groups for grouped convolution.

    Returns:
    - List of tf.keras.layers.Layer: List of layers representing the ResNet model.
    """
    resnet_block = [
      tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=dilation_rate, groups=groups),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Activation('relu'),
      tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides, dilation_rate=dilation_rate, groups=groups),
      tf.keras.layers.BatchNormalization(),
      tf.keras.layers.Add(),
      tf.keras.layers.Activation('relu'),
    ]
    resnet_layers = [layer for _ in range(blocks) for layer in resnet_block]
    return resnet_layers    