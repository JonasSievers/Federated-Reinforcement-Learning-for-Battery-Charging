#Imports
#Tensorflow
import tensorflow as tf

#The ModelGenerator class contains methods to build different models
class ModelGenerator():


  def get_dense_layers(self, layers=3, units=256, dropout=0.2, activation="relu"):
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

  def get_cnn_model(
      self, layers=2, filters=24, kernel_size=3, padding="causal", strides=1, dilation_rate=1, groups=1,
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
  
  def get_lstm_model(self, layers=2, units=20, dropout=0.2):
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
  
  def get_bilstm_model(self, layers=2, units=20, dropout=0.2):
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
       
  def get_cnn_lstm_model(
      self, cov_layers=1, filters=24, kernel_size=3, padding="causal", strides=1, dilation_rate=1, groups=1,
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

  def get_resnet_model(self, blocks=1, filters=24, kernel_size=3, padding="causal", strides=1, dilation_rate=1, groups=1,):
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
    
        