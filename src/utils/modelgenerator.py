#Imports
#Tensorflow
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn import svm
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import pandas as pd
import time

#The ModelGenerator class contains methods to build different models
#Tensorflow models: softgated_moe_model, top1_moe_model, topk_moe_model, lstm_model, bilstm_model, cnn, dense, probability_model, transformer
#SKlearn models: Svm, Elasticnet_regression, Decisiontree, Randomforrest, K_neighbors regression
class ModelGenerator():
  
  #Builds the expert models for the MoE Layer
  def build_expert_network(self, expert_units):
      expert = keras.Sequential([
              layers.Dense(expert_units, activation="relu"), 
              ])
      return expert
  

  #Builds a MoE model with soft gating
  def build_soft_dense_moe_model(self, X_train, batch_size, horizon, dense_units,  expert_units, num_experts, m1):
    #Input of shape (batch_size, sequence_length, features)
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size, name='input_layer') 
    x = inputs
   
    #EMBEDDED MOE LAYER
    # Gating network (Routing Softmax)
    routing_logits = layers.Dense(num_experts, activation='softmax')(x)
    #experts
    experts = [m1.build_expert_network(expert_units=expert_units)(x) for _ in range(num_experts)]
    expert_outputs = tf.stack(experts, axis=1)
    #Add and Multiply expert models with router probability
    moe_output = layers.Lambda(lambda x: tf.einsum('bsn,bnse->bse', x[0], x[1]))([routing_logits, expert_outputs])
    #moe_output = tf.einsum('bsn,bnse->bse', routing_logits, expert_outputs)
    #END MOE LAYER

    x = layers.Dense(dense_units, activation="relu")(moe_output)
    x = layers.Dense(dense_units, activation="relu")(x)

    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(horizon)(x)
    softgated_moe_model = models.Model(inputs=inputs, outputs=outputs, name="soft_dense_moe")

    return softgated_moe_model
  

    #Builds a MoE model with soft gating
  def build_soft_biLSTM_moe_model(self, X_train, batch_size, horizon, lstm_units, num_experts, expert_units, m1):
    #Input of shape (batch_size, sequence_length, features)
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size, name='input_layer') 
    x = inputs

    #EMBEDDED MOE LAYER
    # Gating network (Routing Softmax)
    routing_logits = layers.Dense(num_experts, activation='softmax')(x)
    #experts
    experts = [m1.build_expert_network(expert_units=expert_units)(x) for _ in range(num_experts)]
    expert_outputs = tf.stack(experts, axis=1)
    #Add and Multiply expert models with router probability
    moe_output = layers.Lambda(lambda x: tf.einsum('bsn,bnse->bse', x[0], x[1]))([routing_logits, expert_outputs])
    #moe_output = tf.einsum('bsn,bnse->bse', routing_logits, expert_outputs)
    #END MOE LAYER

    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(moe_output)
    #x = layers.Dense(16)(moe_output)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(horizon)(x)
    softgated_moe_model = models.Model(inputs=inputs, outputs=outputs, name="soft_bilstm_moe")

    return softgated_moe_model
  

  #Builds a MoE model with top_k gating
  def build_topk_bilstm_moe_model(self, X_train, batch_size, horizon, lstm_units, num_experts, top_k, expert_units, m1):
    #Input of shape (batch_size, sequence_length, features)
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size) 
    x = inputs

    router_inputs = x
    router_probs = layers.Dense(num_experts, activation='softmax')(router_inputs)
    expert_gate, expert_index = tf.math.top_k(router_probs, k=top_k)
    expert_idx_mask = tf.one_hot(expert_index, depth=num_experts)
    combined_tensor = layers.Lambda(lambda x: tf.einsum('abc,abcd->abd', x[0], x[1]))([expert_gate, expert_idx_mask])
    #combined_tensor = tf.einsum('abc,abcd->abd', expert_gate, expert_idx_mask)
    expert_inputs = layers.Lambda(lambda x: tf.einsum("abc,abd->dabc", x[0], x[1]))([router_inputs, combined_tensor])
    #expert_inputs = tf.einsum("abc,abd->dabc", router_inputs, combined_tensor) # Instead of (3,4) -> (3, 16, 24, 4)

    expert_input_list = tf.unstack(expert_inputs, axis=0)
    expert_output_list = [
            [m1.build_expert_network(expert_units=expert_units) for _ in range(num_experts)][idx](expert_input)
            for idx, expert_input in enumerate(expert_input_list)
        ]
    expert_outputs = tf.stack(expert_output_list, axis=1)
    expert_outputs_combined = layers.Lambda(lambda x: tf.einsum("abcd,ace->acd", x[0], x[1]))([expert_outputs, combined_tensor])
    #expert_outputs_combined = tf.einsum("abcd,ace->acd", expert_outputs, combined_tensor) #(16, 2, 24, 4) and (16, 24, 3))    
    moe_output = expert_outputs_combined

    x = layers.Bidirectional(layers.LSTM(lstm_units, activation="relu", return_sequences=True))(moe_output)
    #BOTTOM Model
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(horizon)(x)
    topk_moe_model = models.Model(inputs=inputs, outputs=outputs, name="topk_bilstm_moe")

    return topk_moe_model


    #Builds a MoE model with top_k gating
  def build_topk_dense_moe_model(self, X_train, batch_size, horizon, dense_units, num_experts, top_k, expert_units, m1):
    #Input of shape (batch_size, sequence_length, features)
    inputs = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size, name='input_layer') 
    x = inputs

    router_inputs = x
    router_probs = layers.Dense(num_experts, activation='softmax')(router_inputs)
    expert_gate, expert_index = tf.math.top_k(router_probs, k=top_k)
    expert_idx_mask = tf.one_hot(expert_index, depth=num_experts)
    combined_tensor = layers.Lambda(lambda x: tf.einsum('abc,abcd->abd', x[0], x[1]))([expert_gate, expert_idx_mask])
    #combined_tensor = tf.einsum('abc,abcd->abd', expert_gate, expert_idx_mask)
    expert_inputs = layers.Lambda(lambda x: tf.einsum("abc,abd->dabc", x[0], x[1]))([router_inputs, combined_tensor])
    #expert_inputs = tf.einsum("abc,abd->dabc", router_inputs, combined_tensor) # Instead of (3,4) -> (3, 16, 24, 4)
    expert_input_list = tf.unstack(expert_inputs, axis=0)
    expert_output_list = [
            [m1.build_expert_network(expert_units=expert_units) for _ in range(num_experts)][idx](expert_input)
            for idx, expert_input in enumerate(expert_input_list)
        ]
    expert_outputs = tf.stack(expert_output_list, axis=1)
    expert_outputs_combined = layers.Lambda(lambda x: tf.einsum("abcd,ace->acd", x[0], x[1]))([expert_outputs, combined_tensor])
    #expert_outputs_combined = tf.einsum("abcd,ace->acd", expert_outputs, combined_tensor) #(16, 2, 24, 4) and (16, 24, 3)
    moe_output = expert_outputs_combined

    #BOTTOM Model
    x = layers.Dense(dense_units)(moe_output) 
    x = layers.Dense(dense_units, activation="relu")(x)
    
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(horizon)(x)
    topk_moe_model = models.Model(inputs=inputs, outputs=outputs, name="topk_moe")

    return topk_moe_model


  
  #Builds 
  def build_lstm_model(self, X_train, horizon, num_layers, units, batch_size):

    input_data = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size) 
    x =  layers.LSTM(units, return_sequences=True)(input_data)
    for _ in range(num_layers-1):
      x = layers.LSTM(units, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(horizon)(x) 

    lstm_model = tf.keras.Model(inputs=input_data, outputs=output, name="lstm_model")
    
    """model = tf.keras.Sequential([
        layers.LSTM(lstm_cells, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True), #shape (48,3)
        layers.LSTM(lstm_cells, return_sequences=True), #If True, retuns all sequences x with (x, 48,1) shape, if false only the (x,1)
        layers.GlobalAveragePooling1D(),
        layers.Dense(horizon) #Output 1 value
    ])"""

    return lstm_model

  

  def build_bilstm_model(self, X_train, horizon, num_layers, units, batch_size):

    input_data = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size) 
    x =  layers.Bidirectional(layers.LSTM(units, return_sequences=True))(input_data)
    for _ in range(num_layers-1):
      x = layers.Bidirectional(layers.LSTM(units, return_sequences=True))(x)
    x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling1D()(x)
    output = layers.Dense(horizon)(x) 

    bilstm_model = tf.keras.Model(inputs=input_data, outputs=output, name="lstm_model")
  
    """model = tf.keras.Sequential([
        layers.Bidirectional(tf.keras.layers.LSTM(bilstm_cells, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True)), #shape (48,3)
        layers.Bidirectional(tf.keras.layers.LSTM(bilstm_cells, return_sequences=False)), #If True, retuns all sequences x with (x, 48,1) shape, if false only the (x,1)
        layers.Dropout(0.2),
        layers.Dense(horizon) #Output 1 value
    ])"""

    return bilstm_model
  

  def build_cnn_model(self, X_train, horizon, num_layers, filter, kernel_size, dense_units, batch_size):

    input_data = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size) 

    x =  layers.Conv1D(filters=filter, kernel_size=kernel_size)(input_data)
    for _ in range(num_layers-1):
      x = layers.Conv1D(filters=filter, kernel_size=kernel_size)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(dense_units)(x)

    output = layers.Dense(horizon)(x) 

    cnn_model = tf.keras.Model(inputs=input_data, outputs=output, name="lstm_model")

    """model = tf.keras.Sequential([
      layers.Conv1D(filters=filter, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
      layers.MaxPooling1D(pool_size=2),
      layers.Conv1D(filters=filter, kernel_size=kernel_size, activation='relu'),
      layers.MaxPooling1D(pool_size=2),
      layers.Conv1D(filters=filter, kernel_size=kernel_size, activation='relu'),
      layers.MaxPooling1D(pool_size=2),
      layers.Conv1D(filters=filter, kernel_size=kernel_size, activation='relu'),
      
      layers.GlobalAveragePooling1D(), # tf.keras.layers.Flatten()
      layers.Dense(horizon)
    ])"""
    return cnn_model
  
  def resnet_block(self, residual_out, filter, kernel_size): 
    # residual block
    conv_in = layers.Conv1D(filters=filter, kernel_size=kernel_size, padding='same')(residual_out)
    conv = layers.BatchNormalization()(conv_in)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv1D(filters=filter, kernel_size=kernel_size, padding='same')(conv)
    conv = layers.BatchNormalization()(conv)
    residual = layers.Add()([residual_out, conv])
    residual_out = layers.Activation('relu')(residual)

    return residual_out
  
  def build_resnet_model(self, X_train, horizon, resnet_blocks, filter, kernel_size, dense_units, batch_size, m1):

    input_data = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size) 

    #Initial Convolutional Layer
    conv1 = layers.Conv1D(filters=filter, kernel_size=kernel_size, padding='same')(input_data)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)

    # First residual block
    conv_in = layers.Conv1D(filters=filter, kernel_size=kernel_size, padding='same')(conv1)
    conv = layers.BatchNormalization()(conv_in)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv1D(filters=filter, kernel_size=kernel_size, padding='same')(conv)
    conv = layers.BatchNormalization()(conv)
    residual = layers.Add()([conv1, conv])
    x = layers.Activation('relu')(residual)

    for _ in range(resnet_blocks-1):
      x = m1.resnet_block(x, filter, kernel_size)

    x = layers.Dropout(0.2)(x)
    x = layers.Dense(dense_units)(x)

    output = layers.Dense(horizon)(x) 

    cnn_model = tf.keras.Model(inputs=input_data, outputs=output, name="lstm_model")

    """model = tf.keras.Sequential([
      layers.Conv1D(filters=filter, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
      layers.MaxPooling1D(pool_size=2),
      layers.Conv1D(filters=filter, kernel_size=kernel_size, activation='relu'),
      layers.MaxPooling1D(pool_size=2),
      layers.Conv1D(filters=filter, kernel_size=kernel_size, activation='relu'),
      layers.MaxPooling1D(pool_size=2),
      layers.Conv1D(filters=filter, kernel_size=kernel_size, activation='relu'),
      
      layers.GlobalAveragePooling1D(), # tf.keras.layers.Flatten()
      layers.Dense(horizon)
    ])"""
    return cnn_model
  

  def build_dense_model(self, X_train, horizon, num_layers, units, batch_size):

    input_data = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size) 
    x =  layers.Dense(units, activation='relu')(input_data)
    for _ in range(num_layers-1):
      x = layers.Dense(units, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)
    output = layers.Dense(horizon)(x) 

    dense_model = tf.keras.Model(inputs=input_data, outputs=output, name="Dense_model")

    return dense_model


  """  
  import tensorflow_probability as tfp

  def build_probability_model(self, X_train, horizon, units):

    probability_model = tf.keras.Sequential([
      tfp.layers.DenseFlipout(units, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
      tfp.layers.DenseFlipout(units, activation='relu'),
      tfp.layers.DenseFlipout(horizon)
    ])
    return probability_model
  """

  def encoder(self, x, num_heads, num_features):

    ec_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_features)(x, x)
    ec_norm = layers.LayerNormalization(epsilon=1e-6)(x + ec_att)
    ec_ffn = layers.Dense(num_features, activation='relu')(ec_norm) 
    ec_drop = layers.Dropout(0.2)(ec_ffn) 
    ec_out = layers.LayerNormalization(epsilon=1e-6)(ec_norm + ec_drop)

    return ec_out
  
  def decoder(self, input_data, x, num_heads, num_features):

    dc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_features)(input_data, input_data)
    dc_norm = layers.LayerNormalization(epsilon=1e-6)(input_data + dc_att)

    dc_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=num_features)(dc_norm, x)
    dc_att = layers.Dropout(0.2)(dc_att)
    dc_att = layers.LayerNormalization(epsilon=1e-6)(dc_norm + dc_att)

    dc_ffn = layers.Dense(num_features, activation='relu')(dc_att) 
    dc_drop = layers.Dropout(0.2)(dc_ffn) 
    dc_out = layers.LayerNormalization(epsilon=1e-6)(dc_att + dc_drop)

    return dc_out

  def build_transformer_model(self, X_train, horizon, batch_size, sequence_length, num_layers, num_features, num_heads, dense_units, m1):
    
    #Input Layer
    input_data = layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size)  
    positional_encoding = layers.Embedding(input_dim=sequence_length-1, output_dim=num_features)(tf.range(sequence_length-1))
    input = input_data + positional_encoding

    #Encoder
    x = m1.encoder(input, num_heads, num_features)
    for _ in range (num_layers-1): 
      x = m1.encoder(x, num_heads, num_features)

    #Decoder
    x = m1.decoder(input, x, num_heads, num_features)
    for _ in range (num_layers-1): 
      x = m1.decoder(input, x, num_heads, num_features)

    # Global average pooling
    output = tf.keras.layers.GlobalAveragePooling1D()(x)  
    output = layers.Dense(dense_units)(output) 
    output = layers.Dense(horizon)(output)

    transformer_model = tf.keras.Model(inputs=input_data, outputs=output)

    return transformer_model
  

  def build_svm_model(self, kernel):
    svm_model = svm.SVR(kernel=kernel)

    return svm_model

  def build_elasticnet_regression_model(self, alpha, l1_ratio):
    elasticnet_regression_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio) #alpha start small, increase when overfitting

    return elasticnet_regression_model

  def build_decisiontree_model():
    decisiontree_model = DecisionTreeRegressor()

    return decisiontree_model

  def build_randomforrest_model(self, n_estimators):
    randomforrest_model = RandomForestRegressor(n_estimators=n_estimators) #, random_state=42

    return randomforrest_model

  def build_k_neighbors_model(self, n_neighbors):
    k_neighbors_model = KNeighborsRegressor(n_neighbors=n_neighbors)

    return k_neighbors_model

  def build_compile_evaluate_ensemble_model(self, X_train, y_train, X_val, y_val, X_test, y_test, horizon, batch_size, sequence_length, num_features, callbacks, user=""):

    # LSTM
    model_lstm = tf.keras.Sequential([
        layers.Input(shape=(X_train.shape[1], X_train.shape[2]), batch_size=batch_size),
        layers.LSTM(8, return_sequences=True, activation='relu'),
        layers.Dropout(0.2),
        layers.LSTM(8, activation='relu'),
        layers.Dense(horizon)
    ])
    model_lstm.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam())
    model_lstm.fit(X_train, y_train, epochs=100, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)

    # Random Forest Model
    model_rf = RandomForestRegressor(n_estimators=100, random_state=0)
    model_rf.fit(X_train.reshape(-1, (sequence_length-1) * num_features), y_train)

    #  SVM Model
    model_svm = SVR(kernel='linear', C=1, gamma='auto')
    model_svm.fit(X_train.reshape(-1, (sequence_length-1) * num_features), y_train)

    # Predictions from individual models on validation data
    y_pred_lstm_val = model_lstm.predict(X_val)
    y_pred_rf_val = model_rf.predict(X_val.reshape(-1, (sequence_length-1) * num_features))
    y_pred_svm_val = model_svm.predict(X_val.reshape(-1, (sequence_length-1) * num_features))

    # Calculate prediction errors on validation data
    mse_lstm_val = mean_squared_error(y_val, y_pred_lstm_val)
    mse_rf_val = mean_squared_error(y_val, y_pred_rf_val)
    mse_svm_val = mean_squared_error(y_val, y_pred_svm_val)

    # Calculate weights based on prediction errors (e.g., inverse of MSE)
    weight_lstm = 1.0 / mse_lstm_val
    weight_rf = 1.0 / mse_rf_val
    weight_svm = 1.0 / mse_svm_val

    total_weight = weight_lstm + weight_rf + weight_svm
    weight_lstm /= total_weight
    weight_rf /= total_weight
    weight_svm /= total_weight

    # Predictions from individual models on test data
    y_pred_lstm_test = model_lstm.predict(X_test)
    y_pred_rf_test = model_rf.predict(X_test.reshape(-1, (sequence_length-1) * num_features))
    y_pred_svm_test = model_svm.predict(X_test.reshape(-1, (sequence_length-1) * num_features))

    # Ensemble Prediction with weighted aggregation
    ensemble_prediction_test = (
        weight_lstm * tf.cast(tf.squeeze(y_pred_lstm_test), tf.float32) + 
        tf.cast(weight_rf * tf.squeeze(y_pred_rf_test), tf.float32) + 
        tf.cast(weight_svm * tf.squeeze(y_pred_svm_test), tf.float32)
    )

    # Evaluate the ensemble model on test data
    ensemble_mse_test = mean_squared_error(y_test, ensemble_prediction_test)

    model_user_result = pd.DataFrame(data=[[user, "LSTM_SVR_RF", ensemble_mse_test]], columns=["user", "architecture", "mse"])

    return model_user_result
  
  """import lightgbm as lgb
  from lightgbm import Dataset
  def build_compile_evaluate_lightgbm_model(self,train_data, valid_data, X_test, y_test, user):
    
    # Define LightGBM hyperparameters.
    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'mse',

        'learning_rate': 0.0449,
        'max_depth' : 58,
        'min_gain_to_split' : 0.9284, 
        'min_sum_hessian_in_leaf' : 0.0043, 
        'num_leaves': 52,
        'num_iterations' : 1440,
        'lambda_l1' : 0.5006,
        'lambda_l2' : 0.5435,

    }

    num_round = 100
    start_time = time.time()
    bst = lgb.train(params, train_data, num_round, valid_sets=[valid_data])
    end_time = time.time()
    train_time = end_time - start_time
    avg_time_epoch = train_time/100

    # Make predictions on the test data.
    y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)

    # Evaluate the model using Mean Squared Error (MSE).
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    model_user_result = pd.DataFrame(
            data=[[user, "LightGBM", train_time, avg_time_epoch, mse, rmse, mape, mae]], 
            columns=["user", "architecture", "train_time", "avg_time_epoch", "mse", "rmse", "mape", "mae"]
        )

    return model_user_result"""


