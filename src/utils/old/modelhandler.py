#Imports
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import pandas as pd
import numpy as np

from keras.callbacks import ModelCheckpoint

#The Modelhandler class contains usefull methods to compile, fit, evaluate, and plot models
class Modelhandler():

    #This method plots 1. training and validation loss & 2. prediction results
    def plot_model_predictions(self, model, history, y_test, X_test, batch_size, plt_length=200):
        # Plot training and validation loss
        plt.figure(figsize=(15, 3))
        plt.plot(history.history['loss'], label='train_loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        # Make predictions on the test set
        y_pred = model.predict(X_test, batch_size=batch_size)

        # Plot prediction results
        plt.figure(figsize=(10, 3))
        plt.plot(y_test[:plt_length], label='True')
        plt.plot(y_pred[:plt_length], label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Electricity consumption')
        plt.legend()
        plt.show()

    #This method compiles the model using Adam optimizer, fits the model, and evaluates it
    def compile_fit_evaluate_model(self, model, loss, metrics, X_train, y_train, max_epochs, batch_size, X_val, y_val, X_test, y_test, callbacks, user= "", hyper="", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)):
        #Compile the model
        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # Train the model
        history = model.fit(X_train, y_train, epochs=max_epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks, verbose=0,)
        #model = tf.keras.models.load_model('models/best_model.h5')
        #Evaluate the model on test dataset
        test_loss = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)

        train_times = callbacks[1].get_training_times_df()
        total_train_time = train_times["Total Training Time"][0]
        avg_time_epoch = train_times["Epoch Avg Train_time"].iloc[-1]
    
        model_user_result = pd.DataFrame(
            data=[[user, hyper, total_train_time, avg_time_epoch, test_loss[0], test_loss[1], test_loss[2], test_loss[3]]], 
            columns=["user", "architecture", "train_time", "avg_time_epoch", "mse", "rmse", "mape", "mae"]
        )

        return history, model_user_result
    

    
    def evaluate_ensemble(self, y_test, final_predictions, user, hyper, train_time, avg_time_epoch): 
        
        # Calculate Mean Squared Error (MSE)
        mse = mean_squared_error(y_test, final_predictions)
        
        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)          

        # Calculate Mean Absolute Percentage Error (MAPE)
        epsilon = 1e-10  # Small epsilon to avoid division by zero
        mape = np.mean(np.abs((y_test - final_predictions) / (y_test + epsilon))) * 100

        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test, final_predictions)

        model_user_result = pd.DataFrame(
            data=[[user, hyper, train_time, avg_time_epoch, mse, rmse, mape, mae]], 
            columns=["user", "architecture", "train_time", "avg_time_epoch", "mse", "rmse", "mape", "mae"]
        )

        return model_user_result

    #This methods fits, predicts, and plots the results for sklearn models
    def statistical_model_compile_fit_evaluate(self, X_train, y_train, X_test, y_test, model):
        X_train_flattened = X_train.reshape((X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
        model.fit(X_train_flattened, y_train)

        X_test_flattened = X_test.reshape((X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
        y_pred = model.predict(X_test_flattened)

        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

        # Plot the actual and predicted values
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.xlabel('Time')
        plt.ylabel('Electricity consumption')
        plt.title('Model Prediction vs Actual')
        plt.legend()
        plt.show()

class TimingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_time)

    def get_training_times_df(self):
        total_training_time = time.time() - self.start_time
        average_epoch_times = [sum(self.epoch_times[:i+1]) / (i + 1) for i in range(len(self.epoch_times))]
        data = {
            'Epoch': list(range(1, len(self.epoch_times) + 1)),
            'Epoch Train_time': self.epoch_times,
            'Epoch Avg Train_time': average_epoch_times,
            'Total Training Time': total_training_time
        }
        return pd.DataFrame(data)
    

class CustomCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.epoch_times = []
        self.losses = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'test_loss': []
        }

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - self.epoch_start_time
        self.epoch_times.append(epoch_time)

        self.losses['epoch'].append(epoch)
        self.losses['train_loss'].append(logs['loss'])
        self.losses['val_loss'].append(logs['val_loss'])

    def on_test_end(self, logs=None):
        self.losses['test_loss'].append(logs['loss'])

    def get_loss_df(self):
        total_training_time = time.time() - self.start_time
        average_epoch_times = [sum(self.epoch_times[:i+1]) / (i + 1) for i in range(len(self.epoch_times))]
        self.losses['avg_epoch_time'] = average_epoch_times
        self.losses['total_training_time'] = total_training_time
        return pd.DataFrame(self.losses)