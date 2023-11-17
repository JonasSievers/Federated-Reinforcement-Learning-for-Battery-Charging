#Imports
#Normalization
from sklearn.preprocessing import MinMaxScaler
#Data handling
import pandas as pd
#Create data arrays
import numpy as np

#The Datahandler class contains usefull methods for data analysis and sequencing
class Datahandler(): 

    #min_max_scaling
    #Sclaes all columns of the dataframe df to the rang (0,1)
    def min_max_scaling(self, df): #normailizing
        #Min Max Sclaing
        col_names = df.columns
        features = df[col_names]
        scaler = MinMaxScaler().fit(features.values)
        features = scaler.transform(features.values)
        df_scaled = pd.DataFrame(features, columns = col_names, index=df.index)
        return df_scaled

    #standardizing_df
    #Standardizes all columns of the dataframe df by subtracting its mean and devide the result by the standard deviation
    def standardizing_df(df):
        #Normalize Data
        mean = df.mean()
        std = df.std()

        normalized_df = (df - mean) / std
        return normalized_df

    #create_sequences
    #Split the dataframe into datasets with sequences of lngth=Sequence_length
    def create_sequences(self, df, sequence_length):
        sequences = []
        for i in range(len(df) - sequence_length + 1):
            sequence = df.iloc[i:i+sequence_length, :]  # Take all columns
            sequences.append(sequence.values)
        return np.array(sequences)

    #prepare_data
    # Split each sequence into X (features) and Y (labels). 
    # The label Y must be the FIRST column! The last batch is discarded, when < batch_size
    def prepare_data(self, sequences, batch_size):
        X = sequences[:, :-1, :].astype('float32') #For all sequences, Exclude last row of the sequence, take all columns
        y = sequences[:, -1, 0].astype('float32') #For all sequences, Take the last row of the sequence, take the first column

        #As some models need to reshape the inputs, the correct batch_size is important
        #Adjust the dataset_size to be divisible by batch_size by discarding the remaining data points not fitting a complete batch.
        num_batches = len(X) // batch_size
        adjusted_X = X[:num_batches * batch_size]
        adjusted_y = y[:num_batches * batch_size]

        return adjusted_X, adjusted_y
    