import pandas as pd
import numpy as np

def loadCustomerData(customer):
    data = pd.read_csv("data/3final_data/combined_data_"+str(customer)+".csv", header=0)

    train = data[0:17520].set_index(pd.RangeIndex(0,17520))
    eval = data[17520:35088].set_index(pd.RangeIndex(0,17568))
    test = data[35088:52608].set_index(pd.RangeIndex(0,17520))
    
    return train, eval, test