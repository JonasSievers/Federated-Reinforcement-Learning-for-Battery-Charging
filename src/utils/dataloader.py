import pandas as pd
import numpy as np

def loadCustomerData(path):
    data = pd.read_csv(path, header=0)

    train = data[0:17520].set_index(pd.RangeIndex(0,17520))
    eval = data[17520:35088].set_index(pd.RangeIndex(0,17568))
    test = data[35088:52608].set_index(pd.RangeIndex(0,17520))
    
    return train, eval, test