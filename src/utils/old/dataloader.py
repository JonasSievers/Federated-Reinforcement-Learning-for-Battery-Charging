import pandas as pd
import numpy as np

def loadCustomerData(path, customer):
    energy_data = pd.read_csv(path, header=0)
    energy_data.set_index('Date', inplace=True)
    energy_data.fillna(0, inplace=True)
    user_data = energy_data[[f'load_{customer}', f'pv_{customer}', 'price', 'fuelmix']]

    # Split data
    train = user_data[0:35088].set_index(pd.RangeIndex(0,35088))
    test = user_data[35088:52608].set_index(pd.RangeIndex(0,17520))

    return train, test