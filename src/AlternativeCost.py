import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '..')
import utils.dataloader as dataloader

"""
Calculates the alternative cost in the case without a battery
"""
for customer in range(1,2):
   # Load data
    energy_data = pd.read_csv("data/3final_data/Final_Energy_dataset.csv", header=0)
    energy_data.set_index('Date', inplace=True)
    energy_data.fillna(0, inplace=True)
    user_data = energy_data[[f'load_{customer}', f'pv_{customer}', 'price', 'fuelmix']]

    # Split data
    train = user_data[0:17520].set_index(pd.RangeIndex(0,17520))
    eval = user_data[17520:35088].set_index(pd.RangeIndex(0,17568))
    test = user_data[35088:52608].set_index(pd.RangeIndex(0,17520))
    timeslot = 1
    cost = 0.0

    while timeslot <= 17519:
        load = test.iloc[timeslot,0]
        pv = test.iloc[timeslot,1]
        electricity_price = test.iloc[timeslot,2]
        net_load = load - pv
        if net_load < 0:
            cost += net_load * 0.024
        else:
            cost += net_load * electricity_price
        timeslot += 1

    print(customer)
    print(cost)

