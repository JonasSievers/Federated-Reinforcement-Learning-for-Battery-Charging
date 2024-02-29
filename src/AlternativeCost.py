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
    train, eval, test = dataloader.loadCustomerData("data/3final_data/Final_Energy_dataset.csv",customer)
    timeslot = 0
    cost = 0.0

    while timeslot < 17520:
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

