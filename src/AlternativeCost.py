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
    train, test = dataloader.loadCustomerData("data/3final_data/Final_Energy_dataset.csv",customer)
    timeslot = 0
    profit = 0.0

    while timeslot < 17520:
        load = test.iloc[timeslot,0]
        pv = test.iloc[timeslot,1]
        electricity_price = test.iloc[timeslot,2]
        net_load = load - pv
        if net_load < 0:
            profit -= net_load * 0.076
        else:
            profit -= net_load * electricity_price
        timeslot += 1

    print(customer)
    print(profit)

