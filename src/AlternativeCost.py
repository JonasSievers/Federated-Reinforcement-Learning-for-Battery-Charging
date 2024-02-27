import numpy as np

import sys
sys.path.insert(0, '..')
import utils.dataloader as dataloader

"""
Calculates the alternative cost in the case without a battery
"""
for i in range(1,2):
    train, eval, test = dataloader.loadCustomerData("data/3final_data/combined_data_1.csv")
    timeslot = 1
    cost = 0.0

    while timeslot <= 17567:
        load = eval.iloc[timeslot,0]
        pv = eval.iloc[timeslot,1]
        electricity_price = eval.iloc[timeslot,2]
        net_load = load - pv
        if net_load < 0:
            cost += net_load * 0.024
        else:
            cost += net_load * electricity_price
        timeslot += 1

    print(i)
    print(cost)

