import numpy as np

import sys
sys.path.insert(0, '..')
import utils.dataloader as dataloader

"""
Calculates the alternative cost in the case without a battery
"""
for i in range(1,2):
    load_data, pv_data, price_data = dataloader.get_customer_data(dataloader.loadData('./data/load1213.csv'),
                                             dataloader.loadPrice('./data/price.csv'), None, i)
    day = 0
    timeslot = 0
    cost = 0.0
    sum_load = 0.0
    sum_pv = 0.0

    while timeslot <= 17517:
        load = load_data.iloc[timeslot,0]
        pv = pv_data.iloc[timeslot,0]
        electricity_price = price_data.iloc[timeslot,0]
        net_load = load - pv
        if net_load < 0:
            cost += net_load * electricity_price * 0.7
        else:
            cost += net_load * electricity_price
        sum_load += load
        sum_pv += pv
        timeslot += 1
    print(cost)

