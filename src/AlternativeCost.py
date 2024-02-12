import numpy as np

import sys
sys.path.insert(0, '..')
import utils.dataloader as Dataloader
import utils.new_dataloader as new_dataloader

"""
Calculates the alternative cost in the case without a battery
"""
for i in range(1,2):
    train, eval, test = new_dataloader.getCustomerData('./data/load1011.csv','./data/load1112.csv','./data/load1213.csv','./data/price_wo_outlier.csv', i)
    data, electricity_prices, electricity_prices_scaled = eval
    timeslot = 1
    cost = 0.0

    while timeslot <= 17519:
        load = data.iloc[timeslot,1]
        pv = data.iloc[timeslot,2]
        electricity_price = electricity_prices_scaled[timeslot]
        net_load = load - pv
        if net_load < 0:
            cost += net_load * electricity_price * 0.7
        else:
            cost += net_load * electricity_price
        timeslot += 1

    print(i)
    print(cost)

