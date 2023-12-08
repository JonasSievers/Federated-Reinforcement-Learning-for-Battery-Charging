import numpy as np

import sys
sys.path.insert(0, '..')
import utils.dataloader as Dataloader

"""
Calculates the alternative cost in the case without a battery
"""
for i in range(13,14):
    load_data, pv_data, price_data = Dataloader.get_customer_data(Dataloader.loadData('data/load1213.csv'),
                                             Dataloader.loadPrice('data/price.csv'), i)
    day = 0
    timeslot = 0
    cost = 0.0
    sum_load = 0.0
    sum_pv = 0.0

    while day <= 364:
        load = load_data.iloc[day][timeslot]
        pv = pv_data.iloc[day][timeslot]
        electricity_price = price_data.iloc[(day * 48) + timeslot][0]
        net_load = load - pv
        if net_load < 0:
            cost += net_load * electricity_price * 0.7
        else:
            print(net_load)
            cost += np.clip(a=net_load, a_min=0, a_max= 100.0) * electricity_price
        sum_load += load
        sum_pv += pv
        if timeslot == 47:
            timeslot = 0
            day += 1
        else:
            timeslot += 1
    print(i)
    print(sum_pv)
    print(sum_load)
    print(cost)

