import numpy as np
import pandas as pd

feed_in_price = 0.076

def costWithoutPV(data, amountTimeslots):
    _, test = data
    profit = 0.0

    for timeslot in range(0,amountTimeslots):
        load = test.iloc[timeslot,0]
        electricity_price = test.iloc[timeslot,2]
        profit -= load * electricity_price
    return profit

def costWithPV(data, amountTimeslots):
    _, test = data
    profit = 0.0

    for timeslot in range(0,amountTimeslots):
        load = test.iloc[timeslot,0]
        pv = test.iloc[timeslot,1]
        electricity_price = test.iloc[timeslot,2]
        net_load = load - pv
        if net_load < 0:
            profit -= net_load * feed_in_price
        else:
            profit -= net_load * electricity_price
    return profit

def costRuleBased2(data, battery_power=2.3, quantil=20, ecoPriority=0):
    
    # Setup data
    train, test = data
    amountTimeslots=17520
    profit = 0.0
    emissions = 0.0
    battery_soe=0.0
    aging_limit = 0.8
    battery_capacity=13.5*aging_limit
    battery_power=battery_power
    feed_in_price=0.076

    battery_soe_list = []  # List to store SoE values

    for timeslot in range(amountTimeslots):
        
        # Load data for the current timeslot
        p_load = test.iloc[timeslot, 0]
        p_pv = test.iloc[timeslot, 1]
        price_buy = test.iloc[timeslot, 2]
        price_forecast = test.iloc[timeslot:timeslot + 19, 2]  # Adjusted to 18 for forecasting horizon
        emission = test.iloc[timeslot, 3]
        emission_forecast = test.iloc[timeslot:timeslot + 19, 3]  # Adjusted to 18 for forecasting horizon

        # Set thresholds based on the selected priority (cost or emissions)
        if ecoPriority == 0:  # Optimize for cost
            low_threshold = np.percentile(price_forecast.squeeze(), quantil)
            high_threshold = np.percentile(price_forecast.squeeze(), 100 - quantil)
        else:  # Optimize for emissions
            low_threshold = np.percentile(emission_forecast.squeeze(), quantil)
            high_threshold = np.percentile(emission_forecast.squeeze(), 100 - quantil)

        # Determine battery action
        if (ecoPriority == 0 and price_buy < low_threshold) or (ecoPriority == 1 and emission < low_threshold):
            # Charge the battery
            charge_amount = min(battery_power, battery_capacity - battery_soe)
            battery_soe += charge_amount
            p_battery = -charge_amount
        elif (ecoPriority == 0 and price_buy > high_threshold) or (ecoPriority == 1 and emission > high_threshold):
            # Discharge the battery
            discharge_amount = min(battery_power, battery_soe)
            battery_soe -= discharge_amount
            p_battery = discharge_amount
        else:
            p_battery = 0.0

        # Append the current SoE to the list
        battery_soe_list.append(battery_soe)

        # Calculate prosumption
        prosumption = p_load - p_pv - p_battery

        # Calculate energy feed-in and energy from grid based on prosumption
        if prosumption < 0:  # Excess energy
            energy_feed_in = -prosumption
            energy_from_grid = 0.0
        else:  # Need energy from the grid
            energy_feed_in = 0.0
            energy_from_grid = prosumption

        # Calculate profit and emissions
        profit += energy_feed_in * feed_in_price - energy_from_grid * price_buy
        emissions += energy_from_grid * emission

    return profit, emissions, battery_soe_list




def ruleBasedCharging(price_buy, price_forecast, emission_forecast, battery_soe, battery_capacity, fixed_battery_action, ecoPriority=0):
    
    if ecoPriority == 0: 
        low_price = np.percentile(price_forecast.squeeze(), 20)
        high_price = np.percentile(price_forecast.squeeze(), 80)
    else:
        low_price = np.percentile(emission_forecast.squeeze(), 20)
        high_price = np.percentile(emission_forecast.squeeze(), 80)

    #Battery Should Stay between 20 and 80 Percent SoE
    if battery_soe > battery_capacity*0.9:
         return (- fixed_battery_action)
    if battery_soe < battery_capacity*0.1:
         return fixed_battery_action
    
    # (Dis-) Charge based on Price level
    if price_buy <= low_price:
        return fixed_battery_action
    if price_buy >= high_price:
        return (- fixed_battery_action)
    return 0.0


def costRuleBased(data, fixed_battery_action, amountTimeslots, battery_soe=0.0, battery_capacity=13.5, battery_power=2.3, ecoPriority=0):
    
    assert 0.0 <= fixed_battery_action <= battery_power

    _, test = data
    profit = 0.0
    emissions = 0.0
    soe_list = []
    
    for timeslot in range(0,amountTimeslots):
        
        # Load data
        p_load =  test.iloc[timeslot,0]
        p_pv =  test.iloc[timeslot,1]
        price_buy = test.iloc[timeslot,2]
        price_forecast = test.iloc[timeslot:timeslot+19,2] # +48,2]
        emission = test.iloc[timeslot, 3]
        emission_forecast = test.iloc[timeslot:timeslot + 19, 3]

        # Charge the battery with available PV power
        charge_amount = min(p_pv, battery_power, battery_capacity - battery_soe)
        battery_soe += charge_amount
        p_pv -= charge_amount
        battery_power -= charge_amount


        # Decide whether to charge or discharge based on price forecast
        battery_action = ruleBasedCharging(price_buy, price_forecast, emission_forecast, battery_soe, battery_capacity, fixed_battery_action, ecoPriority=ecoPriority)
        battery_action = min(battery_action, battery_power)

        # Update battery state and calculate battery power used
        old_soe = battery_soe
        battery_soe = np.clip(old_soe + battery_action, 0.0, battery_capacity)
        p_battery = old_soe - battery_soe

        soe_list.append(p_battery)

        # Calculate prosumption per household
        prosumption = p_load - p_pv - p_battery

        # Calculate energy buy and sell
        energy_feed_in = -prosumption if prosumption < 0 else 0 #positive
        energy_from_grid = prosumption if prosumption > 0 else 0 #positive

        # Calculate profit and emissions
        profit += energy_feed_in * feed_in_price - energy_from_grid * price_buy
        emissions += energy_from_grid * emission

    return profit, emissions, soe_list

def costEmissionsRuleBased(data, fixed_battery_action, amountTimeslots, battery_soe=0.0, battery_capacity=13.5, battery_power=4.6):
    assert fixed_battery_action >= 0.0
    assert fixed_battery_action <= battery_power/2
    _, test = data
    profit = 0.0
    emission = 0.0
    for timeslot in range(0,amountTimeslots):
        # Load data
        p_load =  test.iloc[timeslot,0]
        p_pv =  test.iloc[timeslot,1]
        price_buy = test.iloc[timeslot,2]
        emissions = test.iloc[timeslot,3]
        price_forecast = test.iloc[timeslot:timeslot+19,2] # +48,2]

        # Reduces every time the battery is used each timeslot
        timeslot_power_left = battery_power/2

        # Clip PV to max battery power
        p_pv_clipped = np.clip(p_pv, 0.0, battery_power/2, dtype=np.float32)
        # Leftover PV can be reused in energy management
        p_pv -= p_pv_clipped
        # Reduce power of battery for the timeslot
        timeslot_power_left -= p_pv_clipped
        # Charge battery with PV
        old_soe = battery_soe
        battery_soe = np.clip(old_soe + p_pv_clipped, a_min=0.0, a_max=battery_capacity, dtype=np.float32)
        # Leftover PV can be reused in energy management
        p_pv += np.abs(battery_soe - old_soe - p_pv_clipped)

        # Decide whether charge or discharge on price forecast
        battery_action = ruleBasedCharging(price_buy, price_forecast, battery_soe, battery_capacity, fixed_battery_action)
        # Adapt to left power of battery
        battery_action = min(battery_action, timeslot_power_left)

        # Calculate new SOE
        old_soe = battery_soe
        battery_soe = np.clip(old_soe + battery_action, a_min=0.0, a_max=battery_capacity, dtype=np.float32)
        p_battery = old_soe - battery_soe

        # Initialize values
        energy_from_grid = 0.0
        energy_feed_in = 0.0  

        energy_management = p_load - p_pv - p_battery
        # Energy from grid needed
        if energy_management > 0:
            energy_from_grid = energy_management
        # Feed in remaining energy
        else:
            energy_feed_in = np.abs(energy_management)

        profit += energy_feed_in * feed_in_price - energy_from_grid * price_buy
        emission += np.abs(energy_from_grid) * emissions
    
    return profit, emission