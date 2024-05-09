import numpy as np

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

def ruleBasedCharching(current_price, price, battery_soe, battery_capacity, fixed_battery_action):
    if price.empty:
        return 0.0
    
    low_price = np.percentile(price.squeeze(), 10)
    high_price = np.percentile(price.squeeze(), 50)

    if battery_soe > battery_capacity*0.8:
         return (- fixed_battery_action)
    if battery_soe < battery_capacity*0.2:
         return fixed_battery_action
    if current_price <= low_price:
        return fixed_battery_action
    if current_price >= high_price:
        return (- fixed_battery_action)
    return 0.0

def costRuleBased(data, fixed_battery_action, amountTimeslots, battery_soe=0.0, battery_capacity=13.5, battery_power=4.6):
    assert fixed_battery_action >= 0.0
    assert fixed_battery_action <= battery_power/2
    _, test = data
    profit = 0.0
    for timeslot in range(0,amountTimeslots):
        # Load data
        p_load =  test.iloc[timeslot,0]
        p_pv =  test.iloc[timeslot,1]
        price_buy = test.iloc[timeslot,2]
        price_forecast = test.iloc[timeslot:timeslot+48,2]

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
        battery_action = ruleBasedCharching(price_buy, price_forecast, battery_soe, battery_capacity, fixed_battery_action)
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
    return profit