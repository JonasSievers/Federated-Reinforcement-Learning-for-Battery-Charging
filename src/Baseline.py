import numpy as np

import utils.dataloader as dataloader

class Baseline:
    def __init__(self, customer=1, battery_soe=0.0, battery_capacity=13.5, battery_power=4.6, feed_in_price=0.076):
        self.train, self.eval, self.test = dataloader.loadCustomerData("data/3final_data/Final_Energy_dataset.csv",customer)
        self.customer = str(customer)
        self.battery_soe = battery_soe
        self.battery_capacity = battery_capacity
        self.battery_power = battery_power
        self.feed_in_price = feed_in_price
        self.fixed_battery_action = 0.0
        self.electricity_cost = 0.0

    def forecast_mean_fn(self, current_price, price_forecast):
        if current_price > price_forecast:
            return (- self.fixed_battery_action)
        elif current_price < price_forecast:
            return self.fixed_battery_action
        else:
            return 0.0
    
    def percentileOverPastTimeslots(self, current_price, price_past):
        if price_past.empty:
            return 0.0
        low_price = np.percentile(price_past.squeeze(), 30)
        high_price = np.percentile(price_past.squeeze(), 90)

        if current_price > high_price:
            return (- self.fixed_battery_action)
        elif current_price < low_price:
            return self.fixed_battery_action
        else:
            return 0.0

    
    def main(self, fixed_battery_action, action_method):
        # Reset cost
        self.electricity_cost = 0.0

        self.fixed_battery_action = fixed_battery_action 
    
        for timeslot in range(0,17520):
            # Load data
            p_load =  self.test["load_"+self.customer].loc[timeslot]
            p_pv =  self.test["pv_"+self.customer].loc[timeslot]
            price_buy = self.test["price"].loc[timeslot]
            # price_forecast = self.test["price"].loc[timeslot+1:timeslot+7].mean()
            price_past = self.test["price"].loc[timeslot-(48*7):timeslot-1]

            # Decide whether charge or discharge on price forecast
            battery_action = action_method(price_buy,price_past)

            # Calculate new SOE
            old_soe = self.battery_soe
            self.battery_soe = np.clip(old_soe + battery_action, a_min=0.0, a_max=self.battery_capacity, dtype=np.float32)
            p_battery = old_soe - self.battery_soe
    
            
            # Initialize values
            energy_from_grid = 0.0
            energy_feed_in = 0.0  

            # Electricy price higher than feed in price
            if price_buy >= self.feed_in_price:
                energy_management = p_load - p_pv - p_battery
                # Energy from grid needed
                if energy_management > 0:
                    energy_from_grid = energy_management
                # Feed in remaining energy
                else:
                    energy_feed_in = np.abs(energy_management)
            # Electricy price lower than feed in price
            else:
                # Discharge battery and sell everything
                if p_battery >= 0:
                    energy_from_grid = p_load
                    energy_feed_in = p_pv + p_battery 
                # Charge battery with energy from grid and feed in pv
                else:
                    energy_from_grid = p_load - p_battery
                    energy_feed_in = p_pv

            cost = energy_from_grid * price_buy
            profit = energy_feed_in * self.feed_in_price
            self.electricity_cost += profit - cost
        
        print(self.electricity_cost)


baseline = Baseline()
for battery_action in np.arange(0.1,2.4,0.1):
    baseline.main(battery_action, baseline.percentileOverPastTimeslots)