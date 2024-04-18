import numpy as np

import utils.dataloader as dataloader

class Baseline:
    def __init__(self, customer=1, battery_soe=0.0, battery_capacity=13.5, battery_power=4.6, feed_in_price=0.076):
        self.train, self.test = dataloader.loadCustomerData("data/3final_data/Final_Energy_dataset.csv",customer)
        self.customer = str(customer)
        self.battery_soe = battery_soe
        self.battery_capacity = battery_capacity
        self.battery_power = battery_power
        self.feed_in_price = feed_in_price
        self.fixed_battery_action = 0.0
        self.electricity_cost = 0.0
    
    def ruleBasedCharching(self, current_price, price, pv):
        if price.empty:
            return 0.0
        
        low_price = np.percentile(price.squeeze(), 20)

        # Strom sehr billig über die nächsten 24h oder PV Erzeugung
        if (current_price <= low_price) | (pv > 0):
            return self.fixed_battery_action
        # Keine PV Erzeugung
        if (pv <= 0):
            return (- self.fixed_battery_action)
        return 0.0

    def main(self, fixed_battery_action):
        # Reset
        self.electricity_cost = 0.0
        self.battery_soe = 0.0

        self.fixed_battery_action = fixed_battery_action 
    
        for timeslot in range(0,17520):
            # Load data
            p_load =  self.test["load_"+self.customer].loc[timeslot]
            p_pv =  self.test["pv_"+self.customer].loc[timeslot]
            price_buy = self.test["price"].loc[timeslot]
            price_forecast = self.test["price"].loc[timeslot:timeslot+48]

            # Decide whether charge or discharge on price forecast
            battery_action = self.ruleBasedCharching(price_buy,price_forecast,p_pv)

            # Calculate new SOE
            old_soe = self.battery_soe
            self.battery_soe = np.clip(old_soe + battery_action, a_min=0.0, a_max=self.battery_capacity, dtype=np.float32)
            p_battery = old_soe - self.battery_soe

            p_battery_left = self.battery_power - np.abs(p_battery)
    
            
            # Initialize values
            energy_from_grid = 0.0
            energy_feed_in = 0.0  

            energy_management = p_load - p_pv - p_battery
            # Energy from grid needed
            if energy_management > 0:
                energy_from_grid = energy_management
            # Feed in remaining energy
            else:
                charge_battery = np.clip(abs(energy_management),a_min=0.0, a_max=min(p_battery_left,self.battery_capacity-self.battery_soe))
                leftover_energy = abs(energy_management)-charge_battery
                self.battery_soe = np.clip(old_soe + charge_battery, a_min=0.0, a_max=self.battery_capacity, dtype=np.float32)
                energy_feed_in = leftover_energy


            cost = energy_from_grid * price_buy
            profit = energy_feed_in * self.feed_in_price
            self.electricity_cost += profit - cost
        
        print(self.electricity_cost)


baseline = Baseline()
# for battery_action in np.arange(0.1,2.4,0.1):
baseline.main(0.2)