import utils.dataloader as dataloader
from IPython.display import display
from enum import Enum
import numpy as np

class Price(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2

class Baseline:
    def __init__(self, customer=1, battery_soe=0.0, battery_capacity=13.5, battery_power=4.6):
        self.load_data, self.pv_data, self.price_data = dataloader.get_customer_data(dataloader.loadData('data/load1213.csv'), dataloader.loadPrice('data/price.csv'), customer)
        self.customer = customer
        self.battery_soe = battery_soe
        self.battery_capacity = battery_capacity
        self.battery_power = battery_power
        self.current_price = 0
        self.charge_discharge_amount = 1
        
    def percentileOverPastTimeslots(self, length, window, low_percentile, high_percentile):
        low_price = np.percentile(window.iloc[0:length].squeeze(), low_percentile)
        high_price = np.percentile(window.iloc[0:length].squeeze(), high_percentile)
        return Price.HIGH if self.current_price > high_price else Price.LOW if self.current_price < low_price else Price.NORMAL
    
    def percentileOverFutureTimeslots(self, length, window, low_percentile, high_percentile):
        low_price = np.percentile(window.iloc[0:length].squeeze(), low_percentile)
        high_price = np.percentile(window.iloc[0:length].squeeze(), high_percentile)
        return Price.HIGH if self.current_price > high_price else Price.LOW if self.current_price < low_price else Price.NORMAL
    
    def calculateBatteryEvents(self, timestep, price_status):
        load = self.load_data.iloc[timestep,0]
        pv = self.pv_data.iloc[timestep,0]
        net_load = load - pv
        cost = 0.0
        profit = 0.0
        used_power = 0.0

        # Process the net load
        # If positive energy needed, otherwise energy abundance
        if net_load > 0:
            match price_status:
                case Price.LOW:
                    # Take energy from grid
                    cost = net_load * self.current_price
                case Price.NORMAL:
                    cost, used_power = self.dischargeBatteryNeed(net_load)
                    pass
                case Price.HIGH:
                    # Take energy from battery
                    cost, used_power = self.dischargeBatteryNeed(net_load)
                    pass
        else:
            match price_status:
                case Price.LOW:
                    # Store leftover energy
                    profit, used_power = self.chargeBatteryLeftover(net_load)
                    pass
                case Price.NORMAL:
                    # Store leftover energy
                    profit, used_power = self.chargeBatteryLeftover(net_load)
                    pass
                case Price.HIGH:
                    # Sell leftover energy
                    profit = abs(net_load) * self.current_price * 0.7
                    pass


        if price_status == Price.LOW:
            # Charge Battery
            cost += self.chargeBatteryCheap(used_power)
        if price_status == Price.HIGH:
            # Discharge Battery
            profit += self.dischargeBatteryExpensive(used_power)
 
        return profit-cost


    def dischargeBatteryNeed(self, net_load):
        cost = 0.0
        # Discharge battery at most with maximum battery power
        discharge = min(net_load, self.battery_power/2)
        # More energy need then battery power can provide
        if net_load > self.battery_power/2:
            # Buy remaining energy from grid
            cost += (net_load - self.battery_power/2) * self.current_price

        new_battery_soe = self.battery_soe - discharge
        # if battery is discharged below 0% => charge"back to 0%
        if new_battery_soe < 0:
            cost += abs(new_battery_soe) * self.current_price
        clipped_soe = np.clip(new_battery_soe, a_min=0.0, a_max=self.battery_capacity)
        used_power = abs(self.battery_soe-clipped_soe)
        self.battery_soe = clipped_soe
        return cost, used_power
    
    def chargeBatteryLeftover(self, net_load):
        profit = 0.0
        production = abs(net_load)
        charge = min(production, self.battery_power/2)
        if production > self.battery_power/2:
            profit += (production - self.battery_power/2) * self.current_price * 0.7
        new_battery_soe = self.battery_soe + charge
        # if battery is charged below 100% => discharge back to 100%
        if new_battery_soe > self.battery_capacity:
            profit += (new_battery_soe - self.battery_capacity) * self.current_price * 0.7
        clipped_soe = np.clip(new_battery_soe, a_min=0.0, a_max=self.battery_capacity)
        used_power = abs(self.battery_soe-clipped_soe)
        self.battery_soe = clipped_soe
        return profit, used_power
    
    def chargeBatteryCheap(self, used_power):
        max_charge = self.battery_capacity - self.battery_soe
        power = self.battery_power/2 - used_power
        charge = min(max_charge, power, self.charge_discharge_amount)
        cost = charge * self.current_price
        self.battery_soe = self.battery_soe + charge
        return cost

    def dischargeBatteryExpensive(self, used_power):
        max_discharge = self.battery_soe
        power = self.battery_power/2 - used_power
        discharge = min(max_discharge, power, self.charge_discharge_amount)
        profit = discharge * self.current_price * 0.7
        self.battery_soe = self.battery_soe - discharge
        return profit


    def slidingWindow(self, length, timestep):
        shifted_price = self.price_data.reindex(index=np.roll(self.price_data.index,length-timestep))
        return shifted_price.iloc[0:length].squeeze()

    def main(self,method):
        for i in np.arange(0.1,2.4,0.1):
            cost = 0.0
            self.charge_discharge_amount=i
            for timestep in range(0,17520):
                self.current_price = self.price_data.iloc[timestep,0].squeeze()
                window = self.slidingWindow(24, timestep)
                price_status = method(24,window,30,90)
                value = self.calculateBatteryEvents(timestep, price_status)
                cost += value
            print(i)
            print(cost)
            print("----")

# Programm
baseline = Baseline()
baseline.main(baseline.percentileOverPastTimeslots)

#    TODO