import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import wandb


class EnergyManagementEnv(py_environment.PyEnvironment):

    def __init__(self, load_data, pv_data, price_data):
 
        #Time steps
        self._current_timestep = -1
        self._max_timesteps = (48 * 365)-1 #Timeslots: 2*24*365, starting at 0 -> -1

        #Battery parameters
        self._capacity = 13.5
        self._power_battery = 4.6/2
        self._init_charge = 0.0
        self._soe = 0.0
        self._power_grid = 25.0
        self._feed_in_tarif = 0.076
        self._total_electricity_bill = 0.0
        self._efficiency = 0.95

        #Observation and Action space
        self._action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=-self._power_battery, maximum=self._power_battery, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(6,), dtype=np.float32, name='observation')

        #Data
        self._load_data = load_data
        self._pv_data = pv_data
        self._electricity_prices = price_data

        #Reset
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._current_timestep = -1 #is set to -1, indicating that the next timeslot will be the first one.
        self._soe = self._init_charge #is set to the initial charge value (_init_charge).
        self._total_electricity_bill = 0.0 #as it accumulates the electricity cost during each episode.
       
        observation = np.array([
            self._soe, 
            self._load_data.iloc[0,0], 
            self._pv_data.iloc[0,0], 
            self._electricity_prices.iloc[0,0], 
            self._electricity_prices.iloc[1:5,0].mean(), # Price Prediction: 2 hour mean
            self._pv_data.iloc[1:5,0].mean()], # PV Prediction: 4 hour mean
            dtype=np.float32)
        self._episode_ended = False
        
        return ts.restart(observation)


    def _step(self, action):
        
        #0. Set time
        self._current_timestep += 1
        #Check for Episode termination to reset
        if self._episode_ended:
            return self.reset()

        # 1. Balance Battery -> Action clipping unnötig
        soe_old = self._soe
        self._soe = np.clip(soe_old + action[0], 0.0, self._capacity, dtype=np.float32)
        penalty_soe  = np.abs(action[0] - (self._soe - soe_old))*5
        
        lower_threshold, upper_threshold = 0.10 * self._capacity, 0.90 * self._capacity
        if self._soe < lower_threshold:
            penalty_aging = np.abs(lower_threshold - self._soe) * 5
        elif self._soe > upper_threshold:
            penalty_aging = np.abs(self._soe - upper_threshold) * 5
        else:
            penalty_aging = 0 

        p_battery = soe_old - self._soe #Clipped to actual charging. + -> discharging/ providing energy
        
        #2. Get data
        price_buy = self._electricity_prices.iloc[self._current_timestep, 0]
        p_load = self._load_data.iloc[self._current_timestep, 0] 
        p_pv = self._pv_data.iloc[self._current_timestep, 0] 
        #2.1 Get forecasts
        price_forecast_1 = self._electricity_prices.iloc[self._current_timestep+1 : self._current_timestep+5, 0].mean() # Mean of next 2 hours
        p_pv_forecast_1 = self._pv_data.iloc[self._current_timestep+1 : self._current_timestep+5, 0].mean() # Mean of next 2 hours
        
        #3. Balance Grid
        if price_buy >= self._feed_in_tarif: # Hoher Kaufpreis, daher Eigennutzung
            grid = p_load - p_pv - p_battery
            grid_buy = grid if grid > 0 else 0
            grid_sell = abs(grid) if grid < 0 else 0
        elif price_buy < self._feed_in_tarif: # Hoher Verkaufspreis, daher alles verkaufen
            grid_buy = p_load
            grid_sell = p_pv + p_battery

        #4. Calculate profit
        profit = (grid_sell*self._feed_in_tarif) - (grid_buy*price_buy)
        self._total_electricity_bill += profit

        #5. Calculate reward
        reward = profit - penalty_soe - penalty_aging
        
        #6. Create observation
        observation = np.array([self._soe, p_load, p_pv, price_buy, price_forecast_1, p_pv_forecast_1], dtype=np.float32)
        
        if self._current_timestep % 100 == 0:
            #Logging
            wandb.log({
                'Action [2.3, -2.3]': action[0], 
                'SoE [0, 13.5]': self._soe, 
                'Imbalance': penalty_soe ,
                'Profit (+ profit, - cost)': profit,
                'Total Profit': self._total_electricity_bill,
                'Current Timestep' : self._current_timestep,
                'Reward' : reward,
                })
        # Check for episode end
        if self._current_timestep >= self._max_timesteps-8: 
            #wandb.log({'Profit': self._total_electricity_bill})
            self._episode_ended = True
            return ts.termination(observation=observation,reward=reward)          
        else:
            return ts.transition(observation=observation,reward=reward)