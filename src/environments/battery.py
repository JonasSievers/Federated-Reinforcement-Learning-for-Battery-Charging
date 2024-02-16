from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import wandb


class Battery(py_environment.PyEnvironment):

    """
    Initialize the environment. Default values simulate tesla's powerwall (13.5 kWh, with 4.6 kW power, 2.3 charge, -2.3 discharge, and grid 25 kW)

    :param data: load, pv and electricity price data
    :param init_charge: initial state of charge of the battery in kWh
    :param timeslots_per_day: count of timeslots in 24h interval
    :param max_days: count of days to train on
    :param capacity: capacity of the battery in kWh
    :param power_battery: power of the battery in kW
    :param power_grid: power of the electricity grid in kW
    :param test: activates test logs
    """
    def __init__(
            self, 
            data, #load, pv and electricity price data
            init_charge=0.0, #initial state of charge of the battery in kWh
            timeslots_per_day=48, #count of timeslots in 24h interval
            days=365, #count of days to train on
            capacity=13.5, #capacity of the battery in kWh
            power_battery=4.6, #power of the battery in kW
            power_grid=25.0, #power of the electricity grid in kW
            test=False
        ):
 
        self._current_timestep = -1 #Tracks the current timeslot in the simulation.
        self._timeslots_per_day = timeslots_per_day #Maximum number of timeslots in a 24-hour interval.
        self._max_timesteps = timeslots_per_day * days #Maximum number of days to train on.
        self._capacity = capacity #Capacity of the battery.
        self._power_battery = power_battery # Power of the battery.
        self._init_charge = init_charge #Initial state of charge of the battery.
        self._soe = init_charge #State of charge of the battery (initialized based on init_charge).
        self._power_grid = power_grid #Power of the electricity grid.
        self._episode_ended = False #Boolean flag indicating whether the current episode has ended.
        self._electricity_cost = 0.0 #Cumulative electricity cost incurred during the simulation.
        self._test = test #Boolean flag indicating whether the environment is in test mode.
        self._feed_in_price = 0.076

        # Continuous action space battery=[-2.3,2.3]
        self._action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=-self._power_battery/2, maximum=self._power_battery/2, name='action')

        # Observation space [day,timeslot,soe,load,pv,price]
        self._observation_spec = array_spec.BoundedArraySpec(shape=(6,), dtype=np.float32, minimum=0.0, name='observation')

        #Hold the load, PV, and electricity price data, respectively, passed during initialization.
        self._data = data

    """
    Return the action spec
    :return: action spec
    """
    def action_spec(self):

        return self._action_spec

    """
    Return the oberservation spec
    :return: oberservation spec
    """
    def observation_spec(self):

        return self._observation_spec

    """
    Reset the environment, preparing it for a new episode.
    :return: initial TimeStep
    """
    def _reset(self):
        self._current_timestep = 0
        
        # weekday = self._load_pv_data.iloc[self._current_timestep,0]
        load = self._data.iloc[self._current_timestep,0]
        pv = self._data.iloc[self._current_timestep,1]
        electricity_price = self._data.iloc[self._current_timestep,2]
        fuelmix = self._data.iloc[self._current_timestep,3]

        pv_forecast = self._data.iloc[self._current_timestep+1 : self._current_timestep+5, 1].mean()
        electricity_price_forecast = self._data.iloc[self._current_timestep+1 : self._current_timestep+5,2].mean()

        self._soe = self._init_charge
        self._episode_ended = False
        self._electricity_cost = 0.0

        observation = np.array([self._soe, load, pv, pv_forecast, electricity_price, electricity_price_forecast], dtype=np.float32)
        return ts.restart(observation)

    """
    Simulate the effect of the battery action on the environment, including updating timeslot, loading data, 
    balancing energy, calculating reward, and creating an observation. 

    :param action: action taken by the policy
    :return: next TimeStep
    """
    def _step(self, action):

        #Update the timeslot
        self._current_timestep += 1

        #Check for Episode termination to reset
        if self._episode_ended:
            return self.reset()
        
        #Get the agents action
        battery_action = action[0]

        # load data for current step
        # weekday = self._load_pv_data.iloc[self._current_timestep,0]
        load = self._data.iloc[self._current_timestep,0]
        pv = self._data.iloc[self._current_timestep,1]
        electricity_price = self._data.iloc[self._current_timestep,2]
        fuelmix = self._data.iloc[self._current_timestep,3]

        # load data for forecast
        pv_forecast = self._data.iloc[self._current_timestep+1 : self._current_timestep+5, 1].mean()
        electricity_price_forecast = self._data.iloc[self._current_timestep+1 : self._current_timestep+5,2].mean()

        #Balance energy
        old_soe = self._soe
        energy_from_grid = 0.0
        energy_feed_in = 0.0        

        new_soe = np.clip(old_soe + battery_action, a_min=0.0, a_max=self._capacity, dtype=np.float32)
        amount_charged_discharged = (new_soe - old_soe)
        self._soe = new_soe

        # Stromkauf teuer
        if electricity_price >= self._feed_in_price:
            energy_management = load - pv + amount_charged_discharged
            # Noch mehr Strom benötigt
            if energy_management > 0:
                energy_from_grid = energy_management
            # Strom übrig
            else:
                energy_feed_in = np.abs(energy_management)
        # Stromkauf billig
        else:
            if battery_action < 0:
                energy_from_grid = load
                energy_feed_in = pv + np.abs(amount_charged_discharged)
            else:
                energy_from_grid = load + amount_charged_discharged
                energy_feed_in = pv
                
        # Battery wear model
        if new_soe < (0.2 * self._capacity):
            battery_wear_cost = (0.2 * self._capacity) - new_soe
        elif new_soe > (0.8 * self._capacity):
            battery_wear_cost = new_soe - (0.8 * self._capacity)
        else:
            battery_wear_cost = 0.0
                
        # Calculate Costs and Profits
        cost = energy_from_grid * electricity_price
        profit = energy_feed_in * self._feed_in_price
        self._electricity_cost += profit - cost

        # Calculate reward
        current_reward = 10 * (profit - cost) - battery_wear_cost

        observation = np.array([self._soe, load, pv, pv_forecast, electricity_price, electricity_price_forecast], dtype=np.float32)

        # Log test
        if self._test:
            # wandb.log({'battery action': battery_action, 'soe': new_soe, 'energy leftover or missing': energy_leftover_missing})
            wandb.log({'battery action': battery_action, 'soe': new_soe})

        """
        If the last timeslot is reached and the maximum number of training days has been reached, 
        the episode is marked as ended. The method returns a termination TimeStep object.
        """
        if self._current_timestep >= self._max_timesteps - 5:
            self._episode_ended = True
            if self._test:
                wandb.log({'profit': self._electricity_cost})           
            return ts.termination(observation=observation,reward=current_reward)

        #Return observation and reward
        return ts.transition(observation=observation,reward=current_reward)