from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
import wandb


class Household(py_environment.PyEnvironment):

    """
    Initialize the environment. Default values simulate tesla's powerwall

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
            init_charge, #initial state of charge of the battery in kWh
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

        # Continuous action space battery=[-12.5,12.5]
        self._action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=-self._power_grid/2, maximum=self._power_grid/2, name='action')

        # Observation space
        self._observation_spec = array_spec.BoundedArraySpec(shape=(5,), dtype=np.float32, name='observation')

        #Hold the load, PV, and electricity price data, respectively, passed during initialization.
        self._load_pv_data, self._electricity_prices, self._electricity_prices_scaled = data

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
        
        weekday = self._load_pv_data.iloc[self._current_timestep,0]
        load = self._load_pv_data.iloc[self._current_timestep,1]
        pv = self._load_pv_data.iloc[self._current_timestep,2]
        electricity_price = self._electricity_prices[self._current_timestep]
        net_load = load - pv

        self._soe = self._init_charge
        self._episode_ended = False
        self._electricity_cost = 0.0

        # Weekday, Timeslot, SoE, Load, PV, Price
        observation = np.array([weekday, 0, self._soe, net_load, electricity_price], dtype=np.float32)
        return ts.restart(observation)

    """
    The _step method simulates the agent's action in the environment, 
    updates the state based on the action taken, and returns the next TimeStep object, 
    which encapsulates the new state, reward, and whether the episode has ended.
    :param action: action taken by the policy
    :return: next TimeStep
    """
    def _step(self, action):

        #Update Timeslot:
        self._current_timestep += 1

        #Check for episode end
        if self._episode_ended:
            return self.reset()
        
        # manage actions
        grid_action = action[0]
        
        # load data for current step
        weekday = self._load_pv_data.iloc[self._current_timestep,0]
        load = self._load_pv_data.iloc[self._current_timestep,1]
        pv = self._load_pv_data.iloc[self._current_timestep,2]
        net_load = load - pv
        electricity_price = self._electricity_prices[self._current_timestep]

        # balance energy
        old_soe = self._soe
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(old_soe, grid_action, net_load)
        self._soe = new_soe

        # Calculate Costs and Profits
        cost = energy_from_grid * electricity_price
        profit = energy_feed_in * electricity_price * 0.7
        self._electricity_cost += profit - cost

      
        # Battery wear model
        if new_soe < (0.2 * self._capacity):
            battery_wear_cost = (0.2 * self._capacity) - new_soe
        elif new_soe > (0.8 * self._capacity):
            battery_wear_cost = new_soe - (0.8 * self._capacity)
        else:
            battery_wear_cost = 0.0


        # Calculate reward
        current_reward = 10*(profit - cost) - energy_missing - energy_leftover - battery_wear_cost

        # Day calculation
        timeslot = self._current_timestep % self._timeslots_per_day

        observation = np.array([weekday, timeslot, self._soe, net_load, electricity_price], dtype=np.float32)
    
        # Log test
        if self._test:
            wandb.log({'grid action': grid_action, 'soe': new_soe, 'energy leftover': energy_leftover, 'energy missing': energy_missing})

        """
        If the last timeslot is reached and the maximum number of training days has been reached, 
        the episode is marked as ended. The method returns a termination TimeStep object.
        """
        if self._current_timestep >= self._max_timesteps - 1:
            self._episode_ended = True
            if self._test:
                wandb.log({'profit': self._electricity_cost})    
            return ts.termination(observation=observation,reward=current_reward)

        """
        For regular time steps, a transition TimeStep object is returned, representing the new state, 
        the negative of the current reward (as the environment aims to minimize costs), 
        and a discount factor of 1.0 (indicating no discounting in this scenario).
        """
        return ts.transition(observation=observation,reward=current_reward)

    def _chargeDischargeBattery(self, amount, old_soe):
        clipped_amount = np.clip(amount, -self._power_battery/2, self._power_battery/2)
        new_soe = np.clip(old_soe + clipped_amount, 0.0, self._capacity)
        soe_change = (new_soe - old_soe)
        energy_leftover_missing = np.abs(amount - soe_change)
        return new_soe, energy_leftover_missing

    def _balanceHousehold(self, old_soe, action, net_load):
        energy_from_grid = 0.0
        energy_missing = 0.0
        energy_leftover = 0.0
        energy_feed_in = 0.0

        energy_management = action - net_load
        # Sell energy
        if action < 0:
            # Need battery -> discharge battery
            if energy_management < 0:
                new_soe, energy_missing = self._chargeDischargeBattery(energy_management, old_soe)
                energy_feed_in = np.clip(np.abs(action) - energy_missing, 0.0, np.abs(action))
            # Do not need battery
            elif energy_management > 0:
                 # Sell action
                energy_feed_in = np.abs(action)
                # Store leftover energy
                new_soe, energy_leftover = self._chargeDischargeBattery(energy_management, old_soe)
            else:
                new_soe = old_soe
                energy_feed_in = np.abs(action)
        # Buy energy or do nothing
        else:
            # Need battery -> discharge battery
            if energy_management < 0:
                new_soe, energy_missing = self._chargeDischargeBattery(energy_management, old_soe)
                energy_from_grid = action
            # More energy than needed -> charge battery
            elif energy_management > 0:
                new_soe, energy_leftover = self._chargeDischargeBattery(energy_management, old_soe)
                energy_from_grid = np.clip(action - energy_leftover, 0.0, action)
            else:
                new_soe = old_soe
                energy_from_grid = action
        
        if np.isclose([energy_missing], [0.0]):
            energy_missing = 0.0
        if np.isclose([energy_leftover], [0.0]):
            energy_leftover = 0.0

        return new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover