from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


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
        self._test_writer = None #TensorFlow summary writer used for test logs.

        if self._test:
            self._test_writer = tf.compat.v2.summary.create_file_writer(
                logdir='./log/test/ddpg', flush_millis=10000
            )

        # Continuous action space battery=[-12.5,12.5]
        self._action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=-self._power_grid/2, maximum=self._power_grid/2, name='action')

        # Observation space
        # self._observation_spec = array_spec.BoundedArraySpec(shape=(3,5), dtype=np.float32, name='observation')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(26,), dtype=np.float32, name='observation')

        #Hold the load, PV, and electricity price data, respectively, passed during initialization.
        self._load_data, self._pv_data, self._electricity_prices = data

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
        # Forecast values
        self._current_timestep = -1 #is set to -1, indicating that the next timeslot will be the first one.
        # load_forecast = self._load_data.iloc[0:1, 0]
        # pv_forecast = self._pv_data.iloc[0:1, 0]
        electricity_price_forecast = self._electricity_prices.iloc[0:24, 0]
        # net_load_forecast = np.subtract(load_forecast, pv_forecast)

        # Reset environemnt
        self._soe = self._init_charge #is set to the initial charge value (_init_charge).
        self._episode_ended = False #signaling the start of a new episode.
        self._electricity_cost = 0.0 #as it accumulates the electricity cost during each episode.

        # Observation
        # observation = np.array([np.repeat(self._init_charge, 25), np.concatenate(([0.0], net_load_forecast)), np.concatenate(([0.0], electricity_price_forecast))], dtype=np.float32)
        # observation = np.concatenate(([self._init_charge, 0.0], np.concatenate(([0.0], electricity_price_forecast))), dtype=np.float32)
        observation = np.concatenate(([self._init_charge], np.concatenate(([0.0], electricity_price_forecast))), dtype=np.float32)
        # observation = np.array([self._init_charge, 0.0], dtype=np.float32)

        #The method returns an initial TimeStep object using the ts.restart function
        return ts.restart(observation)

    """
    The _step method simulates the agent's action in the environment, 
    updates the state based on the action taken, and returns the next TimeStep object, 
    which encapsulates the new state, reward, and whether the episode has ended.
    :param action: action taken by the policy
    :return: next TimeStep
    """
    def _step(self, action):

        """
        Update Timeslot:
        We first update the current timeslot and day in the simulation.
        If the current timeslot exceeds the maximum allowed timeslots for a day, 
        it increments the current day and resets the timeslot to the first one. This simulates the transition to a new day.
        """
        self._current_timestep += 1

        """
        If the episode has already ended (_episode_ended is True), the environment is reset to its initial state by calling the reset method.
        """
        if self._episode_ended:
            return self.reset()
        
        # load data for current step
        # load = self._load_data.iloc[self._current_timestep, 0]
        # load_forecast = self._load_data.iloc[self._current_timestep+1 : self._current_timestep+2, 0]
        # pv = self._pv_data.iloc[self._current_timestep, 0]
        # pv_forecast = self._pv_data.iloc[self._current_timestep+1 : self._current_timestep+2, 0]
        electricity_price = self._electricity_prices.iloc[self._current_timestep, 0]
        electricity_price_forecast= self._electricity_prices.iloc[self._current_timestep+1 : self._current_timestep+25, 0]
        # net_load = load - pv
        # net_load_forecast = np.subtract(load_forecast, pv_forecast)

        # balance energy
        old_soe = self._soe
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(old_soe, action[0], 0.0)
        self._soe = new_soe

        # Calculate Costs and Profits
        cost = energy_from_grid * electricity_price
        profit = energy_feed_in * electricity_price * 0.7
        self._electricity_cost += cost - profit

      
        # Battery wear model
        if new_soe < (0.2 * self._capacity):
            battery_wear_cost = (0.2 * self._capacity) - new_soe
        elif new_soe > (0.8 * self._capacity):
            battery_wear_cost = new_soe - (0.8 * self._capacity)
        else:
            battery_wear_cost = 0.0


        # Calculate reward
        current_reward = (profit - cost) - battery_wear_cost - energy_missing - energy_leftover

        # Observation
        # observation = np.array([new_soe, electricity_price], dtype=np.float32)
        # observation = np.concatenate(([new_soe, load], load_forecast, [pv], pv_forecast, [electricity_price], electricity_price_forecast), dtype=np.float32)
        # observation = np.array([np.repeat(new_soe, 25), np.concatenate(([net_load], net_load_forecast)), np.concatenate(([electricity_price], electricity_price_forecast))], dtype=np.float32)
        # observation = np.array([np.repeat(new_soe, 25), np.concatenate(([electricity_price], electricity_price_forecast))], dtype=np.float32)
        observation = np.concatenate(([new_soe], np.concatenate(([electricity_price], electricity_price_forecast))), dtype=np.float32)
        # observation = np.concatenate(([new_soe, net_load], np.concatenate(([electricity_price], electricity_price_forecast))), dtype=np.float32)
    
        # Log test
        if self._test:
            with self._test_writer.as_default(step=self._current_timestep):
                tf.summary.scalar(name='action', data=action[0])
                tf.summary.scalar(name='soe', data=new_soe)
                tf.summary.scalar(name='energy missing', data=energy_missing)
                tf.summary.scalar(name='energy leftover', data=energy_leftover)
                # tf.summary.scalar(name='net_load', data=net_load)
                tf.summary.scalar(name='price', data=electricity_price)
                tf.summary.scalar(name='current reward', data=current_reward)

        """
        If the last timeslot is reached and the maximum number of training days has been reached, 
        the episode is marked as ended. The method returns a termination TimeStep object.
        """
        if self._current_timestep >= self._max_timesteps - 25:
            self._episode_ended = True
            if self._test:
                with self._test_writer.as_default(step=self._current_timestep):
                    tf.summary.scalar(name='cost', data=self._electricity_cost)
            
            return ts.termination(observation=observation,reward=current_reward)

        """
        For regular time steps, a transition TimeStep object is returned, representing the new state, 
        the negative of the current reward (as the environment aims to minimize costs), 
        and a discount factor of 1.0 (indicating no discounting in this scenario).
        """
        return ts.transition(observation=observation,reward=current_reward)

    def _chargeBattery(self, amount_to_charge, old_soe):
        clipped_amount_to_charge = np.clip(amount_to_charge, 0.0, self._power_battery/2)
        new_soe = np.clip(clipped_amount_to_charge + old_soe, 0.0, self._capacity, dtype=np.float32)
        amount_charged = (new_soe - old_soe)
        energy_leftover = amount_to_charge - amount_charged
        return new_soe, energy_leftover

    def _dischargeBattery(self, amount_to_discharge, old_soe):
        clipped_amount_to_discharge = np.clip(amount_to_discharge, 0.0, self._power_battery/2) 
        new_soe = np.clip(old_soe - clipped_amount_to_discharge, 0.0, self._capacity, dtype=np.float32)
        amount_discharged = (old_soe - new_soe)
        energy_missing = amount_to_discharge - amount_discharged
        return new_soe, energy_missing

    def _balanceHousehold(self, old_soe, action, net_load):
        energy_from_grid = 0.0
        energy_missing = 0.0
        energy_leftover = 0.0
        energy_feed_in = 0.0

        energy_management = net_load - action
        # Sell energy
        if action < 0:
            # Do not need battery
            if energy_management < 0:
                # Sell action
                energy_feed_in = np.abs(action)
                # Store leftover energy
                amount_to_charge = np.abs(energy_management)
                new_soe, energy_leftover = self._chargeBattery(amount_to_charge, old_soe)
            # Need battery
            elif energy_management > 0:
                new_soe, energy_missing = self._dischargeBattery(energy_management, old_soe)
                energy_feed_in = np.clip(np.abs(action) - energy_missing, 0.0, None, dtype=np.float32)
            else:
                new_soe = old_soe
                energy_feed_in = np.abs(action)
        # Buy energy or do nothing
        else:
            # More energy than needed -> charge battery
            if energy_management < 0:
                amount_to_charge = np.abs(energy_management)
                new_soe, energy_leftover = self._chargeBattery(amount_to_charge, old_soe)
                energy_from_grid = np.clip(action - energy_leftover, 0.0, None)
            # Need battery
            elif energy_management > 0:
                energy_from_grid = action
                amount_to_discharge = energy_management
                new_soe, energy_missing = self._dischargeBattery(amount_to_discharge, old_soe)
            else:
                new_soe = old_soe
                energy_from_grid = action
        
        if np.isclose([energy_missing], [0.0]):
            energy_missing = 0.0
        if np.isclose([energy_leftover], [0.0]):
            energy_leftover = 0.0

        return new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover