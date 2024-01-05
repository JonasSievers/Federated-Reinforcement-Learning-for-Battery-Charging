from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


class Environment(py_environment.PyEnvironment):

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
        self._soe = np.float32(init_charge) #State of charge of the battery (initialized based on init_charge).
        self._power_grid = power_grid #Power of the electricity grid.
        self._episode_ended = False #Boolean flag indicating whether the current episode has ended.
        self._electricity_cost = 0.0 #Cumulative electricity cost incurred during the simulation.
        self._test = test #Boolean flag indicating whether the environment is in test mode.
        self._test_writer = None #TensorFlow summary writer used for test logs.

        if self._test:
            self._test_writer = tf.compat.v2.summary.create_file_writer(
                logdir='./log/test/ddpg', flush_millis=10000
            )

        # Continuous action space battery=[-2.3,2.3]
        #self._action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=-self._power_battery/2, maximum=self._power_battery/2, name='action')
        self._action_spec = array_spec.BoundedArraySpec(shape=(1,), dtype=np.float32, minimum=-self._power_battery / 2,
                                                        maximum=self._power_grid / 2, name='action')
        
        # Observation space [soe,load,pv,price, price_forecast]
        # self._observation_spec = array_spec.BoundedArraySpec(shape=(5,), dtype=np.float32, name='observation')
        self._observation_spec = array_spec.BoundedArraySpec(shape=(4,), dtype=np.float32, minimum=0.0,
                                                             name='observation')
        
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

        self._current_timestep = -1 #is set to -1, indicating that the next timeslot will be the first one.
        electricity_price_forecast = self._electricity_prices.iloc[0, 0]
        self._soe = self._init_charge #is set to the initial charge value (_init_charge).
        self._episode_ended = False #signaling the start of a new episode.
        self._electricity_cost = 0.0 #as it accumulates the electricity cost during each episode.
        
        #The method returns an initial TimeStep object using the ts.restart function. [SoE, load, pv, price]
        # return ts.restart(np.array([self._init_charge, 0.0, 0.0, 0.0, electricity_price_forecast], dtype=np.float32))
        return ts.restart(
            np.array([self._init_charge, 0.0, 0.0, 0.0], dtype=np.float32)
            )

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
        
        # manage actions
        battery_action = action[0]

        # load data for current step
        load = self._load_data.iloc[self._current_timestep, 0]
        pv = self._pv_data.iloc[self._current_timestep, 0]
        electricity_price = self._electricity_prices.iloc[self._current_timestep, 0]
        net_load = load - pv
        # fuel_mix = self._fuel_mix.iloc[self._current_timestep]

        # load data for forecast
        # load_forecast = self._load_data.iloc[self._current_timestep +1, 0]
        # pv_forecast = self._pv_data.iloc[self._current_timestep +1, 0]
        # electricity_price_forecast = self._electricity_prices.iloc[self._current_timestep+1, 0]
        # fuel_mix = self._fuel_mix.iloc[self._current_timestep]

        # balance energy
        old_soe = self._soe
        # new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(old_soe, battery_action, net_load)
        

        new_soe, energy_from_grid, energy_missing, energy_leftover, energy_feed_in = self.balanceHousehold(old_soe,
                                                                                                           load,
                                                                                                           pv,
                                                                                                           battery_action)
        self._soe = new_soe

        # Calculate Costs and Profits
        cost = energy_from_grid * electricity_price
        profit = energy_feed_in * electricity_price * 0.7
        self._electricity_cost += cost - profit

        # Battery wear model
        """
        A battery wear cost is calculated based on the current state of charge. 
        If the state of charge is below 20% or above 80% of the battery capacity, wear costs are incurred.
        """
        if new_soe < (0.2 * self._capacity):
            battery_wear_cost = (0.2 * self._capacity) - new_soe
        elif new_soe > (0.8 * self._capacity):
            battery_wear_cost = new_soe - (0.8 * self._capacity)
        else:
            battery_wear_cost = 0.0

        # Calculate Environmental Score 
        # sum_generated_energy = fuel_mix.sum()
        # sum_bad_energy = fuel_mix.iloc[7] + fuel_mix.iloc[8]+fuel_mix.iloc[9]+fuel_mix.iloc[1]

        # environment_score = sum_bad_energy / sum_generated_energy

        # Calculate reward
        # current_reward = (cost - profit) + energy_missing + energy_leftover + battery_wear_cost
        current_reward = 10 * (cost - profit) + 10 * energy_feed_in + 4 * battery_wear_cost + 4 * energy_missing + 15 * energy_leftover

        # Log test
        if self._test:
            with self._test_writer.as_default(step=self._current_timestep):
                tf.summary.scalar(name='battery action', data=battery_action)
                tf.summary.scalar(name='soe', data=new_soe)
                tf.summary.scalar(name='energy missing', data=energy_missing)
                tf.summary.scalar(name='energy leftover', data=energy_leftover)
                tf.summary.scalar(name='pv', data=pv)
                tf.summary.scalar(name='load', data=load)
                tf.summary.scalar(name='price', data=electricity_price)
                tf.summary.scalar(name='current reward', data=-current_reward)

        """
        If the last timeslot is reached and the maximum number of training days has been reached, 
        the episode is marked as ended. The method returns a termination TimeStep object.
        """
        if self._current_timestep >= self._max_timesteps - 2:
            self._episode_ended = True
            if self._test:
                with self._test_writer.as_default(step=self._current_timestep):
                    tf.summary.scalar(name='cost', data=self._electricity_cost)
            
            # return ts.termination(np.array([new_soe, load, pv, electricity_price, electricity_price_forecast], dtype=np.float32),reward=-current_reward)
            return ts.termination(np.array([new_soe, load, pv, electricity_price], dtype=np.float32),
                                  reward=-current_reward)

        """
        For regular time steps, a transition TimeStep object is returned, representing the new state, 
        the negative of the current reward (as the environment aims to minimize costs), 
        and a discount factor of 1.0 (indicating no discounting in this scenario).
        """
        # return ts.transition(np.array([new_soe, load, pv, electricity_price, electricity_price_forecast], dtype=np.float32), reward=-current_reward)
        return ts.transition(np.array([new_soe, load, pv, electricity_price], dtype=np.float32), reward=-current_reward)



    # def _chargeBattery(self, amount_to_charge, old_soe):
    #     new_soe = np.clip(amount_to_charge + old_soe, 0.0, self._capacity, dtype=np.float32)
    #     amount_charged = (new_soe - old_soe)
    #     energy_leftover = amount_to_charge - amount_charged
    #     return new_soe, amount_charged, energy_leftover

    # def _dischargeBattery(self, amount_to_discharge, old_soe):
    #     energy_provided = 0.0
    #     # Battery soe can handle needed energy
    #     if old_soe >= amount_to_discharge:
    #         new_soe = old_soe - amount_to_discharge
    #         energy_provided = amount_to_discharge
    #     # Battery soe cant handle needed energy
    #     elif old_soe < amount_to_discharge:
    #         new_soe = 0.0
    #         energy_provided = old_soe
    #     energy_missing = amount_to_discharge - energy_provided
    #     return new_soe, energy_provided, energy_missing

    # def _balanceHousehold(self, old_soe, battery_action, net_load):
    #     # battery_action > 0 -> charge battery
    #     # battery_action < 0 -> discharge battery
    #     # energy_management > 0 -> need energy
    #     # energy_management < 0 -> leftover energy
    #     energy_from_grid = 0.0
    #     energy_feed_in = 0.0
    #     energy_missing = 0.0
    #     energy_leftover = 0.0
    #     energy_management = net_load

    #     # Discharge battery
    #     if battery_action < 0:
    #         new_soe, energy_provided, energy_missing = self._dischargeBattery(np.abs(battery_action), old_soe)
    #         energy_management -= energy_provided
    #     # Charge battery
    #     elif battery_action > 0:
    #         new_soe, energy_used, energy_leftover = self._chargeBattery(battery_action, old_soe)
    #         energy_management += energy_used
    #     # Neutral case
    #     else:
    #         new_soe = old_soe

    #     # Sell energy
    #     if energy_management < 0:
    #         energy_feed_in = np.abs(energy_management)
    #     # Buy energy
    #     elif energy_management > 0:
    #         energy_from_grid = energy_management

    #     return new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover
    
    def chargeBattery(self, amount_to_charge, old_soe, load, pv, action_taken):
        """
        Charge the battery with the given amount
        :param amount_to_charge: amount to charge
        :param old_soe: old state of energy of the battery
        :param load: current load
        :param pv: current pv generation
        :param action_taken: the action taken by the policy
        :return: new state of energy, amount of energy taken from grid, energy leftover
        """
        clipped_amount_to_charge = np.clip(amount_to_charge, 0.0, self._power_battery / 2, dtype=np.float32)
        new_soe = np.clip(clipped_amount_to_charge + old_soe, 0.0, self._capacity, dtype=np.float32)
        amount_charged = (new_soe - old_soe)
        if action_taken < 0:
            energy_from_grid = 0.0
        else:
            energy_from_grid = amount_charged + (load - pv)
        energy_leftover = amount_to_charge - amount_charged
        if np.isclose([energy_leftover], [0]):
            energy_leftover = 0.0
        return new_soe, energy_from_grid, energy_leftover

    def dischargeBattery(self, amount_to_discharge, old_soe):
        """
        Discharge the battery with the given amount
        :param amount_to_discharge: amount to discharge
        :param old_soe: old state of energy of the battery
        :return: new state of energy, energy missing to fulfil to load
        """
        energy_missing = 0.0
        negative_delta_energy = np.abs(amount_to_discharge)
        # Battery power and soe can handle needed energy
        if (negative_delta_energy <= self._power_battery) / 2 and (old_soe >= negative_delta_energy):
            new_soe = old_soe - negative_delta_energy
        # Battery power and soe cant handle needed energy
        elif (self._power_battery / 2 < negative_delta_energy) and (old_soe < negative_delta_energy):
            min_value = min(self._power_battery / 2, old_soe)
            new_soe = old_soe - min_value
            energy_missing = negative_delta_energy - min_value
        # Battery power cant handle needed energy
        elif self._power_battery / 2 < negative_delta_energy <= old_soe:
            new_soe = old_soe - self._power_battery / 2
            energy_missing = negative_delta_energy - self._power_battery / 2
        # Battery soe cant handle needed energy
        elif old_soe < negative_delta_energy <= self._power_battery / 2:
            new_soe = 0.0
            energy_missing = negative_delta_energy - old_soe
        return new_soe, energy_missing

    def balanceHousehold(self, old_soe, load, pv, action_taken):
        """
        Charge and/or discharges the battery under the given situation
        :param old_soe: old state of energy
        :param load: current load
        :param pv: current pv generation
        :param action_taken: the action taken by the policy
        :return: new state of energy, amount of energy taken from grid, energy missing to fulfil to load, energy
            leftover, energy feed into the grid
        """
        energy_from_grid = 0.0
        energy_missing = 0.0
        energy_leftover = 0.0
        energy_feed_in = 0.0
        # Sell energy
        if action_taken < 0:
            delta_energy = pv - load - np.abs(action_taken)
            # Do not need battery
            if delta_energy > 0:
                amount_to_charge = delta_energy
                new_soe, energy_from_grid, energy_leftover = self.chargeBattery(amount_to_charge, old_soe, load, pv,
                                                                                action_taken)
                energy_feed_in = np.abs(action_taken)
            # Need battery
            elif delta_energy < 0:
                amount_to_discharge = delta_energy
                new_soe, energy_missing = self.dischargeBattery(amount_to_discharge, old_soe)
                energy_feed_in = np.abs(action_taken) - energy_missing
            else:
                new_soe = old_soe
        # Buy energy
        else:
            delta_energy = pv + action_taken - load
            # Do not need battery
            if delta_energy > 0:
                amount_to_charge = delta_energy
                new_soe, energy_from_grid, energy_leftover = self.chargeBattery(amount_to_charge, old_soe, load, pv,
                                                                                action_taken)
            # Need battery
            elif delta_energy < 0:
                amount_to_discharge = delta_energy
                new_soe, energy_missing = self.dischargeBattery(amount_to_discharge, old_soe)
            else:
                new_soe = old_soe
        return new_soe, energy_from_grid, energy_missing, energy_leftover, energy_feed_in