import unittest

import numpy as np
import pandas as pd

import Environment as Env

import sys
sys.path.insert(0, '..')
import utils.Dataloader as Dataloader


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        self.init_charge = 10.0
        self.max_timeslots = 48
        self.max_days = 365
        self.capacity = 13.5
        self.power_battery = 4.6
        self.power_grid = 25.0
        self.data = Dataloader.get_customer_data(Dataloader.loadData('../data.csv'), 1)

    def test_reset(self):
        self.environment = Env.Environment(init_charge=self.init_charge, data=self.data,
                                           max_timeslots=self.max_timeslots,
                                           max_days=self.max_days, capacity=self.capacity,
                                           power_battery=self.power_battery, power_grid=self.power_grid)
        self.environment.reset()
        time_step = self.environment.step(np.array([1.0], dtype=np.float32))
        self.assertAlmostEqual(time_step.observation[0], 0.0)
        self.assertAlmostEqual(time_step.observation[1], 0.0)
        self.assertLess(time_step.observation[2], 10.0)
        time_step = self.environment.reset()
        self.assertAlmostEqual(time_step.observation[0], 0.0)
        self.assertAlmostEqual(time_step.observation[1], -1.0)
        self.assertAlmostEqual(time_step.observation[2], 10.0)

    # positive delta, battery not full
    def test_env_01(self):
        load = pd.DataFrame([1.85], dtype=np.float32)
        pv = pd.DataFrame([1.2], dtype=np.float32)
        electricity_price = np.float32(1.0)
        test_action = np.array([3.5], dtype=np.float32)
        environment = Env.Environment(init_charge=self.init_charge, data=(load, pv, electricity_price),
                                      max_timeslots=2,
                                      max_days=1, capacity=self.capacity,
                                      power_battery=self.power_battery, power_grid=self.power_grid)
        environment.reset()
        new_soe = 12.3
        time_step = environment.step(test_action)
        grid_cost = (0.65 + 2.3) * electricity_price
        battery_cost = 0.0
        if new_soe > self.capacity * 0.8 or new_soe < self.capacity * 0.2:
            battery_cost += 10.0
        reward = grid_cost + battery_cost
        self.assertAlmostEqual(time_step.observation[2], new_soe, places=5)
        self.assertAlmostEqual(time_step.reward, -reward, places=5)

    # positive delta, battery nearly full
    def test_env_02(self):
        load = pd.DataFrame([1.85], dtype=np.float32)
        pv = pd.DataFrame([1.2], dtype=np.float32)
        electricity_price = np.float32(1.0)
        test_action = np.array([2.7], dtype=np.float32)
        environment = Env.Environment(init_charge=13.0, data=(load, pv, electricity_price),
                                      max_timeslots=2,
                                      max_days=1, capacity=self.capacity,
                                      power_battery=self.power_battery, power_grid=self.power_grid)
        environment.reset()
        new_soe = 13.5
        time_step = environment.step(test_action)
        grid_cost = (0.65 + 0.5) * electricity_price
        battery_cost = 0.0
        if new_soe > self.capacity * 0.8 or new_soe < self.capacity * 0.2:
            battery_cost += 10.0
        reward = grid_cost + battery_cost
        self.assertAlmostEqual(time_step.observation[2], new_soe, places=5)
        self.assertAlmostEqual(time_step.reward, -reward, places=5)

    # positive delta, battery full
    def test_env_03(self):
        load = pd.DataFrame([1.85], dtype=np.float32)
        pv = pd.DataFrame([1.2], dtype=np.float32)
        electricity_price = np.float32(1.0)
        test_action = np.array([2.7], dtype=np.float32)
        environment = Env.Environment(init_charge=13.5, data=(load, pv, electricity_price),
                                      max_timeslots=2,
                                      max_days=1, capacity=self.capacity,
                                      power_battery=self.power_battery, power_grid=self.power_grid)
        environment.reset()
        new_soe = 13.5
        time_step = environment.step(test_action)
        grid_cost = 0.65 * electricity_price
        battery_cost = 0.0
        if new_soe > self.capacity * 0.8 or new_soe < self.capacity * 0.2:
            battery_cost += 10.0
        reward = grid_cost + battery_cost
        self.assertAlmostEqual(time_step.observation[2], new_soe, places=5)
        self.assertAlmostEqual(time_step.reward, -reward, places=5)

    # negative delta, battery in between
    def test_env_04(self):
        load = pd.DataFrame([1.85], dtype=np.float32)
        pv = pd.DataFrame([1.2], dtype=np.float32)
        electricity_price = np.float32(1.0)
        test_action = np.array([0.3], dtype=np.float32)
        environment = Env.Environment(init_charge=self.init_charge, data=(load, pv, electricity_price),
                                      max_timeslots=2,
                                      max_days=1, capacity=self.capacity,
                                      power_battery=self.power_battery, power_grid=self.power_grid)
        environment.reset()
        new_soe = 9.65
        time_step = environment.step(test_action)
        grid_cost = 0.3 * electricity_price
        battery_cost = 0.0
        if new_soe > self.capacity * 0.8 or new_soe < self.capacity * 0.2:
            battery_cost += 10.0
        reward = grid_cost + battery_cost
        self.assertAlmostEqual(time_step.observation[2], new_soe, places=6)
        self.assertAlmostEqual(time_step.reward, -reward, places=6)

    # negative delta, battery full
    def test_env_05(self):
        load = pd.DataFrame([1.85], dtype=np.float32)
        pv = pd.DataFrame([1.2], dtype=np.float32)
        electricity_price = np.float32(1.0)
        test_action = np.array([0.3], dtype=np.float32)
        environment = Env.Environment(init_charge=13.5, data=(load, pv, electricity_price),
                                      max_timeslots=2,
                                      max_days=1, capacity=self.capacity,
                                      power_battery=self.power_battery, power_grid=self.power_grid)
        environment.reset()
        new_soe = 13.15
        time_step = environment.step(test_action)
        grid_cost = 0.3 * electricity_price
        battery_cost = 0.0
        if new_soe > self.capacity * 0.8 or new_soe < self.capacity * 0.2:
            battery_cost += 10.0
        reward = grid_cost + battery_cost
        self.assertAlmostEqual(time_step.observation[2], new_soe, places=6)
        self.assertAlmostEqual(time_step.reward, -reward, places=6)

    # negative delta, nearly empty, energy not enough
    def test_env_06(self):
        load = pd.DataFrame([1.85], dtype=np.float32)
        pv = pd.DataFrame([1.2], dtype=np.float32)
        electricity_price = np.float32(1.0)
        test_action = np.array([0.3], dtype=np.float32)
        environment = Env.Environment(init_charge=0.3, data=(load, pv, electricity_price),
                                      max_timeslots=2,
                                      max_days=1, capacity=self.capacity,
                                      power_battery=self.power_battery, power_grid=self.power_grid)
        environment.reset()
        new_soe = 0.0
        time_step = environment.step(test_action)
        grid_cost = 0.3 * electricity_price
        battery_cost = 0.0
        if new_soe > self.capacity * 0.8 or new_soe < self.capacity * 0.2:
            battery_cost += 10.0
        reward = grid_cost + battery_cost
        self.assertAlmostEqual(time_step.observation[2], new_soe, places=6)
        self.assertAlmostEqual(time_step.reward, -reward, places=6)

    # negative delta, empty, energy not enough
    def test_env_07(self):
        load = pd.DataFrame([1.85], dtype=np.float32)
        pv = pd.DataFrame([1.2], dtype=np.float32)
        electricity_price = np.float32(1.0)
        test_action = np.array([0.3], dtype=np.float32)
        environment = Env.Environment(init_charge=0.0, data=(load, pv, electricity_price),
                                      max_timeslots=2,
                                      max_days=1, capacity=self.capacity,
                                      power_battery=self.power_battery, power_grid=self.power_grid)
        environment.reset()
        new_soe = 0.0
        time_step = environment.step(test_action)
        grid_cost = 0.3 * electricity_price
        battery_cost = 0.0
        if new_soe > self.capacity * 0.8 or new_soe < self.capacity * 0.2:
            battery_cost += 10.0
        reward = grid_cost + battery_cost
        self.assertAlmostEqual(time_step.observation[2], new_soe, places=6)
        self.assertAlmostEqual(time_step.reward, -reward, places=6)

    # negative delta, battery in between, power not enough
    def test_env_08(self):
        load = pd.DataFrame([3.85], dtype=np.float32)
        pv = pd.DataFrame([1.2], dtype=np.float32)
        electricity_price = np.float32(1.0)
        test_action = np.array([0.3], dtype=np.float32)
        environment = Env.Environment(init_charge=self.init_charge, data=(load, pv, electricity_price),
                                      max_timeslots=2,
                                      max_days=1, capacity=self.capacity,
                                      power_battery=self.power_battery, power_grid=self.power_grid)
        environment.reset()
        new_soe = 7.7
        time_step = environment.step(test_action)
        grid_cost = 0.3 * electricity_price
        battery_cost = 0.0
        if new_soe > self.capacity * 0.8 or new_soe < self.capacity * 0.2:
            battery_cost += 10.0
        reward = grid_cost + battery_cost
        self.assertAlmostEqual(time_step.observation[2], new_soe, places=6)
        self.assertAlmostEqual(time_step.reward, -reward, places=6)

    # negative delta, battery nearly empty, power and energy not enough
    def test_env_09(self):
        load = pd.DataFrame([3.85], dtype=np.float32)
        pv = pd.DataFrame([1.2], dtype=np.float32)
        electricity_price = np.float32(1.0)
        test_action = np.array([0.3], dtype=np.float32)
        environment = Env.Environment(init_charge=2.0, data=(load, pv, electricity_price),
                                      max_timeslots=2,
                                      max_days=1, capacity=self.capacity,
                                      power_battery=self.power_battery, power_grid=self.power_grid)
        environment.reset()
        new_soe = 0.0
        time_step = environment.step(test_action)
        grid_cost = 0.3 * electricity_price
        battery_cost = 0.0
        if new_soe > self.capacity * 0.8 or new_soe < self.capacity * 0.2:
            battery_cost += 10.0
        reward = grid_cost + battery_cost
        self.assertAlmostEqual(time_step.observation[2], new_soe, places=6)
        self.assertAlmostEqual(time_step.reward, -reward, places=6)

    # negative delta, battery full, power and energy not enough
    def test_env_10(self):
        load = pd.DataFrame([3.85], dtype=np.float32)
        pv = pd.DataFrame([1.2], dtype=np.float32)
        electricity_price = np.float32(1.0)
        test_action = np.array([0.3], dtype=np.float32)
        environment = Env.Environment(init_charge=13.5, data=(load, pv, electricity_price),
                                      max_timeslots=2,
                                      max_days=1, capacity=self.capacity,
                                      power_battery=self.power_battery, power_grid=self.power_grid)
        environment.reset()
        new_soe = 11.2
        time_step = environment.step(test_action)
        grid_cost = 0.3 * electricity_price
        battery_cost = 0.0
        if new_soe > self.capacity * 0.8 or new_soe < self.capacity * 0.2:
            battery_cost += 10.0
        reward = grid_cost + battery_cost
        self.assertAlmostEqual(time_step.observation[2], new_soe, places=6)
        self.assertAlmostEqual(time_step.reward, -reward, places=6)

    # negative delta, battery empty, power and energy not enough
    def test_env_11(self):
        load = pd.DataFrame([3.85], dtype=np.float32)
        pv = pd.DataFrame([1.2], dtype=np.float32)
        electricity_price = np.float32(1.0)
        test_action = np.array([0.3], dtype=np.float32)
        environment = Env.Environment(init_charge=0.0, data=(load, pv, electricity_price),
                                      max_timeslots=2,
                                      max_days=1, capacity=self.capacity,
                                      power_battery=self.power_battery, power_grid=self.power_grid)
        environment.reset()
        new_soe = 0.0
        time_step = environment.step(test_action)
        grid_cost = 0.3 * electricity_price
        battery_cost = 0.0
        if new_soe > self.capacity * 0.8 or new_soe < self.capacity * 0.2:
            battery_cost += 10.0
        reward = grid_cost + battery_cost
        self.assertAlmostEqual(time_step.observation[2], new_soe, places=6)
        self.assertAlmostEqual(time_step.reward, -reward, places=6)

    if __name__ == '__main__':
        unittest.main()
