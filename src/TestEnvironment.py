import unittest

import numpy as np
import pandas as pd


class TestEnvironment(unittest.TestCase):

    def _chargeBattery(self, amount_to_charge, old_soe):
        new_soe = np.clip(amount_to_charge + old_soe, 0.0, 13.5, dtype=np.float32)
        amount_charged = (new_soe - old_soe)
        energy_leftover = amount_to_charge - amount_charged
        return new_soe, amount_charged, energy_leftover

    def _dischargeBattery(self, amount_to_discharge, old_soe):
        new_soe = np.clip(old_soe - amount_to_discharge, 0.0, 13.5, dtype=np.float32)
        amount_discharged = (old_soe - new_soe)
        energy_missing = amount_to_discharge - amount_discharged
        return new_soe, amount_discharged, energy_missing

    def _balanceHousehold(self, old_soe, battery_action, net_load):
        # battery_action > 0 -> charge battery
        # battery_action < 0 -> discharge battery
        # energy_management > 0 -> need energy
        # energy_management < 0 -> leftover energy
        energy_from_grid = 0.0
        energy_feed_in = 0.0
        energy_missing = 0.0
        energy_leftover = 0.0
        energy_management = net_load

        # Discharge battery
        if battery_action < 0:
            new_soe, energy_provided, energy_missing = self._dischargeBattery(np.abs(battery_action), old_soe)
            energy_management -= energy_provided
        # Charge battery
        elif battery_action > 0:
            new_soe, energy_used, energy_leftover = self._chargeBattery(battery_action, old_soe)
            energy_management += energy_used
        # Neutral case
        else:
            new_soe = old_soe

        # Sell energy
        if energy_management < 0:
            energy_feed_in = np.abs(energy_management)
        # Buy energy
        elif energy_management > 0:
            energy_from_grid = energy_management

        return new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover

    def test_01(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 2.3, 0.0)
        self.assertAlmostEqual(new_soe, 2.3)
        self.assertAlmostEqual(energy_from_grid, 2.3)
        self.assertAlmostEqual(energy_feed_in, 0.0)
        self.assertAlmostEqual(energy_missing, 0.0)
        self.assertAlmostEqual(energy_leftover, 0.0)
        
    def test_02(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 2.3, 2.3)
        self.assertAlmostEqual(new_soe, 2.3)
        self.assertAlmostEqual(energy_from_grid, 4.6)
        self.assertAlmostEqual(energy_feed_in, 0.0)
        self.assertAlmostEqual(energy_missing, 0.0)
        self.assertAlmostEqual(energy_leftover, 0.0)
    
    def test_03(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 2.3, -2.3)
        self.assertAlmostEqual(new_soe, 2.3)
        self.assertAlmostEqual(energy_from_grid, 0.0)
        self.assertAlmostEqual(energy_feed_in, 0.0)
        self.assertAlmostEqual(energy_missing, 0.0)
        self.assertAlmostEqual(energy_leftover, 0.0)

    def test_04(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 2.3, -2.5)
        self.assertAlmostEqual(new_soe, 2.3)
        self.assertAlmostEqual(energy_from_grid, 0.0)
        self.assertAlmostEqual(energy_feed_in, 0.2)
        self.assertAlmostEqual(energy_missing, 0.0)
        self.assertAlmostEqual(energy_leftover, 0.0)


if __name__ == '__main__':
    unittest.main()
