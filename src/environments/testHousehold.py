import unittest

import numpy as np
import pandas as pd


class TestHousehold(unittest.TestCase):

    def _chargeBattery(self, amount_to_charge, old_soe):
        clipped_amount_to_charge = np.clip(amount_to_charge, 0.0, 2.3)
        new_soe = np.clip(clipped_amount_to_charge + old_soe, 0.0, 13.5, dtype=np.float32)
        amount_charged = (new_soe - old_soe)
        energy_leftover = amount_to_charge - amount_charged
        return new_soe, energy_leftover

    def _dischargeBattery(self, amount_to_discharge, old_soe):
        clipped_amount_to_discharge = np.clip(amount_to_discharge, 0.0, 2.3, dtype=np.float32) 
        new_soe = np.clip(old_soe - clipped_amount_to_discharge, 0.0, 13.5, dtype=np.float32)
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
        return new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover
    
    # ---------------
    # Without action
    # ---------------
    def test_01(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 0.0, 0.0)
        self.assertAlmostEqual(new_soe, 0.0)
        self.assertAlmostEqual(energy_from_grid, 0.0)
        self.assertAlmostEqual(energy_feed_in, 0.0)
        self.assertAlmostEqual(energy_missing, 0.0)
        self.assertAlmostEqual(energy_leftover, 0.0)

    def test_02(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 0.0, 1.0)
        self.assertAlmostEqual(new_soe, 0.0)
        self.assertAlmostEqual(energy_from_grid, 0.0)
        self.assertAlmostEqual(energy_feed_in, 0.0)
        self.assertAlmostEqual(energy_missing, 1.0)
        self.assertAlmostEqual(energy_leftover, 0.0)
    
    def test_03(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 0.0, -1.0)
        self.assertAlmostEqual(new_soe, 1.0)
        self.assertAlmostEqual(energy_from_grid, 0.0)
        self.assertAlmostEqual(energy_feed_in, 0.0)
        self.assertAlmostEqual(energy_missing, 0.0)
        self.assertAlmostEqual(energy_leftover, 0.0)
    
    # ---------------
    # Action: Sell
    # ---------------
    # Battery empty, no energy
    def test_04(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -1.0, 0.0)
        self.assertAlmostEqual(new_soe, 0.0, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 0.0, 6)
        self.assertAlmostEqual(energy_missing, 1.0, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)

    def test_05(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -7.5, 0.0)
        self.assertAlmostEqual(new_soe, 0.0, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 0.0, 6)
        self.assertAlmostEqual(energy_missing, 7.5, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)

    def test_06(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -15, 0.0)
        self.assertAlmostEqual(new_soe, 0.0, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 0.0, 6)
        self.assertAlmostEqual(energy_missing, 15.0, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)
    
    # Battery half full, no energy
    def test_04(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(7.0, -1.0, 0.0)
        self.assertAlmostEqual(new_soe, 6.0, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 1.0, 6)
        self.assertAlmostEqual(energy_missing, 0.0, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)

    def test_05(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(7.0, -7.5, 0.0)
        self.assertAlmostEqual(new_soe, 4.7, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 2.3, 6)
        self.assertAlmostEqual(energy_missing, 5.2, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)

    def test_06(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(7.0, -15, 0.0)
        self.assertAlmostEqual(new_soe, 4.7, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 2.3, 6)
        self.assertAlmostEqual(energy_missing, 12.7, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)
    
    # Battery full, no energy
    def test_07(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(13.5, -1.0, 0.0)
        self.assertAlmostEqual(new_soe, 12.5, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 1.0, 6)
        self.assertAlmostEqual(energy_missing, 0.0, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)

    def test_08(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(13.5, -7.5, 0.0)
        self.assertAlmostEqual(new_soe, 11.2, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 2.3, 6)
        self.assertAlmostEqual(energy_missing, 5.2, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)

    def test_09(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(13.5, -15, 0.0)
        self.assertAlmostEqual(new_soe, 11.2, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 2.3, 6)
        self.assertAlmostEqual(energy_missing, 12.7, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)

    # Battery empty, energy producing little
    def test_10(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -1.0, -1.0)
        self.assertAlmostEqual(new_soe, 0.0, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 1.0, 6)
        self.assertAlmostEqual(energy_missing, 0.0, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)

    def test_11(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -7.5, -1.0)
        self.assertAlmostEqual(new_soe, 0.0, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 1.0, 6)
        self.assertAlmostEqual(energy_missing, 6.5, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)

    def test_12(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -15, -1.0)
        self.assertAlmostEqual(new_soe, 0.0, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 1.0, 6)
        self.assertAlmostEqual(energy_missing, 14.0, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)
        
    # Battery empty, energy producing middle
    def test_13(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -1.0, -4.0)
        self.assertAlmostEqual(new_soe, 2.3, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 1.0, 6)
        self.assertAlmostEqual(energy_missing, 0.0, 6)
        self.assertAlmostEqual(energy_leftover, 0.7, 6)

    def test_14(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -7.5, -4.0)
        self.assertAlmostEqual(new_soe, 0.0, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 4.0, 6)
        self.assertAlmostEqual(energy_missing, 3.5, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)

    def test_15(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -15, -4.0)
        self.assertAlmostEqual(new_soe, 0.0, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 4.0, 6)
        self.assertAlmostEqual(energy_missing, 11.0, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)

    # Battery empty, energy producing high
    def test_14(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -1.0, -8.0)
        self.assertAlmostEqual(new_soe, 2.3, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 1.0, 6)
        self.assertAlmostEqual(energy_missing, 0.0, 6)
        self.assertAlmostEqual(energy_leftover, 4.7, 6)

    def test_15(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -7.5, -8.0)
        self.assertAlmostEqual(new_soe, 0.5, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 7.5, 6)
        self.assertAlmostEqual(energy_missing, 0.0, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)

    def test_16(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -15, -8.0)
        self.assertAlmostEqual(new_soe, 0.0, 6)
        self.assertAlmostEqual(energy_from_grid, 0.0, 6)
        self.assertAlmostEqual(energy_feed_in, 8.0, 6)
        self.assertAlmostEqual(energy_missing, 7.0, 6)
        self.assertAlmostEqual(energy_leftover, 0.0, 6)
    
    # def test_05(self):
    #     new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -1.0, -2.0)
    #     self.assertAlmostEqual(new_soe, 1.0)
    #     self.assertAlmostEqual(energy_from_grid, 0.0)
    #     self.assertAlmostEqual(energy_feed_in, 1.0)
    #     self.assertAlmostEqual(energy_missing, 0.0)
    #     self.assertAlmostEqual(energy_leftover, 0.0)

    # def test_06(self):
    #     new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(1.0, -1.0, 0.0)
    #     self.assertAlmostEqual(new_soe, 0.0)
    #     self.assertAlmostEqual(energy_from_grid, 0.0)
    #     self.assertAlmostEqual(energy_feed_in, 1.0)
    #     self.assertAlmostEqual(energy_missing, 0.0)
    #     self.assertAlmostEqual(energy_leftover, 0.0)
        
    # def test_07(self):
    #     new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 1.0, 0.0)
    #     self.assertAlmostEqual(new_soe, 1.0)
    #     self.assertAlmostEqual(energy_from_grid, 1.0)
    #     self.assertAlmostEqual(energy_feed_in, 0.0)
    #     self.assertAlmostEqual(energy_missing, 0.0)
    #     self.assertAlmostEqual(energy_leftover, 0.0)

    # def test_08(self):
    #     new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 1.0, 2.0)
    #     self.assertAlmostEqual(new_soe, 0.0)
    #     self.assertAlmostEqual(energy_from_grid, 1.0)
    #     self.assertAlmostEqual(energy_feed_in, 0.0)
    #     self.assertAlmostEqual(energy_missing, 1.0)
    #     self.assertAlmostEqual(energy_leftover, 0.0)
        
    # def test_09(self):
    #     new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, -10.0, 2.0)
    #     self.assertAlmostEqual(new_soe, 0.0)
    #     self.assertAlmostEqual(energy_from_grid, 0.0)
    #     self.assertAlmostEqual(energy_feed_in, 0.0)
    #     self.assertAlmostEqual(energy_missing, 12.0)
    #     self.assertAlmostEqual(energy_leftover, 0.0)
    
    # def test_10(self):
    #     new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(5.0, -10.0, 2.0)
    #     self.assertAlmostEqual(new_soe, 2.7)
    #     self.assertAlmostEqual(energy_from_grid, 0.0)
    #     self.assertAlmostEqual(energy_feed_in, 0.3)
    #     self.assertAlmostEqual(energy_missing, 9.7)
    #     self.assertAlmostEqual(energy_leftover, 0.0)

    # def test_11(self):
    #     new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 2.3, 0.0)
    #     self.assertAlmostEqual(new_soe, 2.3)
    #     self.assertAlmostEqual(energy_from_grid, 2.3)
    #     self.assertAlmostEqual(energy_feed_in, 0.0)
    #     self.assertAlmostEqual(energy_missing, 0.0)
    #     self.assertAlmostEqual(energy_leftover, 0.0)

    # def test_12(self):
    #     new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 2.4, 0.0)
    #     self.assertAlmostEqual(new_soe, 2.3)
    #     self.assertAlmostEqual(energy_from_grid, 2.4)
    #     self.assertAlmostEqual(energy_feed_in, 0.0)
    #     self.assertAlmostEqual(energy_missing, 0.0)
    #     self.assertAlmostEqual(energy_leftover, 0.1)

    # def test_12(self):
    #     new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(13.0, 1.5, 0.0)
    #     self.assertAlmostEqual(new_soe, 13.5)
    #     self.assertAlmostEqual(energy_from_grid, 0.5)
    #     self.assertAlmostEqual(energy_feed_in, 0.0)
    #     self.assertAlmostEqual(energy_missing, 0.0)
    #     self.assertAlmostEqual(energy_leftover, 1.0)

    # def test_13(self):
    #     new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(13.0, 1.5, -10.0)
    #     self.assertAlmostEqual(new_soe, 13.5)
    #     self.assertAlmostEqual(energy_from_grid, 0.0)
    #     self.assertAlmostEqual(energy_feed_in, 0.0)
    #     self.assertAlmostEqual(energy_missing, 0.0)
    #     self.assertAlmostEqual(energy_leftover, 11.0)


    # def test_02(self):
    #     new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 2.3, 2.3)
    #     self.assertAlmostEqual(new_soe, 2.3)
    #     self.assertAlmostEqual(energy_from_grid, 4.6)
    #     self.assertAlmostEqual(energy_feed_in, 0.0)
    #     self.assertAlmostEqual(energy_missing, 0.0)
    #     self.assertAlmostEqual(energy_leftover, 0.0)
    
    # def test_03(self):
    #     new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 2.3, -2.3)
    #     self.assertAlmostEqual(new_soe, 2.3)
    #     self.assertAlmostEqual(energy_from_grid, 0.0)
    #     self.assertAlmostEqual(energy_feed_in, 0.0)
    #     self.assertAlmostEqual(energy_missing, 0.0)
    #     self.assertAlmostEqual(energy_leftover, 0.0)

    # def test_04(self):
    #     new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self._balanceHousehold(0.0, 2.3, -2.5)
    #     self.assertAlmostEqual(new_soe, 2.3)
    #     self.assertAlmostEqual(energy_from_grid, 0.0)
    #     self.assertAlmostEqual(energy_feed_in, 0.2)
    #     self.assertAlmostEqual(energy_missing, 0.0)
    #     self.assertAlmostEqual(energy_leftover, 0.0)


if __name__ == '__main__':
    unittest.main()
