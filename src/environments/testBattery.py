import unittest

import numpy as np


class TestHousehold(unittest.TestCase):

    def _balanceHousehold(self, old_soe, battery_action, load, pv, priceHigh):
        energy_from_grid = 0.0
        energy_feed_in = 0.0        

        new_soe = np.clip(old_soe + battery_action, 0.0, 13.5, dtype=np.float32)
        amount_charged_discharged = (new_soe - old_soe)

        # Stromkauf teuer
        if priceHigh:
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
                energy_feed_in = pv + np.abs(amount_charged_discharged)
            else:
                energy_from_grid = load - pv + amount_charged_discharged
            
        
        return new_soe, energy_from_grid, energy_feed_in
    
    # ---------------
    # Without action
    # ---------------
    def test_01(self):
        new_soe, energy_from_grid, energy_feed_in = self._balanceHousehold(0.0, 0.0, 0.0, 0.0, True)
        self.assertAlmostEqual(new_soe, 0.0)
        self.assertAlmostEqual(energy_from_grid, 0.0)
        self.assertAlmostEqual(energy_feed_in, 0.0)
    
    # ---------------
    # 
    # ---------------
    def test_02(self):
        new_soe, energy_from_grid, energy_feed_in = self._balanceHousehold(0.0, 1.0, 2.0, 0.5, True)
        self.assertAlmostEqual(new_soe, 1.0)
        self.assertAlmostEqual(energy_from_grid, 2.5)
        self.assertAlmostEqual(energy_feed_in, 0.0)

   

if __name__ == '__main__':
    unittest.main()