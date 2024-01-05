import numpy as np
import unittest


class HouseholdTest(unittest.TestCase):

    def setUp(self):
        self.capacity = 13.5

    def chargeBattery(self, amount_to_charge, old_soe):
        new_soe = np.clip(amount_to_charge + old_soe, 0.0, self.capacity, dtype=np.float32)
        amount_charged = (new_soe - old_soe)
        energy_leftover = amount_to_charge - amount_charged
        return new_soe, amount_charged, energy_leftover

    def dischargeBattery(self, amount_to_discharge, old_soe):
        energy_provided = 0.0
        # Battery soe can handle needed energy
        if old_soe >= amount_to_discharge:
            new_soe = old_soe - amount_to_discharge
            energy_provided = amount_to_discharge
        # Battery soe cant handle needed energy
        elif old_soe < amount_to_discharge:
            new_soe = 0.0
            energy_provided = old_soe
        energy_missing = amount_to_discharge - energy_provided
        return new_soe, energy_provided, energy_missing

    def balanceHousehold(self, old_soe, grid_action, battery_action, net_load):
        # grid_action > 0 -> buy energy from grid
        # grid_action < 0 -> sell energy to grid
        # battery_action > 0 -> charge battery
        # battery_action < 0 -> discharge battery
        # energy_management > 0 -> need energy
        # energy_management < 0 -> leftover energy
        energy_from_grid = 0.0
        energy_feed_in = 0.0
        energy_provided = 0.0
        energy_used = 0.0
        energy_missing = 0.0
        energy_leftover = 0.0
        energy_management = net_load

        # Neutral case
        if battery_action == 0.0:
            new_soe = old_soe

        # Provide energy
        # Discharge battery
        if battery_action < 0:
            new_soe, energy_provided, energy_missing = self.dischargeBattery(np.abs(battery_action), old_soe)
            energy_management -= energy_provided
        
        # Buy energy
        if grid_action > 0:
            energy_management -= grid_action
            energy_from_grid = grid_action
        
        # Use energy
        # Charge battery
        if battery_action > 0:
            # Leftover energy -> charge battery
            if energy_management < 0:
                charge_amount = np.clip(np.abs(energy_management), 0.0, battery_action)
                if charge_amount < battery_action:
                    energy_missing += battery_action - charge_amount
                new_soe, energy_used, energy_leftover_charge = self.chargeBattery(charge_amount, old_soe)
                energy_management += energy_used
                energy_leftover += energy_leftover_charge
            # Not enough energy -> dont charge battery
            else:
                new_soe = old_soe
                energy_missing += battery_action

        # Sell energy
        if grid_action < 0:
            energy_management -= grid_action
            # Not enough energy
            if energy_management > 0:
                energy_feed_in = np.clip(np.abs(grid_action)-energy_management, 0.0, np.abs(grid_action))
            # Feed in energy
            elif energy_management < 0:
                energy_feed_in = np.abs(grid_action)

         # Not enough energy
        if energy_management > 0:
            energy_missing += energy_management
         # Leftover energy
        elif energy_management < 0:
            energy_leftover += np.abs(energy_management)

        return new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover

    def test_01(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self.balanceHousehold(10.0,-5.0,2.3,-1.0)
        current_reward = (energy_from_grid - energy_feed_in) + energy_missing + energy_leftover
        self.assertAlmostEqual(new_soe, 11.0)
        self.assertAlmostEqual(energy_from_grid, 0.0)
        self.assertAlmostEqual(energy_feed_in, 0.0)
        self.assertAlmostEqual(energy_missing, 6.3)
        self.assertAlmostEqual(energy_leftover, 0.0)
        self.assertAlmostEqual(current_reward, 6.3)
    
    def test_02(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self.balanceHousehold(13.5,-5.0,2.3,-1.0)
        current_reward = (energy_from_grid - energy_feed_in) + energy_missing + energy_leftover
        self.assertAlmostEqual(new_soe, 13.5)
        self.assertAlmostEqual(energy_from_grid, 0.0)
        self.assertAlmostEqual(energy_feed_in, 1.0)
        self.assertAlmostEqual(energy_missing, 5.3)
        self.assertAlmostEqual(energy_leftover, 1.0)
        self.assertAlmostEqual(current_reward, 5.3)
    
    def test_03(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self.balanceHousehold(0.0,-5.0,2.3,-1.0)
        self.assertAlmostEqual(new_soe, 1.0)
        self.assertAlmostEqual(energy_from_grid, 0.0)
        self.assertAlmostEqual(energy_feed_in, 0.0)
        self.assertAlmostEqual(energy_missing, 6.3)
        self.assertAlmostEqual(energy_leftover, 0.0)
    
    def test_04(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self.balanceHousehold(13.5,-5.0,2.3,1.0)
        self.assertAlmostEqual(new_soe, 13.5)
        self.assertAlmostEqual(energy_from_grid, 0.0)
        self.assertAlmostEqual(energy_feed_in, 0.0)
        self.assertAlmostEqual(energy_missing, 8.3)
        self.assertAlmostEqual(energy_leftover, 0.0)

    def test_05(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self.balanceHousehold(13.5,5.0,2.3,1.0)
        self.assertAlmostEqual(new_soe, 13.5)
        self.assertAlmostEqual(energy_from_grid, 5.0)
        self.assertAlmostEqual(energy_feed_in, 0.0)
        self.assertAlmostEqual(energy_missing, 0.0)
        self.assertAlmostEqual(energy_leftover, 6.3)

    def test_06(self):
        new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = self.balanceHousehold(5.7265,0.5347,2.2121,0.271-1.006)
        current_reward = 4.632*(energy_from_grid - 0.7*energy_feed_in) + energy_missing + energy_leftover
        print(self.balanceHousehold(5.7265,0.5347,2.2121,0.271-1.006))
        print(current_reward)



if __name__ == '__main__':
    unittest.main()

# new_soe, energy_from_grid, energy_feed_in, energy_missing, energy_leftover = balanceHousehold(0.0,-4.0,-1.0,-1.0)
# print(new_soe)
# print(energy_from_grid)
# print(energy_feed_in)
# print(energy_missing)
# print(energy_leftover)