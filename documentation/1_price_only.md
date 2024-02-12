# Only price arbitrage without any load consumption included (household env)

## Experiment 01 (outdated)
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [(soe,...,soe),(price0,price1,...,price24)]
- Reward = (profit - cost) - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 1.500
- Household: X
### Metrics
![avg return](./1_ex_01/avg_return.png)
![loss](./1_ex_01/loss.png)
- Evaluation: Costs = -1,4523

## Experiment 02
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [soe,price0,price1,...,price24]
- Reward = (profit - cost) - battery_wear_cost - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 1.500
- Household: X
### Metrics
![avg return](./1_ex_02/avg_return.png)
![loss](./1_ex_02/loss.png)
- Evaluation: Costs = -1,5341$ 
- 
## Experiment 03
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [soe,price0]
- Reward = (profit - cost) - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 1.500
- Household: X
- Special: Reduced dataset to compare against experiment 1&2
### Metrics
![avg return](./1_ex_03/avg_return.png)
![loss](./1_ex_03/loss.png)
- Evaluation: Costs = -3,624$ 
- 
## Experiment 04
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [(soe,...,soe),(price0,price1,...,price48)]
- Reward = (profit - cost) - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 5.000
- Household: X
### Metrics
![avg return](./1_ex_04/avg_return.png)
![loss](./1_ex_04/loss.png)
- Evaluation: Costs = -6,3192$

## Experiment 05
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [(soe,...,soe),(price0,price1,...,price48)]
- Reward = 100*(profit - cost) - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 1.550
- Household: X
### Metrics
![avg return](./1_ex_05/avg_return.png)
![loss](./1_ex_05/loss.png)
- Evaluation: Costs = -4,2815$

## Experiment 06
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [(soe,...,soe),(price0,price1,...,price24)]
- Reward = (25*profit - cost) - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 1.550
- Household: X
### Metrics
![avg return](./1_ex_06/avg_return.png)
![loss](./1_ex_06/loss.png)
- Evaluation: Costs = 481,0556$  

## Experiment 07
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [(soe,...,soe),(price0,price1,...,price24)]
- Reward = (5*profit - cost) - energy_feed_in - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 1.500
- Household: X
### Metrics
![avg return](./1_ex_07/avg_return.png)
![loss](./1_ex_07/loss.png)
- Evaluation: Costs = -6,7549$  
  
## Experiment 08
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [(soe,...,soe),(price0,price1,...,price24)]
- Reward = (5*profit - cost) - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 1.500
- Household: X
### Metrics
![avg return](./1_ex_08/avg_return.png)
![loss](./1_ex_08/loss.png)
- Evaluation: Costs = 466,2188$  

## Experiment 09
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [(soe,...,soe),(price0,price1,...,price24)]
- Reward = (3*profit - cost) - energy_feed_in - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 1.500
- Household: X
### Metrics
![avg return](./1_ex_09/avg_return.png)
![loss](./1_ex_09/loss.png)
- Evaluation: Costs = -2,5368$  