# Only price forecast with load consumption included

## Experiment 01
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [(soe,...,soe),(price0,price1,...,price24)]
- Reward = (5*profit - cost) - energy_feed_in - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 1.500
- Household: 13
### Metrics
![avg return](./2_ex_01/avg_return.png)
![loss](./2_ex_01/loss.png)
- Evaluation: Costs = 126,5157$  

## Experiment 02
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [(soe,...,soe),(price0,price1,...,price24)]
- Reward = (5*profit - cost) - energy_feed_in - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 3.000
- Household: 1
### Metrics
![avg return](./2_ex_02/avg_return.png)
![loss](./2_ex_01/loss.png)
- Evaluation: Costs = 227,378$

## Experiment 03
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [(soe,net_load,price0,price1,...,price24)]
- Reward = (profit - cost) - energy_feed_in - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 3.000
- Household: 1
### Metrics
![avg return](./2_ex_02/avg_return.png)
![loss](./2_ex_01/loss.png)
- Evaluation: Costs = 227,378$  