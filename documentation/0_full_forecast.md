# Forecast all values

## Experiment 01
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [(soe,...,soe),(net_load0,net_load1,...,net_load24),(price0,price1,...,price24)]
- Reward = (profit - cost) - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 10.000
- Household: 1
### Metrics
![avg return](./0_ex_01/avg_return.png)
![loss](./0_ex_01/loss.png)
- Evaluation: Costs = 225,1422$ (bei 6000 ca. 180$)

## Experiment 02
### Setup
- Action Space = [-grid_power/2,grid_power/2]
- Observation Space = [(soe,...,soe),(net_load0,net_load1,...,net_load24),(price0,price1,...,price24)]
- Reward = (profit - cost) - energy_missing - energy_leftover
- Agent: Standard DDPG
- Train-iterations: 6.000
- Household: 13
### Metrics
<!-- ![avg return](./ex_01/avg_return.png)
![loss](./ex_01/loss.png)
- Evaluation: Costs = 225,1422$ (bei 6000 ca. 180$) -->

  