# Battery Environemnt

## Experiment 01 -- Price only
### Setup
- Action Space = [-battery_power/2,battery_power/2]
- Observation Space = [(soe)]
- Reward = profit - cost - energy_leftover_missing
- Agent: Standard DDPG
- Train-iterations: 3.000
- Household: X
### Metrics
![avg return](./3_ex_01/avg_return.png)
![loss](./3_ex_01/loss.png)
- Evaluation: Profit = -0,0011$
- 
## Experiment 02 -- Price only
### Setup
- Action Space = [-battery_power/2,battery_power/2]
- Observation Space = [(soe,price0)]
- Reward = profit - cost - energy_leftover_missing
- Agent: Standard DDPG
- Train-iterations: 3.000
- Household: X
### Metrics
![avg return](./3_ex_02/avg_return.png)
![loss](./3_ex_02/loss.png)
- Evaluation: Profit = 5,879$
- 
## Experiment 03 -- Price only
### Setup
- Action Space = [-battery_power/2,battery_power/2]
- Observation Space = [(soe,price0,price1)]
- Reward = profit - cost - energy_leftover_missing
- Agent: Standard DDPG
- Train-iterations: 3.000
- Household: X
### Metrics
![avg return](./3_ex_03/avg_return.png)
![loss](./3_ex_03/loss.png)
- Evaluation: Profit = 0,0709$
- 
## Experiment 04 -- Price only
### Setup
- Action Space = [-battery_power/2,battery_power/2]
- Observation Space = [(soe,price0,price1,price2)]
- Reward = profit - cost - energy_leftover_missing
- Agent: Standard DDPG
- Train-iterations: 3.000
- Household: X
### Metrics
![avg return](./3_ex_04/avg_return.png)
![loss](./3_ex_04/loss.png)
- Evaluation: Profit = 6,9221$
  
## Experiment 05 -- Price only
### Setup
- Action Space = [-battery_power/2,battery_power/2]
- Observation Space = [(soe,price0,price1,price2)]
- Reward = profit - cost - energy_leftover_missing
- Agent: Standard DDPG
- Train-iterations: 3.000
- Household: X
### Metrics
![avg return](./3_ex_05/avg_return.png)
![loss](./3_ex_05/loss.png)
- Evaluation: Profit = 11,5285$
  
## Experiment 06 -- Price only
### Setup
- Action Space = [-battery_power/2,battery_power/2]
- Observation Space = [(soe,price0,price1,price2)]
- Reward = profit - cost - energy_leftover_missing
- Agent: Standard DDPG
- Train-iterations: 10.000
- Household: X
### Metrics
![avg return](./3_ex_06/avg_return.png)
![loss](./3_ex_06/loss.png)
- Evaluation: Profit = 33,0074$
  
## Experiment 07 -- Price only
### Setup
- Action Space = [-battery_power/2,battery_power/2]
- Observation Space = [(soe,price0,price1,price2,price3,price4)]
- Reward = profit - cost - energy_leftover_missing
- Agent: Standard DDPG
- Train-iterations: 10.000
- Household: X
### Metrics
![avg return](./3_ex_07/avg_return.png)
![loss](./3_ex_07/loss.png)
- Evaluation: Profit = 25,0298$

## Experiment 08 -- Price only
### Setup
- Action Space = [-battery_power/2,battery_power/2]
- Observation Space = [(soe,price0,price1,price2,price3,price4)]
- Reward = profit - cost - energy_leftover_missing
- Agent: Standard DDPG
- Train-iterations: 15.000
- Household: X
### Metrics
![avg return](./3_ex_08/avg_return.png)
![loss](./3_ex_08/loss.png)
- Evaluation: Profit = 72,4569$
  
## Experiment 09 -- Price only
### Setup
- Action Space = [-battery_power/2,battery_power/2]
- Observation Space = [(soe,price0,price1,price2,price3,price4,price5,price6)]
- Reward = profit - cost - energy_leftover_missing
- Agent: Standard DDPG
- Train-iterations: 10.000
- Household: X
### Metrics
- Evaluation: Profit = $

## Experiment 10 -- Price only
### Setup
- Action Space = [-battery_power/2,battery_power/2]
- Observation Space = [(soe,price0,price1,price2,price3,price4,price5,price6,price7,price8)]
- Reward = profit - cost - energy_leftover_missing
- Agent: Standard DDPG
- Train-iterations: 10.000
- Household: X
### Metrics
- Evaluation: Profit = $

## Experiment 11
### Setup
- Action Space = [-battery_power/2,battery_power/2]
- Observation Space = [(soe,net_load0,net_load1,net_load2,price0,price1,price2,price3,price4,price5,price6)]
- Reward = profit - cost - energy_leftover_missing
- Agent: Standard DDPG
- Train-iterations: 20.000
- Household: X
### Metrics
- Evaluation: Profit = $

## Experiment 12
### Setup
- Action Space = [-battery_power/2,battery_power/2]
- Observation Space = [(soe,net_load0,net_load1,net_load2,price0,price1,price2,price3,price4,price5,price6)]
- Reward = 10*profit - 10*cost - 15*energy_leftover_missing
- Agent: Standard DDPG
- Train-iterations: 20.000
- Household: X
### Metrics
- Evaluation: Profit = $