import pandas as pd
import os
from tf_agents.environments import tf_py_environment
import tensorflow as tf
from tf_agents.agents import ddpg
from tf_agents.agents.ddpg import ddpg_agent  
from tf_agents.utils import common
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
import wandb
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_CONSOLE'] = 'off'

import sys
sys.path.insert(0, '..')
from environments.EnergyManagementEnv import EnergyManagementEnv

def setup_energymanagement_environments(num_buildings=30, path_energy_data="../../data/3final_data/Final_Energy_dataset.csv", return_dataset=False):
    
    energy_data = pd.read_csv(path_energy_data).fillna(0).set_index('Date')

    dataset = {"train": {}, "eval": {}, "test": {}}
    environments = {"train": {}, "eval": {}, "test": {}}
   
    for idx in range(num_buildings):
        user_data = energy_data[[f'load_{idx+1}', f'pv_{idx+1}', 'price', 'fuelmix']]
        
        dataset["train"][f"building_{idx+1}"] = user_data[0:17520].set_index(pd.RangeIndex(0,17520))
        dataset["eval"][f"building_{idx+1}"] = user_data[17520:35088].set_index(pd.RangeIndex(0,17568))
        dataset["test"][f"building_{idx+1}"] = user_data[35088:52608].set_index(pd.RangeIndex(0,17520))

        environments["train"][f"building_{idx+1}"] = tf_py_environment.TFPyEnvironment(EnergyManagementEnv(init_charge=0.0, data=dataset["train"][f"building_{idx+1}"]))
        environments["eval"][f"building_{idx+1}"] = tf_py_environment.TFPyEnvironment(EnergyManagementEnv(init_charge=0.0, data=dataset["eval"][f"building_{idx+1}"]))
        environments["test"][f"building_{idx+1}"] = tf_py_environment.TFPyEnvironment(EnergyManagementEnv(init_charge=0.0, data=dataset["test"][f"building_{idx+1}"], logging=True))

    observation_spec = environments["train"][f"building_1"].observation_spec()
    action_spec = environments["train"][f"building_1"].action_spec()

    if return_dataset:
        return environments, observation_spec, action_spec, dataset
    else:
        return environments, observation_spec, action_spec

  
def initialize_ddpg_agent(observation_spec, action_spec, global_step, environments): 
    
    actor_net = ddpg.actor_network.ActorNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec, 
        fc_layer_params=(256, 256),
        activation_fn=tf.keras.activations.relu)
     
    critic_net = ddpg.critic_network.CriticNetwork(
        input_tensor_spec=(observation_spec, action_spec),
        joint_fc_layer_params=(256, 256),
        activation_fn=tf.keras.activations.relu)

    target_actor_network = ddpg.actor_network.ActorNetwork(
        input_tensor_spec=observation_spec,
        output_tensor_spec=action_spec, fc_layer_params=(256, 256),
        activation_fn=tf.keras.activations.relu)

    target_critic_network = ddpg.critic_network.CriticNetwork(
        input_tensor_spec=(observation_spec, action_spec),
        joint_fc_layer_params=(256, 256),
        activation_fn=tf.keras.activations.relu)
    

    agent_params = {
        "time_step_spec": environments["train"][f"building_{1}"].time_step_spec(),
        "action_spec": environments["train"][f"building_{1}"].action_spec(),
        "actor_network": actor_net,
        "critic_network": critic_net,
        "actor_optimizer": tf.compat.v1.train.AdamOptimizer(learning_rate=1e-3), #1e-3
        "critic_optimizer": tf.compat.v1.train.AdamOptimizer(learning_rate=1e-4), #1e-2
        "ou_stddev": 0.9, #0.9,
        "ou_damping": 0.15,
        "target_actor_network": target_actor_network,
        "target_critic_network": target_critic_network,
        "target_update_tau": 0.05,
        "target_update_period": 100, #5,
        "dqda_clipping": 0.5,
        "td_errors_loss_fn": tf.compat.v1.losses.huber_loss,
        "gamma": 1, #0.99,
        "reward_scale_factor": 1,
        "train_step_counter": global_step,
    }

    # Create the DdpgAgent with unpacked parameters
    ddpg_tf_agent = ddpg_agent.DdpgAgent(**agent_params)

    ddpg_tf_agent.initialize()
    eval_policy = ddpg_tf_agent.policy
    collect_policy = ddpg_tf_agent.collect_policy

    return ddpg_tf_agent, eval_policy, collect_policy

def setup_rl_training_pipeline(tf_agent, env_train, replay_buffer_capacity,collect_policy, initial_collect_steps, collect_steps_per_iteration, batch_size):
    
    #Setup replay buffer -> TFUniform to give each sample an equal selection chance
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=tf_agent.collect_data_spec,
            batch_size= env_train.batch_size,
            max_length=replay_buffer_capacity,
        )

    # Populate replay buffer with inital experience before actual training (for num_steps times)
    initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
        env=env_train,
        policy=collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=initial_collect_steps,
    )

    # After the initial collection phase, the collect driver takes over for the continuous collection of data during the training process
    collect_driver = dynamic_step_driver.DynamicStepDriver(
        env=env_train,
        policy=collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration,
    )

    # For better performance
    initial_collect_driver.run = common.function(initial_collect_driver.run)
    collect_driver.run = common.function(collect_driver.run)
    tf_agent.train = common.function(tf_agent.train)

    # Collect initial replay data
    initial_collect_driver.run()
    #initial_collect_driver.run()
    time_step = env_train.reset()
    policy_state = collect_policy.get_initial_state(env_train.batch_size)

    # The dataset is created from the replay buffer in a more structured and efficient way to provide mini-batches
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=tf.data.experimental.AUTOTUNE, 
        sample_batch_size=batch_size, num_steps=2).prefetch(tf.data.experimental.AUTOTUNE)
    
    #Feed batches of experience to the agent for training
    iterator = iter(dataset)

    return iterator, collect_driver, time_step, policy_state

def initialize_wandb_logging(project="DDPG_battery_testing", name="Exp", num_iterations=1500, batch_size=1, a_lr="1e-4", c_lr="1e-3"):
    wandb.login()
    wandb.init(
        project="DDPG_battery_testing",
        job_type="train_eval_test",
        name=name,
        config={
            "train_steps": num_iterations,
            "batch_size": batch_size,
            "actor_learning_rate": 1e-3,
            "critic_learning_rate": 1e-2}
    )
    artifact = wandb.Artifact(name='save', type="checkpoint")

    """train_checkpointer = common.Checkpointer(
            ckpt_dir='checkpoints/ddpg/',
            max_to_keep=1,
            agent=tf_agent,
            policy=tf_agent.policy,
            replay_buffer=replay_buffer,
            global_step=global_step
        )
        train_checkpointer.initialize_or_restore()"""

    return artifact

def end_and_log_wandb(metrics, artifact):
    wandb.log(metrics)
    #artifact.add_dir(local_path='checkpoints/ddpg/')
    wandb.log_artifact(artifact)
    wandb.finish()