from tf_agents.agents import ddpg

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment, py_environment, batched_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
import matplotlib.pyplot as plt
import wandb

def get_ddpg_agent(observation_spec, action_spec, custom_layers, global_step, environments): 
    
    """actor_net = ActorNetwork(observation_spec=observation_spec, action_spec=action_spec, custom_layers=custom_layers)

    critic_net = CriticNetwork(observation_spec=observation_spec, action_spec=action_spec, custom_layers=custom_layers)
    
    target_actor_network = ActorNetwork(observation_spec=observation_spec, action_spec=action_spec, custom_layers=custom_layers)

    target_critic_network = CriticNetwork(observation_spec=observation_spec, action_spec=action_spec, custom_layers=custom_layers)
    """

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
    tf_agent = ddpg_agent.DdpgAgent(**agent_params)

    tf_agent.initialize()
    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    return tf_agent, eval_policy, collect_policy

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