import tensorflow as tf
from tf_agents.agents import ddpg
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
import wandb

import utils.dataloader as dataloader
import environments.battery as battery_env
import environments.household as household_env

"""
Train and evaluate a DDPG agent
"""

# Param for iteration
num_iterations = 5000
customer = 1
# 0 = battery, 1 = household
env = 0
# Params for collect
initial_collect_steps = 1000
collect_steps_per_iteration = 2000
replay_buffer_capacity = 1000000
ou_stddev = 0.2
ou_damping = 0.15

# Params for target update
target_update_tau = 0.05
target_update_period = 5

# Params for train
train_steps_per_iteration = 1
batch_size = 48 * 7
actor_learning_rate = 1e-4
critic_learning_rate = 1e-3
dqda_clipping = None
td_errors_loss_fn = tf.compat.v1.losses.huber_loss
gamma = 0.99
reward_scale_factor = 1.0
gradient_clipping = None

# Params for eval and checkpoints
num_eval_episodes = 1
num_test_episodes = 1
eval_interval = 50

# Load data
data_train = dataloader.get_customer_data(dataloader.loadData('./data/load1011.csv'),
                                          dataloader.loadPrice('./data/price_wo_outlier.csv'), dataloader.loadMix("./data/fuel2021.csv"), customer)
data_eval = dataloader.get_customer_data(dataloader.loadData('./data/load1112.csv'),
                                         dataloader.loadPrice('./data/price_wo_outlier.csv'), dataloader.loadMix("./data/fuel2122.csv"), customer)
data_test = dataloader.get_customer_data(dataloader.loadData('./data/load1213.csv'),
                                         dataloader.loadPrice('./data/price_wo_outlier.csv'), dataloader.loadMix("./data/fuel2223.csv"), customer)

# Initiate env
if env == 0:
    tf_env_train = tf_py_environment.TFPyEnvironment(battery_env.Battery(init_charge=0.0, data=data_train))
    tf_env_eval = tf_py_environment.TFPyEnvironment(battery_env.Battery(init_charge=0.0, data=data_eval))
else:
    tf_env_train = tf_py_environment.TFPyEnvironment(household_env.Household(init_charge=0.0, data=data_train))
    tf_env_eval = tf_py_environment.TFPyEnvironment(household_env.Household(init_charge=0.0, data=data_eval))

# Prepare runner
global_step = tf.compat.v1.train.get_or_create_global_step()

actor_net = ddpg.actor_network.ActorNetwork(
    input_tensor_spec=tf_env_train.observation_spec(),
    output_tensor_spec=tf_env_train.action_spec(), fc_layer_params=(400, 300),
    activation_fn=tf.keras.activations.relu)

critic_net = ddpg.critic_network.CriticNetwork(
    input_tensor_spec=(tf_env_train.observation_spec(), tf_env_train.action_spec()),
    joint_fc_layer_params=(400, 300),
    activation_fn=tf.keras.activations.relu)

tf_agent = ddpg_agent.DdpgAgent(
    tf_env_train.time_step_spec(),
    tf_env_train.action_spec(),
    actor_network=actor_net,
    critic_network=critic_net,
    actor_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=actor_learning_rate
    ),
    critic_optimizer=tf.compat.v1.train.AdamOptimizer(
        learning_rate=critic_learning_rate
    ),
    ou_stddev=ou_stddev,
    ou_damping=ou_damping,
    target_update_tau=target_update_tau,
    target_update_period=target_update_period,
    dqda_clipping=dqda_clipping,
    td_errors_loss_fn=td_errors_loss_fn,
    gamma=gamma,
    reward_scale_factor=reward_scale_factor,
    gradient_clipping=gradient_clipping,
    debug_summaries=False,
    summarize_grads_and_vars=False,
    train_step_counter=global_step,
)

tf_agent.initialize()

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

print("Batch: ", tf_env_train.batch_size)

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    tf_agent.collect_data_spec,
    batch_size=tf_env_train.batch_size,
    max_length=replay_buffer_capacity,
)

initial_collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_env_train,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=initial_collect_steps,
)

collect_driver = dynamic_step_driver.DynamicStepDriver(
    tf_env_train,
    collect_policy,
    observers=[replay_buffer.add_batch],
    num_steps=collect_steps_per_iteration,
)

wandb.login()
wandb.init(
    project="DDPG_battery",
    job_type="train_eval_test",
    name="3_ex_09",
    config={
        "train_steps": num_iterations,
        "batch_size": batch_size,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": critic_learning_rate}
)

artifact = wandb.Artifact(name='save', type="checkpoint")

eval_metrics = [
    tf_metrics.AverageReturnMetric(name="AverageReturnEvaluation", buffer_size=num_eval_episodes)
]

test_metrics = [
    tf_metrics.AverageReturnMetric(name="AverageReturnTest", buffer_size=num_eval_episodes)
]

train_checkpointer = common.Checkpointer(
    ckpt_dir='checkpoints/ddpg/',
    max_to_keep=1,
    agent=tf_agent,
    policy=tf_agent.policy,
    replay_buffer=replay_buffer,
    global_step=global_step
)

train_checkpointer.initialize_or_restore()

global_step = tf.compat.v1.train.get_global_step()

# For better performance
initial_collect_driver.run = common.function(initial_collect_driver.run)
collect_driver.run = common.function(collect_driver.run)
tf_agent.train = common.function(tf_agent.train)

# Collect initial replay data
initial_collect_driver.run()

time_step = tf_env_train.reset()
policy_state = collect_policy.get_initial_state(tf_env_train.batch_size)

# Dataset generates trajectories with shape [Bx2x...]
# pipeline which will feed data to the agent
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
).prefetch(3)
iterator = iter(dataset)

# Train and evaluate
print("Start training ...")
while global_step.numpy() < num_iterations:
    time_step, policy_state = collect_driver.run(
        time_step=time_step,
        policy_state=policy_state,
    )
    experience, _ = next(iterator)
    train_loss = tf_agent.train(experience)
    metrics = {}
    if global_step.numpy() % eval_interval == 0:
        train_checkpointer.save(global_step)
        metrics = metric_utils.eager_compute(
            eval_metrics,
            tf_env_eval,
            eval_policy,
            num_episodes=num_eval_episodes,
            train_step=global_step,
            summary_writer=None,
            summary_prefix='',
            use_function=True)
    
    metrics["loss"] = train_loss.loss
    wandb.log(metrics)

# Initiate test env
if env == 0:
    tf_env_test = tf_py_environment.TFPyEnvironment(battery_env.Battery(init_charge=0.0, data=data_test, test=True))
else:
    tf_env_test = tf_py_environment.TFPyEnvironment(household_env.Household(init_charge=0.0, data=data_test, test=True))

print("Start testing ...")
metrics = metric_utils.eager_compute(
    test_metrics,
    tf_env_test,
    eval_policy,
    num_episodes=num_test_episodes,
    train_step=None,
    summary_writer=None,
    summary_prefix='',
    use_function=True)
wandb.log(metrics)
artifact.add_dir(local_path='checkpoints/ddpg/')
wandb.log_artifact(artifact)
wandb.finish()