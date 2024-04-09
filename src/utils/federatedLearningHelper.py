import numpy as np
import os
import pandas as pd
import wandb
from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils

def load_clustered_buildings(num_clusters=10):

    #Catch non-clustered cluster sizes
    if num_clusters < 2 or num_clusters > 20:
        print("Currently clustering has been done from cluster sizes within the range of 2 to 20.")
        return
    
    #Load data from pre-clustered buildings
    cluster_data  = np.loadtxt(f'../../data/3final_data/Clusters_KMeans_dtw_c{num_clusters}.csv', delimiter=',').astype(int)
    
    # Iterate through each cluster
    clustered_buildings = {i: [] for i in range(num_clusters)}
    for cluster_number in range(num_clusters):
        # Find indices of buildings in the current cluster
        buildings_in_cluster = np.where(cluster_data  == cluster_number)[0] +1
        clustered_buildings[cluster_number] = buildings_in_cluster
    
    return clustered_buildings

def save_agent_weights(global_tf_agent, model_dir):
    np.savez(os.path.join(model_dir, "actor_weights.npz"), *global_tf_agent._actor_network.get_weights())
    np.savez(os.path.join(model_dir, "critic_weights.npz"), *global_tf_agent._critic_network.get_weights())
    np.savez(os.path.join(model_dir, "target_actor_weights.npz"), *global_tf_agent._target_actor_network.get_weights())
    np.savez(os.path.join(model_dir, "target_critic_weights.npz"), *global_tf_agent._target_critic_network.get_weights())

def set_weights_to_ddpg_agent(local_ddpg_agent, model_dir):
    # Extract the arrays using the keys corresponding to their order
    with np.load(os.path.join(model_dir, "actor_weights.npz"), allow_pickle=True) as data:
        actor_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ddpg_agent._actor_network.set_weights(actor_weights)
    
    with np.load(os.path.join(model_dir, "critic_weights.npz"), allow_pickle=True) as data:
        critic_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ddpg_agent._critic_network.set_weights(critic_weights)
    
    with np.load(os.path.join(model_dir, "target_actor_weights.npz"), allow_pickle=True) as data:
        target_actor_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ddpg_agent._target_actor_network.set_weights(target_actor_weights)
    
    with np.load(os.path.join(model_dir, "target_critic_weights.npz"), allow_pickle=True) as data:
        target_critic_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ddpg_agent._target_critic_network.set_weights(target_critic_weights)

    return local_ddpg_agent

def save_federated_weights(model_dir, average_actor_weights, average_critic_weights, average_target_actor_weights, average_target_critic_weights):
    np.savez(os.path.join(model_dir, "actor_weights.npz"), *average_actor_weights)
    np.savez(os.path.join(model_dir, "critic_weights.npz"), *average_critic_weights)
    np.savez(os.path.join(model_dir, "target_actor_weights.npz"), *average_target_actor_weights)
    np.savez(os.path.join(model_dir, "target_critic_weights.npz"), *average_target_critic_weights)

def local_agent_training_and_evaluation(
        local_iterator, local_collect_driver, local_time_step, local_policy_state, global_step, local_ddpg_agent, 
        local_eval_policy, local_storage, building_index, num_iterations, environments): 
                    
    eval_metrics = [tf_metrics.AverageReturnMetric()]
    test_metrics = [tf_metrics.AverageReturnMetric()]

    while global_step.numpy() < num_iterations:
        local_time_step, local_policy_state = local_collect_driver.run(time_step=local_time_step, policy_state=local_policy_state)
        local_experience, _ = next(local_iterator)
        local_train_loss = local_ddpg_agent.train(local_experience)
    
    #4.Evaluate training
    eval_metric = metric_utils.eager_compute(test_metrics, environments["eval"][f"building_{building_index}"], local_eval_policy)
    local_storage["performance_metrics"].append(eval_metric["AverageReturn"].numpy())
    print("Return: ", eval_metric["AverageReturn"].numpy())

    local_storage["actor_weights"].append(local_ddpg_agent._actor_network.get_weights())
    local_storage["critic_weights"].append(local_ddpg_agent._critic_network.get_weights())
    local_storage["target_actor_weights"].append(local_ddpg_agent._target_actor_network.get_weights())
    local_storage["target_critic_weights"].append(local_ddpg_agent._target_critic_network.get_weights())

    return  local_ddpg_agent, local_storage

def agent_retraining_and_evaluation(global_step, num_test_iterations, collect_driver, time_step, policy_state, iterator, 
 tf_ddpg_agent, eval_policy, building_index, result_df, eval_interval, environments): 
                    
    eval_metrics = [tf_metrics.AverageReturnMetric()]
    test_metrics = [tf_metrics.AverageReturnMetric()]

    while global_step.numpy() < num_test_iterations:
        
        #Training
        time_step, policy_state = collect_driver.run(time_step=time_step, policy_state=policy_state)
        experience, _ = next(iterator)
        train_loss = tf_ddpg_agent.train(experience)

        #Evaluation
        metrics = {}
        if global_step.numpy() % eval_interval == 0:
            eval_metric = metric_utils.eager_compute(eval_metrics,environments["eval"][f"building_{building_index}"], eval_policy, train_step=global_step)
        if global_step.numpy() % 2 == 0:
            metrics["loss"] = train_loss.loss
            wandb.log(metrics)
    
    #Testing
    test_metrics = metric_utils.eager_compute(test_metrics,environments["test"][f"building_{building_index}"], eval_policy, train_step=global_step)
    result_df = pd.concat([result_df, pd.DataFrame({'Building': [building_index], 'Total Profit': [wandb.summary["Final Profit"]]})], ignore_index=True)
    print('Building: ', building_index, ' - Total Profit: ', wandb.summary["Final Profit"])

    return result_df, metrics