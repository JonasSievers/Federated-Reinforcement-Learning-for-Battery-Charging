import numpy as np
import os
import pandas as pd
import wandb
from tf_agents.metrics import tf_metrics
from tf_agents.eval import metric_utils
import pickle

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

def prosumption_clustered_buildings(num_clusters=10):

    # Catch non-clustered cluster sizes
    if num_clusters < 2 or num_clusters > 20:
        print("Currently, clustering has been done from cluster sizes within the range of 2 to 20.")
        return

    with open(f'../../data/3final_data/cluster_labels.pkl', 'rb') as file:
        cluster_data = pickle.load(file)

    # Retrieve cluster data from the provided dictionary
    cluster_data = cluster_data[num_clusters]

    # Iterate through each cluster
    clustered_buildings = {i: [] for i in range(num_clusters)}
    for cluster_number in range(num_clusters):
        # Find indices of buildings in the current cluster
        buildings_in_cluster = np.where(cluster_data == cluster_number)[0] + 1
        clustered_buildings[cluster_number] = buildings_in_cluster

    return clustered_buildings

def save_ddpg_weights(global_tf_agent, model_dir):
    np.savez(os.path.join(model_dir, "actor_weights.npz"), *global_tf_agent._actor_network.get_weights())
    np.savez(os.path.join(model_dir, "critic_weights.npz"), *global_tf_agent._critic_network.get_weights())
    np.savez(os.path.join(model_dir, "target_actor_weights.npz"), *global_tf_agent._target_actor_network.get_weights())
    np.savez(os.path.join(model_dir, "target_critic_weights.npz"), *global_tf_agent._target_critic_network.get_weights())

def save_sac_weights(global_tf_agent, model_dir):
    np.savez(os.path.join(model_dir, "actor_weights.npz"), *global_tf_agent._actor_network.get_weights())
    np.savez(os.path.join(model_dir, "critic_weights_1.npz"), *global_tf_agent._critic_network_1.get_weights())
    np.savez(os.path.join(model_dir, "critic_weights_2.npz"), *global_tf_agent._critic_network_2.get_weights())

def save_td3_weights(global_td3_agent, model_dir):
    np.savez(os.path.join(model_dir, "actor_weights.npz"), *global_td3_agent._actor_network.get_weights())
    np.savez(os.path.join(model_dir, "critic_weights_1.npz"), *global_td3_agent._critic_network_1.get_weights())

def save_ppo_weights(global_ppo_agent, model_dir):
    np.savez(os.path.join(model_dir, "actor_weights.npz"), *global_ppo_agent._actor_net.get_weights())
    np.savez(os.path.join(model_dir, "value_weights.npz"), *global_ppo_agent._value_net.get_weights())

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

def set_weights_to_sac_agent(local_sac_agent, model_dir):
    # Extract the arrays using the keys corresponding to their order
    with np.load(os.path.join(model_dir, "actor_weights.npz"), allow_pickle=True) as data:
        actor_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_sac_agent._actor_network.set_weights(actor_weights)
    
    with np.load(os.path.join(model_dir, "critic_weights_1.npz"), allow_pickle=True) as data:
        critic_weights_1 = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_sac_agent._critic_network_1.set_weights(critic_weights_1)
    
    with np.load(os.path.join(model_dir, "critic_weights_2.npz"), allow_pickle=True) as data:
        critic_weights_2 = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_sac_agent._critic_network_2.set_weights(critic_weights_2)

    return local_sac_agent

def set_weights_to_td3_agent(local_td3_agent, model_dir):
    # Extract the arrays using the keys corresponding to their order
    with np.load(os.path.join(model_dir, "actor_weights.npz"), allow_pickle=True) as data:
        actor_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_td3_agent._actor_network.set_weights(actor_weights)
    
    with np.load(os.path.join(model_dir, "critic_weights_1.npz"), allow_pickle=True) as data:
        critic_weights_1 = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_td3_agent._critic_network_1.set_weights(critic_weights_1)
    
    return local_td3_agent

def set_weights_to_ppo_agent(local_ppo_agent, model_dir):
    # Extract the arrays using the keys corresponding to their order
    with np.load(os.path.join(model_dir, "actor_weights.npz"), allow_pickle=True) as data:
        actor_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ppo_agent._actor_net.set_weights(actor_weights)
    
    with np.load(os.path.join(model_dir, "value_weights.npz"), allow_pickle=True) as data:
        value_weights = [data[f'arr_{i}'] for i in range(len(data.files))]
        local_ppo_agent._value_net.set_weights(value_weights)
    
    return local_ppo_agent

def save_federated_ddpg_weights(model_dir, average_actor_weights, average_critic_weights, average_target_actor_weights, average_target_critic_weights):
    np.savez(os.path.join(model_dir, "actor_weights.npz"), *average_actor_weights)
    np.savez(os.path.join(model_dir, "critic_weights.npz"), *average_critic_weights)
    np.savez(os.path.join(model_dir, "target_actor_weights.npz"), *average_target_actor_weights)
    np.savez(os.path.join(model_dir, "target_critic_weights.npz"), *average_target_critic_weights)

def save_federated_sac_weights(model_dir, average_actor_weights, average_critic_weights_1, average_critic_weights_2):
    np.savez(os.path.join(model_dir, "actor_weights.npz"), *average_actor_weights)
    np.savez(os.path.join(model_dir, "critic_weights_1.npz"), *average_critic_weights_1)
    np.savez(os.path.join(model_dir, "critic_weights_2.npz"), *average_critic_weights_2)

def save_federated_td3_weights(model_dir, average_actor_weights, average_critic_weights_1):
    np.savez(os.path.join(model_dir, "actor_weights.npz"), *average_actor_weights)
    np.savez(os.path.join(model_dir, "critic_weights_1.npz"), *average_critic_weights_1)

def save_federated_ppo_weights(model_dir, average_actor_weights, average_value_weights):
    np.savez(os.path.join(model_dir, "actor_weights.npz"), *average_actor_weights)
    np.savez(os.path.join(model_dir, "value_weights.npz"), *average_value_weights)

def local_agent_training_and_evaluation(
        iterator, collect_driver, time_step, policy_state, global_step, tf_agent, 
        eval_policy, local_storage, building_index, num_iterations, environments, agent_type): 
                    
    eval_metrics = [tf_metrics.AverageReturnMetric()]
    test_metrics = [tf_metrics.AverageReturnMetric()]

    while global_step.numpy() < num_iterations:
        time_step, policy_state = collect_driver.run(time_step=time_step, policy_state=policy_state)
        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience)
    
    #4.Evaluate training
    eval_metric = metric_utils.eager_compute(test_metrics, environments["eval"][f"building_{building_index}"], eval_policy)
    local_storage["performance_metrics"].append(eval_metric["AverageReturn"].numpy())
    print("Return: ", eval_metric["AverageReturn"].numpy())
    
    if agent_type == "ddpg":
        local_storage = append_ddpg_weights_to_local_storage(tf_agent, local_storage)
    elif agent_type =="sac":
        local_storage = append_sac_weights_to_local_storage(tf_agent, local_storage)
    elif agent_type == "td3":
        local_storage = append_td3_weights_to_local_storage(tf_agent, local_storage)
    elif agent_type == "ppo":
        local_storage = append_ppo_weights_to_local_storage(tf_agent, local_storage)
     
    return  tf_agent, local_storage

def append_ddpg_weights_to_local_storage(tf_agent, local_storage): 
    local_storage["actor_weights"].append(tf_agent._actor_network.get_weights())
    local_storage["critic_weights"].append(tf_agent._critic_network.get_weights())
    local_storage["target_actor_weights"].append(tf_agent._target_actor_network.get_weights())
    local_storage["target_critic_weights"].append(tf_agent._target_critic_network.get_weights())
    return local_storage

def append_sac_weights_to_local_storage(tf_agent, local_storage):
    local_storage["actor_weights"].append(tf_agent._actor_network.get_weights())
    local_storage["critic_weights_1"].append(tf_agent._critic_network_1.get_weights())
    local_storage["critic_weights_2"].append(tf_agent._critic_network_2.get_weights())
    return local_storage

def append_td3_weights_to_local_storage(tf_agent, local_storage):
    local_storage["actor_weights"].append(tf_agent._actor_network.get_weights())
    local_storage["critic_weights_1"].append(tf_agent._critic_network_1.get_weights())
    return local_storage

def append_ppo_weights_to_local_storage(tf_agent, local_storage):
    local_storage["actor_weights"].append(tf_agent._actor_net.get_weights())
    local_storage["value_weights"].append(tf_agent._value_net.get_weights())
    return local_storage
