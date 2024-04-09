import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def geometric_median(X, eps=1e-5):
    """
    Compute the geometric median for a set of points (X) using Weiszfeld's algorithm.
    Note: This is a simplified and approximate implementation.
    Args:
        X (tf.Tensor): A 2D tensor where each row represents a point.
        eps (float): Convergence criterion.
    Returns:
        tf.Tensor: The geometric median of the points.
    """
    y = tf.reduce_mean(X, axis=0)
    for _ in range(100):  # max iterations
        D = tf.norm(X - y, axis=1)
        nonzeros = tf.cast(D > eps, tf.float32)
        Dinv = tf.where(D > eps, 1 / D, 0)
        W = Dinv / tf.reduce_sum(Dinv)
        T = tf.reduce_sum(W[:, None] * X, axis=0)
        if tf.norm(T - y) < eps:
            return T
        y = T
    return y

class FederatedAggregation():
    
    #Momentum? Which update performed best
    #Baysian? Use Baysian inference

    def federated_average_aggregation(weights_list, noise_stddev=0.0, clipping=None, aggregation_method='mean'):
        """
        This function averages the weights of models from different sources, optionally applies clipping to the
        averaged weights to control the influence of outliers, and adds Gaussian noise for differential privacy.

        Args:
            weights_list (list): A nested list, where each inner list contains the network weights (same architectures required).
            noise (float, optional): The standard deviation of the Gaussian noise to be added for differential privacy.
            clipping (float, optional): Clipping treshhold for weights.
        Returns:
            list: A list of tensors representing the averaged, optionally clipped and noised weights of the network.
    """
        aggregated_weights = []
        for weight_pair in zip(*weights_list):
            # Convert weight pairs to tensor
            weight_tensor = tf.convert_to_tensor(weight_pair)

            #1. Apply clipping if threshold is specified           
            if clipping is not None:
                weight_tensor = tf.clip_by_norm(weight_tensor, clipping)
            """ -> Finer control
            if clipping is not None:
                norms = tf.norm(weight_tensor, axis=0)
                desired_norms = tf.minimum(norms, clipping)
                weight_tensor = weight_tensor * (desired_norms / (norms + 1e-6))
            """

            #2. Averaging
            if aggregation_method == 'mean':
                aggregated_weight = tf.math.reduce_mean(weight_tensor, axis=0)
            elif aggregation_method == 'median':
                aggregated_weight = tfp.stats.percentile(weight_tensor, 50.0, axis=0, interpolation='midpoint')
            elif aggregation_method == 'geometric_median':
                aggregated_weight = geometric_median(weight_tensor)
            else:
                raise ValueError("Unsupported aggregation method: {}".format(aggregation_method))
            
            #3. Adding noise
            noise = tf.random.normal(shape=aggregated_weight.shape, mean=0.0, stddev=noise_stddev)
            aggregated_weight = aggregated_weight + noise

            aggregated_weights.append(aggregated_weight)

        return aggregated_weights
    
    def federated_weigthed_aggregation(weights_list, performance_metrics, aggregation_method='mean', clipping=None, noise_stddev=0.0):
        
        """
        Aggregates weights from multiple models based on their performance metrics.
        
        Parameters:
        - weights_list: List of lists, where each sublist contains the weights of a model.
        - performance_metrics: List of metrics corresponding to the models' performances.
        - aggregation_method: Method to use for aggregation. Supports 'mean' and 'weighted_mean'.
        - clipping: Maximum norm for each weight vector. If None, no clipping is applied.
        - noise_stddev: Standard deviation of Gaussian noise to be added for differential privacy. If 0, no noise is added.
        
        Returns:
        - aggregated_weights: List of aggregated weights.
        """
        performance_metrics_tensor = tf.convert_to_tensor(performance_metrics, dtype=tf.float32)
        weights_list_tensor = [[tf.convert_to_tensor(layer) for layer in model] for model in weights_list]
       
        # Calulate the weights based on performance metrics and weighting method
        if aggregation_method == 'mean':
            offset = abs(min(performance_metrics)) + 1e-6
            transformed_metrics = [metric + offset for metric in performance_metrics]
            total_performance = sum(transformed_metrics)
            performance_weights = [metric / total_performance for metric in transformed_metrics]
        elif aggregation_method == 'softmax':
            performance_weights = tf.nn.softmax(performance_metrics_tensor)
        elif aggregation_method == 'top1':
            _, top_indices = tf.nn.top_k(performance_metrics_tensor, k=1)
            performance_weights = tf.cast(tf.reduce_sum(tf.one_hot(top_indices, depth=tf.size(performance_metrics_tensor)), axis=0), tf.float32)
        else: 
            print("select an aggregation method from: mean, softmax, top1")
            return
        
        #Federated weighted aggregation for each layer
        aggregated_weights = []
        num_layers = len(weights_list[0])
        for layer_idx in range(num_layers):
            layer_weights = np.array([model[layer_idx] for model in weights_list_tensor])
            
            #1. Apply clipping if threshold is specified
            if clipping is not None:
                #norms = np.linalg.norm(layer_weights, axis=1, keepdims=True)
                #desired = np.clip(norms, None, clipping)
                #layer_weights = layer_weights * (desired / np.maximum(norms, 1e-16))
                layer_weights = tf.clip_by_norm(layer_weights, clipping)
            
            #2. Average weights
            weighted_sum = np.tensordot(performance_weights, layer_weights, axes=([0], [0]))
                       
            #3. Add noise
            noise = tf.random.normal(shape=weighted_sum.shape, mean=0.0, stddev=noise_stddev)
            weighted_sum += noise
            
            aggregated_weights.append(weighted_sum)
        
        return aggregated_weights
            
