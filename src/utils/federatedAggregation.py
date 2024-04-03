import tensorflow as tf
import tensorflow_probability as tfp

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
        Performs federated weighted aggregation on a list of model weights, with various aggregation methods.

        Args:
            weights_list (list): A list of lists, where each inner list contains the weights (as numpy arrays or tensors)
                                of an actor's network.
            performance_metrics (list): A list of performance metrics corresponding to each set of weights.
            aggregation_method (str): Method of aggregation - 'mean', 'median', 'geometric_median', 'softmax', 'top1', 'top2'.
            clipping (float, optional): Clipping threshold for the L2 norm of the weights.
            noise_scale (float, optional): Standard deviation of Gaussian noise added for privacy.

        Returns:
            list: A list of tensors representing the aggregated weights.
        """
        
        aggregated_weights = []
        performance_tensor = tf.convert_to_tensor(performance_metrics, dtype=tf.float32)
        # Shift the performance metrics to ensure all are positive.
        min_performance = tf.reduce_min(performance_tensor)
        performance_tensor = performance_tensor - min_performance + 1
        
        #0. Normalize performance metric to get importance based on the aggregation
        if aggregation_method == 'mean': 
            performance_weights = performance_tensor / tf.reduce_sum(performance_tensor)
        elif aggregation_method == 'softmax':
            performance_weights = tf.nn.softmax(performance_tensor)
        elif aggregation_method == 'top_1':
            top_k_values, top_k_indices = tf.nn.top_k(performance_tensor, k=1)
            performance_weights = tf.one_hot(top_k_indices[0], depth=len(performance_metrics))
            performance_weights /= tf.reduce_sum(performance_weights)
        elif aggregation_method == "top_2":
            top_k_values, top_k_indices = tf.nn.top_k(performance_tensor, k=2)
            performance_weights = tf.reduce_sum(tf.one_hot(top_k_indices, depth=len(performance_metrics)), axis=0)
            performance_weights /= tf.reduce_sum(performance_weights)  # Normalize weights
        else:
            raise ValueError("Unsupported aggregation method: {}".format(aggregation_method))
            

        #1. Align weights: Unpack weights_list and pair up the corresponding elements of each sublist
        for weight_pair in zip(*weights_list):
            
            weight_tensor = tf.convert_to_tensor(weight_pair, dtype=tf.float32)
            
            #1. Apply clipping if threshold is specified           
            if clipping is not None:
                weight_tensor = tf.clip_by_norm(weight_tensor, clipping)

            #2. Average weights
            aggregated_weight = tf.reduce_sum(weight_tensor * performance_weights[:, tf.newaxis, tf.newaxis], axis=0)

            #3. Add noise
            noise = tf.random.normal(shape=aggregated_weight.shape, mean=0.0, stddev=noise_stddev)
            aggregated_noisy_weight = aggregated_weight + noise

            aggregated_weights.append(aggregated_noisy_weight)
        
        return aggregated_weights 