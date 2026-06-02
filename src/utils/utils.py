import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Detach torch tensor and move to NumPy."""
    return x.detach().cpu().numpy()

def kmeans_initialize(node_data, K, random_state=None):
    local_centers = []
    for node_idx, node in enumerate(node_data):
        seed = None if random_state is None else (random_state + node_idx)
        km = KMeans(n_clusters=K, init='k-means++', n_init=10, random_state=seed)
        km.fit(node)
        local_centers.append(km.cluster_centers_)
    return local_centers

def match_clusters(mu_ref, mu_target):
    """
    Align clusters of mu_target to mu_ref using Hungarian algorithm
    based on Euclidean distance.
    """
    cost = np.linalg.norm(mu_ref[:, None, :] - mu_target[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    return mu_target[col_ind], col_ind

def est_error(pred_means: np.ndarray, true_means: np.ndarray) -> float:
    """
    Compute parameter estimation error between predicted and true GMM means
    using Hungarian matching.
    
    Args:
        pred_means: np.ndarray of shape (K, D)
        true_means: np.ndarray of shape (K, D)
    
    Returns:
        avg_error: float, average Euclidean distance after matching
    """
    # Compute cost matrix (Euclidean distances)
    cost_matrix = np.linalg.norm(pred_means[:, None, :] - true_means[None, :, :], axis=2)
    
    # Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Match predicted and true
    matched_pred = pred_means[row_ind]
    matched_true = true_means[col_ind]
    
    # Average Euclidean distance per component
    avg_error = np.linalg.norm(matched_pred - matched_true, axis=1).mean()
    return avg_error