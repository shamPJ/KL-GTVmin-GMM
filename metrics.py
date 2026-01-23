import numpy as np
from scipy.optimize import linear_sum_assignment

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