
import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from src.utils import initialize_gmm_params_kmeans, compute_responsibilities, compute_gmm_updates, bhattacharyya_distance
from src.evaluation import compute_errors, compute_consensus_error, compute_metrics

####################################################################
#                            GRAPHFED_EM                           #
####################################################################

def federated_gmm_training(data, data_val, y_val, N_i_val, A, K, 
                           mu_true=None, cov_true=None, T=100, T_local=5, 
                           mode='fed', beta=1, epsilon=1e-6):
    """
    Federated EM training of a Gaussian Mixture Model (GMM) with graph-based 
    aggregation and Hungarian alignment.

    Parameters
    ----------
    data, data_val : ndarray
        Training and validation data (nodes x samples x features).
    y_val : ndarray
        Validation labels.
    N_i_val : int
        Number of validation samples per node.
    A : ndarray
        Communication adjacency matrix.
    K : int
        Number of Gaussian components.
    mu_true, cov_true : ndarray, optional
        Ground-truth parameters for error analysis.
    T, T_local : int
        Global and local EM iterations.
    mode : str
        Evaluation mode.
    beta : float
        Aggregation blending factor.
    epsilon : float
        Numerical stability.

    Returns
    -------
    results_df : pandas.DataFrame
        Training metrics per iteration.
    pi_est_, mu_est_, sigma_est_ : ndarray
        Estimated mixing coefficients, means, and covariances.
    """
    c, n_ds, N_i, d = data.shape
    n = int(c*n_ds)
    use_shrinkage = (d/N_i) >= 0.2

    # Initialize estimates per node
    pi_est_, mu_est_, sigma_est_ = initialize_gmm_params_kmeans(data, K)
    # Init eval metrics
    ll_, nmi_, ari_ = np.zeros((T,)), np.zeros((T,)), np.zeros((T,))
    # Distance from true parameters
    mu_error_, cov_error_ = np.zeros((T,)), np.zeros((T,))
    # Distance from neib. parameters
    mu_err_consensus_, cov_err_consensus_ = np.zeros((T,)), np.zeros((T,))

    # Baseline on iter zero
    ll_0, nmi_0, ari_0 = compute_metrics(data_val, y_val, N_i_val, pi_est_, mu_est_, sigma_est_, mode)
    mu_err_0, cov_err_0 = compute_errors(mu_true, cov_true, mu_est_, sigma_est_)
    mu_err_consensus_0, cov_err_consensus_0, _ = compute_consensus_error(data, A, pi_est_, mu_est_, sigma_est_,epsilon=1e-6)
    
    ll_[0], nmi_[0], ari_[0] = ll_0, nmi_0, ari_0
    mu_error_[0], cov_error_[0] = mu_err_0, cov_err_0
    mu_err_consensus_[0], cov_err_consensus_[0] = mu_err_consensus_0, cov_err_consensus_0

    for t in range(1,T):
        """
        ====================================
        Local EM steps
        ====================================
        """
        for _ in range(T_local):   
            """
            ====================================
             E-step: Compute responsibilities
            ====================================
            """
            gamma = np.zeros((c, n_ds, N_i, K))
            for c_i in range(c):
                for ds in range(n_ds):
                    X, pi, mu, sigma = data[c_i, ds], pi_est_[c_i, ds], mu_est_[c_i, ds], sigma_est_[c_i, ds]
                    gamma[c_i, ds] = compute_responsibilities(X, pi, mu, sigma)      
    
            """
            ====================================
             M-step: Update parameters
            ====================================
            """
            ll = 0.0
            # init arrays to store M-step updates
            # pi_est_t, mu_est_t, sigma_est_t = np.zeros_like(pi_est_), np.zeros_like(mu_est_), np.zeros_like(sigma_est_)
            for c_i in range(c):
                for ds in range(n_ds):
                    X, pi, mu, sigma = data[c_i, ds], pi_est_[c_i, ds], mu_est_[c_i, ds], sigma_est_[c_i, ds]
                    gamma_i = gamma[c_i, ds]
                    
                    # compute closed form update for each local dataset
                    pi, mu, sigma = compute_gmm_updates(X, gamma_i, pi, mu, sigma, use_shrinkage=use_shrinkage)
                    pi_est_[c_i, ds] = pi
                    mu_est_[c_i, ds] = mu
                    sigma_est_[c_i, ds] = sigma
        
        
        """
        =========================================
        Aggregation step via adjacency matrix A  
        =========================================
        """
        if beta == 0:
            pass
        else:
            # temp vars not to overwrite params
            pi_est_t = pi_est_.copy()
            mu_est_t = mu_est_.copy()
            sigma_est_t = sigma_est_.copy()
        
            for c_i in range(c):
                for ds in range(n_ds):
                    i = c_i*n_ds + ds # current node's id
                    A_i = A[i] # adjacency vector for node i; shape (n,)
                    mu_i, pi_i, sigma_i = mu_est_t[c_i, ds], pi_est_t[c_i, ds], sigma_est_t[c_i, ds]
    
                    # Effective counts N_k for current node `i`, 
                    # i.e. for each component `k` n.o. datapoints it is "responsible" for
                    Nk_i = np.sum(gamma[c_i, ds], axis=0) + epsilon # shape (K,)
            
                    # Initialize accumulators for weighted averages
                    # by weighting parameters value with Nk_i
                    mu_accum = Nk_i[:, None] * mu_i # shape (K,d)
                    # pi_accum = Nk_i * pi_i          # shape (K,)
                    # accumulate effective counts instead of Nk * pi
                    pi_accum = Nk_i.copy()          # just the counts N_{i,k}
                    sigma_accum = Nk_i[:, None, None] * sigma_i # shape (K,d,d);
            
                    total_weight = Nk_i.copy() # shape (K,)
                    # loop over neighbors nodes
                    for j, A_ij in enumerate(A_i):
                        if A_ij <= 0:
                            continue
    
                        c_j, ds_j = divmod(j, n_ds)
                        mu_j, pi_j, sigma_j  = mu_est_t[c_j, ds_j], pi_est_t[c_j, ds_j], sigma_est_t[c_j, ds_j]
                        # effective counts N_k neighbors node `j`, 
                        Nk_j = np.sum(gamma[c_j, ds_j], axis=0) + epsilon
            
                        # Compute distance matrix to find the best matching component
                        D = np.zeros((K, K))
                        for k1 in range(K):
                            for k2 in range(K):
                                D[k1, k2] = bhattacharyya_distance(mu_i[k1], sigma_i[k1], mu_j[k2], sigma_j[k2])
                        
                        row_ind, col_ind = linear_sum_assignment(D)
    
                        for k in range(K):
                            l = col_ind[k] # `l` is idx of the best matching component
                            # w_ik, w_jl = Nk_i[k], Nk_j[l] # retrieve effective size of matched components `k`,`l`
            
                            # add weighted param of node `j` for component `k`
                            mu_accum[k] += A_ij * Nk_j[l] * mu_j[l]
                            pi_accum[k] += A_ij * Nk_j[l]
                            sigma_accum[k] += A_ij * Nk_j[l] * sigma_j[l]
                            total_weight[k] += A_ij * Nk_j[l]
            
                    # controlled update (blending with previous estimates)
                    mu_agg = mu_accum / total_weight[:, None]
                    pi_agg = pi_accum / (np.sum(pi_accum) + epsilon)
                    sigma_agg = sigma_accum / total_weight[:, None, None]
    
                    mu_est_[c_i, ds] = (1 - beta) * mu_est_t[c_i, ds] + beta * mu_agg
                    pi_est_[c_i, ds] = (1 - beta) * pi_est_t[c_i, ds] + beta * pi_agg
                    sigma_est_[c_i, ds] = (1 - beta) * sigma_est_t[c_i, ds] + beta * sigma_agg
    
                    # Symmetrize and regularize
                    for k in range(K):
                        sigma_k = sigma_est_[c_i, ds][k]
                        sigma_k = (sigma_k + sigma_k.T) / 2
                        sigma_k += 1e-6 * np.eye(d)
                        sigma_est_[c_i, ds][k] = sigma_k 
                    
        #====================Performance metrics on validation set====================#
        ll, nmi, ari = compute_metrics(data_val, y_val, N_i_val, pi_est_, mu_est_, sigma_est_, mode)
        mu_error, cov_error = compute_errors(mu_true, cov_true, mu_est_, sigma_est_)
        
        ll_[t], nmi_[t], ari_[t]  = ll, nmi, ari 
        mu_error_[t], cov_error_[t] = mu_error, cov_error 

        mu_err_consensus, cov_err_consensus, _ = compute_consensus_error(data, A, pi_est_, mu_est_, sigma_est_, epsilon=1e-6)
        mu_err_consensus_[t], cov_err_consensus_[t] = mu_err_consensus, cov_err_consensus 
        
    results = {
        'Algorithm': ['GraphFed-EM']*(T),
        'LogLikelihood': ll_,
        'NMI': nmi_,
        'ARI': ari_,
        'mu err': mu_error_,
        'cov err': cov_error_,
        'T': np.arange(T)
    }

    if np.any(A) and len(A)!=1:
        results['mu err consensus'] = mu_err_consensus_
        results['cov err consensus'] = cov_err_consensus_
    else:
        results['mu err consensus'] = None
        results['cov err consensus'] = None
 
    return pd.DataFrame(results), mu_est_, sigma_est_