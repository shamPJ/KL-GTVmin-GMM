import numpy as np
import pandas as pd
import warnings
from scipy.optimize import linear_sum_assignment
from scipy.linalg import cholesky, solve_triangular
from src.utils import initialize_gmm_params_kmeans, compute_responsibilities, make_spd, stabilize_cov, is_spd
from src.evaluation import compute_errors, compute_consensus_error, compute_metrics

####################################################################
#                             FEDGREM                              #
####################################################################

def grad_step(X, pi, mu, Sigma, gamma, lr, eps=1e-6):
    """
    One gradient step for Gaussian mixture parameters using Cholesky solves.

    X: (N_i, d) data for one client
    pi: (K,) mixture weights
    mu: (K, d) means
    Sigma: (K, d, d) covariances
    gamma: (N_i, K) responsibilities
    lr: step size
    """

    N_i, d = X.shape
    K, _ = mu.shape
    
    grad_mu = np.zeros_like(mu)       # (K, d)
    grad_Sigma = np.zeros_like(Sigma) # (K, d, d)
    I = np.eye(d)
    
    for k in range(K):
        diff = X - mu[k].reshape(1,-1)   # (N_i, d)
        s_k = np.sum(gamma[:, k][:, None] * diff, axis=0)  # weighted sum (d,)
        S_k = diff.T @ (gamma[:, k][:, None] * diff) # (d,d)
        Nk = np.sum(gamma[:, k]) + eps

        # compute inverse of cov from Cholesky
        L = cholesky(Sigma[k] + eps * np.eye(d), lower=True)
        Z = solve_triangular(L, I, lower=True)
        cov_inv = solve_triangular(L.T, Z, lower=False)

        # --------- Mean gradient ---------
        grad_mu[k] = (cov_inv @ s_k.reshape(d,1)).ravel()

        # --------- Covariance gradient ---------
        covinv_Sk_covinv = cov_inv @ S_k @ cov_inv   
        grad_Sigma[k] = 0.5 * (covinv_Sk_covinv - Nk * cov_inv)

    # Gradient update
    mu_new = mu + lr * grad_mu            # maximizing Q → ascent
    Sigma_new = Sigma + lr * grad_Sigma

    # Ensure positive-definiteness of covariances
    for k in range(K):
        Sigma_new[k] = make_spd(Sigma_new[k], eps=eps)

    return mu_new, Sigma_new

def match_clusters(mu_ref, mu_targets, sigma_targets):
    """
    Align clusters of mu_target to mu_ref using Hungarian algorithm
    based on Euclidean distance.
    mu_ref shape (K, d)
    mu_targets shape (n_nodes, K, d)
    """
    n_nodes, K, d = mu_targets.shape
    mu_aligned, sigma_aligned = [], []
    
    for i in range(n_nodes):
        mu_target, sigma_target = mu_targets[i], sigma_targets[i] 
        cost = np.linalg.norm(mu_ref[:, None, :] - mu_target[None, :, :], axis=2)
        # check for NaN, inf
        # replace any inf/nan with a large finite number so Hungarian won't fail
        if not np.isfinite(cost).all():
            # choose a large positive value bigger than any finite distance observed
            finite_max = np.nanmax(cost[np.isfinite(cost)]) if np.isfinite(cost).any() else 1.0
            large_val = max(1e6, finite_max * 1e6)
            cost = np.nan_to_num(cost, nan=large_val, posinf=large_val, neginf=large_val)
            warnings.warn(f"Cost matrix for node {i} contained NaN/Inf; replaced with {large_val}.")

        row_ind, col_ind = linear_sum_assignment(cost)
        mu_aligned.append(mu_target[col_ind])
        sigma_aligned.append(sigma_target[col_ind])
    
    return np.array(mu_aligned),  np.array(sigma_aligned)
    
def central_update(mu_est, sigma_est, N_i, max_iter=5, epsilon=1e-6):
    """
    Central update for FedGrEM: proximal update of local GMM parameters (mu and sigma)
    
    Parameters
    ----------
    mu_est : np.array
        Local means from all nodes, shape (n_clusters, n_ds, K, d)
    sigma_est : np.array
        Local covariances from all nodes, shape (n_clusters, n_ds, K, d, d)
    N_i : scalar
        Local sample size
    epsilon : float
        Small constant for numerical stability
    max_iter : int
        Maximum number of alternating optimization iterations.
        From paper "We iterate (ii) and (iii) for a few times until convergence."
    
    Returns
    -------
    mu_central : np.array
        Updated means, shape (n_nodes, K, d)
    sigma_central : np.array
        Updated covariances, shape (n_nodes, K, d, d)
    mu_bar : np.array
        Global center for means, shape (K, d)
    sigma_bar : np.array
        Global center for covariances, shape (K, d, d)
    """

    
    n_clusters, n_ds, K, d = mu_est.shape
    n_nodes = int(n_clusters*n_ds)
    mu_est = mu_est.reshape(n_nodes, K, d)
    sigma_est = sigma_est.reshape(n_nodes, K, d, d)
    
    # Initialize deviations
    Delta_mu = np.zeros_like(mu_est)
    Delta_sigma = np.zeros_like(sigma_est)
    
    # Initialize global centers
    mu_bar = np.mean(mu_est, axis=0)
    sigma_bar = np.mean(sigma_est, axis=0)

    # Init penalty coefficient
    lambda_t = 1
    # Compute penalty update const term p.18 https://arxiv.org/abs/2310.15330
    pen_kappa = 0.1
    pen_C = 2
    pen_const = pen_C*np.sqrt(d + np.log(K))

    # Align components
    mu_ref = mu_est[0]
    mu_est, sigma_est = match_clusters(mu_ref, mu_est, sigma_est)
    
    for it in range(max_iter): 
        # Step 1: update global means and covariances
        mu_bar = np.mean(mu_est - Delta_mu, axis=0)
        sigma_bar = np.mean(sigma_est - Delta_sigma, axis=0)
        
        # Step 2: update deviations using vector soft-thresholding
        for i in range(n_nodes):
            for k in range(K):
                # --- Update mu ---
                a_mu = mu_est[i, k] - mu_bar[k] # (d,)
                norm_a_mu = np.linalg.norm(a_mu)
                threshold = lambda_t / np.sqrt(N_i)
                if norm_a_mu < threshold:
                    Delta_mu[i, k] = 0.0
                else:
                    Delta_mu[i, k] = (1 - threshold / norm_a_mu) * a_mu
                
                # --- Update sigma (Frobenius norm soft-thresholding) ---
                a_sigma = sigma_est[i, k] - sigma_bar[k]
                norm_a_sigma = np.linalg.norm(a_sigma, 'fro')
                
                if norm_a_sigma < threshold:
                    Delta_sigma[i, k] = 0.0
                else:
                    Delta_sigma[i, k] = (1 - threshold / norm_a_sigma) * a_sigma
        # upd penalty coeff
        lambda_t = pen_kappa*lambda_t + pen_const
        
    # Compute final centralized estimates
    mu_central = mu_bar + Delta_mu
    sigma_central = sigma_bar + Delta_sigma
    
    # Ensure sigma is symmetric and positive definite
    for i in range(n_nodes):
        sigma = sigma_central[i]
        for k in range(sigma.shape[0]):
            cov_k = sigma[k]
            cov_k = stabilize_cov(cov_k, eps=1e-6, cond_max=1e10)
            if not is_spd(cov_k):
                cov_k = make_spd(cov_k, eps=1e-6)
            sigma_central[i, k] = cov_k

    return mu_central.reshape(n_clusters, n_ds, K, d), sigma_central.reshape(n_clusters, n_ds, K, d, d)

def FedGrEM(data, data_val, y_val, A, K, mu_true=None, cov_true=None, T=100, lrate=0.01, epsilon=1e-6):

    """
    Perform one gradient ascent step on GMM parameters (FedGrEM baseline).

    Parameters updated:
    - mu_k: component means
    - Sigma_k: component covariances

    Given local data X and responsibilities gamma, the expected complete
    log-likelihood is:

        Q(theta) = E_gamma[ log p(X, Z | theta) ]

    Gradients per component k:

        ∇_{mu_k} Q = Sigma_k^{-1} sum_i gamma_{ik} (x_i - mu_k)
        ∇_{Sigma_k} Q = 0.5 * Sigma_k^{-1} [sum_i gamma_{ik} (x_i - mu_k)(x_i - mu_k)^T - N_k * Sigma_k]

    where N_k = sum_i gamma_{ik}.

    Covariance inversion uses Cholesky decomposition for stability:

        Sigma_k + eps * I = L * L^T
        Sigma_k^{-1} = (L^T)^{-1} * L^{-1}

    Code snippet:

        L = cholesky(Sigma[k] + eps * np.eye(d), lower=True)
        Z = solve_triangular(L, I, lower=True)
        cov_inv = solve_triangular(L.T, Z, lower=False)
    """

    c, n_ds, N_i, d = data.shape
    N_i_val = data_val.shape[2]
    n = int(c*n_ds)

    # Initialize estimates per node
    pi_est, mu_est, sigma_est = initialize_gmm_params_kmeans(data, K)

    ll_, nmi_, ari_       = np.zeros((T,)), np.zeros((T,)), np.zeros((T,))
    mu_error_, cov_error_ = np.zeros((T,)), np.zeros((T,))
    mu_err_consensus_, cov_err_consensus_ = np.zeros((T,)), np.zeros((T,))

    # Baseline on iter zero
    ll_0, nmi_0, ari_0  = compute_metrics(data_val, y_val, N_i_val, pi_est, mu_est, sigma_est)
    mu_err_0, cov_err_0 = compute_errors(mu_true, cov_true, mu_est, sigma_est)
    mu_err_consensus_0, cov_err_consensus_0, _ = compute_consensus_error(data, A, pi_est, mu_est, sigma_est, epsilon=1e-6)
    
    ll_[0], nmi_[0], ari_[0] = ll_0, nmi_0, ari_0
    mu_error_[0], cov_error_[0] = mu_err_0, cov_err_0
    mu_err_consensus_[0], cov_err_consensus_[0] = mu_err_consensus_0, cov_err_consensus_0

    for t in range(1,T):
        """
        ====================================
         E-step: Compute responsibilities
        ====================================
        """
        gamma = np.zeros((c, n_ds, N_i, K))
        for c_i in range(c):
            for ds in range(n_ds):
                X, pi, mu, sigma = data[c_i, ds], pi_est[c_i, ds], mu_est[c_i, ds], sigma_est[c_i, ds]
                gamma[c_i, ds] = compute_responsibilities(X, pi, mu, sigma)      

        """
        ====================================
         M-step: Update parameters
        ====================================
        """ 
        for c_i in range(c):
            for ds in range(n_ds):
                X, pi, mu, sigma = data[c_i, ds], pi_est[c_i, ds], mu_est[c_i, ds], sigma_est[c_i, ds]
                pi = np.mean(gamma[c_i, ds], axis=0) # update mixture weights

                # avoid component collapse:
                # if any weight is extremely small, reinit that component (mean + cov)
                for k in range(K):
                    if pi[k] <= 1e-4:               # threshold
                        # reinitialize: mean from a random data point, cov from data covariance
                        mu[k] = X[np.random.randint(len(X))]
                        sigma[k] = np.cov(X.T) + 1e-3 * np.eye(d)
                        pi[k] = 1e-3
                # renormalize again
                pi /= pi.sum()
                
                # gradient step
                mu, sigma = grad_step(X, pi, mu, sigma, gamma[c_i, ds], lrate, epsilon)  
                pi_est[c_i, ds] = pi
                mu_est[c_i, ds] = mu
                sigma_est[c_i, ds] = sigma
        
        """
        =========================================
        Central update
        =========================================
        """
        mu_est, sigma_est = central_update(mu_est, sigma_est, N_i)

        #====================Performance metrics on validation set====================#
        ll, nmi, ari = compute_metrics(data_val, y_val, N_i_val, pi_est, mu_est, sigma_est)
        mu_error, cov_error = compute_errors(mu_true, cov_true, mu_est, sigma_est)
        mu_err_consensus, cov_err_consensus, _ = compute_consensus_error(data, A, pi_est, mu_est, sigma_est, epsilon=1e-6)
        ll_[t], nmi_[t], ari_[t]  = ll, nmi, ari 
        mu_error_[t], cov_error_[t] = mu_error, cov_error 
        mu_err_consensus_[t], cov_err_consensus_[t] = mu_err_consensus, cov_err_consensus 
        
    results = {
        'Algorithm': ['FedGrEM']*(T),
        'LogLikelihood': ll_,
        'NMI': nmi_,
        'ARI': ari_,
        'mu err': mu_error_,
        'cov err': cov_error_,
        'mu err consensus': mu_err_consensus_,
        'cov err consensus': cov_err_consensus_,
        'T': np.arange(T)
    }
    
    return pd.DataFrame(results)
    


