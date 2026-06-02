

def compute_log_likelihood(X, pi, mu, sigma, verbose=False):
    """
    Compute the average log-likelihood of the data under a Gaussian Mixture Model (GMM).

    Parameters
    ----------
    X : ndarray of shape (N_i, d)
        Data matrix where `N_i` is the number of samples and `d` is the dimensionality.

    pi : ndarray of shape (K,)
        Mixing coefficients (prior probabilities) for each of the `K` Gaussian components.

    mu : ndarray of shape (K, d)
        Mean vectors for each Gaussian component.

    sigma : ndarray of shape (K, d, d)
        Covariance matrices for each Gaussian component.

    verbose : bool, optional
        If True, prints diagnostic messages about numerical issues.

    Returns
    -------
    log_likelihood : float
        The average log-likelihood of the dataset under the current GMM parameters.
    """
    N_i, K = X.shape[0], pi.shape[0]

    # Precompute likelihoods for each component
    pdfs = np.zeros((N_i, K))
    for k in range(K):
        try:
            cov = sigma[k]
            cov = (cov + cov.T) / 2           # enforce symmetry
            cov += 1e-6 * np.eye(cov.shape[0])  # regularization

            if not np.all(np.isfinite(mu[k])) or not np.all(np.isfinite(cov)):
                raise ValueError("Non-finite parameters in log-likelihood")

            pdfs[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mu[k], cov=cov, allow_singular=True).astype(float)

        except Exception as e:
            if verbose:
                print(f"[Warning] Skipping component {k} due to error: {e}")
            pdfs[:, k] = 1e-8  # tiny fallback to avoid zero likelihood

    # Weighted mixture probability for each point
    probs = pdfs.sum(axis=1)

    log_likelihood = np.mean(np.log(probs + 1e-12))  # avoid log(0)
    return log_likelihood


def fed_gmm_grad(partition_store, A, T=10, alpha=0.1, lr=0.01, seed=0):
    params_store = {}
    ll = {}

    for dataset_name, (node_data, _, _, K) in partition_store.items():
        num_nodes = len(node_data)
        G = nx.from_numpy_array(A)
        D = node_data[0].shape[1]

        # Initialize cluster centers using KMeans++
        mu_init = kmeans_initialize(node_data, K, random_state=seed)

        # Initialize parameters
        node_params = []
        for i in range(num_nodes):
            mu = mu_init[i]
            Sigma = np.array([np.eye(D) for _ in range(K)])
            pi = np.ones(K) / K
            node_params.append({'mu': mu, 'Sigma': Sigma, 'pi': pi})

        # Federated rounds
        for t in range(T):
            new_node_params = []
            for i in range(num_nodes):
                # ===== (1) Local E-step ====
                Xi = node_data[i]
                mu, Sigma, pi = node_params[i]['mu'], node_params[i]['Sigma'], node_params[i]['pi']
                N_i = Xi.shape[0]
                gamma = np.zeros((N_i, K))
                for k in range(K):
                    try:
                        # Regularize covariance matrix slightly
                        cov = Sigma[k]
                        cov = (cov + cov.T) / 2  # symmetrize
                        cov += 1e-6 * np.eye(cov.shape[0])  # regularize
                        gamma[:, k] = pi[k] * multivariate_normal.pdf(Xi, mean=mu[k], cov=cov, allow_singular=True)
                    except:
                        gamma[:, k] = 1e-12
                gamma /= gamma.sum(axis=1, keepdims=True) 

                # ===== (2) Gradients =====
                # Mixture weights
                # shapes (N_i, K), (K)
                grad_pi = -np.sum(gamma, axis=0) / (pi+ 1e-12)

                # Means
                grad_mu = np.zeros_like(mu) # shape (K,d)
                for k in range(K):
                    diff = Xi - mu[k].reshape(1,-1) # shape (Ni,d)
                    # (Ni,d)x(d,d) --> Ni,d
                    cov_inv = diff @ np.linalg.inv(Sigma[k])
                    grad_mu[k] = -np.sum(gamma[:, k][:, None] * cov_inv, axis=0)
                    
                # Covariances
                grad_Sigma = np.zeros_like(Sigma)
                for k in range(K):
                    diff = Xi - mu[k].reshape(1, -1)  # (Ni,d)
                    cov_inv = np.linalg.inv(Sigma[k])
                    for n in range(N_i):
                        outer = np.outer(diff[n], diff[n])
                        grad_Sigma[k] += 0.5 * gamma[n, k] * (cov_inv - cov_inv @ outer @ cov_inv)
                
                # ===== (3) Graph penalty =====
                neighbors = list(G.neighbors(i))
                if len(neighbors) > 0:
                    norm_factor = 1.0 / len(neighbors)
                    for j in neighbors:
                        mu_j, Sigma_j, pi_j = (
                            node_params[j]['mu'],
                            node_params[j]['Sigma'],
                            node_params[j]['pi'],
                        )
                        # print(mu, mu_j)
                        mu_aligned, col_ind = match_clusters(mu, mu_j)
                        pi_aligned = pi_j[col_ind]
                        Sigma_aligned = Sigma_j[col_ind]
                        
                        grad_pi += alpha * norm_factor * (pi - pi_aligned)
                        grad_mu += alpha * norm_factor * (mu - mu_aligned)
                        grad_Sigma += alpha * norm_factor * (Sigma - Sigma_aligned)

                # ===== (4) Gradient step =====
                pi_new = pi - lr * grad_pi
                pi_new = np.maximum(pi_new, 1e-8)
                pi_new /= pi_new.sum()

                mu_new = mu - lr * grad_mu
                Sigma_new = Sigma - lr * grad_Sigma
                # Clip eigenvalues
                for k in range(K):
                    eigvals, eigvecs = np.linalg.eigh(Sigma_new[k])
                    eigvals = np.clip(eigvals, 1e-6, None)
                    Sigma_new[k] = eigvecs @ np.diag(eigvals) @ eigvecs.T

                new_node_params.append({'mu': mu_new, 'Sigma': Sigma_new, 'pi': pi_new})

            node_params = new_node_params

        ll_fed = 0
        for i in range(num_nodes):
            params = node_params[i]
            ll_fed += compute_log_likelihood(node_data[i], params['pi'], params['mu'], params['Sigma'])
        
        params_store[dataset_name] = node_params
        ll[dataset_name] = ll_fed / num_nodes
    return params_store, ll