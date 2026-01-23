import numpy as np
from sklearn.datasets import make_blobs

def generate_federated_data(n_clients=3,
                            n_samples=30,
                            n_features=2,
                            n_clusters=3,
                            seed=None,
                            non_iid=False,
                            alpha=0.5):
    """
    Generate federated dataset, split across clients, with local train/val sets.
    
    Parameters
    ----------
    n_clients : int
        Number of federated clients.
    n_samples : int or list of ints
        Total samples. If int, evenly split among clusters. If list, use as cluster sizes.
    n_features : int
        Number of features.
    n_clusters : int
        Number of Gaussian clusters (GMM components).
    seed : int or None
        Random seed.
    non_iid : bool
        If True, generate label-skewed client data using Dirichlet distribution.
    alpha : float
        Dirichlet concentration parameter for non-IID splits. Smaller alpha = more skewed.
    
    Returns
    -------
    client_train_data : np.ndarray (n_clients, n_samples/n_clients, n_features)
    client_train_labels : np.ndarray (n_clients, n_samples/n_clients)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # --- Step 1: generate full dataset ---
    X, y, means = make_blobs(n_samples=n_samples,
                      n_features=n_features,
                      centers=n_clusters,
                      return_centers=True,
                      random_state=seed)
    
    # --- Step 2: split among clients ---
    client_data, client_labels = [], []
    
    if not non_iid:
        # IID split: equal samples per client
        n_total = X.shape[0]
        perm = np.random.permutation(n_total)
        X, y = X[perm], y[perm]
        splits = np.array_split(np.arange(n_total), n_clients)
        for idx in splits:
            client_data.append(X[idx])
            client_labels.append(y[idx])
    else:
        # Non-IID split using Dirichlet distribution over labels
        label_indices = [np.where(y == k)[0] for k in range(n_clusters)]
        client_data = [[] for _ in range(n_clients)]
        client_labels = [[] for _ in range(n_clients)]
        
        for k, indices in enumerate(label_indices):
            n_k = len(indices)
            proportions = np.random.dirichlet(alpha*np.ones(n_clients))
            proportions = np.round(proportions * n_k).astype(int)
            # Adjust rounding errors
            diff = n_k - np.sum(proportions)
            for i in range(abs(diff)):
                proportions[i % n_clients] += np.sign(diff)
            
            start = 0
            for i, p in enumerate(proportions):
                idx = indices[start:start+p]
                client_data[i].append(X[idx])
                client_labels[i].append(y[idx])
                start += p
        
        # concatenate each client's chunks
        client_data = [np.vstack(chunks) for chunks in client_data]
        client_labels = [np.hstack(chunks) for chunks in client_labels]
    
    return np.array(client_data), np.array(client_labels), means

def er_adjacency_matrix(n_nodes, p, seed=None, directed=False):
    """
    Generate a random Erdős-Rényi adjacency matrix.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    p : float
        Probability of edge creation (0 <= p <= 1).
    seed : int or None
        Random seed for reproducibility.
    directed : bool
        If True, generate a directed graph; otherwise undirected.

    Returns
    -------
    A : np.ndarray
        n_nodes x n_nodes adjacency matrix (0/1 entries, zeros on diagonal).
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Start with zeros
    A = np.zeros((n_nodes, n_nodes), dtype=int)
    
    # Fill upper triangle for undirected graph
    if not directed:
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if np.random.rand() < p:
                    A[i, j] = 1
                    A[j, i] = 1  # symmetric
    else:
        # directed: all off-diagonal entries independent
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and np.random.rand() < p:
                    A[i, j] = 1
    
    # Row-normalize
    row_sums = A.sum(axis=1, keepdims=True)
    # Avoid division by zero for isolated nodes
    row_sums[row_sums == 0] = 1
    A_norm = A / row_sums
    
    return A_norm
