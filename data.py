import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split


def generate_data_iid(n_clients=3,
                        n_samples=30,
                        n_samples_val=500,
                        n_features=2,
                        n_clusters=3,
                        seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # --- Step 1: generate full dataset ---
    n_total = n_clients*(n_samples+n_samples_val)

    X, y, means = make_blobs(n_samples=n_total,
                      n_features=n_features,
                      centers=n_clusters,
                      return_centers=True,
                      random_state=seed)
    
    # --- Step 2: split among clients ---
    client_data,  client_labels, client_data_val, client_labels_val = [], [], [], []
    
    # IID split: equal samples per client
    perm = np.random.permutation(n_total)
    X, y = X[perm], y[perm]
    splits = np.array_split(np.arange(n_total), n_clients)
    for idx in splits:
        X_i, y_i = X[idx], y[idx]
        X_train, X_val, y_train, y_val = train_test_split(X_i, y_i, test_size=n_samples_val, stratify=y_i)
        client_data.append(X_train)
        client_data_val.append(X_val)
        client_labels.append(y_train)
        client_labels_val.append(y_val)
        
    return np.array(client_data), np.array(client_data_val), np.array(client_labels), np.array(client_labels_val), means
    
def generate_data_dirichlet(n_clients=3,
                            n_samples=30,
                            n_samples_val=500,
                            n_features=2,
                            n_clusters=3,
                            seed=0,
                            alpha=0.5):

    N_i = n_samples+n_samples_val

    client_data,  client_labels, client_data_val, client_labels_val = [], [], [], []
    means_list = []
    for c in range(n_clients):
        np.random.seed(seed+c)
        props = np.random.dirichlet(alpha*np.ones(n_clusters))
        samples_per_class = np.round(N_i * props).astype(int)

        diff = N_i - np.sum(samples_per_class)
        for i in range(abs(diff)):
            samples_per_class[i % n_clusters] += np.sign(diff) 

        assert sum(samples_per_class) == N_i
        try:
            X, y, means = make_blobs(n_samples=samples_per_class,
                    n_features=n_features,
                    centers=None,
                    return_centers=True,
                    random_state=seed)   
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=n_samples_val, stratify=y)
        except ValueError:
            print("ValueError")
            for k in range(n_clusters):
                if len(y[y==k]) == 1:
                    idx = np.where(y!=k)[0]
                    X_new = X[idx] 
                    y_new = y[idx]
                    X_train, X_val, y_train, y_val = train_test_split(X_new, y_new, test_size=n_samples_val-1, stratify=y_new)
               
        client_data.append(X_train)
        client_data_val.append(X_val[:n_samples_val-1])
        client_labels.append(y_train)
        client_labels_val.append(y_val[:n_samples_val-1])
        means_list.append(means)
    return np.array(client_data), np.array(client_data_val), np.array(client_labels), np.array(client_labels_val), means_list

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

def generate_data_mean_shift(n_clients=3,
                            n_samples=30,
                            n_samples_val=500,
                            n_features=2,
                            n_clusters=3,
                            seed=None,
                            shift_scale=0.5):
    if seed is not None:
        np.random.seed(seed)
    
    n_total = n_clients*(n_samples+n_samples_val)
    # Global dataset
    X, y, means = make_blobs(n_samples=n_total,
                      n_features=n_features,
                      centers=n_clusters,
                      return_centers=True,
                      random_state=seed)
    
    # Datasets with mean shift
    client_data,  client_labels, client_data_val, client_labels_val = [], [], [], []
    shifts = shift_scale*np.random.randn(n_clients, n_clusters, n_features)
    for c in range(n_clients):
        Xi, yi = make_blobs(
            n_samples=n_samples+n_samples_val,
            n_features=n_features,
            centers=means+shifts[c],
            random_state=seed+c)

        X_train, X_val, y_train, y_val = train_test_split(Xi, yi, test_size=n_samples_val, stratify=yi)
        client_data.append(X_train)
        client_data_val.append(X_val)
        client_labels.append(y_train)
        client_labels_val.append(y_val)
        
    return np.array(client_data), np.array(client_data_val), np.array(client_labels), np.array(client_labels_val), means

