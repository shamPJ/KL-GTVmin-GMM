import numpy as np
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
import umap

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

def generate_data_iid(n_clients=3,
                        n_samples=30,
                        n_samples_val=500,
                        n_features=2,
                        n_clusters=3,
                        seed=None):
    """
    Returns IID data for n_clients, each with n_samples and n_samples_val samples for training and validation.
    Data is generated from a global mixture of n_clusters Gaussians, and then split IID
    Shapes: 
        - client_data: (n_clients, n_samples, n_features)
        - client_data_val: (n_clients, n_samples_val, n_features)
        - client_labels: (n_clients, n_samples)
        - client_labels_val: (n_clients, n_samples_val)
        - means: (n_clusters, n_features)
    """
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

def generate_data_mean_shift(n_clients=10,
                            n_samples=50,
                            n_samples_val=500,
                            n_features=2,
                            n_clusters=3,
                            shift_scale=0.5,
                            var_scale=1.0,
                            cov_type='diag',
                            seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    # Global means
    means = np.random.randn(n_clusters, n_features) * 5
    
    client_data, client_labels = [], []
    client_data_val, client_labels_val = [], []
    
    # Mean heterogeneity
    shifts = shift_scale * np.random.randn(n_clients, n_clusters, n_features)
    means_clients = means[None, :, :] + shifts

    covs_clients = []
    
    for c in range(n_clients):
        covs = []
        for k in range(n_clusters):
            
            if cov_type == 'diag':
                # diag = var_scale * np.abs(1 + 0.5*np.random.randn(n_features))
                # cov_k = np.diag(diag)

                if np.random.rand() < 0.5:
                    # sharp cluster
                    diag = var_scale * np.random.uniform(0.05, 0.5, size=n_features)
                else:
                    # wide cluster
                    diag = var_scale * np.random.uniform(2, 5, size=n_features)
                cov_k = np.diag(diag)
            
            elif cov_type == 'full':
                A = np.random.randn(n_features, n_features)
                cov_k = A @ A.T
                cov_k *= var_scale
            
            covs.append(cov_k)
        
        covs_clients.append(covs)

        # ---- sampling ----
        X_list, y_list = [], []
        n_total = n_samples + n_samples_val
        per_cluster = n_total // n_clusters

        for k in range(n_clusters):
            Xk = np.random.multivariate_normal(
                mean=means_clients[c, k],
                cov=covs[k],
                size=per_cluster
            )
            yk = np.full(per_cluster, k)
            
            X_list.append(Xk)
            y_list.append(yk)

        Xi = np.vstack(X_list)
        yi = np.hstack(y_list)

        X_train, X_val, y_train, y_val = train_test_split(
            Xi, yi,
            test_size=n_samples_val,
            stratify=yi
        )

        client_data.append(X_train)
        client_data_val.append(X_val)
        client_labels.append(y_train)
        client_labels_val.append(y_val)

    return (np.array(client_data),
            np.array(client_data_val),
            np.array(client_labels),
            np.array(client_labels_val),
            means_clients,
            np.array(covs_clients))

def generate_cluster_dropout(
                        n_clients=10,
                        n_clusters=6,
                        keep_per_client=2,
                        n_samples=100,
                        n_samples_val=500,
                        n_features=2,
                        std=0.5,
                        seed=0,
    ):
    rng = np.random.default_rng(seed)
    samples_per_cluster = int(n_samples / keep_per_client)

    means = rng.normal(0, 5, size=(n_clusters, n_features))
    X_val, y_val = make_blobs(n_samples=n_clients*n_samples_val, n_features=n_features, centers=means, cluster_std=std, random_state=seed)
    client_data_val = [
        X_val[c*n_samples_val: c*n_samples_val + n_samples_val] 
        for c in range(n_clients)
    ]
    client_labels_val = [
        y_val[c*n_samples_val: c*n_samples_val + n_samples_val] 
        for c in range(n_clients)
    ]

    client_data,  client_labels = [], []

    for i in range(n_clients):
        keep = rng.choice(n_clusters, keep_per_client, replace=False)
        Xi, yi = [], []

        for k in keep:
            Xi.append(
                rng.normal(means[k], std, size=(samples_per_cluster, n_features)) 
            )
            yi.append(np.full(samples_per_cluster, k))

        client_data.append(np.vstack(Xi))
        client_labels.append(np.concatenate(yi))

    return (np.array(client_data), 
            np.array(client_data_val), 
            np.array(client_labels), 
            np.array(client_labels_val), 
            means
    )

# def generate_data_var(n_clients=3,
#                     n_samples=30,
#                     n_samples_val=500,
#                     n_features=4,
#                     n_clusters=3,
#                     seed=None):
#     if seed is not None:
#         np.random.seed(seed)
    
#     n_total = n_clients*(n_samples+n_samples_val)
#     # Global dataset
#     X, y, means = make_blobs(n_samples=n_total,
#                             n_features=n_features,
#                             centers=n_clusters,
#                             return_centers=True,
#                             random_state=seed)
    
#     # Datasets with varying variance
#     client_data,  client_labels, client_data_val, client_labels_val = [], [], [], []
#     # shifts = shift_scale*np.random.randn(n_clients, n_clusters, n_features)
#     vars = np.random.uniform(0.01, 10, (n_clients, n_clusters))
#     for c in range(n_clients):
#         Xi, yi = make_blobs(
#             n_samples=n_samples+n_samples_val,
#             n_features=n_features,
#             centers=means,
#             cluster_std = vars[c],
#             random_state=seed+c)

#         X_train, X_val, y_train, y_val = train_test_split(Xi, yi, test_size=n_samples_val, stratify=yi)
#         client_data.append(X_train)
#         client_data_val.append(X_val)
#         client_labels.append(y_train)
#         client_labels_val.append(y_val)
        
#     return np.array(client_data), np.array(client_data_val), np.array(client_labels), np.array(client_labels_val), means

# def generate_data_het(n_clients=3,
#                     n_samples=30,
#                     n_samples_val=500,
#                     n_features=4,
#                     n_clusters=3,
#                     seed=None,
#                     heterogeneity=0.5):
#     if seed is not None:
#         np.random.seed(seed)
    
#     n_groups = max(1, int(n_clients * heterogeneity))

#     means_list = []
#     client_data,  client_labels, client_data_val, client_labels_val = [], [], [], []
#     group_ids = np.random.choice(n_groups, size=n_clients, replace=True)
    
#     for c in range(n_clients):
#         g = group_ids[c]
#         X, y, means = make_blobs(n_samples=n_samples+n_samples_val,
#                                 n_features=n_features,
#                                 centers=n_clusters,
#                                 return_centers=True,
#                                 random_state=seed+g)
        
#         X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=n_samples_val, stratify=y)
#         client_data.append(X_train)
#         client_data_val.append(X_val)
#         client_labels.append(y_train)
#         client_labels_val.append(y_val)
    
#         means_list.append(means)
    
#     return np.array(client_data), np.array(client_data_val), np.array(client_labels), np.array(client_labels_val), means_list

def load_raw_mnist():
    mnist = datasets.MNIST(root="./data", train=True, download=True)
    X = mnist.data.numpy().reshape(-1, 28*28) / 255.0
    y = mnist.targets.numpy()
    return X, y


def split_client_data(X_i, y_i, n_samples, n_samples_val, rng):
    """
    Split per-client data preserving class proportions approximately,
    but limit total samples to n_samples + n_samples_val.
    """
    N = n_samples + n_samples_val

    # Global index pool
    all_idxs = np.arange(len(X_i))

    X_train, X_val, y_train, y_val = [], [], [], []
    
    # Get class proportions for client
    classes, counts = np.unique(y_i, return_counts=True) # e.g. (array([0, 1, 2, 3, 4]), array([3, 3, 2, 1, 1]))
    proportions = counts / counts.sum()
    
    # Allocate number of samples per class based on proportions
    n_per_class = (proportions * N).astype(int)
    # Fix rounding mismatch
    diff = N - n_per_class.sum()
    if diff != 0:
        adjust_classes = rng.choice(len(n_per_class), size=abs(diff), replace=True)
        for i in adjust_classes:
            n_per_class[i] += np.sign(diff)
    # remove negative sample count
    n_per_class = np.clip(n_per_class, 0, None)
    used_idxs = set()

    # Sample per class
    for c, n_c_target in zip(classes, n_per_class):
        idxs_c = np.where(y_i == c)[0]
        rng.shuffle(idxs_c)
        idxs_c = idxs_c[:n_c_target]

        used_idxs.update(idxs_c.tolist())

        n_train_c = int(len(idxs_c) * n_samples / N)

        X_train.append(X_i[idxs_c[:n_train_c]])
        y_train.append(y_i[idxs_c[:n_train_c]])

        X_val.append(X_i[idxs_c[n_train_c:]])
        y_val.append(y_i[idxs_c[n_train_c:]])
    # account for zero sample class with np.empty
    X_train = np.concatenate(X_train) if X_train else np.empty((0, X_i.shape[1]))
    y_train = np.concatenate(y_train) if y_train else np.empty((0,), dtype=y_i.dtype)

    X_val = np.concatenate(X_val) if X_val else np.empty((0, X_i.shape[1]))
    y_val = np.concatenate(y_val) if y_val else np.empty((0,), dtype=y_i.dtype)

    # Remaining unused indices
    remaining_idxs = np.array(list(set(all_idxs) - used_idxs))
    rng.shuffle(remaining_idxs)

    # Top up train set
    if len(X_train) < n_samples:
        needed = n_samples - len(X_train)
        extra = remaining_idxs[:needed]
        remaining_idxs = remaining_idxs[needed:]

        X_train = np.concatenate([X_train, X_i[extra]])
        y_train = np.concatenate([y_train, y_i[extra]])

    # Top up validation set
    if len(X_val) < n_samples_val:
        needed = n_samples_val - len(X_val)
        extra = remaining_idxs[:needed]

        X_val = np.concatenate([X_val, X_i[extra]])
        y_val = np.concatenate([y_val, y_i[extra]])

    return X_train[:n_samples], X_val[:n_samples_val], y_train[:n_samples], y_val[:n_samples_val]

def generate_data_mnist(n_clients=10,
                        n_clusters=6,
                        n_samples=100,
                        n_samples_val=500,
                        n_features=2,
                        seed=0,
                        alpha=0.5,
                        dimred='umap'):
        
    rng = np.random.default_rng(seed=seed)
    X_raw, y = load_raw_mnist() # shapes (60000, 784) and (60000,)

    # ----------------------------------------------------------------------
    # Split global dataset: reserve 10k stratified for dimensionality reduction
    # ----------------------------------------------------------------------
    X_dimred, X_split, y_dimred, y_split = train_test_split(
        X_raw, y, train_size=10000, stratify=y)

    # ----------------------------------------------------------------------
    # Prepare class-wise index groups from the remaining data (50k)
    # ----------------------------------------------------------------------

    # Group indices by class
    indices_by_class = {k: np.where(y_split == k)[0] for k in range(10)}
    # Shuffle samples within the class
    for idxs in indices_by_class.values():
        rng.shuffle(idxs)
        
    counter = 0
    valid_split = False
    while (counter < 20) and (valid_split == False):
        client_indices = [[] for _ in range((n_clients))] # create empty lists
        label_proportions = np.zeros((n_clients, 10)) # proportions of ALL classes for each local node 

        for k in range(n_clusters):
            idxs = indices_by_class[k] # get indices for class `k`
            # create list of len(n_clients) filled with val alpha and use it
            # to get simplex for class `k` over n_ds datasets, e.g. array([0.003, 0.812, 0.002, 0.176, 0.007])
            proportions = rng.dirichlet([alpha] * n_clients)
            """
            - multiply proportions by the total number of samples in this class.
            This tells roughly where to cut the sorted indices for digit k:
            e.g. [0.2, 0.5, 0.3] --> [0.2, 0.7, 1. ]* 90 -> [ 18.0,  63.0, 90.0]
            - convert to int and drop last ellement -> [18, 63]
            - split_idxs = np.split(idxs, [18, 63]). This results in 3 chunks:
            idxs[0:18] -> 18 samples to client 0
            idxs[18:63] -> 45 samples to client 1
            idxs[63:90] -> 27 samples to client 2
            That matches the Dirichlet proportions: [0.2, 0.5, 0.3]
            """
            proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
            split_idxs = np.split(idxs, proportions)
            for client_id, idx_set in enumerate(split_idxs):
                client_indices[client_id].extend(idx_set)
                label_proportions[client_id, k] = len(idx_set)
            
        # check if there are enough datapoints
        valid_split = all(len(idxs) >= n_samples + n_samples_val for idxs in client_indices)
        counter+=1
    if not valid_split: raise ValueError("Failed to generate valid client splits after 20 attempts.")
    # normalize to get class distribution per client
    label_proportions = label_proportions / label_proportions.sum(axis=1, keepdims=True)

    # ----------------------------------------------------------------------
    # Fit dimensionality reducer on the reserved 10k subset
    # ----------------------------------------------------------------------
    if dimred == 'pca':
        reducer = PCA(n_components=n_features)
        reducer.fit(X_dimred)
    elif dimred == 'umap':
        reducer = umap.UMAP(n_components=n_features, random_state=seed)
        reducer.fit(X_dimred)
    else:
        raise ValueError("dimred must be 'umap' or 'pca'.")
        
    # ----------------------------------------------------------------------
    # Generate per-client datasets
    # ----------------------------------------------------------------------
    client_data, client_data_val, = [], []
    client_labels, client_labels_val = [], []

    for i, idxs in enumerate(client_indices):
        X_i, y_i = X_split[idxs], y_split[idxs]
        X_train, X_val, y_train, y_val = split_client_data(X_i, y_i, n_samples, n_samples_val, rng)

        client_data.append(reducer.transform(X_train))
        client_data_val.append(reducer.transform(X_val))
        
        client_labels.append(y_train)
        client_labels_val.append(y_val)

    return np.array(client_data), np.array(client_data_val), np.array(client_labels), np.array(client_labels_val), label_proportions
