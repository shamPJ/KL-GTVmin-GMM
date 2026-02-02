import numpy as np
from sklearn.datasets import make_blobs
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

def generate_data_var(n_clients=3,
                    n_samples=30,
                    n_samples_val=500,
                    n_features=4,
                    n_clusters=3,
                    seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    n_total = n_clients*(n_samples+n_samples_val)
    # Global dataset
    X, y, means = make_blobs(n_samples=n_total,
                            n_features=n_features,
                            centers=n_clusters,
                            return_centers=True,
                            random_state=seed)
    
    # Datasets with varying variance
    client_data,  client_labels, client_data_val, client_labels_val = [], [], [], []
    # shifts = shift_scale*np.random.randn(n_clients, n_clusters, n_features)
    vars = np.random.uniform(0.01, 10, (n_clients, n_clusters))
    for c in range(n_clients):
        Xi, yi = make_blobs(
            n_samples=n_samples+n_samples_val,
            n_features=n_features,
            centers=means,
            cluster_std = vars[c],
            random_state=seed+c)

        X_train, X_val, y_train, y_val = train_test_split(Xi, yi, test_size=n_samples_val, stratify=yi)
        client_data.append(X_train)
        client_data_val.append(X_val)
        client_labels.append(y_train)
        client_labels_val.append(y_val)
        
    return np.array(client_data), np.array(client_data_val), np.array(client_labels), np.array(client_labels_val), means

def generate_data_het(n_clients=3,
                        n_samples=30,
                        n_samples_val=500,
                        n_features=4,
                        n_clusters=3,
                        seed=None,
                        heterogeneity=0.5):
    if seed is not None:
        np.random.seed(seed)
    
    n_groups = max(1, int(n_clients * heterogeneity))

    means_list = []
    client_data,  client_labels, client_data_val, client_labels_val = [], [], [], []
    group_ids = np.random.choice(n_groups, size=n_clients, replace=True)
    
    for c in range(n_clients):
        g = group_ids[c]
        X, y, means = make_blobs(n_samples=n_samples+n_samples_val,
                                n_features=n_features,
                                centers=n_clusters,
                                return_centers=True,
                                random_state=seed+g)
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=n_samples_val, stratify=y)
        client_data.append(X_train)
        client_data_val.append(X_val)
        client_labels.append(y_train)
        client_labels_val.append(y_val)
    
        means_list.append(means)
    
    return np.array(client_data), np.array(client_data_val), np.array(client_labels), np.array(client_labels_val), means_list

def load_raw_mnist():
    transform = transforms.ToTensor()
    mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    X = mnist.data.numpy().reshape(-1, 28*28) / 255.0
    y = mnist.targets.numpy()
    return X, y
# def generate_data_mnist(n_clients=3,
#                         n_samples=30,
#                         n_samples_val=500,
#                         n_features=2,
#                         n_clusters=3,
#                         seed=0,
#                         alpha=0.5):
        
    # X_raw, y = load_raw_mnist() # shapes (60000, 784) and (60000,)

    # # ----------------------------------------------------------------------
    # # Split global dataset: reserve 10k stratified for dimensionality reduction
    # # ----------------------------------------------------------------------
    # X_dimred, X_split, y_dimred, y_split = train_test_split(
    #     X_raw, y, train_size=10000, stratify=y)

    # # ----------------------------------------------------------------------
    # # Prepare class-wise index groups from the remaining data (50k)
    # # ----------------------------------------------------------------------
    # if cluster_map is None:
    #     # default 3 clusters (1-3, 4-6, 7-9)
    #     cluster_map = {0: [1,2,3], 1: [4,5,6], 2: [7,8,9]}

    # n_clusters = len(cluster_map)
    
    # # Group indices by class
    # indices_by_class = {k: np.where(y_split == k)[0] for k in range(10)}
    # # Shuffle samples within the class
    # for idxs in indices_by_class.values():
    #     rng.shuffle(idxs)
        
    # counter = 0
    # valid_split = False
    # while (counter < 20) and (valid_split == False):
    #     client_indices = [[] for _ in range(n_clusters*n_ds)] # create empty lists
    #     label_proportions = np.zeros((n_clusters*n_ds, 10)) # proportions of ALL classes for each local node 

    #     for cluster_id, digits in cluster_map.items():
    #         cluster_clients = range(cluster_id * n_ds,
    #                             (cluster_id + 1) * n_ds)
    #         # only distribute samples from the allowed digits to these clients
    #         for k in digits:
    #             idxs = indices_by_class[k] # get indices for class `k`
    #             # create list of len(n_ds) filled with val alpha and use it
    #             # to get simplex for class `k` over n_ds datasets, e.g. array([0.003, 0.812, 0.002, 0.176, 0.007])
    #             proportions = rng.dirichlet([alpha] * len(cluster_clients))
    #             """
    #             - multiply proportions by the total number of samples in this class.
    #             This tells roughly where to cut the sorted indices for digit k:
    #             e.g. [0.2, 0.5, 0.3] --> [0.2, 0.7, 1. ]* 90 -> [ 18.0,  63.0, 90.0]
    #             - convert to int and drop last ellement -> [18, 63]
    #             - split_idxs = np.split(idxs, [18, 63]). This results in 3 chunks:
    #             idxs[0:18] -> 18 samples to client 0
    #             idxs[18:63] -> 45 samples to client 1
    #             idxs[63:90] -> 27 samples to client 2
    #             That matches the Dirichlet proportions: [0.2, 0.5, 0.3]
    #             """
    #             proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
    #             split_idxs = np.split(idxs, proportions)
    #             for client_id, idx_set in zip(cluster_clients, split_idxs):
    #                 client_indices[client_id].extend(idx_set)
    #                 label_proportions[client_id, k] = len(idx_set)
            
    #     # check if there are enough datapoints
    #     valid_split = all(len(idxs) >= N_i + N_i_val for idxs in client_indices)
    #     counter+=1
    # if not valid_split: raise ValueError("Failed to generate valid client splits after 20 attempts.")
    # # normalize to get class distribution per client
    # label_proportions = label_proportions / label_proportions.sum(axis=1, keepdims=True)

    # # ----------------------------------------------------------------------
    # # Fit dimensionality reducer on the reserved 10k subset
    # # ----------------------------------------------------------------------
    # if dimred == 'pca':
    #     reducer = PCA(n_components=d)
    #     reducer.fit(X_dimred)
    # elif dimred == 'umap':
    #     reducer = umap.UMAP(n_components=d)
    #     reducer.fit(X_dimred)
    # else:
    #     raise ValueError("dimred must be 'umap' or 'pca'.")
        
    # # ----------------------------------------------------------------------
    # # Generate per-client datasets
    # # ----------------------------------------------------------------------
    # data_train = np.zeros((1, n_clusters*n_ds, N_i, d))
    # data_val   = np.zeros((1, n_clusters*n_ds, N_i_val, d))
    # labels_val = np.zeros((1, n_clusters*n_ds, N_i_val), dtype=int)

    # for i, idxs in enumerate(client_indices):
    #     X_i, y_i = X_split[idxs], y_split[idxs]
    #     X_train, X_val, y_train, y_val = split_client_data(X_i, y_i, N_i, N_i_val, rng)

    #     data_train[0, i] = reducer.transform(X_train)
    #     data_val[0, i]   = reducer.transform(X_val)
        
    #     labels_val[0, i] = y_val

    # return data_train.reshape(n_clusters, n_ds, N_i, d), data_val.reshape(n_clusters, n_ds, N_i_val, d), \
    #                         labels_val.reshape(n_clusters, n_ds, N_i_val), label_proportions.reshape(n_clusters, n_ds, 10), reducer, X_dimred, y_dimred
