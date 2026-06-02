def fedAvg_kmeans(partition_store, T=10, seed=0):
    """
    Federated KMeans with FedAvg-style aggregation via Hungarian matching.
    Each round: local KMeans -> Hungarian alignment -> global averaging.
    """
    params_store = {}

    for dataset_name, (node_data, _, _, K) in partition_store.items():
        num_nodes = len(node_data)
        D = node_data[0].shape[1]
        
        # store cluster centers per node
        node_params = np.zeros((num_nodes, K, D))
        
        for t in range(T):
            # --- Local KMeans update ---
            for i in range(num_nodes):
                km = KMeans(n_clusters=K, init='k-means++', n_init=10,
                            random_state=seed+t).fit(node_data[i])
                node_params[i] = km.cluster_centers_
            
            # --- Alignment step (pick reference = node 0) ---
            mu_ref = node_params[0]
            aligned_params = []
            for j in range(num_nodes):
                mu_aligned, _ = match_clusters(mu_ref, node_params[j])
                aligned_params.append(mu_aligned)
            aligned_params = np.stack(aligned_params)  # (num_nodes, K, D)
            
            # --- Averaging step (all nodes identical after averaging) ---
            global_avg = np.mean(aligned_params, axis=0)  # (K, D)
            node_params[:] = global_avg[None, :, :]       # broadcast to all nodes
            
        params_store[dataset_name] = [{'mu': node} for node in node_params]

    return params_store
