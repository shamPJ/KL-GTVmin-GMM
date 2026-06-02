
import torch
from sklearn.mixture import GaussianMixture
from utils.utils import to_numpy

def centralized(
    X: torch.Tensor,
    K: int,
    seed: int = 0,
):
    """
    Centralized (non-FL) GMM baseline.
    Pools all local datasets and fits a single GMM.
    """
    # Pool all data
    n_clients, N, D = X.shape
    X_np = to_numpy(X)
    X_all = X_np.reshape(-1,D)

    gmm = GaussianMixture(
        n_components=K,
        covariance_type="diag",
        random_state=seed,
        max_iter=20,
    )
    gmm.fit(X_all)

    # Average log-likelihood per node (for fair comparison with FL version)
    # gmm.score(X) computes per-sample average log-likelihood of the given data X
    ll = {}
    for i in range(n_clients):
        ll[i] = gmm.score(X_np[i])  
    avg_ll = sum(ll.values()) / len(ll) # across nodes average

    return gmm, ll, avg_ll
    
def run(args, data):
    gmm, ll, avg_ll = centralized(
        data["X"], K=args.K, seed=args.seed
    )

    return {"ll": avg_ll, "models": {0: gmm}}