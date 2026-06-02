import torch
import numpy as np
from sklearn.mixture import GaussianMixture
from utils.utils import to_numpy

def local(
    X: torch.Tensor,
    K: int,
    seed: int = 0,
):
    """
    Local-only baseline: one independent GMM per client.
    Returns fitted models + extracted parameters + likelihoods.
    """
    models = {}
    ll = {}
    pred_means = {}

    n_clients, N, D = X.shape
    X_np = to_numpy(X)

    for i in range(n_clients):
        gmm = GaussianMixture(
            n_components=K,
            covariance_type="diag",
            random_state=seed,
            max_iter=20,
        )
        gmm.fit(X_np[i])

        models[i] = gmm
        ll[i] = gmm.score(X_np[i])

        # ---- predicted parameters (important addition)
        pred_means[i] = gmm.means_.copy()

    avg_ll = float(np.mean(list(ll.values())))

    return {
        "models": models,
        "ll_per_client": ll,
        "ll": avg_ll,
        "pred_means": pred_means,
    }

def run(args, data):
    out = local(
        data["X"],
        K=args.K,
        seed=args.seed,
    )

    return {
        "algo": "local",
        "ll": out["ll"],
        "ll_per_client": out["ll_per_client"],
        "pred_means": out["pred_means"],
    }