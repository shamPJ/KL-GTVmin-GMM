import json
import os
import numpy as np
import torch

from data.data import generate_data_mean_shift, er_adjacency_matrix
from utils.parser import parse_args
from utils.metrics import est_error
from utils.utils import save_metrics, filter_args
from algos.registry import get_algo, ALGO_ARG_MAP

def run_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    X, X_val, y, y_val, means = generate_data_mean_shift(
        n_clients=args.n_clients,
        n_samples=args.N,
        n_samples_val=args.N_val,
        n_features=args.D,
        n_clusters=args.K,
        seed=args.seed,
        shift_scale=args.shift_scale
    )

    A = er_adjacency_matrix(
        args.n_clients,
        p=args.p_in,
    )

    data = {
        "X": torch.from_numpy(X).to(args.device),
        "X_val": torch.from_numpy(X_val).to(args.device),
        "y": torch.from_numpy(y).to(args.device),
        "y_val": torch.from_numpy(y_val).to(args.device),
        "A": A,
        "means": means,   # ground truth; shape (n_clients, n_clusters, n_features)
    }

    # ----------------------
    # Algorithm
    # ----------------------
    algo_fn = get_algo(args.algo)
    allowed = ALGO_ARG_MAP[args.algo]
    algo_args = filter_args(args, allowed)

    out = algo_fn(algo_args, data)
    # ----------------------
    # Result 
    # ----------------------
    config = vars(args)

    result = {
    "config": config,
    "algo": args.algo,
    "seed": args.seed,
    "ll_rounds": out.get("ll_rounds", None),
    "NMI": out.get("NMI", None),
    "AMI": out.get("AMI", None)
    }
    
    MSE_w = []
    pred_means = out["pred_means"]
    for i in range(args.n_clients):
        pred_mean = pred_means[i]
        true_mean = means[i]
        mse = est_error(pred_mean, true_mean)
        MSE_w.append(mse)

    result["MSE_w"] = MSE_w

    return result

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    
    args.outdir = os.path.join(
    args.outdir,
    args.algo
    )
    os.makedirs(args.outdir, exist_ok=True)

    result = run_experiment(args)

    base = (
    f"D{args.D}_N{args.N}_"
    f"shift_scale{args.shift_scale}_"
    f"seed{args.seed}"
    )

    # ---------- JSON (config + scalars) ----------
    json_path = os.path.join(args.outdir, base + ".json")

    json_result = {
        "config": result["config"],
        "algo": result["algo"],
        "seed": result["seed"],
        "NMI": result["NMI"],
        "AMI": result["AMI"],
    }

    with open(json_path, "w") as f:
        json.dump(json_result, f, indent=2)

    # ---------- CSV files ----------
    save_metrics(
    result.get("ll_rounds"),
    result.get("MSE_w"),
    os.path.join(args.outdir, base + "_metrics.csv")
    )

    print(f"[DONE] Results saved with base name {base}")
