import argparse
import json
import os
import numpy as np
import torch

# ============================================================
# Import project code
# ============================================================
from data.data import generate_data_iid, er_adjacency_matrix
from algos.registry import get_algo

# ============================================================
# Argument parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser("Federated GMM experiments")

    # Algorithm selection
    parser.add_argument("--algo", type=str, required=True)

    # Sweep params
    parser.add_argument("--reg_term", type=float, required=True, help="lambda coupling")
    parser.add_argument("--p_in", type=float, default=1.0)
    parser.add_argument("--p_out", type=float, default=0.0)
    parser.add_argument("--lrate", type=float, default=1e-3)

    # Experiment params
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--N_val", type=int, default=1000)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    # System
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outdir", type=str, default="results")

    return parser.parse_args()

def run_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ----------------------
    # Data
    # ----------------------
    X, X_val, y, y_val, means = generate_data_iid(
        n_clients=args.n_clients,
        n_samples=args.N,
        n_samples_val=args.N_val,
        n_features=args.D,
        n_clusters=args.K,
        seed=args.seed
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
        "means": means,   # ground truth (evaluation only)
    }

    # ----------------------
    # Algorithm
    # ----------------------
    algo_fn = get_algo(args.algo)
    out = algo_fn(args, data)

    # ----------------------
    # Result 
    # ----------------------
    config = vars(args)

    result = {
    "config": config,
    "algo": args.algo,
    "seed": args.seed,
    "ll": out["ll"],
    "ll_log": out.get("ll_log", None),
    "pred_means": out["pred_means"],
}

    return result

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    result = run_experiment(args)

    fname = (
        f"{args.algo}_"
        f"D{args.D}_N{args.N}_"
        f"seed{args.seed}.json"
    )

    path = os.path.join(args.outdir, fname)

    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x)

    print(f"[DONE] Results saved to {path}")