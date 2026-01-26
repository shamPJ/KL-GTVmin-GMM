import argparse
import json
import os
import time
import numpy as np
import torch

# ============================================================
# Import your project code
# ============================================================
from data import generate_federated_data, er_adjacency_matrix
from training import DiagGMM, run_federated_skl_gtvmin, centralized_gmm_baseline, local_gmm_baseline
from metrics import est_error


# ============================================================
# Argument parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser("Federated GMM experiments")

    # Sweep-controlled params (from SLURM)
    parser.add_argument("--reg_term", type=float, required=True, help="lambda coupling")
    parser.add_argument("--p_in", type=float, default=1.0)
    parser.add_argument("--p_out", type=float, default=0.0)
    parser.add_argument("--lrate", type=float, default=1e-3)

    # Experiment params
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    # System
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outdir", type=str, default="results")

    return parser.parse_args()


# ============================================================
# Single experiment
# ============================================================
def run_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ----------------------
    # Data
    # ----------------------
    client_data, _, means = generate_federated_data(
        n_clients=args.n_clients,
        n_samples=args.n_clients * args.N,
        n_features=args.D,
        n_clusters=args.K,
        seed=args.seed,
        non_iid=False,
        alpha=0.3,
    )

    A = er_adjacency_matrix(
        args.n_clients,
        p=args.p_in,      # you can generalize this to block models if needed
    )

    # Convert from NumPy to torch tensors and move to device
    client_data = torch.from_numpy(client_data).to(args.device)


    # ----------------------
    # Models
    # ----------------------
    models = {
        i: DiagGMM(K=args.K, D=args.D, init_scale=1.0).to(args.device)
        for i in range(args.n_clients)
    }

    models_aug = {
        i: DiagGMM(K=args.K, D=args.D, init_scale=1.0).to(args.device)
        for i in range(args.n_clients)
    }
    # ----------------------
    # Federated training
    # ----------------------
    start = time.time()
    run_federated_skl_gtvmin(
        A=A,
        X=client_data,
        models=models,
        rounds=args.rounds,
        lam=args.reg_term,
        M_self=512,
        M_nbr=512,
        local_steps=250,
        lr=args.lrate,
        batch_size=256,
        device=args.device,
        use_forward_term=True,
    )
    fl_time = time.time() - start

    # ----------------------
    # No Forward KL term
    # ----------------------
    run_federated_skl_gtvmin(
        A=A,
        X=client_data,
        models=models_aug,
        rounds=args.rounds,
        lam=args.reg_term,
        M_self=512,
        M_nbr=512,
        local_steps=250,
        lr=args.lrate,
        batch_size=256,
        device=args.device,
        use_forward_term=False,
    )

    # ----------------------
    # Baselines
    # ----------------------
    gmm_c, _, _ = centralized_gmm_baseline(client_data, K=args.K)
    gmm_l, _, _ = local_gmm_baseline(client_data, K=args.K)

    # ----------------------
    # Errors
    # ----------------------
    fl_vs_true = []
    fl_vs_central = []

    for m in models.values():
        mu = m.means.detach().cpu().numpy()
        fl_vs_true.append(est_error(mu, means))
        fl_vs_central.append(est_error(mu, gmm_c.means_))

    fl_vs_true_aug = []
    fl_vs_central_aug = []

    for m in models_aug.values():
        mu = m.means.detach().cpu().numpy()
        fl_vs_true_aug.append(est_error(mu, means))
        fl_vs_central_aug.append(est_error(mu, gmm_c.means_))

    local_vs_true = []
    local_vs_central = []
    for gmm in gmm_l.values():
        local_vs_true.append(est_error(gmm.means_, means))
        local_vs_central.append(est_error(gmm.means_,gmm_c.means_))

    result = {
        "D": args.D,
        "N": args.N,
        "K": args.K,
        "n_clients": args.n_clients,
        "lambda": args.reg_term,
        "lr": args.lrate,
        "seed": args.seed,
        "centralized_vs_true": float(est_error(gmm_c.means_, means)),
        "local_vs_true": float(np.mean(local_vs_true)),
        "local_vs_central": float(np.mean(local_vs_central)),
        "fl_vs_true": float(np.mean(fl_vs_true)),
        "fl_vs_central": float(np.mean(fl_vs_central)),
        "fl_vs_true_aug": float(np.mean(fl_vs_true_aug)),
        "fl_vs_central_aug": float(np.mean(fl_vs_central_aug)),
        "runtime_sec": fl_time,
    }

    return result


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    result = run_experiment(args)

    # Unique filename per job
    fname = (
        f"D{args.D}_N{args.N}_"
        f"lam{args.reg_term}_"
        f"seed{args.seed}.json"
    )

    path = os.path.join(args.outdir, fname)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] Results saved to {path}")
