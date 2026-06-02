import argparse
import json
import os
import time
import numpy as np
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_mutual_info_score

from data import generate_data_mnist, er_adjacency_matrix
from training import DiagGMM, run_federated_skl_gtvmin, centralized_gmm_baseline, local_gmm_baseline

# ============================================================
# Argument parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser("Federated GMM experiments")

    # Sweep-controlled params (from SLURM)
    parser.add_argument("--alpha", type=float, default=0.2, help="dirichlet coeff")
    parser.add_argument("--p_in", type=float, default=0.8)
    parser.add_argument("--lrate", type=float, default=5e-4)

    # Experiment params
    parser.add_argument("--D", type=int, default=10)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--N_val", type=int, default=1000)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)

    # System
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outdir", type=str, default="results_mnist")

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
    client_data, client_data_val, client_labels, client_labels_val, label_proportions = generate_data_mnist(
        n_clients=args.n_clients,
        n_samples=args.N,
        n_samples_val=args.N_val,
        n_features=args.D,
        n_clusters=args.K,
        seed=args.seed,
        alpha=args.alpha)

    # Pool all data
    X_val_all = client_data_val.reshape(args.n_clients*args.N_val, args.D)
    y_val_all = client_labels_val.reshape(args.n_clients*args.N_val,)

    A = er_adjacency_matrix(
        args.n_clients,
        p=args.p_in,      # you can generalize this to block models if needed
    )

    # Convert from NumPy to torch tensors and move to device
    client_data = torch.from_numpy(client_data).to(args.device)
    client_data_val = torch.from_numpy(client_data_val).to(args.device)

    # ----------------------
    # Models
    # ----------------------
    models = {
        i: DiagGMM(K=args.K, D=args.D, covariance_type = "full", init_scale=1.0).to(args.device)
        for i in range(args.n_clients)
    }

    models_aug = {
        i: DiagGMM(K=args.K, D=args.D, covariance_type = "full", init_scale=1.0).to(args.device)
        for i in range(args.n_clients)
    }
    # ----------------------
    # Federated training
    # ----------------------
    lmb = 1
    start = time.time()
    fl_ll = run_federated_skl_gtvmin(
            A=A,
            X=client_data,
            X_val=client_data_val,
            models=models,
            rounds=args.rounds,
            lam=lmb,
            M_self=100,
            M_nbr=int(100/args.n_clients),
            local_steps=100,
            lr=args.lrate,
            batch_size=256,
            device=args.device,
            use_forward_term=True,
    )
    fl_time = time.time() - start

    # ----------------------
    # No Forward KL term
    # ----------------------
    fl_aug_ll = run_federated_skl_gtvmin(
                A=A,
                X=client_data,
                X_val=client_data_val,
                models=models_aug,
                rounds=args.rounds,
                lam=lmb,
                M_self=100,
                M_nbr=int(100/args.n_clients),
                local_steps=100,
                lr=args.lrate,
                batch_size=256,
                device=args.device,
                use_forward_term=False,
    )

    # ----------------------
    # Baselines
    # ----------------------
    gmm_c, _, avg_ll_c = centralized_gmm_baseline(client_data, client_data_val, K=args.K, seed=args.seed, covariance_type = "full")
    gmm_l, _, avg_ll_l = local_gmm_baseline(client_data, client_data_val, K=args.K, seed=args.seed, covariance_type = "full")

    # ----------------------
    # Errors
    # ----------------------
    NMI, AMI = np.zeros((args.n_clients)), np.zeros((args.n_clients))
    X_val_all_torch = torch.from_numpy(X_val_all).to(args.device)
    for i, m in enumerate(models.values()):
        y_probs = m.log_prob_components(X_val_all_torch) # (B, K)
        y_preds = np.argmax(y_probs.detach().cpu().numpy(), axis=1)
        NMI[i] = normalized_mutual_info_score(y_val_all, y_preds)
        AMI[i] = adjusted_mutual_info_score(y_val_all, y_preds)

    NMI_aug, AMI_aug = np.zeros((args.n_clients)), np.zeros((args.n_clients))
    for i, m in enumerate(models_aug.values()):
        y_probs = m.log_prob_components(X_val_all_torch) # (B, K)
        y_preds = np.argmax(y_probs.detach().cpu().numpy(), axis=1)
        NMI_aug[i] = normalized_mutual_info_score(y_val_all, y_preds)
        AMI_aug[i] = adjusted_mutual_info_score(y_val_all, y_preds)

    y_preds_c = gmm_c.predict(X_val_all)
    NMI_central = normalized_mutual_info_score(y_val_all, y_preds_c)
    AMI_central = adjusted_mutual_info_score(y_val_all, y_preds_c)

    NMI_local, AMI_local = np.zeros((args.n_clients)), np.zeros((args.n_clients))
    for i, gmm in enumerate(gmm_l.values()):
        y_preds_l = gmm.predict(X_val_all)
        NMI_local[i] = normalized_mutual_info_score(y_val_all, y_preds_l)
        AMI_local[i] = adjusted_mutual_info_score(y_val_all, y_preds_l)

    result = {
        "D": args.D,
        "N": args.N,
        "K": args.K,
        "n_clients": args.n_clients,
        "alpha": args.alpha,
        "lambda": lmb,
        "lr": args.lrate,
        "seed": args.seed,
        "ll_c": avg_ll_c,
        "ll_l": avg_ll_l,
        "fl_ll": fl_ll,
        "fl_aug_ll": fl_aug_ll,
        "centralized_nmi": NMI_central,
        "centralized_ami": AMI_central,
        "local_nmi": np.mean(NMI_local),
        "local_ami": np.mean(AMI_local),
        "fl_nmi": np.mean(NMI),
        "fl_ami": np.mean(AMI),
        "fl_aug_nmi": np.mean(NMI_aug),
        "fl_aug_ami": np.mean(AMI_aug),
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
        f"alpha{args.alpha}_"
        f"seed{args.seed}.json"
    )

    path = os.path.join(args.outdir, fname)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[DONE] Results saved to {path}")
