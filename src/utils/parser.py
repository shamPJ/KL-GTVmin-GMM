import argparse

def parse_args():
    parser = argparse.ArgumentParser("Federated GMM experiments")

    # Algorithm selection
    parser.add_argument("--algo", type=str, required=True)

    # Data params
    parser.add_argument("--cov", type=str, choices=["diag", "full"], help="Covariance type")

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

    # distrGTV-GB specific params
    parser.add_argument("--reg_term", type=float, default=1.0, help="coupling term coeff")
    parser.add_argument("--m_self", type=int, default=512)
    parser.add_argument("--m_nbr", type=int, default=512)

    parser.add_argument("--local_steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument(
        "--use_self_term",
        action="store_true",
        help="enable self term"
    )

    parser.add_argument(
        "--no_self_term",
        action="store_false",
        dest="use_self_term",
        help="disable self term"
    )

    parser.set_defaults(use_self_term=True) 

    return parser.parse_args()