
def get_algo(name):
    if name == "distrGTVMinKL":
        from algos.distrGTVMinKL import run
        return run

    if name == "centralized":
        from algos.centralGMM import run
        return run

    if name == "local":
        from algos.localGMM import run
        return run

    raise ValueError(f"Unknown algo: {name}")

ALGO_ARG_MAP = {
    "distrGTVMinKL": [
        "lrate",
        "reg_term",
        "rounds",
        "local_steps",
        "batch_size",
        "m_self",
        "m_nbr",
        "use_forward_term",
        "device",
        "K",
        "D",
        "cov"
    ],
    "kmeans": [
        "rounds",
        "seed",
        "device",
    ],
}