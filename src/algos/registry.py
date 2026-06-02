
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