import math
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F
from torch import nn

# ============================================================
# 1) GMM model (diagonal covariances)
# ============================================================
class DiagGMM(nn.Module):
    """
    K-component Gaussian mixture model with diagonal covariances.

    Parameters (trainable):
      - logits: (K,) mixture logits, pi = softmax(logits)
      - means:  (K, D)
      - log_std:(K, D)  std = exp(log_std) + eps  (diagonal covariance)

    We implement:
      log p(x) = log sum_k pi_k * N(x | mu_k, diag(std_k^2))
    """

    def __init__(self, K: int, D: int, init_scale: float = 1.0):
        super().__init__()
        self.K = K
        self.D = D

        # Initialize mixture weights uniform (logits=0), means random, std=1
        self.logits = nn.Parameter(torch.zeros(K))
        self.means = nn.Parameter(init_scale * torch.randn(K, D))
        self.log_std = nn.Parameter(torch.zeros(K, D))

    def mixture_weights(self) -> torch.Tensor:
        """Return pi_k as a probability vector (K,)."""
        return F.softmax(self.logits, dim=0)

    def log_prob_components(self, x: torch.Tensor) -> torch.Tensor:
        """
        Per-component log density for each x.

        Args:
          x: (B, D)
        Returns:
          log N_k(x): (B, K)
        """
        B, D = x.shape
        assert D == self.D

        std = torch.exp(self.log_std) + 1e-6         # (K, D)
        var = std * std                              # (K, D)

        # Expand to broadcast:
        x_e = x[:, None, :]                          # (B, 1, D)
        mu = self.means[None, :, :]                  # (1, K, D)
        var_e = var[None, :, :]                      # (1, K, D)
        log_std_e = self.log_std[None, :, :]         # (1, K, D)

        # log N(x | mu, diag(var)) = -0.5*( sum_d ((x-mu)^2/var) + sum_d log(2*pi*var) )
        quad = ((x_e - mu) ** 2) / var_e             # (B, K, D)
        log_var = 2.0 * log_std_e                    # log(var) = 2 log_std
        const = math.log(2.0 * math.pi)

        return -0.5 * (quad.sum(dim=-1) + (log_var + const).sum(dim=-1))  # (B, K)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Mixture log probability log p(x) for each x. Returns shape (B,)."""
        log_pi = F.log_softmax(self.logits, dim=0)            # (K,)
        logp_comp = self.log_prob_components(x)               # (B, K)
        return torch.logsumexp(logp_comp + log_pi[None, :], dim=1)

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """
        Sample n points from the mixture. Returns (n, D).
        """
        pi = self.mixture_weights()                           # (K,)
        comp = torch.distributions.Categorical(probs=pi).sample((n,))  # (n,)

        std = torch.exp(self.log_std) + 1e-6                  # (K, D)
        eps = torch.randn(n, self.D, device=self.means.device)

        mu = self.means[comp, :]                              # (n, D)
        s = std[comp, :]                                      # (n, D)
        return mu + s * eps


@torch.no_grad()
def clone_gmm(model: DiagGMM) -> DiagGMM:
    """Deep-copy GMM parameters (for synchronous rounds / frozen sampling)."""
    out = DiagGMM(model.K, model.D).to(model.means.device)
    out.logits.copy_(model.logits)
    out.means.copy_(model.means)
    out.log_std.copy_(model.log_std)
    return out

# ============================================================
# 3) Local SKL surrogate loss (paper-close structure)
# ============================================================
def local_skl_surrogate_loss(
    gmm_i: DiagGMM,
    x_private: torch.Tensor,
    x_nbr_list: List[torch.Tensor],
    x_self: torch.Tensor,
    beta_i: float,
    gamma_list: List[float],
    delta_list: List[float],
    use_forward_term: bool = True,
) -> torch.Tensor:
    """
    Local surrogate objective to MINIMIZE:

      beta_i * E_{x in private}[ -log p_i(x) ]
    + sum_j gamma_ij * E_{x~p_j^t}[ -log p_i(x) ]              (reverse KL contribution)
    + sum_j delta_ij * E_{x~p_i^t}[ +log p_i(x) ]              (forward KL majorization)

    If use_forward_term=False, we drop the last term (baseline: "augmented dataset MLE-like").
    """
    device = gmm_i.means.device

    # (A) private negative log-likelihood
    nll_priv = (-gmm_i.log_prob(x_private).mean()) if x_private.numel() > 0 else torch.tensor(0.0, device=device)

    # (B) neighbor negative log-likelihoods (cross-entropy pull)
    nll_nbr = torch.tensor(0.0, device=device)
    for x_nbr, gamma in zip(x_nbr_list, gamma_list):
        if x_nbr.numel() > 0 and gamma != 0.0:
            nll_nbr = nll_nbr + float(gamma) * (-gmm_i.log_prob(x_nbr).mean())

    # (C) forward KL majorization term via frozen self-samples
    #     This is +log p_i (not -log p_i), so it is NOT a standard MLE term.
    self_term = torch.tensor(0.0, device=device)
    if use_forward_term and x_self.numel() > 0:
        delta_sum = float(sum(delta_list))
        if delta_sum != 0.0:
            self_term = delta_sum * (gmm_i.log_prob(x_self).mean())

    return float(beta_i) * nll_priv + nll_nbr + self_term


# ============================================================
# 4) One node update (gradient-based "EMM" / GEM-like in spirit)
# ============================================================
def local_update_emm(
    gmm_i: DiagGMM,
    gmm_i_prev: DiagGMM,
    neighbor_models_prev: List[DiagGMM],
    x_private: torch.Tensor,
    gamma_list: List[float],
    delta_list: List[float],
    beta_i: float,
    lam: float,
    # sampling
    M_self: int,
    M_nbr: int,
    # optimization
    steps: int,
    lr: float,
    batch_size: int,
    device: str,
    use_forward_term: bool = True,
) -> DiagGMM:
    """
    Perform a local update of theta_i by minimizing the SKL surrogate
    using gradient-based optimization in PyTorch.

    - gmm_i_prev: frozen previous iterate for generating self-samples
    - neighbor_models_prev: frozen neighbor models from previous round for neighbor-samples
    """
    gmm_i = gmm_i.to(device)
    gmm_i_prev = gmm_i_prev.to(device)
    x_private = x_private.to(device)

    # Generate frozen sample pools (paper: MC approximation of KL terms)
    with torch.no_grad():
        x_self = gmm_i_prev.sample(M_self).to(device)
        x_nbr_list = [m.to(device).sample(M_nbr) for m in neighbor_models_prev]

    # Optimizer for the local surrogate
    opt = torch.optim.Adam(gmm_i.parameters(), lr=lr)

    N_priv = x_private.shape[0]
    N_self = x_self.shape[0]
    N_nbrs = [x.shape[0] for x in x_nbr_list]

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)

        # Mini-batch from private data
        if N_priv > 0:
            idx = torch.randint(0, N_priv, (min(batch_size, N_priv),), device=device)
            xb_priv = x_private[idx]
        else:
            xb_priv = x_private

        # Mini-batch from self-samples
        idx = torch.randint(0, N_self, (min(batch_size, N_self),), device=device)
        xb_self = x_self[idx]

        # Mini-batches from each neighbor pool
        xb_nbr_list = []
        for x_nbr, Nn in zip(x_nbr_list, N_nbrs):
            idx = torch.randint(0, Nn, (min(batch_size, Nn),), device=device)
            xb_nbr_list.append(x_nbr[idx])

        # Surrogate loss
        loss = local_skl_surrogate_loss(
            gmm_i=gmm_i,
            x_private=xb_priv,
            x_nbr_list=xb_nbr_list,
            x_self=xb_self,
            beta_i=beta_i,
            gamma_list=gamma_list,
            delta_list=delta_list,
            use_forward_term=use_forward_term,
        )

        loss.backward()
        opt.step()

        # Numerical safety: clamp log_std (prevents near-singular covariances)
        with torch.no_grad():
            gmm_i.log_std.clamp_(min=-5.0, max=5.0)

    return gmm_i

# ============================================================
# 5) Synchronous FL rounds (paper-like)
# ============================================================
def run_federated_skl_gtvmin(
    A: torch.Tensor,
    X: torch.Tensor,
    models: Dict[int, DiagGMM],
    rounds: int = 10,
    lam: float = 0.25,
    M_self: int = 512,
    M_nbr: int = 512,
    local_steps: int = 200,
    lr: float = 5e-3,
    batch_size: int = 256,
    device: str = "cpu",
    use_forward_term: bool = True,
):
    """
    Synchronous FL loop:
      - At round t, freeze all models -> {theta_i^t}
      - Each node i updates theta_i^{t+1} using:
            private data D_i
            neighbor samples from {theta_j^t, j in N(i)}
            self samples from theta_i^t
        and paper-close weights beta_i, gamma_ij, delta_ij.

    The coupling parameter lam corresponds to the paper's regularization parameter (lambda / alpha).
    """
    for t in range(rounds):
        # Freeze all models for synchronous round t
        prev = {i: clone_gmm(m).to(device) for i, m in models.items()}

        # Update nodes (can be parallelized; here sequential for simplicity)
        N = A.shape[0]
        for i in range(N):
            nbrs = np.where(A[i] != 0)[0]
            a_ij = A[i][nbrs]

            neighbor_models_prev = [prev[j] for j in nbrs]

            # Paper-close weights:
            beta_i = 1.0
            # gamma_ij, delta_ij carry the SKL 1/2 factor and lambda scaling
            gamma_list = [0.5 * lam * a for a in a_ij]  # reverse KL part
            delta_list = [0.5 * lam * a for a in a_ij]  # forward KL part (via frozen sampling)

            models[i] = local_update_emm(
                gmm_i=models[i],
                gmm_i_prev=prev[i],
                neighbor_models_prev=neighbor_models_prev,
                x_private=X[i],
                gamma_list=gamma_list,
                delta_list=delta_list,
                beta_i=beta_i,
                lam=lam,
                M_self=M_self,
                M_nbr=M_nbr,
                steps=local_steps,
                lr=lr,
                batch_size=batch_size,
                device=device,
                use_forward_term=use_forward_term,
            )

        # Simple paper-friendly scalar diagnostic:
        # average log-likelihood on private data (higher is better), averaged over nodes.
        with torch.no_grad():
            ll = []
            for i in range(N):
                ll.append(models[i].log_prob(X[i].to(device)).mean().item())
            avg_ll = sum(ll) / len(ll)
            print(f"[round {t+1:02d}/{rounds}] avg private log-likelihood = {avg_ll:.3f}")


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Detach torch tensor and move to NumPy."""
    return x.detach().cpu().numpy()

def centralized_gmm_baseline(
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
    
def local_gmm_baseline(
    X: torch.Tensor,
    K: int,
    seed: int = 0,
):
    """
    Local-only baseline: one independent GMM per node.
    """
    models = {}
    ll = {}

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

    avg_ll = sum(ll.values()) / len(ll)
    return models, ll, avg_ll
