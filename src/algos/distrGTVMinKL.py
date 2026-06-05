import math
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_mutual_info_score
from typing import Dict, List, Optional, Tuple
import torch.nn.functional as F
from torch import nn

class GMM_torch(nn.Module):
    """
    K-component Gaussian mixture model implemented in PyTorch.

    Parameters (trainable):
      - logits: (K,) mixture logits, pi = softmax(logits)
      - means:  (K, D)
      - log_std:(K, D)  std = exp(log_std) + eps  (diagonal covariance)

    We implement:
      log p(x) = log sum_k pi_k * N(x | mu_k, diag(std_k^2))
    """

    def __init__(self, K: int, D: int, covariance_type: str = "diag", init_scale: float = 1.0):
        super().__init__()
        self.K = K
        self.D = D

        # Initialize mixture weights uniform (logits=0), means random, std=1
        self.logits = nn.Parameter(torch.zeros(K))
        self.means = nn.Parameter(init_scale * torch.randn(K, D))
        self.covariance_type = covariance_type

        if covariance_type == "diag":
            self.log_std = nn.Parameter(torch.zeros(K, D))
        else:  # full covariance
            self.raw_L = nn.Parameter(torch.zeros(K, D, D))

    @torch.no_grad()
    def initialize_params(self, X):
        X_np = X.detach().cpu().numpy()

        # ---------- KMeans ----------
        kmeans = KMeans(n_clusters=self.K).fit(X_np)
        labels = torch.tensor(kmeans.labels_, device=X.device)
        means = torch.tensor(kmeans.cluster_centers_, dtype=X.dtype, device=X.device)

        # ---------- mixture weights ----------
        counts = torch.bincount(labels, minlength=self.K).float()
        eps = 1e-8
        probs = (counts + eps) / (counts.sum() + eps * self.K)

        self.means.copy_(means)
        self.logits.copy_(torch.log(probs))

        # ---------- covariance ----------
        eps_cov = 1e-6

        if self.covariance_type == "diag":
            vars = torch.zeros(self.K, self.D, device=X.device)

            for k in range(self.K):
                cluster = X[labels == k]

                if cluster.shape[0] > 0:
                    vars[k] = cluster.var(dim=0, correction=0)
                else:
                    vars[k] = X.var(dim=0, correction=0)

            self.log_std.copy_(0.5 * torch.log(vars + eps_cov))

        else:  # full covariance
            for k in range(self.K):
                cluster = X[labels == k]

                if cluster.shape[0] > 1:
                    cov = torch.cov(cluster.T, correction=0)
                else:
                    cov = torch.cov(X.T, correction=0)

                # regularize covariance by adding small value to diagonal (--> all eigenvalues > 0)
                cov = cov + eps_cov * torch.eye(self.D, device=X.device)

                L = torch.linalg.cholesky(cov) # \Sigma = LL^T, L is lower-triangular, works for PD symm matrix

                # store unconstrained Cholesky params
                # We store the diagonal of the Cholesky factor in log-space (unconstrained),
                # and exponentiate it every time we use the covariance (log-prob, sampling, etc.).
                # log|LL^T| = 2*sum(log(diag(L)))
                self.raw_L[k].copy_(
                    torch.tril(L, -1) + # exclude main diagonal
                    torch.diag(torch.log(torch.diagonal(L))) # log-parameterize the diagonal
                )

    def log_prob_components(self, x: torch.Tensor) -> torch.Tensor:
        B, D = x.shape
        assert D == self.D
        x_e = x[:, None, :]              # (B, 1, D)
        mu = self.means[None, :, :]      # (1, K, D)

        if self.covariance_type == "diag":
            # log N(x | mu, diag(var)) = -0.5*( sum_d ((x-mu)^2/var) + sum_d log(2*pi*var) )
            log_var = 2.0 * self.log_std
            quad = ((x_e - mu) ** 2) * torch.exp(-log_var)[None, :, :] # (B, K, D) - (1, K, D)
            const = D * math.log(2 * math.pi)
            return -0.5 * (quad.sum(dim=-1) + log_var[None, :, :].sum(dim=-1) + const)
        else:  # full
            # \Sigma = LL^T -> det(\Sigma) = det(LL^T) = det(L)det(L^T)
            # For a triangular matrix, the determinant is the product of the diagonal.
            L = torch.tril(self.raw_L) # shape (K, D, D)
            # extract the diagonal elements from the last two dimensions of a tensor, 
            # while keeping the other batch dimensions intact.
            diag = torch.diagonal(L, dim1=-2, dim2=-1) # Shape: (K, D) for K components
            # zero out dig -> L - torch.diag_embed(diag) 
            # torch.exp(diag) guarantees all diagonal entries are strictly positive
            # note, that we store the diagonal in log-space, thus use of exp(diag)
            L = L - torch.diag_embed(diag) + torch.diag_embed(torch.exp(diag))
            # Build covariance
            Sigma = L @ L.transpose(-1, -2)
            # Stabilize
            eps = 1e-4 # set higher than eps_cov used to do Cholesky/inversion 
            Sigma = Sigma + eps * torch.eye(self.D, device=Sigma.device)
            # Re-factor stabilized covariance
            L_stable = torch.linalg.cholesky(Sigma)

            diff = x_e - mu
            # solve y=L^{-1}(x-mu_k) --> (B,K,D)
            # shapes --> L: (K,D,D), diff: (B,K,D) --> unsqueeze L to (1,K,D,D) and diff to (B,K,D,1) 
            y = torch.linalg.solve_triangular(L_stable[None, :, :, :], diff[..., None], upper=False).squeeze(-1)
            # Mahalanobis distance --> (B,K)
            quad = (y ** 2).sum(dim=-1)
            log_det = 2.0 * torch.log(torch.diagonal(L_stable, dim1=-2, dim2=-1)).sum(dim=-1)
            const = D * math.log(2 * math.pi)
            return -0.5 * (quad + log_det[None, :] + const)

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Mixture log probability log p(x) for each x. Returns shape (B,)."""
        log_pi = F.log_softmax(self.logits, dim=0)            # (K,)
        logp_comp = self.log_prob_components(x)               # (B, K)
        return torch.logsumexp(logp_comp + log_pi[None, :], dim=1)
    
    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        pi = F.softmax(self.logits, dim=0)
        comp = torch.distributions.Categorical(probs=pi).sample((n,))
        mu = self.means[comp]

        if self.covariance_type == "diag":
            std = torch.exp(self.log_std) + 1e-6
            s = std[comp]
            eps = torch.randn(n, self.D, device=mu.device)
            return mu + s * eps
        else:
            L = torch.tril(self.raw_L)
            diag = torch.diagonal(L, dim1=-2, dim2=-1)
            # ensure that evalues are positive
            L = L - torch.diag_embed(diag) + torch.diag_embed(torch.exp(diag))
            Sigma = L @ L.transpose(-1, -2)
            # Stabilize
            eps = 1e-4
            Sigma = Sigma + eps * torch.eye(self.D, device=Sigma.device)

            dist = torch.distributions.MultivariateNormal(
                loc=mu,
                covariance_matrix=Sigma[comp]
            )
            return dist.sample()
        
    @torch.no_grad()  
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Predict hard cluster assignments for each x. Returns shape (B,)."""
        log_pi = F.log_softmax(self.logits, dim=0)            # (K,)
        logp_comp = self.log_prob_components(x)               # (B, K)
        logp = logp_comp + log_pi[None, :]
        return torch.argmax(logp, dim=1)

@torch.no_grad()
def clone_gmm(model):
    """Deep-copy GMM parameters (for synchronous rounds / frozen sampling)."""
    out = type(model)(
        K=model.K,
        D=model.D,
        covariance_type=model.covariance_type,
    ).to(model.means.device)

    # shared params
    out.logits.copy_(model.logits)
    out.means.copy_(model.means)

    # covariance-specific params
    if model.covariance_type == "diag":
        out.log_std.copy_(model.log_std)
    else:  # full
        out.raw_L.copy_(model.raw_L)

    return out

# ============================
# Local SKL surrogate loss 
# ============================
def local_skl_surrogate_loss(
    gmm_i: GMM_torch,
    x_local: torch.Tensor,
    x_nbr_list: List[torch.Tensor],
    x_self: torch.Tensor,
    beta_i: float,
    gamma_list: List[float],
    delta_list: List[float],
    use_self_term: bool = True,
) -> torch.Tensor:
    """
    Local surrogate objective to MINIMIZE:

      beta_i * E_{x in local}[ -log p_i(x) ]
    + sum_j gamma_ij * E_{x~p_j^t}[ -log p_i(x) ]              (nbr KL)
    + sum_j delta_ij * E_{x~p_i^t}[ +log p_i(x) ]              (self KL)

    If use_self_term=False, we drop the last term (i.e. augmented dataset).
    """
    device = gmm_i.means.device

    # (A) local NLL
    nll_i = -gmm_i.log_prob(x_local).mean()

    # (B) neighbor NLL (cross-entropy pull)
    nll_nbr = torch.tensor(0.0, device=device)
    for x_nbr, gamma in zip(x_nbr_list, gamma_list):
        if x_nbr.numel() > 0 and gamma != 0.0:
            nll_nbr = nll_nbr + float(gamma) * (-gmm_i.log_prob(x_nbr).mean())

    # (C) "self" term via frozen self-samples
    #     This is +log p_i (not -log p_i), so it is NOT a standard MLE term.
    self_term = torch.tensor(0.0, device=device)
    if use_self_term and x_self.numel() > 0:
        delta_sum = float(sum(delta_list))
        if delta_sum != 0.0:
            self_term = delta_sum * (gmm_i.log_prob(x_self).mean())

    return float(beta_i) * nll_i + nll_nbr + self_term


# ============================================================
# 4) One node update (gradient-based EM updates)
# ============================================================
def local_update(
    gmm_i: GMM_torch,
    gmm_i_prev: GMM_torch,
    neighbor_models_prev: List[GMM_torch],
    x_local: torch.Tensor,
    gamma_list: List[float],
    delta_list: List[float],
    beta_i: float,
    # sampling
    M_self: int,
    M_nbr: int,
    # optimization
    steps: int,
    lr: float,
    batch_size: int,
    device: str,
    use_self_term: bool = True,
) -> GMM_torch:
    """
    Perform a local update of theta_i by minimizing the SKL surrogate
    using gradient-based optimization in PyTorch.

    - gmm_i_prev: frozen previous iterate for generating self-samples
    - neighbor_models_prev: frozen neighbor models from previous round for neighbor-samples
    """
    gmm_i = gmm_i.to(device)
    gmm_i_prev = gmm_i_prev.to(device)
    x_local = x_local.to(device)

    # Sample from `fixed` distr's
    with torch.no_grad():
        x_self = gmm_i_prev.sample(M_self).to(device)
        x_nbr_list = [m.sample(M_nbr) for m in neighbor_models_prev]

    # Optimizer for the local surrogate
    opt = torch.optim.Adam(gmm_i.parameters(), lr=lr)

    N_i = x_local.shape[0]
    N_self = x_self.shape[0]
    N_nbrs = [x.shape[0] for x in x_nbr_list]

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)

        # Mini-batch from local data
        idx = torch.randint(0, N_i, (min(batch_size, N_i),), device=device)
        xb_local = x_local[idx]

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
            x_local=xb_local,
            x_nbr_list=xb_nbr_list,
            x_self=xb_self,
            beta_i=beta_i,
            gamma_list=gamma_list,
            delta_list=delta_list,
            use_self_term=use_self_term,
        )

        loss.backward()
        opt.step()

        # Numerical safety: clamp log_std (prevents near-singular covariances)
        with torch.no_grad():
            if gmm_i.covariance_type == "diag":
                gmm_i.log_std.clamp_(min=-5.0, max=5.0)
            if gmm_i.covariance_type == "full":
                diag = torch.diagonal(gmm_i.raw_L, dim1=-2, dim2=-1)
                diag.clamp_(min=-10.0, max=10.0)

    return gmm_i

# ============================================================
# 5) Synchronous FL rounds
# ============================================================
def run(
        args,
        data
):
    """
    Synchronous FL loop:
      - At round t, freeze all models -> {theta_i^t}
      - Each node i updates theta_i^{t+1} using:
            local data D_i
            neighbor samples from {theta_j^t, j in N(i)}
            self samples from theta_i^t
        and weights for each term beta_i, gamma_ij, delta_ij.

    The coupling parameter lam corresponds to the paper's regularization parameter (lambda / alpha).
    """

    # unpack data
    X = data["X"] # shape (N, Ni, D) list of tensors
    X_val = data["X_val"]
    y_val = data["y_val"]
    A = data["A"]

    # unpack args
    K = args.K
    D = args.D
    covariance_type = args.cov
    rounds = args.rounds
    lam = args.reg_term
    M_self = args.m_self
    M_nbr = args.m_nbr
    local_steps = args.local_steps
    lr = args.lrate
    batch_size = args.batch_size
    device = args.device
    use_self_term = args.use_self_term   

    # Init params
    N = A.shape[0]
    models = {}

    # init with KMeans
    for i in range(N):
        gmm = GMM_torch(K=K, D=D, covariance_type=covariance_type).to(device)
        gmm.initialize_params(X[i])
        models[i] = gmm

    ll_rounds = np.zeros((rounds,))
    for t in range(rounds):
        # Freeze all models for synchronous round t
        prev = {i: clone_gmm(m).to(device) for i, m in models.items()}

        # Update nodes (can be parallelized; here sequential for simplicity)
        for i in range(N):
            nbrs = np.where(A[i] != 0)[0]
            a_ij = A[i][nbrs]

            neighbor_models_prev = [prev[j] for j in nbrs]

            # weights:
            beta_i = 1.0
            # gamma_ij, delta_ij carry the SKL 1/2 factor and lambda scaling
            gamma_list = [0.5 * lam * a for a in a_ij]  # nbr term
            delta_list = [0.5 * lam * a for a in a_ij]  # self term

            models[i] = local_update(
                gmm_i=models[i],
                gmm_i_prev=prev[i],
                neighbor_models_prev=neighbor_models_prev,
                x_local=X[i],
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
                use_self_term=use_self_term,
            )

        # per-sample average log-likelihood on local data, averaged over nodes.
        with torch.no_grad():
            ll = []
            for i in range(N):
                ll.append(models[i].log_prob(X_val[i]).mean().item())
            ll_rounds[t] = sum(ll) / len(ll)
            # print(f"[round {t+1:02d}/{rounds}] avg validation log-likelihood = {avg_ll:.3f}")
    
    pred_means = {i: models[i].means.detach().cpu().numpy() for i in models}
    pred_val = {i: models[i].predict(X_val[i]) for i in models}
    NMI = [normalized_mutual_info_score(y_val[i].cpu(), pred_val[i].cpu()) for i in models]
    AMI = [adjusted_mutual_info_score(y_val[i].cpu(), pred_val[i].cpu()) for i in models]

    return {
        "ll_rounds": ll_rounds,
        "models": models,
        "pred_means": pred_means,
        "NMI": sum(NMI) / len(NMI), # average NMI across nodes
        "AMI": sum(AMI) / len(AMI)
    }
