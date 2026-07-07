"""
Multi-source multi-fidelity fusion for point-cloud deformation prediction.

Pipeline:
  point clouds (n parallel sources)  -> shared PointNet encoder -> latents z_i
  set of latents {z_i}               -> attention fusion (handles missing sources)
  fused latent + disagreement signal -> heteroscedastic head -> (mean, var)
  K such models                      -> deep ensemble -> epistemic + aleatoric

Uncertainty decomposition (law of total variance):
  predictive_var = mean_k(var_k)        # aleatoric  (avg predicted variance)
                 + var_k(mean_k)        # epistemic  (spread of ensemble means)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. Shared per-source encoder (PointNet-style; swap for ForgeNet's encoder)
#    Same weights for every source so LF/HF clouds of the same object land
#    near each other in latent space. Order- and count-invariant via maxpool.
# ----------------------------------------------------------------------
class PointNetEncoder(nn.Module):
    def __init__(self, in_dim=3, latent_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, 64, 1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, 1),    nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 1),   nn.ReLU(), nn.BatchNorm1d(256),
        )
        self.head = nn.Sequential(nn.Linear(256, latent_dim), nn.ReLU())

    def forward(self, pts):                 # pts: (B, N, in_dim)
        x = pts.transpose(1, 2)             # (B, in_dim, N)
        x = self.mlp(x)                     # (B, 256, N)
        x = torch.max(x, dim=2).values      # (B, 256)  symmetric pooling
        return self.head(x)                 # (B, latent_dim)


# ----------------------------------------------------------------------
# 2. Attention-based fusion over a SET of source latents.
#    Sources are siblings, not a hierarchy. A learned query attends over
#    the available sources; a mask zeros out missing ones so coverage can
#    be ragged (not every case has all n sources).
#    We also append a source-id embedding so the model can learn that
#    "source 3 is the rate-dependent one" etc.
# ----------------------------------------------------------------------
class SourceAttentionFusion(nn.Module):
    def __init__(self, n_sources, latent_dim=128, n_heads=4):
        super().__init__()
        self.src_embed = nn.Embedding(n_sources, latent_dim)
        self.query = nn.Parameter(torch.randn(1, 1, latent_dim))
        self.attn = nn.MultiheadAttention(latent_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, z, mask):
        """
        z:    (B, n_sources, latent_dim)  per-source latents (zeros where missing)
        mask: (B, n_sources) bool         True = source present
        returns: fused latent (B, latent_dim), attn weights (B, n_sources)
        """
        B, n, d = z.shape
        ids = torch.arange(n, device=z.device).unsqueeze(0).expand(B, n)
        z = z + self.src_embed(ids)                       # tag each source
        q = self.query.expand(B, 1, d)                    # (B, 1, d)
        key_padding = ~mask                               # True = ignore
        fused, w = self.attn(q, z, z, key_padding_mask=key_padding,
                             need_weights=True, average_attn_weights=True)
        fused = self.norm(fused.squeeze(1))               # (B, d)
        return fused, w.squeeze(1)                        # weights: (B, n)


def source_disagreement(z, mask):
    """Empirical variance across available source latents -> a free,
    powerful uncertainty feature. High where heuristics fan out."""
    mask_f = mask.unsqueeze(-1).float()                   # (B, n, 1)
    cnt = mask_f.sum(1).clamp(min=1.0)                    # (B, 1)
    mean = (z * mask_f).sum(1) / cnt                      # (B, d)
    var = ((z - mean.unsqueeze(1))**2 * mask_f).sum(1) / cnt
    return var.mean(dim=1, keepdim=True)                  # (B, 1) scalar spread


# ----------------------------------------------------------------------
# 3. Heteroscedastic head: predicts mean AND (log)variance -> aleatoric.
#    Residual structure baked in (HF ~= consensus): predict a correction
#    on top of the fused latent's own decoded estimate if you wish.
#    Here it maps to a target vector (e.g. HF latent or QoI).
# ----------------------------------------------------------------------
class HeteroscedasticHead(nn.Module):
    def __init__(self, latent_dim=128, out_dim=128, hidden=256):
        super().__init__()
        # +1 for the disagreement scalar
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, out_dim)
        self.logvar = nn.Linear(hidden, out_dim)

    def forward(self, fused, disagree):
        h = self.trunk(torch.cat([fused, disagree], dim=-1))
        mean = self.mean(h)
        logvar = self.logvar(h).clamp(-8, 8)              # stability
        return mean, logvar


# ----------------------------------------------------------------------
# 4. One full member (encoder shared across sources within a member).
# ----------------------------------------------------------------------
class FusionMember(nn.Module):
    def __init__(self, n_sources, in_dim=3, latent_dim=128, out_dim=128):
        super().__init__()
        self.encoder = PointNetEncoder(in_dim, latent_dim)
        self.fusion = SourceAttentionFusion(n_sources, latent_dim)
        self.head = HeteroscedasticHead(latent_dim, out_dim)

    def forward(self, source_clouds, mask):
        """
        source_clouds: list of length n_sources, each (B, N_i, in_dim)
                       (N_i may differ per source; use zeros + mask if absent)
        mask:          (B, n_sources) bool
        """
        zs = []
        for i, pts in enumerate(source_clouds):
            zs.append(self.encoder(pts))                  # shared weights
        z = torch.stack(zs, dim=1)                        # (B, n_sources, d)
        z = z * mask.unsqueeze(-1).float()                # null missing
        fused, attn_w = self.fusion(z, mask)
        disagree = source_disagreement(z, mask)
        mean, logvar = self.head(fused, disagree)
        return {"mean": mean, "logvar": logvar,
                "attn": attn_w, "disagree": disagree}


# ----------------------------------------------------------------------
# 5. Deep ensemble -> epistemic uncertainty from spread of member means.
# ----------------------------------------------------------------------
class FusionEnsemble(nn.Module):
    def __init__(self, k=5, **kw):
        super().__init__()
        self.members = nn.ModuleList([FusionMember(**kw) for _ in range(k)])

    def forward(self, source_clouds, mask):
        means, varis = [], []
        for m in self.members:
            out = m(source_clouds, mask)
            means.append(out["mean"])
            varis.append(out["logvar"].exp())
        means = torch.stack(means)                        # (K, B, out_dim)
        varis = torch.stack(varis)                        # (K, B, out_dim)

        pred_mean = means.mean(0)                          # (B, out_dim)
        aleatoric = varis.mean(0)                          # avg predicted var
        epistemic = means.var(0)                           # spread of means
        total_var = aleatoric + epistemic                  # law of total var
        return {"mean": pred_mean,
                "aleatoric": aleatoric,
                "epistemic": epistemic,
                "total_var": total_var}


# ----------------------------------------------------------------------
# 6. Losses.
#    - Heteroscedastic Gaussian NLL trains mean + variance jointly.
#      ONLY on samples that have an HF target (paired). Unpaired LF data
#      trains the encoder via reconstruction (your Stage-1 loss, omitted).
#    - Each ensemble member is trained independently (different seed/shuffle)
#      so their disagreement is a meaningful epistemic signal.
# ----------------------------------------------------------------------
def gaussian_nll(mean, logvar, target):
    # 0.5 * [ logvar + (target-mean)^2 / var ]
    inv_var = torch.exp(-logvar)
    return 0.5 * (logvar + (target - mean)**2 * inv_var).mean()


def train_member(member, loader, opt, device, epochs=50):
    member.train()
    for _ in range(epochs):
        for batch in loader:
            clouds = [c.to(device) for c in batch["source_clouds"]]
            mask = batch["mask"].to(device)              # (B, n_sources)
            has_hf = batch["has_hf"].to(device)          # (B,) bool
            target = batch["hf_target"].to(device)       # (B, out_dim)

            out = member(clouds, mask)
            if has_hf.any():
                loss = gaussian_nll(out["mean"][has_hf],
                                    out["logvar"][has_hf],
                                    target[has_hf])
            else:
                loss = torch.zeros((), device=device, requires_grad=True)
            # (add reconstruction term here for unpaired LF data if joint-training)
            opt.zero_grad(); loss.backward(); opt.step()


# ----------------------------------------------------------------------
# 7. Smoke test with synthetic shapes.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, n_src, N, in_dim, latent, out = 8, 4, 512, 3, 128, 128
    dev = "cpu"

    clouds = [torch.randn(B, N, in_dim) for _ in range(n_src)]
    mask = torch.ones(B, n_src, dtype=torch.bool)
    mask[0, 2] = False                                   # case 0 missing source 2
    mask[3, 0] = False                                   # ragged coverage

    ens = FusionEnsemble(k=5, n_sources=n_src, in_dim=in_dim,
                         latent_dim=latent, out_dim=out).to(dev)
    out = ens(clouds, mask)
    print("mean      ", out["mean"].shape)
    print("aleatoric ", out["aleatoric"].mean().item())
    print("epistemic ", out["epistemic"].mean().item())
    print("total var ", out["total_var"].mean().item())