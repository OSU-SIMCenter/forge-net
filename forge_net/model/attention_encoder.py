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

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax


# ----------------------------------------------------------------------
# 1. Shared per-source encoder (PointNet-style; swap for ForgeNet's encoder)
#    Same weights for every source so LF/HF clouds of the same object land
#    near each other in latent space. Order- and count-invariant via maxpool.
# ----------------------------------------------------------------------
class PointNetEncoder(nn.Module):
    latent_dim: int = 128

    @nn.compact
    def __call__(self, pts: jax.Array, train: bool = True):
        """pts: (B, N, in_dim) -> (B, latent_dim)"""
        he_init = nn.initializers.he_normal()

        x = nn.Dense(64, kernel_init=he_init)(pts)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Dense(128, kernel_init=he_init)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Dense(256, kernel_init=he_init)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = jnp.max(x, axis=1)                             # (B, 256) symmetric pooling
        x = nn.Dense(self.latent_dim, kernel_init=he_init)(x)
        return nn.relu(x)


# ----------------------------------------------------------------------
# 2. Attention-based fusion over a SET of source latents.
#    Sources are siblings, not a hierarchy. A learned query attends over
#    the available sources; a mask zeros out missing ones so coverage can
#    be ragged (not every case has all n sources).
#    We also add a source-id embedding so the model can learn that
#    "source 3 is the rate-dependent one" etc.
#    Implemented by hand (rather than nn.MultiHeadDotProductAttention) so
#    the per-source attention weights are directly returned for logging.
# ----------------------------------------------------------------------
class SourceAttentionFusion(nn.Module):
    n_sources: int
    latent_dim: int = 128
    n_heads: int = 4

    @nn.compact
    def __call__(self, z: jax.Array, mask: jax.Array):
        """
        z:    (B, n_sources, latent_dim)  per-source latents (zeros where missing)
        mask: (B, n_sources) bool         True = source present
        returns: fused latent (B, latent_dim), attn weights (B, n_sources) (head-averaged)
        """
        B, n, d = z.shape
        head_dim = d // self.n_heads
        he_init = nn.initializers.he_normal()

        src_embed = nn.Embed(self.n_sources, d)(jnp.arange(n))
        z = z + src_embed[None, :, :]                      # tag each source

        query = self.param("query", nn.initializers.normal(0.02), (1, 1, d))
        q = jnp.broadcast_to(query, (B, 1, d))

        q_proj = nn.Dense(d, kernel_init=he_init, name="q_proj")(q)   # (B, 1, d)
        k_proj = nn.Dense(d, kernel_init=he_init, name="k_proj")(z)   # (B, n, d)
        v_proj = nn.Dense(d, kernel_init=he_init, name="v_proj")(z)   # (B, n, d)

        def split_heads(x, seq_len):
            return x.reshape(B, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)

        qh = split_heads(q_proj, 1)                         # (B, H, 1, hd)
        kh = split_heads(k_proj, n)                          # (B, H, n, hd)
        vh = split_heads(v_proj, n)                          # (B, H, n, hd)

        logits = jnp.einsum("bhqd,bhkd->bhqk", qh, kh) / jnp.sqrt(head_dim)
        attn_mask = mask[:, None, None, :]                   # (B, 1, 1, n), True = keep
        logits = jnp.where(attn_mask, logits, jnp.finfo(logits.dtype).min)

        weights = jax.nn.softmax(logits, axis=-1)            # (B, H, 1, n)
        # A case with zero present sources softmaxes an all -inf row -> NaN;
        # zero it out instead (fused output is meaningless there regardless).
        weights = jnp.nan_to_num(weights)

        fused_h = jnp.einsum("bhqk,bhkd->bhqd", weights, vh)         # (B, H, 1, hd)
        fused = fused_h.transpose(0, 2, 1, 3).reshape(B, 1, d)
        fused = nn.Dense(d, kernel_init=he_init, name="out_proj")(fused)
        fused = nn.LayerNorm()(fused.squeeze(1))                      # (B, d)

        attn_w = weights.mean(axis=1).squeeze(1)                      # (B, n) avg over heads
        return fused, attn_w


def source_disagreement(z: jax.Array, mask: jax.Array):
    """Empirical variance across available source latents -> a free,
    powerful uncertainty feature. High where heuristics fan out."""
    mask_f = mask[..., None].astype(z.dtype)                # (B, n, 1)
    cnt = jnp.clip(mask_f.sum(axis=1), min=1.0)              # (B, 1)
    mean = (z * mask_f).sum(axis=1) / cnt                    # (B, d)
    var = ((z - mean[:, None, :]) ** 2 * mask_f).sum(axis=1) / cnt
    return var.mean(axis=1, keepdims=True)                   # (B, 1) scalar spread


# ----------------------------------------------------------------------
# 3. Heteroscedastic head: predicts mean AND (log)variance -> aleatoric.
#    Residual structure baked in (HF ~= consensus): predict a correction
#    on top of the fused latent's own decoded estimate if you wish.
#    Here it maps to a target vector (e.g. HF latent or QoI).
# ----------------------------------------------------------------------
class HeteroscedasticHead(nn.Module):
    out_dim: int = 128
    hidden: int = 256

    @nn.compact
    def __call__(self, fused: jax.Array, disagree: jax.Array):
        he_init = nn.initializers.he_normal()
        # +1 (disagree) folded in via concatenation
        h = jnp.concatenate([fused, disagree], axis=-1)
        h = nn.relu(nn.Dense(self.hidden, kernel_init=he_init)(h))
        h = nn.relu(nn.Dense(self.hidden, kernel_init=he_init)(h))

        mean = nn.Dense(self.out_dim, kernel_init=he_init)(h)
        logvar = nn.Dense(self.out_dim, kernel_init=he_init)(h)
        logvar = jnp.clip(logvar, -8, 8)                     # stability
        return mean, logvar


# ----------------------------------------------------------------------
# 4. One full member (encoder shared across sources within a member).
# ----------------------------------------------------------------------
class FusionMember(nn.Module):
    n_sources: int
    latent_dim: int = 128
    out_dim: int = 128

    @nn.compact
    def __call__(self, source_clouds, mask: jax.Array, train: bool = True):
        """
        source_clouds: sequence of length n_sources, each (B, N_i, in_dim)
                       (N_i may differ per source; use zeros + mask if absent)
        mask:          (B, n_sources) bool
        """
        encoder = PointNetEncoder(self.latent_dim, name="encoder")
        zs = [encoder(pts, train=train) for pts in source_clouds]  # shared weights
        z = jnp.stack(zs, axis=1)                            # (B, n_sources, d)
        z = z * mask[..., None].astype(z.dtype)               # null missing

        fused, attn_w = SourceAttentionFusion(self.n_sources, self.latent_dim)(z, mask)
        disagree = source_disagreement(z, mask)
        mean, logvar = HeteroscedasticHead(self.out_dim)(fused, disagree)
        return {"mean": mean, "logvar": logvar,
                "attn": attn_w, "disagree": disagree}


# ----------------------------------------------------------------------
# 5. Deep ensemble -> epistemic uncertainty from spread of member means.
#    Members are just K independently-parameterized FusionMember submodules
#    (their own encoder/fusion/head weights each), fused under one compact
#    call rather than a torch-style ModuleList.
# ----------------------------------------------------------------------
class FusionEnsemble(nn.Module):
    k: int = 5
    n_sources: int = 4
    latent_dim: int = 128
    out_dim: int = 128

    @nn.compact
    def __call__(self, source_clouds, mask: jax.Array, train: bool = True):
        means, varis = [], []
        for i in range(self.k):
            member = FusionMember(self.n_sources, self.latent_dim, self.out_dim,
                                   name=f"member_{i}")
            out = member(source_clouds, mask, train=train)
            means.append(out["mean"])
            varis.append(jnp.exp(out["logvar"]))

        means = jnp.stack(means, axis=0)                     # (K, B, out_dim)
        varis = jnp.stack(varis, axis=0)                     # (K, B, out_dim)

        pred_mean = means.mean(axis=0)                       # (B, out_dim)
        aleatoric = varis.mean(axis=0)                        # avg predicted var
        epistemic = means.var(axis=0)                          # spread of means
        total_var = aleatoric + epistemic                      # law of total var
        return {"mean": pred_mean,
                "aleatoric": aleatoric,
                "epistemic": epistemic,
                "total_var": total_var}


# ----------------------------------------------------------------------
# 6. Losses + optax training step.
#    - Heteroscedastic Gaussian NLL trains mean + variance jointly.
#      ONLY on samples that have an HF target (paired). Unpaired LF data
#      trains the encoder via reconstruction (your Stage-1 loss, omitted).
#    - Each ensemble member is trained independently (different seed/shuffle)
#      so their disagreement is a meaningful epistemic signal.
# ----------------------------------------------------------------------
def gaussian_nll(mean: jax.Array, logvar: jax.Array, target: jax.Array):
    # 0.5 * [ logvar + (target-mean)^2 / var ]
    inv_var = jnp.exp(-logvar)
    return 0.5 * jnp.mean(logvar + (target - mean) ** 2 * inv_var)


class TrainState(train_state.TrainState):
    """Adds BatchNorm running stats on top of optax.TrainState (see trainer.py)."""
    batch_stats: dict


def _make_train_step(apply_fn):
    @jax.jit
    def train_step(state, source_clouds, mask, has_hf, target):
        def compute_loss(params):
            variables = {"params": params, "batch_stats": state.batch_stats}
            out, mutated_vars = apply_fn(
                variables, source_clouds, mask, train=True, mutable=["batch_stats"],
            )
            # jax.jit can't branch on `has_hf.any()` (data-dependent Python bool),
            # so mask the per-sample NLL instead of skipping unpaired rows.
            has_hf_f = has_hf.astype(target.dtype)[:, None]
            inv_var = jnp.exp(-out["logvar"])
            nll = 0.5 * (out["logvar"] + (target - out["mean"]) ** 2 * inv_var)
            denom = jnp.clip(has_hf_f.sum(), min=1.0) * target.shape[-1]
            loss = (nll * has_hf_f).sum() / denom
            return loss, mutated_vars

        (loss, mutated_vars), grads = jax.value_and_grad(compute_loss, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=mutated_vars["batch_stats"])
        return state, loss

    return train_step


def train_member(member, params, batch_stats, loader, optimizer, epochs=50):
    """Optax training loop for one ensemble member (mirrors ForgeNetTrainer's pattern)."""
    state = TrainState.create(apply_fn=member.apply, params=params, tx=optimizer,
                               batch_stats=batch_stats)
    train_step = _make_train_step(member.apply)

    for _ in range(epochs):
        for batch in loader:
            state, _ = train_step(
                state, batch["source_clouds"], batch["mask"],
                batch["has_hf"], batch["hf_target"],
            )
    return state


# ----------------------------------------------------------------------
# 7. Smoke test with synthetic shapes.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    B, n_src, N, in_dim, latent, out_dim = 8, 4, 512, 3, 128, 128

    rng = jax.random.PRNGKey(0)
    rng, data_rng, init_rng = jax.random.split(rng, 3)

    clouds = list(jax.random.normal(data_rng, (n_src, B, N, in_dim)))
    mask = jnp.ones((B, n_src), dtype=bool)
    mask = mask.at[0, 2].set(False)                      # case 0 missing source 2
    mask = mask.at[3, 0].set(False)                      # ragged coverage

    ens = FusionEnsemble(k=5, n_sources=n_src, latent_dim=latent, out_dim=out_dim)
    variables = ens.init(init_rng, clouds, mask, train=False)

    out = ens.apply(variables, clouds, mask, train=False)
    print("mean      ", out["mean"].shape)
    print("aleatoric ", float(out["aleatoric"].mean()))
    print("epistemic ", float(out["epistemic"].mean()))
    print("total var ", float(out["total_var"].mean()))
