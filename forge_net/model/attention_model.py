"""Local-attention ForgeNet variant -- targets the "soft edges near the
press contact, correct overall elongation" failure mode observed on
`model.py`/`small_model.py`'s pure global-maxpool architecture (state
encoder -> ONE pooled vector -> broadcast to every point -> decode). A
single global vector can't carry independent high-frequency detail for
every point, so it low-pass-filters exactly the sharp local discontinuity
a press-contact edge actually has.

`LocalAttentionBlock` adds a k-NN local branch alongside the (UNCHANGED,
kept because it already works) global maxpool branch -- Point-Transformer-
style vector self-attention (Zhao et al. 2021: subtraction-based attention
logits + a relative-position encoding added to both key and value,
softmax-normalized over each point's k nearest geometric neighbors) rather
than a symmetric max/mean pool over the neighborhood, so the network can
LEARN which neighbors matter most for a given point instead of treating
them all the same way -- the mechanism that's actually supposed to recover
edge sharpness, not just more depth.

Two `LocalAttentionBlock`s are stacked (not one) specifically to give the
local branch a two-hop effective receptive field -- a single k-NN hop only
sees geometric neighbors of a point directly, a second hop lets that
information propagate one hop further, closer to seeing the full extent of
a contact region rather than just its immediate boundary.

Action encoder is UNCHANGED from `model.py`'s (small MLP -> global feature,
concatenated the same way) -- this file's scope is the local/global
STATE-encoding split, not the action-space redesign discussed alongside it
(displacement-only scalar action, FiLM conditioning, canonicalized-away
phi/z) -- those are independent, separable design decisions, deliberately
not bundled in here.

Sized deliberately >= `model.py`'s ("large") ~361K params, not smaller --
adding a local-attention branch is meant to ADD a capability (learn which
neighbors matter), not trade off the global path's existing capacity."""

import jax
import jax.numpy as jnp
import flax.linen as nn


def _knn_gather_single(feats: jax.Array, idx: jax.Array) -> jax.Array:
    """`feats`: (N, D), `idx`: (N, k) neighbor indices -> (N, k, D). One
    batch element's worth -- `jax.vmap`'d across the batch dimension by the
    caller, since each batch element has its OWN neighbor indices (fancy
    indexing `feats[idx]` only works cleanly per-element, not batched)."""
    return feats[idx]


def _knn_indices(coords: jax.Array, k: int) -> jax.Array:
    """`coords`: (B, N, 3) -> (B, N, k) nearest-neighbor indices per point.
    Computed ONCE by the caller (`ForgeNet.__call__`) and shared across
    every stacked `LocalAttentionBlock` -- `coords` never changes between
    blocks, so recomputing the full O(N^2) pairwise-distance matrix inside
    each block would be pure waste (confirmed a real cost at this project's
    N=4096/batch_size=32: the (B,N,N,3) `diff` intermediate alone is
    several GB, doubling it for no reason was a real risk to training
    feasibility, not just inefficiency)."""
    diff = coords[:, :, None, :] - coords[:, None, :, :]  # (B, N, N, 3)
    dist2 = jnp.sum(diff ** 2, axis=-1)  # (B, N, N)
    # k NEAREST -> smallest dist2 -> largest -dist2, `jax.lax.top_k` finds
    # the k LARGEST values along the last axis.
    _, knn_idx = jax.lax.top_k(-dist2, k)  # (B, N, k)
    return knn_idx


class LocalAttentionBlock(nn.Module):
    """One Point-Transformer-style local self-attention layer.

    `coords`: (B, N, 3) -- ALWAYS just xyz, even when the caller's per-point
    feature vector also carries e.g. temperature -- neighbor search must
    reflect actual GEOMETRIC proximity, not be skewed by an unrelated
    feature channel's scale.
    `features`: (B, N, D_in) -- per-point features to attend over (NOT
    necessarily `coords` itself; typically the previous block's/the shared
    encoder's output).
    `knn_idx`: (B, N, k) -- precomputed via `_knn_indices` ONCE by the
    caller and shared across every stacked block (see `_knn_indices`'s
    docstring for why this isn't recomputed here).

    Returns `(B, N, latent_size)` -- one attention-aggregated feature per
    point, replacing (not augmenting) `features`' channel count; residual
    connections, if wanted, are the CALLER's job (see `ForgeNet.__call__`,
    which does add one across the two stacked blocks)."""
    latent_size: int

    @nn.compact
    def __call__(self, coords: jax.Array, features: jax.Array, knn_idx: jax.Array) -> jax.Array:
        he_init = nn.initializers.he_normal()

        neighbor_feats = jax.vmap(_knn_gather_single)(features, knn_idx)  # (B, N, k, D_in)
        neighbor_coords = jax.vmap(_knn_gather_single)(coords, knn_idx)  # (B, N, k, 3)

        rel_pos = neighbor_coords - coords[:, :, None, :]  # (B, N, k, 3), relative to the query point
        pos_enc = nn.Dense(self.latent_size, kernel_init=he_init)(rel_pos)  # (B, N, k, latent_size)

        q = nn.Dense(self.latent_size, kernel_init=he_init)(features)  # (B, N, latent_size)
        k_feat = nn.Dense(self.latent_size, kernel_init=he_init)(neighbor_feats)  # (B, N, k, latent_size)
        v_feat = nn.Dense(self.latent_size, kernel_init=he_init)(neighbor_feats)  # (B, N, k, latent_size)

        # Vector (subtraction) attention: logits from q-k directly (not a
        # dot product), same relative-position encoding added to BOTH the
        # key term (before the subtraction) and the value -- so the
        # attention score and the aggregated value both carry "how far/in
        # what direction is this neighbor", not just "how similar is its
        # feature".
        attn_logits = nn.Dense(self.latent_size, kernel_init=he_init)(
            q[:, :, None, :] - (k_feat + pos_enc)
        )  # (B, N, k, latent_size)
        attn_weights = nn.softmax(attn_logits, axis=2)  # normalize OVER THE k NEIGHBORS

        return jnp.sum(attn_weights * (v_feat + pos_enc), axis=2)  # (B, N, latent_size)


class ForgeNet(nn.Module):
    """Local-attention ForgeNet -- see module docstring for the design
    rationale. `use_res`/triple-residual-block style deliberately NOT
    carried over from `model.py`'s "large" arch -- the local-attention
    branch is this file's own capacity lever, not a stand-in for it."""
    latent_size: int
    action_dims: int
    dropout: float = 0.3
    predict_temperature: bool = False
    attn_k: int = 16

    @nn.compact
    def __call__(self, x_t: jax.Array, a_t: jax.Array, train: bool = True):
        """See `model.py`'s `__call__` docstring for the `x_t`/`a_t`/return
        contract -- identical here (xyz or xyz+temp input, `(delta_xyz,
        delta_temp)` tuple iff `predict_temperature`), only the internal
        architecture differs."""
        B, N, C = x_t.shape
        half_latent = self.latent_size // 2
        coords = x_t[..., :3]  # geometric positions for neighbor search -- excludes temp, see LocalAttentionBlock's docstring

        he_init = nn.initializers.he_normal()

        # ====================================================================
        # SHARED PER-POINT BASE ENCODER (feeds BOTH the global and local branches)
        # ====================================================================
        x = nn.Dense(64, kernel_init=he_init)(x_t)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Dense(128, kernel_init=he_init)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Dense(self.latent_size, kernel_init=he_init)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)  # (B, N, latent_size) -- shared base features

        # ====================================================================
        # GLOBAL BRANCH (UNCHANGED mechanism -- already correctly captures
        # overall elongation, kept as-is rather than reworked)
        # ====================================================================
        x_global = jnp.max(x, axis=1)  # (B, latent_size)
        x_global = nn.Dropout(self.dropout, deterministic=not train)(x_global)

        # ====================================================================
        # LOCAL BRANCH (NEW) -- two stacked Point-Transformer-style blocks,
        # residual-connected to each other (not to the shared base -- the
        # base features and attention output live in the same `latent_size`
        # channel count, so a residual add is well-typed and lets the
        # second block refine rather than fully replace the first's output).
        # `knn_idx` computed ONCE here (coords are identical for both
        # blocks) and shared -- see `_knn_indices`'s docstring.
        # ====================================================================
        knn_idx = _knn_indices(coords, self.attn_k)
        x_local = LocalAttentionBlock(latent_size=self.latent_size)(coords, x, knn_idx)
        x_local = nn.relu(x_local)
        x_local = LocalAttentionBlock(latent_size=self.latent_size)(coords, x_local, knn_idx) + x_local
        x_local = nn.relu(x_local)  # (B, N, latent_size)

        # ====================================================================
        # ACTION ENCODER (UNCHANGED from model.py -- see module docstring)
        # ====================================================================
        if a_t.ndim == 1:
            a_t = jnp.expand_dims(a_t, axis=1)

        a = nn.Dense(64, kernel_init=he_init)(a_t)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = nn.relu(a)

        act_identity1 = a
        a = nn.Dense(64, kernel_init=he_init)(a)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = nn.relu(a)
        a = a + act_identity1

        a = nn.Dense(128, kernel_init=he_init)(a)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = nn.relu(a)

        a = nn.Dropout(self.dropout, deterministic=not train)(a)
        a_l = nn.Dense(half_latent, kernel_init=he_init)(a)  # (B, half_latent)

        # ====================================================================
        # POINT-WISE DECODER -- [local (per-point), global (broadcast),
        # action (broadcast), raw coords] -> per-point MLP -> delta
        # ====================================================================
        global_and_action = jnp.concatenate([x_global, a_l], axis=1)  # (B, latent_size + half_latent)
        global_expanded = jnp.broadcast_to(
            jnp.expand_dims(global_and_action, axis=1),
            (B, N, self.latent_size + half_latent),
        )
        combined_features = jnp.concatenate([x_local, global_expanded, x_t], axis=-1)

        d = nn.Dense(512, kernel_init=he_init)(combined_features)
        d = nn.BatchNorm(use_running_average=not train)(d)
        d = nn.relu(d)
        d = nn.Dropout(self.dropout * 0.5, deterministic=not train)(d)

        d = nn.Dense(256, kernel_init=he_init)(d)
        d = nn.BatchNorm(use_running_average=not train)(d)
        d = nn.relu(d)

        d = nn.Dense(128, kernel_init=he_init)(d)
        d = nn.BatchNorm(use_running_average=not train)(d)
        d = nn.relu(d)

        delta = nn.Dense(3, kernel_init=he_init)(d)

        if self.predict_temperature:
            delta_temp = nn.Dense(1, kernel_init=he_init)(d)  # SEPARATE head, see model.py's matching docstring
            return delta, delta_temp

        return delta
