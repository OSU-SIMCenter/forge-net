import jax
import jax.numpy as jnp
import flax.linen as nn


class ForgeNet(nn.Module):
    """
    Reduced ForgeNet (~134K params vs ~610K in the large variant).

    Changes vs ForgeNet:
      - State encoder: 5-layer residual stack (3→64→64→128→256→128) →
        3-layer MLP (3→64→128→half_latent). Raw xyz input doesn't warrant
        the extra depth; max pool still provides global aggregation.
      - Action encoder: 5-layer residual stack → 2-layer MLP (64→half_latent).
        The input is a 1D scalar; the extra capacity was wasted.
      - Decoder: first Dense narrowed 512→256, long-range residual skip removed.
    """
    latent_size: int
    action_dims: int
    dropout: float = 0.3
    predict_temperature: bool = False

    @nn.compact
    def __call__(self, x_t: jax.Array, a_t: jax.Array, train: bool = True):
        B, N, C = x_t.shape
        half_latent = self.latent_size // 2

        he_init = nn.initializers.he_normal()

        # ====================================================================
        # STATE ENCODER  (~25K params, down from ~123K)
        # ====================================================================
        x = nn.Dense(64, kernel_init=he_init)(x_t)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Dense(128, kernel_init=he_init)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        x = nn.Dense(half_latent, kernel_init=he_init)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        x_l = jnp.max(x, axis=1)
        x_l = nn.Dropout(self.dropout, deterministic=not train)(x_l)

        # ====================================================================
        # ACTION ENCODER  (~9K params, down from ~122K)
        # 1D scalar input doesn't benefit from a 4-block residual stack.
        # ====================================================================
        if a_t.ndim == 1:
            a_t = jnp.expand_dims(a_t, axis=1)

        a = nn.Dense(64, kernel_init=he_init)(a_t)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = nn.relu(a)
        a_l = nn.Dense(half_latent, kernel_init=he_init)(a)

        # ====================================================================
        # POINT-WISE DECODER  (~100K params, down from ~365K)
        # First Dense narrowed 512→256; long-range skip removed.
        # ====================================================================
        global_latent = jnp.concatenate([x_l, a_l], axis=1)
        global_expanded = jnp.broadcast_to(
            jnp.expand_dims(global_latent, axis=1),
            (B, N, half_latent * 2)
        )
        combined_features = jnp.concatenate([global_expanded, x_t], axis=-1)

        d = nn.Dense(256, kernel_init=he_init)(combined_features)
        d = nn.BatchNorm(use_running_average=not train)(d)
        d = nn.relu(d)
        d = nn.Dropout(self.dropout * 0.5, deterministic=not train)(d)

        d = nn.Dense(128, kernel_init=he_init)(d)
        d = nn.BatchNorm(use_running_average=not train)(d)
        d = nn.relu(d)

        delta = nn.Dense(3, kernel_init=he_init)(d)

        if self.predict_temperature:
            delta_temp = nn.Dense(1, kernel_init=he_init)(d)  # SEPARATE head, see model.py's matching docstring
            return delta, delta_temp

        return delta