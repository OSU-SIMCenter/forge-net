"""XL ForgeNet -- `model.py`'s exact architecture (residual-block PointNet,
global-maxpool state/action encoders, no local attention -- the "old"
design, deliberately not the `attention_model.py` variant) with every
internal Dense width scaled up ~1.5x, landing at roughly double `model.py`'s
361,284 params. NOT a uniform 2x width scale -- Dense-layer parameter count
scales roughly with width^2 for a layer whose input AND output both grow
(most of this stack), so a straight 2x width multiplier would have
quadrupled params, not doubled them; 1.5x width was chosen empirically
(instantiated + counted, see this file's own test) to land close to 2x
total, not derived from a single global multiplier applied to model.py's
config `latent_size` (most of model.py's widths are HARDCODED literals, not
derived from `latent_size` at all, so changing that config value alone
would barely move the total count)."""

import jax
import jax.numpy as jnp
import flax.linen as nn


class ForgeNet(nn.Module):
    latent_size: int
    action_dims: int
    dropout: float = 0.3
    use_res: bool = True
    predict_temperature: bool = False

    @nn.compact
    def __call__(self, x_t: jax.Array, a_t: jax.Array, train: bool = True):
        """Same `x_t`/`a_t`/return contract as `model.py`'s `ForgeNet` -- see
        its `__call__` docstring. Only the internal widths differ (96/192/
        384/768 in place of 64/128/256/512)."""
        B, N, C = x_t.shape
        half_latent = self.latent_size // 2

        he_init = nn.initializers.he_normal()

        # ====================================================================
        # STATE ENCODER
        # ====================================================================
        x = nn.Dense(96, kernel_init=he_init)(x_t)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        identity1 = x
        x = nn.Dense(96, kernel_init=he_init)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        if self.use_res:
            x = x + identity1

        identity2 = x
        if self.use_res:
            identity2 = nn.Dense(192, kernel_init=he_init)(identity2)
            identity2 = nn.BatchNorm(use_running_average=not train)(identity2)

        x = nn.Dense(192, kernel_init=he_init)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        if self.use_res:
            x = x + identity2

        identity3 = x
        if self.use_res:
            identity3 = nn.Dense(384, kernel_init=he_init)(identity3)
            identity3 = nn.BatchNorm(use_running_average=not train)(identity3)

        x = nn.Dense(384, kernel_init=he_init)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        if self.use_res:
            x = x + identity3

        x = nn.Dense(half_latent, kernel_init=he_init)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        # Global context: Max pooling over the N (points) dimension
        x_l = jnp.max(x, axis=1)
        x_l = nn.Dropout(self.dropout, deterministic=not train)(x_l)

        # ====================================================================
        # ACTION ENCODER
        # ====================================================================
        if a_t.ndim == 1:
            a_t = jnp.expand_dims(a_t, axis=1)

        a = nn.Dense(96, kernel_init=he_init)(a_t)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = nn.relu(a)

        act_identity1 = a
        a = nn.Dense(96, kernel_init=he_init)(a)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = nn.relu(a)
        if self.use_res:
            a = a + act_identity1

        act_identity2 = a
        if self.use_res:
            act_identity2 = nn.Dense(192, kernel_init=he_init)(act_identity2)
            act_identity2 = nn.BatchNorm(use_running_average=not train)(act_identity2)

        a = nn.Dense(192, kernel_init=he_init)(a)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = nn.relu(a)
        if self.use_res:
            a = a + act_identity2

        act_identity3 = a
        if self.use_res:
            act_identity3 = nn.Dense(384, kernel_init=he_init)(act_identity3)
            act_identity3 = nn.BatchNorm(use_running_average=not train)(act_identity3)

        a = nn.Dense(384, kernel_init=he_init)(a)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = nn.relu(a)
        if self.use_res:
            a = a + act_identity3

        a = nn.Dropout(self.dropout, deterministic=not train)(a)
        a_l = nn.Dense(half_latent, kernel_init=he_init)(a)

        # ====================================================================
        # POINT-WISE DECODER
        # ====================================================================
        global_latent = jnp.concatenate([x_l, a_l], axis=1)
        global_expanded = jnp.broadcast_to(
            jnp.expand_dims(global_latent, axis=1),
            (B, N, half_latent * 2)
        )
        combined_features = jnp.concatenate([global_expanded, x_t], axis=-1)

        d = nn.Dense(768, kernel_init=he_init)(combined_features)
        d = nn.BatchNorm(use_running_average=not train)(d)
        d = nn.relu(d)
        d = nn.Dropout(self.dropout * 0.5, deterministic=not train)(d)

        dec_identity1 = d
        if self.use_res:
            dec_identity1 = nn.Dense(192, kernel_init=he_init)(dec_identity1)
            dec_identity1 = nn.BatchNorm(use_running_average=not train)(dec_identity1)

        d = nn.Dense(384, kernel_init=he_init)(d)
        d = nn.BatchNorm(use_running_average=not train)(d)
        d = nn.relu(d)

        d = nn.Dense(192, kernel_init=he_init)(d)
        d = nn.BatchNorm(use_running_average=not train)(d)
        d = nn.relu(d)
        if self.use_res:
            d = d + dec_identity1

        delta = nn.Dense(3, kernel_init=he_init)(d)

        if self.predict_temperature:
            delta_temp = nn.Dense(1, kernel_init=he_init)(d)  # SEPARATE head, see model.py's matching docstring
            return delta, delta_temp

        return delta
