import jax
import jax.numpy as jnp
import flax.linen as nn

class ForgeNet(nn.Module):
    latent_size: int
    action_dims: int
    dropout: float = 0.3
    use_res: bool = True

    @nn.compact
    def __call__(self, x_t: jax.Array, a_t: jax.Array, train: bool = True):
        """
        Forward pass for ForgeNet.
        
        Args:
            x_t: Point cloud input of shape (B, N, 3).
            a_t: Action input of shape (B, action_dims).
            train: Boolean flag for BatchNorm and Dropout behavior.
        """
        B, N, C = x_t.shape
        half_latent = self.latent_size // 2

        he_init = nn.initializers.he_normal()

        # ====================================================================
        # STATE ENCODER
        # ====================================================================
        x = nn.Dense(64, kernel_init=he_init)(x_t)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        identity1 = x
        x = nn.Dense(64, kernel_init=he_init)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        if self.use_res:
            x = x + identity1

        identity2 = x
        if self.use_res:
            identity2 = nn.Dense(128, kernel_init=he_init)(identity2)
            identity2 = nn.BatchNorm(use_running_average=not train)(identity2)

        x = nn.Dense(128, kernel_init=he_init)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        if self.use_res:
            x = x + identity2

        identity3 = x
        if self.use_res:
            identity3 = nn.Dense(256, kernel_init=he_init)(identity3)
            identity3 = nn.BatchNorm(use_running_average=not train)(identity3)

        x = nn.Dense(256, kernel_init=he_init)(x)
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
        # Ensure a_t is at least 2D: (B, action_dims)
        if a_t.ndim == 1:
            a_t = jnp.expand_dims(a_t, axis=1)

        a = nn.Dense(64, kernel_init=he_init)(a_t)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = nn.relu(a)

        act_identity1 = a
        a = nn.Dense(64, kernel_init=he_init)(a)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = nn.relu(a)
        if self.use_res:
            a = a + act_identity1

        act_identity2 = a
        if self.use_res:
            act_identity2 = nn.Dense(128, kernel_init=he_init)(act_identity2)
            act_identity2 = nn.BatchNorm(use_running_average=not train)(act_identity2)

        a = nn.Dense(128, kernel_init=he_init)(a)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = nn.relu(a)
        if self.use_res:
            a = a + act_identity2

        act_identity3 = a
        if self.use_res:
            act_identity3 = nn.Dense(256, kernel_init=he_init)(act_identity3)
            act_identity3 = nn.BatchNorm(use_running_average=not train)(act_identity3)

        a = nn.Dense(256, kernel_init=he_init)(a)
        a = nn.BatchNorm(use_running_average=not train)(a)
        a = nn.relu(a)
        if self.use_res:
            a = a + act_identity3

        # Dropout applied before the final projection in your PyTorch code
        a = nn.Dropout(self.dropout, deterministic=not train)(a)
        a_l = nn.Dense(half_latent, kernel_init=he_init)(a)

        # ====================================================================
        # POINT-WISE DECODER
        # ====================================================================
        # Combine global features: (B, half_latent * 2)
        global_latent = jnp.concatenate([x_l, a_l], axis=1)

        # Expand global context across all points: (B, N, half_latent * 2)
        global_expanded = jnp.broadcast_to(
            jnp.expand_dims(global_latent, axis=1),
            (B, N, half_latent * 2)
        )

        # Concatenate Global Context with Point Positions: (B, N, half_latent*2 + 3)
        combined_features = jnp.concatenate([global_expanded, x_t], axis=-1)

        d = nn.Dense(512, kernel_init=he_init)(combined_features)
        d = nn.BatchNorm(use_running_average=not train)(d)
        d = nn.relu(d)
        d = nn.Dropout(self.dropout * 0.5, deterministic=not train)(d)

        dec_identity1 = d
        if self.use_res:
            dec_identity1 = nn.Dense(128, kernel_init=he_init)(dec_identity1)
            dec_identity1 = nn.BatchNorm(use_running_average=not train)(dec_identity1)

        d = nn.Dense(256, kernel_init=he_init)(d)
        d = nn.BatchNorm(use_running_average=not train)(d)
        d = nn.relu(d)

        d = nn.Dense(128, kernel_init=he_init)(d)
        d = nn.BatchNorm(use_running_average=not train)(d)
        d = nn.relu(d)
        if self.use_res:
            d = d + dec_identity1

        # Predict Point-wise Deltas
        delta = nn.Dense(3, kernel_init=he_init)(d) # Final output is directly (B, N, 3)

        return delta