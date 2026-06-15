"""Print parameter counts and memory footprints for ForgeNet small and large."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import jax
import jax.numpy as jnp
from forge_net.model.model import ForgeNet as LargeForgeNet
from forge_net.model.small_model import ForgeNet as SmallForgeNet

LATENT_SIZE = 512
ACTION_DIMS = 1
N_POINTS = 2048
BATCH = 1
BYTES_PER_PARAM = 4  # float32


def count_params(params):
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def init_model(model_cls, rng):
    dummy_x = jnp.ones((BATCH, N_POINTS, 3))
    dummy_a = jnp.ones((BATCH, ACTION_DIMS))
    variables = model_cls.init({"params": rng, "dropout": rng}, dummy_x, dummy_a, train=False)
    return variables["params"]


def fmt_bytes(n_bytes):
    if n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.2f} KB"
    return f"{n_bytes / 1024 ** 2:.2f} MB"


def report(name, params):
    n = count_params(params)
    mem = n * BYTES_PER_PARAM
    print(f"  {name}")
    print(f"    Parameters : {n:,}")
    print(f"    Memory     : {fmt_bytes(mem)}  ({mem:,} bytes @ float32)")


rng = jax.random.PRNGKey(0)

large = LargeForgeNet(latent_size=LATENT_SIZE, action_dims=ACTION_DIMS)
small = SmallForgeNet(latent_size=LATENT_SIZE, action_dims=ACTION_DIMS)

large_params = init_model(large, rng)
small_params = init_model(small, rng)

print(f"\nForgeNet memory footprint  (latent={LATENT_SIZE}, action_dims={ACTION_DIMS}, N={N_POINTS})")
print("=" * 60)
report("Large (model.py)", large_params)
print()
report("Small (small_model.py)", small_params)
print()
large_n = count_params(large_params)
small_n = count_params(small_params)
print(f"  Reduction: {large_n / small_n:.1f}x fewer params  "
      f"({(1 - small_n / large_n) * 100:.1f}% smaller)\n")
