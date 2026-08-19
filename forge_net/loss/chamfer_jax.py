from typing import Union, Tuple, Optional, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

class ChamferResult(NamedTuple):
    loss:          Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]
    cham_x_to_y:   jnp.ndarray
    cham_y_to_x:   Optional[jnp.ndarray] = None
    loss_normals:  Optional[Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]] = None
    disps_x_to_y:  Optional[jnp.ndarray] = None
    disps_y_to_x:  Optional[jnp.ndarray] = None


def knn_points_simple_jax(
    p1: jnp.ndarray,
    p2: jnp.ndarray,
    lengths1: Optional[jnp.ndarray] = None,
    lengths2: Optional[jnp.ndarray] = None,
    norm: int = 2,
    K: int = 1,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    N, P1, D = p1.shape
    P2 = p2.shape[1]

    if lengths1 is None:
        lengths1 = jnp.full((N,), P1, dtype=jnp.int32)
    if lengths2 is None:
        lengths2 = jnp.full((N,), P2, dtype=jnp.int32)

    if norm == 2:
        diff = p1[:, :, None, :] - p2[:, None, :, :]
        dist_mat = jnp.sum(diff ** 2, axis=-1)
    elif norm == 1:
        diff = p1[:, :, None, :] - p2[:, None, :, :]
        dist_mat = jnp.sum(jnp.abs(diff), axis=-1)
    else:
        raise ValueError("norm must be 1 or 2")

    # Mask out padding in p2
    mask = jnp.arange(P2)[None, :] >= lengths2[:, None]  # (N, P2)
    mask = jnp.broadcast_to(mask[:, None, :], dist_mat.shape)
    dist_mat = jnp.where(mask, jnp.inf, dist_mat)

    # Take K nearest (jax.lax.top_k returns largest, so we negate)
    neg_dists, idx = jax.lax.top_k(-dist_mat, K)
    dists = -neg_dists

    # Zero-out distances for padding positions in p1
    p1_mask = jnp.arange(P1)[None, :] >= lengths1[:, None]  # (N, P1)
    p1_mask = jnp.broadcast_to(p1_mask[:, :, None], dists.shape)
    dists = jnp.where(p1_mask, 0.0, dists)

    return dists, idx


def knn_points_k1_chunked_jax(
    p1: jnp.ndarray,
    p2: jnp.ndarray,
    lengths1: Optional[jnp.ndarray] = None,
    lengths2: Optional[jnp.ndarray] = None,
    norm: int = 2,
    chunk_size: int = 256,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Exact K=1 nearest-neighbor search, same result as
    `knn_points_simple_jax(p1, p2, ..., K=1)` but computed via `jax.lax.scan`
    over chunks of p2 instead of materializing the full (N, P1, P2) distance
    matrix at once. Peak memory is O(N, P1, chunk_size) instead of
    O(N, P1, P2) -- this is what lets n_search_points scale up (e.g. to
    4096) without OOMing. Same total FLOPs as the dense version, just spread
    over sequential chunks; not an approximation.

    The scan body is wrapped in `jax.checkpoint` -- confirmed directly this
    matters, not just theoretical: without it, `jax.lax.scan`'s default
    reverse-mode AD (needed by `jacrev`/`grad` for CMA-MEGA's gradient
    scoring) saves the per-chunk diff/dist residuals STACKED across every
    chunk for the backward pass, which adds back up to ~dense-sized memory
    (measured: chunk_size=1024 -> still OOM'd requesting 33GB, chunk_size=256
    -> pathological multi-minute XLA compile that never finished). With
    `jax.checkpoint`, the backward pass recomputes each chunk's forward
    diff/dist instead of storing it -- trades some backward-pass FLOPs for
    the peak-memory reduction that's the entire point of chunking here.
    """
    N, P1, D = p1.shape
    P2 = p2.shape[1]

    if lengths1 is None:
        lengths1 = jnp.full((N,), P1, dtype=jnp.int32)
    if lengths2 is None:
        lengths2 = jnp.full((N,), P2, dtype=jnp.int32)

    n_chunks = -(-P2 // chunk_size)  # ceil div
    pad = n_chunks * chunk_size - P2
    p2_padded = jnp.pad(p2, ((0, 0), (0, pad), (0, 0))) if pad > 0 else p2
    # (n_chunks, N, chunk_size, D) so scan iterates over chunks
    p2_chunks = p2_padded.reshape(N, n_chunks, chunk_size, D).transpose(1, 0, 2, 3)
    offsets = jnp.arange(n_chunks) * chunk_size

    @jax.checkpoint
    def body(carry, xs):
        best_dist, best_idx = carry
        p2_chunk, offset = xs
        diff = p1[:, :, None, :] - p2_chunk[:, None, :, :]
        if norm == 2:
            dist = jnp.sum(diff ** 2, axis=-1)
        else:
            dist = jnp.sum(jnp.abs(diff), axis=-1)

        local_idx = jnp.arange(chunk_size)[None, None, :] + offset  # (1, 1, chunk)
        valid = local_idx < lengths2[:, None, None]  # (N, 1, chunk)
        dist = jnp.where(valid, dist, jnp.inf)

        chunk_min = jnp.min(dist, axis=-1)  # (N, P1)
        chunk_argmin = jnp.argmin(dist, axis=-1).astype(jnp.int32) + offset  # (N, P1)

        improve = chunk_min < best_dist
        best_dist = jnp.where(improve, chunk_min, best_dist)
        best_idx = jnp.where(improve, chunk_argmin, best_idx)
        return (best_dist, best_idx), None

    init = (jnp.full((N, P1), jnp.inf, dtype=p1.dtype), jnp.zeros((N, P1), dtype=jnp.int32))
    (best_dist, best_idx), _ = jax.lax.scan(body, init, (p2_chunks, offsets))

    p1_mask = jnp.arange(P1)[None, :] >= lengths1[:, None]  # (N, P1)
    best_dist = jnp.where(p1_mask, 0.0, best_dist)

    # match knn_points_simple_jax's (N, P1, K) shape convention with K=1
    return best_dist[..., None], best_idx[..., None]


def knn_gather_jax(
    x: jnp.ndarray,
    idx: jnp.ndarray,
    lengths: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    N, M, U = x.shape
    _N, L, K = idx.shape

    if lengths is None:
        lengths = jnp.full((N,), M, dtype=jnp.int32)

    # Advanced indexing to gather features
    batch_idx = jnp.arange(N)[:, None, None]
    x_out = x[batch_idx, idx]  # (N, L, K, U)

    # Mask out invalid indices based on lengths
    mask = lengths[:, None] <= jnp.arange(K)[None, :]  # (N, K)
    mask = jnp.broadcast_to(mask[:, None, :, None], x_out.shape)
    x_out = jnp.where(mask, 0.0, x_out)

    return x_out


def cosine_similarity_jax(a: jnp.ndarray, b: jnp.ndarray, axis: int = -1, eps: float = 1e-6) -> jnp.ndarray:
    dot = jnp.sum(a * b, axis=axis)
    norm_a = jnp.linalg.norm(a, axis=axis)
    norm_b = jnp.linalg.norm(b, axis=axis)
    return dot / jnp.maximum(norm_a * norm_b, eps)


def _chamfer_single_direction_jax(
    x, y, x_lengths, y_lengths, x_normals, y_normals,
    weights, point_reduction, norm, abs_cosine, return_displacements,
    chunk_size=None,
):
    return_normals = x_normals is not None and y_normals is not None
    N, P1, D = x.shape

    if chunk_size is not None:
        dists, idx = knn_points_k1_chunked_jax(x, y, x_lengths, y_lengths, norm=norm, chunk_size=chunk_size)
    else:
        dists, idx = knn_points_simple_jax(x, y, x_lengths, y_lengths, norm=norm, K=1)
    cham_x = dists[..., 0]

    disps = None
    if return_displacements:
        batch_idx = jnp.arange(N)[:, None, None]
        nearest_y = y[batch_idx, idx][..., 0, :]  # (N, P1, D)
        disps = nearest_y - x

    x_mask = jnp.arange(P1)[None, :] >= x_lengths[:, None]  # (N, P1)
    cham_x = jnp.where(x_mask, 0.0, cham_x)

    if weights is not None:
        cham_x *= weights.reshape(N, 1)

    cham_norm_x = jnp.zeros(())
    if return_normals:
        x_normals_near = knn_gather_jax(y_normals, idx, y_lengths)[..., 0, :]
        cosine_sim = cosine_similarity_jax(x_normals, x_normals_near, axis=2, eps=1e-6)
        cham_norm_x = 1.0 - (jnp.abs(cosine_sim) if abs_cosine else cosine_sim)

        cham_norm_x = jnp.where(x_mask, 0.0, cham_norm_x)
        if weights is not None:
            cham_norm_x *= weights.reshape(N, 1)

    if point_reduction == "max":
        cham_x = jnp.max(cham_x, axis=1)
    elif point_reduction is not None:
        cham_x = jnp.sum(cham_x, axis=1)
        if return_normals:
            cham_norm_x = jnp.sum(cham_norm_x, axis=1)
        if point_reduction == "mean":
            x_lengths_clamped = jnp.maximum(x_lengths, 1)
            cham_x /= x_lengths_clamped
            if return_normals:
                cham_norm_x /= x_lengths_clamped

    return cham_x, cham_norm_x if return_normals else None, disps if return_displacements else None


def _apply_batch_reduction_jax(cham_x, cham_norm_x, weights, batch_reduction):
    if batch_reduction is None:
        return cham_x, cham_norm_x
    N = cham_x.shape[0]
    cham_x = jnp.sum(cham_x)
    if cham_norm_x is not None:
        cham_norm_x = jnp.sum(cham_norm_x)
    if batch_reduction == "mean":
        div = max(N, 1) if weights is None else jnp.maximum(jnp.sum(weights), 1e-8)
        cham_x /= div
        if cham_norm_x is not None:
            cham_norm_x /= div
    return cham_x, cham_norm_x


def chamfer_distance_jax(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_lengths: Optional[jnp.ndarray] = None,
    y_lengths: Optional[jnp.ndarray] = None,
    x_normals: Optional[jnp.ndarray] = None,
    y_normals: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    lambda_x_to_y: float = 1.0,
    lambda_y_to_x: float = 1.0,
    batch_reduction: Optional[str] = "mean",
    point_reduction: Optional[str] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
    return_displacements: bool = False,
    chunk_size: Optional[int] = None,
) -> ChamferResult:

    if batch_reduction is not None and batch_reduction not in ("mean", "sum"):
        raise ValueError('batch_reduction must be "mean", "sum", or None')
    if point_reduction is not None and point_reduction not in ("mean", "sum", "max"):
        raise ValueError('point_reduction must be "mean", "sum", "max", or None')
    if point_reduction is None and batch_reduction is not None:
        raise ValueError("batch_reduction must be None when point_reduction is None")
    if norm not in (1, 2):
        raise ValueError("norm must be 1 or 2")

    N, P1, D = x.shape
    P2 = y.shape[1]

    if x_lengths is None:
        x_lengths = jnp.full((N,), P1, dtype=jnp.int32)
    if y_lengths is None:
        y_lengths = jnp.full((N,), P2, dtype=jnp.int32)

    cham_x, cham_norm_x, disps_x_to_y = _chamfer_single_direction_jax(
        x, y, x_lengths, y_lengths, x_normals, y_normals,
        weights, point_reduction, norm, abs_cosine, return_displacements,
        chunk_size=chunk_size,
    )

    if single_directional:
        reduced, reduced_normals = _apply_batch_reduction_jax(
            cham_x, cham_norm_x, weights, batch_reduction
        )
        return ChamferResult(
            loss=reduced,
            cham_x_to_y=reduced,
            cham_y_to_x=None,
            loss_normals=reduced_normals,
            disps_x_to_y=disps_x_to_y,
        )

    cham_y, cham_norm_y, disps_y_to_x = _chamfer_single_direction_jax(
        y, x, y_lengths, x_lengths, y_normals, x_normals,
        weights, point_reduction, norm, abs_cosine, return_displacements,
        chunk_size=chunk_size,
    )

    cham_x_w = cham_x * lambda_x_to_y
    cham_y_w = cham_y * lambda_y_to_x

    if point_reduction == "max":
        loss = jnp.maximum(cham_x_w, cham_y_w)
        loss_normals = None
    elif point_reduction is not None:
        loss = cham_x_w + cham_y_w
        loss_normals = (cham_norm_x + cham_norm_y) if cham_norm_x is not None else None
    else:
        loss = (cham_x_w, cham_y_w)
        loss_normals = (cham_norm_x, cham_norm_y) if cham_norm_x is not None else None

    # Handle the tuple case separately if point_reduction is None
    if isinstance(loss, tuple):
        loss_0, loss_norm_0 = _apply_batch_reduction_jax(loss[0], loss_normals[0] if loss_normals else None, weights, batch_reduction)
        loss_1, loss_norm_1 = _apply_batch_reduction_jax(loss[1], loss_normals[1] if loss_normals else None, weights, batch_reduction)
        loss = (loss_0, loss_1)
        loss_normals = (loss_norm_0, loss_norm_1) if loss_normals else None
    else:
        loss, loss_normals = _apply_batch_reduction_jax(loss, loss_normals, weights, batch_reduction)

    cham_x_reduced, _ = _apply_batch_reduction_jax(cham_x, None, weights, batch_reduction)
    cham_y_reduced, _ = _apply_batch_reduction_jax(cham_y, None, weights, batch_reduction)

    return ChamferResult(
        loss=loss,
        cham_x_to_y=cham_x_reduced,
        cham_y_to_x=cham_y_reduced,
        loss_normals=loss_normals,
        disps_x_to_y=disps_x_to_y,
        disps_y_to_x=disps_y_to_x,
    )

if __name__ == "__main__":

    #Test jax vs pytorch chamfer implementations
    import torch
    import numpy as np
    from forge_net.loss.chamfer import chamfer_distance

    B, P1, P2, D = 2, 128, 96, 3
    
    np.random.seed(42)
    x_np = np.random.randn(B, P1, D).astype(np.float32)
    y_np = np.random.randn(B, P2, D).astype(np.float32)
    
    x_lengths_np = np.array([128, 80], dtype=np.int32)
    y_lengths_np = np.array([96, 50], dtype=np.int32)

    # 2. PyTorch execution
    x_pt = torch.tensor(x_np)
    y_pt = torch.tensor(y_np)
    x_len_pt = torch.tensor(x_lengths_np, dtype=torch.int64)
    y_len_pt = torch.tensor(y_lengths_np, dtype=torch.int64)

    res_pt = chamfer_distance(
        x_pt, y_pt, 
        x_lengths=x_len_pt, y_lengths=y_len_pt, 
        return_displacements=True
    )

    # 3. JAX execution (JIT-compiled for sanity check)
    x_jx = jnp.array(x_np)
    y_jx = jnp.array(y_np)
    x_len_jx = jnp.array(x_lengths_np)
    y_len_jx = jnp.array(y_lengths_np)
    
    # We can jit the JAX version!
    jitted_chamfer = jax.jit(chamfer_distance_jax, static_argnames=['return_displacements'])

    res_jx = jitted_chamfer(
        x_jx, y_jx, 
        x_lengths=x_len_jx, y_lengths=y_len_jx, 
        return_displacements=True
    )
    print(res_jx.loss, res_pt.loss)
    # 4. Compare outputs
    np.testing.assert_allclose(res_pt.loss.numpy(), np.array(res_jx.loss), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_pt.cham_x_to_y.numpy(), np.array(res_jx.cham_x_to_y), rtol=1e-5, atol=1e-5)
    
    # Check displacements arrays
    np.testing.assert_allclose(res_pt.disps_x_to_y.numpy(), np.array(res_jx.disps_x_to_y), rtol=1e-5, atol=1e-5)

    print("Success! JAX and PyTorch compute identical Chamfer Distances and Displacements.")
