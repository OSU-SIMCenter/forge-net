from typing import Union, Tuple, Optional
import torch
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class ChamferResult:
    loss:          torch.Tensor        # weighted combined scalar
    cham_x_to_y:   torch.Tensor        # x → y direction (unreduced or reduced per batch_reduction)
    cham_y_to_x:   torch.Tensor        # y → x direction (coverage penalty term)
    loss_normals:  Optional[torch.Tensor] = None
    disps_x_to_y:  Optional[torch.Tensor] = None  # (B, N, D) per-point displacements
    disps_y_to_x:  Optional[torch.Tensor] = None

#PyTorch3d Cahmfer distance implementation converted by Claude Sonnet 4.6
def knn_points_simple(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: Optional[torch.Tensor] = None,
    lengths2: Optional[torch.Tensor] = None,
    norm: int = 2,
    K: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Naive pure-PyTorch KNN returning SQUARED distances (matching pytorch3d behaviour).

    Args:
        p1: (N, P1, D)
        p2: (N, P2, D)
        lengths1: (N,) number of valid points in p1, or None (all valid)
        lengths2: (N,) number of valid points in p2, or None (all valid)
        norm: 1 for L1, 2 for L2
        K: number of nearest neighbours

    Returns:
        dists: (N, P1, K) squared (norm=2) or absolute (norm=1) distances
        idx:   (N, P1, K) indices into p2
    """
    N, P1, D = p1.shape
    P2 = p2.shape[1]

    if lengths1 is None:
        lengths1 = torch.full((N,), P1, dtype=torch.int64, device=p1.device)
    if lengths2 is None:
        lengths2 = torch.full((N,), P2, dtype=torch.int64, device=p1.device)

    # Compute pairwise distances: (N, P1, P2)
    if norm == 2:
        # Squared L2 via expand — memory-efficient vs cdist for large clouds
        diff = p1[:, :, None, :] - p2[:, None, :, :]   # (N, P1, P2, D)
        dist_mat = (diff ** 2).sum(-1)                   # (N, P1, P2)  — squared L2
    elif norm == 1:
        diff = p1[:, :, None, :] - p2[:, None, :, :]
        dist_mat = diff.abs().sum(-1)                    # (N, P1, P2)  — L1
    else:
        raise ValueError("norm must be 1 or 2")

    # Mask out padding in p2 so padded points are never selected as NN
    if lengths2.min() < P2:
        mask = torch.arange(P2, device=p2.device)[None] >= lengths2[:, None]  # (N, P2)
        dist_mat[:, :, :][mask[:, None, :].expand_as(dist_mat)] = float("inf")

    # Take K nearest
    dists, idx = dist_mat.topk(K, dim=2, largest=False, sorted=True)  # (N, P1, K)

    # Zero-out distances for padding positions in p1
    if lengths1.min() < P1:
        p1_mask = torch.arange(P1, device=p1.device)[None] >= lengths1[:, None]  # (N, P1)
        dists[p1_mask] = 0.0

    return dists, idx


def knn_gather(
    x: torch.Tensor,
    idx: torch.Tensor,
    lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Gather features from x using knn indices.

    Args:
        x:   (N, M, U)
        idx: (N, L, K)
        lengths: (N,) valid lengths in x, or None

    Returns:
        (N, L, K, U)
    """
    N, M, U = x.shape
    _N, L, K = idx.shape

    if lengths is None:
        lengths = torch.full((N,), M, dtype=torch.int64, device=x.device)

    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, U)          # (N, L, K, U)
    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)

    if lengths.min() < K:
        mask = lengths[:, None] <= torch.arange(K, device=x.device)[None]  # (N, K)
        mask = mask[:, None, :, None].expand(-1, L, -1, U)
        x_out[mask] = 0.0

    return x_out


def _chamfer_single_direction(
    x: torch.Tensor,
    y: torch.Tensor,
    x_lengths: torch.Tensor,
    y_lengths: torch.Tensor,
    x_normals: Optional[torch.Tensor],
    y_normals: Optional[torch.Tensor],
    weights: Optional[torch.Tensor],
    point_reduction: Optional[str],
    norm: int,
    abs_cosine: bool,
    return_displacements: bool
):
    return_normals = x_normals is not None and y_normals is not None
    N, P1, D = x.shape

    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]  # (N, P1)

    dists, idx = knn_points_simple(x, y, x_lengths, y_lengths, norm=norm, K=1)
    cham_x = dists[..., 0]

    disps = None

    if return_displacements:
        nearest_y = y.gather(1, idx.expand(-1, -1, D))       # (B, N, D)
        disps = nearest_y - x 

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)

    cham_norm_x = x.new_zeros(())
    if return_normals:
        x_normals_near = knn_gather(y_normals, idx, y_lengths)[..., 0, :]  # (N, P1, D)
        cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if weights is not None:
            cham_norm_x *= weights.view(N, 1)

    if point_reduction == "max":
        assert not return_normals
        cham_x = cham_x.max(1).values                  # (N,)
    elif point_reduction is not None:
        cham_x = cham_x.sum(1)                          # (N,)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x /= x_lengths_clamped
            if return_normals:
                cham_norm_x /= x_lengths_clamped

    return cham_x, cham_norm_x if return_normals else None, disps if return_displacements else None


def _apply_batch_reduction(cham_x, cham_norm_x, weights, batch_reduction):
    if batch_reduction is None:
        return cham_x, cham_norm_x
    N = cham_x.shape[0]
    cham_x = cham_x.sum()
    if cham_norm_x is not None:
        cham_norm_x = cham_norm_x.sum()
    if batch_reduction == "mean":
        div = max(N, 1) if weights is None else max(weights.sum().item(), 1e-8)
        cham_x /= div
        if cham_norm_x is not None:
            cham_norm_x /= div
    return cham_x, cham_norm_x


def chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    x_lengths: Optional[torch.Tensor] = None,
    y_lengths: Optional[torch.Tensor] = None,
    x_normals: Optional[torch.Tensor] = None,
    y_normals: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    lambda_x_to_y: float = 1.0,
    lambda_y_to_x: float = 1.0,
    batch_reduction: Optional[str] = "mean",
    point_reduction: Optional[str] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
    return_displacements: bool = False,
) -> ChamferResult:
    """
    Chamfer distance. Distances are SQUARED for norm=2.

    Args:
        x:               (B, P1, D)
        y:               (B, P2, D)
        lambda_x_to_y:   weight on x→y term (accuracy)
        lambda_y_to_x:   weight on y→x term (coverage) — set > 1 to penalise over-compression
        return_displacements: if True, populates disps_x_to_y and disps_y_to_x in result

    Returns:
        ChamferResult with fields: loss, cham_x_to_y, cham_y_to_x,
                                   loss_normals, disps_x_to_y, disps_y_to_x
    """
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
        x_lengths = torch.full((N,), P1, dtype=torch.int64, device=x.device)
    if y_lengths is None:
        y_lengths = torch.full((N,), P2, dtype=torch.int64, device=y.device)

    # ── x → y ────────────────────────────────────────────────────────────────
    cham_x, cham_norm_x, disps_x_to_y = _chamfer_single_direction(
        x, y, x_lengths, y_lengths, x_normals, y_normals,
        weights, point_reduction, norm, abs_cosine, return_displacements
    )

    if single_directional:
        reduced, reduced_normals = _apply_batch_reduction(
            cham_x, cham_norm_x, weights, batch_reduction
        )
        return ChamferResult(
            loss=reduced,
            cham_x_to_y=reduced,
            cham_y_to_x=None,
            loss_normals=reduced_normals,
            disps_x_to_y=disps_x_to_y,
        )

    # ── y → x ────────────────────────────────────────────────────────────────
    cham_y, cham_norm_y, disps_y_to_x = _chamfer_single_direction(
        y, x, y_lengths, x_lengths, y_normals, x_normals,
        weights, point_reduction, norm, abs_cosine, return_displacements
    )

    # ── apply asymmetric weights ──────────────────────────────────────────────
    cham_x_w = cham_x * lambda_x_to_y
    cham_y_w = cham_y * lambda_y_to_x

    # ── combine ───────────────────────────────────────────────────────────────
    if point_reduction == "max":
        loss = torch.maximum(cham_x_w, cham_y_w)
        loss_normals = None
    elif point_reduction is not None:
        loss = cham_x_w + cham_y_w
        loss_normals = (cham_norm_x + cham_norm_y) if cham_norm_x is not None else None
    else:
        loss = (cham_x_w, cham_y_w)
        loss_normals = (cham_norm_x, cham_norm_y) if cham_norm_x is not None else None

    loss, loss_normals = _apply_batch_reduction(loss, loss_normals, weights, batch_reduction)
    cham_x_reduced, _ = _apply_batch_reduction(cham_x, None, weights, batch_reduction)
    cham_y_reduced, _ = _apply_batch_reduction(cham_y, None, weights, batch_reduction)

    return ChamferResult(
        loss=loss,
        cham_x_to_y=cham_x_reduced,   # unweighted, for logging
        cham_y_to_x=cham_y_reduced,   # unweighted, for logging
        loss_normals=loss_normals,
        disps_x_to_y=disps_x_to_y,
        disps_y_to_x=disps_y_to_x,
    )
