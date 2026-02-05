import torch

def l1_penalty(net):

    l1_loss = 0.0
    for param in net.parameters():
        l1_loss += torch.sum(torch.abs(param))
    
    return(l1_loss)

import torch
import torch.nn as nn

class AdaptiveSlicedWasserstein(nn.Module):
    def __init__(self, N0=50, s=20, max_projections=500, epsilon=0.01, k=2.0, p=2, device='cuda'):
        super().__init__()
        self.N0 = N0
        self.s = s
        self.max_projections = max_projections
        self.epsilon = epsilon
        self.k = k
        self.p = p
        self.device = device
        
        # Pre-generate a large pool of projection directions to avoid 
        # generating random numbers inside the loop (faster)
        # We'll regenerate this pool if we run out, but usually this is enough.
        self.theta_pool_size = max_projections * 10 
        self.theta_pool = None 

    def _get_projections(self, n, dim):
        """Fetches n random projection vectors from the pool or generates new ones."""
        if self.theta_pool is None or self.theta_pool.shape[1] < n:
            self.theta_pool = torch.randn(dim, self.theta_pool_size, device=self.device)
            self.theta_pool /= torch.norm(self.theta_pool, dim=0, keepdim=True)
            self.pool_idx = 0
            
        # Circular buffer usage
        if self.pool_idx + n > self.theta_pool_size:
            self.pool_idx = 0
            
        theta = self.theta_pool[:, self.pool_idx : self.pool_idx + n]
        self.pool_idx += n
        return theta

    def forward(self, pred, gt):
        """
        Args:
            pred: [B, N, D] (Predicted point cloud)
            gt:   [B, N, D] (Ground truth point cloud)
        """
        B, N, D = pred.shape
        print(pred.shape)
        
        # 1. State Tensors (Dense, for all batch items)
        # We store the running sums here.
        val_sum = torch.zeros(B, device=self.device)
        val_sq_sum = torch.zeros(B, device=self.device)
        current_n = torch.zeros(B, device=self.device)
        
        # 2. Active Indices
        # We only compute for indices that appear in this list.
        # Initially, all batch items [0, 1, ..., B-1] are active.
        active_indices = torch.arange(B, device=self.device)
        
        # Loop until all batches are converged or we hit max projections
        total_projections = 0
        
        while total_projections < self.max_projections and active_indices.numel() > 0:
            # Determine step size: N0 for first step, s for subsequent
            n_sample = self.N0 if total_projections == 0 else self.s
            
            # --- OPTIMIZATION: MASKING ---
            # Gather only the ACTIVE batch items. 
            # This drastically reduces compute if half the batch converges early.
            pred_active = pred[active_indices] # [B_active, N, D]
            gt_active = gt[active_indices]     # [B_active, N, D]
            
            # 3. Project and Sort (The Expensive Part)
            theta = self._get_projections(n_sample, D) # [D, n_sample]
            
            # Project: [B_active, N, D] @ [D, n_sample] -> [B_active, N, n_sample]
            # Sorting is O(N log N). Doing this only for active items is the speedup.
            pred_proj = torch.matmul(pred_active, theta)
            gt_proj = torch.matmul(gt_active, theta)
            
            pred_proj_sorted, _ = torch.sort(pred_proj, dim=1)
            gt_proj_sorted, _ = torch.sort(gt_proj, dim=1)
            
            # 4. Compute Distance on Slice
            # L_p distance between sorted projections
            # Result: [B_active, n_sample]
            w_dist = torch.mean(torch.abs(pred_proj_sorted - gt_proj_sorted).pow(self.p), dim=1)
            
            # 5. Update Statistics
            # Sum over the n_sample projections: [B_active]
            batch_sum = w_dist.sum(dim=1)
            batch_sq_sum = w_dist.pow(2).sum(dim=1)
            
            # Scatter updates back to the global state tensors
            val_sum.index_add_(0, active_indices, batch_sum)
            val_sq_sum.index_add_(0, active_indices, batch_sq_sum)
            current_n.index_add_(0, active_indices, torch.full((len(active_indices),), float(n_sample), device=self.device))
            
            total_projections += n_sample
            
            # 6. Check Convergence (Vectorized)
            # We check convergence ONLY for the currently active indices
            # Global mean/var for active items
            c_n = current_n[active_indices]
            mean = val_sum[active_indices] / c_n
            
            # Variance calculation (avoid div by zero for first step)
            # Mask out cases where c_n <= 1 to avoid NaN
            valid_var_mask = c_n > 1
            var = torch.zeros_like(mean)
            
            if valid_var_mask.any():
                # Standard sample variance formula
                var[valid_var_mask] = (val_sq_sum[active_indices][valid_var_mask] - c_n[valid_var_mask] * mean[valid_var_mask].pow(2)) / (c_n[valid_var_mask] - 1)
            
            # Error bound (Confidence Interval)
            std_dev = torch.sqrt(torch.clamp(var, min=0.0))
            error = (self.k * std_dev) / torch.sqrt(c_n)
            
            # Who has NOT converged?
            # Condition: Error > epsilon OR we haven't done enough projections yet (safety)
            not_converged_mask = error > self.epsilon
            
            # Update active_indices to keep only those who failed the check
            active_indices = active_indices[not_converged_mask]

        # Final result calculation for all batch items
        final_means = val_sum / current_n
        return torch.pow(final_means, 1.0 / self.p)

# Example Usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Simulate a training batch
    pred = torch.randn(512, 1024, 3, device=device)
    gt = torch.randn(512, 1024, 3, device=device)
    
    criterion = AdaptiveSlicedWasserstein(device=device)
    
    # Returns [512] scalar distances
    loss_batch = criterion(pred, gt)
    final_loss = loss_batch.mean()
    
    print(f"Loss: {final_loss.item()}")
    print(f"Batch shape: {loss_batch.shape}")