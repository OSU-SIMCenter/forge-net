import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class SingleStepMeshTransitionDataset(Dataset):
    def __init__(self, coords_t, coords_tp1, actions, delta_scalar=100, cell_type=None, point_cloud_noise_std=0.0,
                 temp_t=None, temp_tp1=None):
        """
        coords_t:   numpy array [num_samples, N_points, 3]
        coords_tp1: numpy array [num_samples, N_points, 3]
        actions:    numpy array [num_samples, action_dims]
        delta_scalar: int to scale the deltas by for numeric stability of gradients
        cell_type: optional numpy array [num_samples] of strings ('tetra'/'triangle'/
            'point_cloud', see process_data_forge_common.py) -- when given together
            with point_cloud_noise_std > 0, only samples tagged 'point_cloud' get the
            noise augmentation below. `None` (default) means no sample is noised,
            regardless of point_cloud_noise_std -- both must be set to opt in.
        point_cloud_noise_std: std (mm) of Gaussian jitter applied to point_cloud
            samples, applied PER-EPOCH here in __getitem__ (not baked into the
            dataset once) so it's genuine augmentation, not a fixed perturbation
            the model could just as easily overfit to. Critically, the SAME noise
            draw is added to x_t AND x_tp1 (not independent draws) -- that
            perturbs what the model sees as input while leaving delta_t =
            x_tp1 - x_t (the actual training label) exactly equal to the true
            physical displacement. Independent per-frame noise would instead
            corrupt delta_t itself with a random (noise_tp1 - noise_t) term,
            actively teaching the wrong dynamics -- do not do that. Point_cloud
            samples' raw particle positions are also the only ones in this
            dataset sampled on a regular MPM lattice (see genesis_forge_adapter.py)
            rather than randomized within a mesh cell; this jitter also breaks
            that grid-alignment tell so the network can't shortcut "which source
            is this" off point regularity.
        temp_t, temp_tp1: optional numpy arrays [num_samples, N_points], degrees C
            (see process_data_forge_common.py's `temp_t`/`temp_tp1`, all-zero
            placeholder where no real temperature was recorded). When given,
            `temp_t` is CONCATENATED onto `x_t` as a 4th input channel (the model
            needs to know each point's CURRENT temperature to predict how it
            changes) and `delta_temp = temp_tp1 - temp_t` (UNSCALED -- unlike the
            position delta, no `delta_scalar` multiply; temperature deltas are
            already O(1-100) in degrees C, not sub-mm) is returned as a 5th
            element for `ForgeNetTrainer`'s separate temperature loss head. Both
            `None` (default) means `x_t` stays xyz-only (3 channels) and no temp
            delta is returned, for full backward compatibility with mechanical-
            only datasets/older configs.

        Each sample is:
            (coords_t[i], actions[i]) -> coords_tp1[i]
        """
        assert len(coords_t) == len(coords_tp1) == len(actions), \
            "coords_t, coords_tp1, and actions must have same length"
        if cell_type is not None:
            assert len(cell_type) == len(coords_t), "cell_type must have same length as coords_t"
        if (temp_t is None) != (temp_tp1 is None):
            raise ValueError("temp_t and temp_tp1 must be given together (both or neither)")

        self.coords_t = coords_t
        self.coords_tp1 = coords_tp1
        self.actions = actions
        self.delta_scalar = delta_scalar
        self.cell_type = cell_type
        self.point_cloud_noise_std = point_cloud_noise_std
        self.temp_t = temp_t
        self.temp_tp1 = temp_tp1

    def __len__(self):
        return len(self.coords_t)

    def __getitem__(self, idx):
        x_t = torch.from_numpy(self.coords_t[idx]).float()
        x_tp1 = torch.from_numpy(self.coords_tp1[idx]).float()

        if (self.cell_type is not None and self.point_cloud_noise_std > 0
                and self.cell_type[idx] == "point_cloud"):
            noise = torch.from_numpy(
                np.random.normal(0.0, self.point_cloud_noise_std, size=x_t.shape)
            ).float()
            x_t = x_t + noise
            x_tp1 = x_tp1 + noise  # SAME draw as x_t -- see class docstring

        delta_t = self.delta_scalar * (x_tp1 - x_t)
        a = torch.from_numpy(self.actions[idx]).float()

        if self.temp_t is None:
            return x_t, a.unsqueeze(0), delta_t.unsqueeze(0), x_tp1.unsqueeze(0)

        temp_t = torch.from_numpy(self.temp_t[idx]).float()
        temp_tp1 = torch.from_numpy(self.temp_tp1[idx]).float()
        delta_temp = temp_tp1 - temp_t  # UNSCALED, see docstring
        x_t_aug = torch.cat([x_t, temp_t.unsqueeze(-1)], dim=-1)  # (N, 4)

        return x_t_aug, a.unsqueeze(0), delta_t.unsqueeze(0), x_tp1.unsqueeze(0), delta_temp.unsqueeze(0)

def RandomSplit(dataset, train_set_percentage):
    n_train = int(len(dataset) * train_set_percentage)
    lengths = [n_train, len(dataset) - n_train]
    return random_split(dataset, lengths)


def GetSingleStepDataLoaders(
    coords_t,
    coords_tp1,
    actions,
    batch_size,
    train_set_percentage=0.9,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    cell_type=None,
    point_cloud_noise_std=0.0,
    temp_t=None,
    temp_tp1=None,
    delta_scalar=100,
    ):

    dataset = SingleStepMeshTransitionDataset(
        coords_t, coords_tp1, actions, delta_scalar=delta_scalar,
        cell_type=cell_type, point_cloud_noise_std=point_cloud_noise_std,
        temp_t=temp_t, temp_tp1=temp_tp1,
    )

    train_set, test_set = RandomSplit(dataset, train_set_percentage)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
        )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
        )

    return train_loader, test_loader
