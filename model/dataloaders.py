import torch
from torch.utils.data import Dataset, DataLoader, random_split


class SingleStepMeshTransitionDataset(Dataset):
    def __init__(self, coords_t, coords_tp1, actions):
        """
        coords_t:   numpy array [num_samples, N_points, 3]
        coords_tp1: numpy array [num_samples, N_points, 3]
        actions:    numpy array [num_samples, action_dim]

        Each sample is:
            (coords_t[i], actions[i]) -> coords_tp1[i]
        """
        assert len(coords_t) == len(coords_tp1) == len(actions), \
            "coords_t, coords_tp1, and actions must have same length"

        self.coords_t = coords_t
        self.coords_tp1 = coords_tp1
        self.actions = actions

    def __len__(self):
        return len(self.coords_t)

    def __getitem__(self, idx):
        x_t = torch.from_numpy(self.coords_t[idx]).float()
        x_tp1 = torch.from_numpy(self.coords_tp1[idx]).float()
        delta_t = (x_tp1 - x_t) * 100
        a = torch.from_numpy(self.actions[idx]).float()

        return x_t, a.unsqueeze(0), delta_t.unsqueeze(0), x_tp1.unsqueeze(0)

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
    pin_memory=True
    ):

    dataset = SingleStepMeshTransitionDataset(coords_t, coords_tp1, actions)

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