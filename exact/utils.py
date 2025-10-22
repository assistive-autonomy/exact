import ot
import torch


def wasserstein_distance(X: torch.Tensor, Y: torch.Tensor) -> float:
    """wasserstein_distance computes the 2-Wasserstein distance between two point clouds"""
    X_norm = X.pow(2).sum(-1).reshape(-1, 1)
    Y_norm = Y.pow(2).sum(-1).reshape(1, -1)
    val = X_norm + Y_norm - 2 * torch.matmul(X, Y.T)
    # clamp is needed to avoid negative values due to numerical errors
    M = torch.sqrt(torch.clamp(val, min=0))

    fst_pot = torch.ones(X.shape[0]) / X.shape[0]
    snd_pot = torch.ones(Y.shape[0]) / Y.shape[0]
    return ot.emd2(fst_pot, snd_pot, M)

