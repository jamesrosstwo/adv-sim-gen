import torch


def get_mask(obs: torch.Tensor) -> torch.Tensor:
    """
    Obtains a binary mask of editable pixels given an observation
    :param obs: batch observation. Shape: [batch_size, 3, 96, 96]
    :return: Binary mask of shape [batch_size, 96, 96]
    """
    base_mask = torch.ones_like(obs[:, 0, :, :]).bool()
    base_mask[:, 84:, :] = 0
    return base_mask