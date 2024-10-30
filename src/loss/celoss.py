import torch
import torch.nn.functional as F
from torch import nn


class CELoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self, vocab_size, seq_len):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, logits: torch.Tensor, texts: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): model output predictions.
            texts (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """

        return {
            "loss": F.cross_entropy(
                logits.view(-1, self.vocab_size), texts[:, 1:].flatten()
            )
        }
