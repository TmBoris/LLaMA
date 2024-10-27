import torch
from torch import nn


class CELoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, texts: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): model output predictions.
            texts (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        target = torch.nn.functional.one_hot(texts[:, 1:].long(), num_classes=32000).transpose(-1, -2).to(torch.float32)

        return {
            "loss": self.loss(logits.view(logits.shape[0], 32000, 257), target)
        }
