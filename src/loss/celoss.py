import torch
from torch import nn


class CELoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self, vocab_size, seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, texts: torch.Tensor, **batch):
        """
        Args:
            logits (Tensor): model output predictions.
            texts (Tensor): ground-truth labels.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """

        return {
            "loss": self.loss(torch.argmax(logits.view(logits.shape[0], self.seq_len, self.vocab_size), dim=-1), texts[:, 1:])
        }
