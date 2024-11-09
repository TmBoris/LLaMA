from torch import nn
from transformers import PreTrainedModel

from .LLaMA import LLaMA
from .LLaMA_config import LLaMA_config


class LLaMA_model(PreTrainedModel):
    config_class = LLaMA_config

    def __init__(self, config):
        super().__init__(config)
        self.model = LLaMA(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            pre_train_seq_len=config.seq_len,
            expected_seq_len=config.expected_seq_len,
            inter_dim=config.inter_dim,
            n_layers=config.n_layers,
            use_xformers=config.use_xformers,
        )

    def forward(self, tensor):
        return self.model(tensor)
