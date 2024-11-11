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
            n_ropes=config.n_ropes,
            rope_coef=config.rope_coef,
            inter_dim=config.inter_dim,
            n_layers=config.n_layers,
            max_seq_len=config.max_seq_len,
            use_xformers=config.use_xformers,
        )

    def forward(self, tensor):
        return self.model(tensor)
