from typing import List

from transformers import PretrainedConfig


class LLaMA_config(PretrainedConfig):
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,
        d_model=768,
        n_heads=16,
        seq_len=256, # train seq_len
        expected_seq_len=1024, # set to 1024 for ft
        inter_dim=1024,
        n_layers=16,
        use_xformers=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.seq_len = seq_len
        self.expected_seq_len = expected_seq_len
        self.inter_dim = inter_dim
        self.n_layers = n_layers
        self.use_xformers = use_xformers

        super().__init__(**kwargs)
