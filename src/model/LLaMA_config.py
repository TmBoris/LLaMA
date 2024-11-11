from typing import List

from transformers import PretrainedConfig


class LLaMA_config(PretrainedConfig):
    model_type = "llama"

    def __init__(
        self,
        vocab_size=32000,
        d_model=768,
        n_heads=16,
        n_ropes=3300,
        rope_coef=0.25,
        inter_dim=1024,
        n_layers=16,
        max_seq_len=1024,
        use_xformers=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_ropes = n_ropes
        self.rope_coef = rope_coef
        self.inter_dim = inter_dim
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.use_xformers = use_xformers

        super().__init__(**kwargs)
