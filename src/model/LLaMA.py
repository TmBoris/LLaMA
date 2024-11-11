import torch
from torch import nn

from .llama_block import LlamaBlock


class ModelAnswer:
  def __init__(self, logits):
    self.logits = logits

  def __getattr__(self, name):
    if name == 'logits':
      return self.logits
    else:
      raise AttributeError(f"Attribute '{name}' not found")

  def __getitem__(self, key):
    if key == 'logits':
      return self.logits
    else:
      raise KeyError(f"Key '{key}' not found")


class LLaMA(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_heads,
        n_ropes,
        rope_coef,
        inter_dim,
        n_layers,
        max_seq_len,
        use_xformers,
    ):
        super().__init__()

        self.embeds = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList(
            [
                LlamaBlock(
                    n_heads, d_model, inter_dim, n_ropes, use_xformers, rope_coef
                )
                for _ in range(n_layers)
            ]
        )

        self.rms = nn.RMSNorm([d_model])
        self.linear = nn.Linear(d_model, vocab_size)

        print("model initialized")

    def forward(self, texts, **batch):
        # b, length = texts.shape

        x = self.embeds(texts[:, :-1])  # cut eos

        for block in self.blocks:
            x = block(x)

        x = self.linear(self.rms(x))

        return ModelAnswer(x)

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
