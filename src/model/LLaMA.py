import torch
from torch import nn

from .llama_layers.llama_block import LlamaBlock


class LLaMA(nn.Module):
    def __init__(
        self, vocab_size, d_model, n_heads, seq_len, inter_dim, n_layers, device
    ):
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.embeds = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList(
            [
                LlamaBlock(n_heads, d_model, seq_len, inter_dim, device)
                for i in range(n_layers)
            ]
        )

        self.rms = nn.RMSNorm([seq_len, d_model])
        self.linear = nn.Linear(d_model, vocab_size)

        print("model initialized")

    def forward(self, texts, **batch):
        # b, length = texts.shape

        x = self.embeds(texts[:, :-1])  # cut eos

        for block in self.blocks:
            x = block(x)

        x = self.linear(self.rms(x))

        return {"logits": x}

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
