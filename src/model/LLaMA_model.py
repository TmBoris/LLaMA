from torch import nn
from transformers import PreTrainedModel

from .LLaMA import LLaMA
from .LLaMA_config import LLaMA_config


class LLaMAModel(PreTrainedModel):
    config_class = LLaMA_config

    def __init__(self, config):
        super().__init__(config)
        self.model = LLaMA()

    def forward(self, tensor):
        return self.model(tensor)
