import logging
import random
from typing import List

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):

    def __init__(
        self, index, limit=None
    ):
        """
        Args:
            dataset: instance of hg datasets.dataset
        """
        self._index = index[:limit] if limit is not None else index

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.
        """
        return torch.load(self._index[ind]['text_path'], weights_only=True)

    def __len__(self):
        """
        Get length of the dataset (length of the dataset).
        """
        return len(self._index)
