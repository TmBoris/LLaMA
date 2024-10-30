import logging
import random
from typing import List

import torch
from torch.utils.data import Dataset
from src.utils.io_utils import ROOT_PATH
from datasets import load_dataset
from datasets import load_from_disk
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class EasyDataset(Dataset):

    def __init__(
        self, limit=None
    ):
        """
        Args:
            dataset: instance of hg datasets.dataset
        """
        data_dir = ROOT_PATH / "data" / "datasets" / "small_openwebtext"
        if data_dir.exists():
            self.dataset = load_from_disk('data/datasets/small_openwebtext')
        else:
            self.dataset = load_dataset("ashaba1in/small_openwebtext")
            self.dataset.save_to_disk('data/datasets/small_openwebtext')
        self.dataset = self.dataset['train']
        self.limit = limit if limit is not None else len(self.dataset)
        self.tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')



    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.
        """
        return self.tokenizer(self.dataset[ind]['text'])['input_ids']

    def __len__(self):
        """
        Get length of the dataset (length of the dataset).
        """
        return self.limit
