import json
import os

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from datasets import load_dataset, load_from_disk
from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH

#  huggingface-cli login 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


class Dataset(BaseDataset):
    def __init__(self, seq_len, dir, n_save, *args, **kwargs):
        self.seq_len = seq_len
        self.n_save = n_save
        data_dir = ROOT_PATH / "data" / "datasets" / dir
        self._data_dir = data_dir
        self.tokenizer = torch.load("data/tokenizer/mistral_tokenizer.pt")

        dataset = self._get_or_load_index("train")
        super().__init__(dataset, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._data_dir / f"{part}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)

        return index

    def _create_index(self, part):
        print("Loading dataset...")
        dataset = load_dataset("ashaba1in/small_openwebtext")["train"]
        if self.n_save is not None:
            dataset = dataset.select(range(self.n_save))

        save_dir = self._data_dir / part
        save_dir.mkdir(parents=True, exist_ok=True)
        absolut_i = 0
        buffer = []
        for sample in tqdm(dataset, desc="tokenizing + saving", total=len(dataset)):
            tok_text = self.tokenizer(sample["text"])["input_ids"]
            i = 0
            while (start := i * self.seq_len) + self.seq_len <= len(tok_text):
                torch.save(
                    torch.tensor(tok_text[start : start + self.seq_len]).to(
                        torch.int16
                    ),
                    f"{save_dir}/text_{absolut_i}.pt",
                )
                i += 1
                absolut_i += 1

            assert len(tok_text) - start < self.seq_len

            buffer.extend(tok_text[start:])
            if len(buffer) >= self.seq_len:
                torch.save(
                    torch.tensor(buffer[: self.seq_len]).to(torch.int16),
                    f"{save_dir}/text_{absolut_i}.pt",
                )
                absolut_i += 1
                buffer = buffer[self.seq_len :]

        index = []
        for text_path in os.listdir(save_dir):
            index.append({"text_path": str(save_dir / text_path)})
        return index

    # def _tokenize_function(self, sample):
    #     return tokenizer(sample["text"])
