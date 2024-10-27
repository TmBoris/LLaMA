import json
import os
import torch

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH
from datasets import load_dataset
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm

#  huggingface-cli login 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
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')


class Dataset(BaseDataset):
    def __init__(self, data_dir=None, *args, **kwargs):
        if data_dir is None:
            data_dir = ROOT_PATH / "data" / "datasets" / "extrasmall_openwebtext_tok"
        self._data_dir = data_dir
        dataset = self._get_or_load_index('train')
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
        dataset = load_dataset("ashaba1in/small_openwebtext")
        dataset_tok = dataset.map(self._tokenize_function, batched=True)

        save_dir = self._data_dir / part
        absolut_i = 0
        for tok_text in dataset_tok['input_ids']:
            i = 0
            while (start := i * 256) + 256 <= len(tok_text):
                torch.save(tok_text[start: start + 256], f'{save_dir}/text_{absolut_i}.pt')
                i += 1
                absolut_i += 1
            
            assert len(tok_text) - start < 256

            torch.save(tok_text[start: ], f'{save_dir}/text_{absolut_i}.pt')
            absolut_i += 1


        index = []
        for text_path in os.listdir(save_dir):
            index.append(
                {
                    "text_path": str(save_dir / text_path)
                }
            )
        return index
    
    def _tokenize_function(sample):
            return tokenizer(sample)
