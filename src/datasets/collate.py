import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch['lengths'] = torch.tensor([len(x) for x in dataset_items])
    result_batch['texts'] = pad_sequence(
        [torch.tensor([1] + x + [2]) for x in dataset_items], batch_first=True
    )

    return result_batch
