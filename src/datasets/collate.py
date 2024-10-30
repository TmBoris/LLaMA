import torch
from torch import tensor
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict], expected_seq_len):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
        expected_seq_len (int): max context length of the llm
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    result_batch["lengths"] = torch.tensor([len(x) for x in dataset_items])
    result_batch["texts"] = pad_sequence(
        [
            torch.cat(
                [
                    torch.cat((tensor([1]), t, tensor([2])))
                    for t in x.view(-1, expected_seq_len - 1)
                ]
            )
            for x in dataset_items
        ],
        batch_first=True,
    ).squeeze(0)

    # print('result_batch["texts"].shape', result_batch["texts"].shape)

    return result_batch
