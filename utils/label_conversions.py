import torch


def convert_multiclass_to_binary_labels_torch(multiclass_labels):
    """
    Converts multiclass segmentation labels into a binary mask.
    """
    binary_labels = torch.zeros_like(multiclass_labels)
    binary_labels[multiclass_labels != 0] = 1

    return binary_labels