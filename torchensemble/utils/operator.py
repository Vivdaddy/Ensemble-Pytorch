"""This module collects operations on tensors used in Ensemble-PyTorch."""


import torch
import torch.nn.functional as F

__all__ = [
    "average",
    "averaged_std"
    "sum_with_multiplicative",
    "onehot_encoding",
    "pseudo_residual_classification",
    "pseudo_residual_regression",
]


def average(outputs):
    """Compute the average over a list of tensors with the same size."""
    return sum(outputs) / len(outputs)


def averaged_std(outputs):
    """
    Computes averaged standard deviation across list of tensors with the similar size
    Step 1: Std across members of ensemble
    Step 2: Average Across all datapoints
    Step 3: Average Across all item
    """
    #print("Outputs ", outputs)
    outputs = torch.stack(outputs)
    #print("Stacked outputs ", outputs)
    step1 = torch.std(outputs, dim=0)
    step2 = torch.mean(step1, dim=0)
    step3 = torch.mean(step2, dim=0)
    #print("step 3 is ", step3)
    return step3.tolist()


def sum_with_multiplicative(outputs, factor):
    """
    Compuate the summation on a list of tensors, and the result is multiplied
    by a multiplicative factor.
    """
    return factor * sum(outputs)


def onehot_encoding(label, n_classes):
    """Conduct one-hot encoding on a label vector."""
    label = label.view(-1)
    onehot = torch.zeros(label.size(0), n_classes).float().to(label.device)
    onehot.scatter_(1, label.view(-1, 1), 1)

    return onehot


def pseudo_residual_classification(target, output, n_classes):
    """
    Compute the pseudo residual for classification with cross-entropyloss."""
    y_onehot = onehot_encoding(target, n_classes)

    return y_onehot - F.softmax(output, dim=1)


def pseudo_residual_regression(target, output):
    """Compute the pseudo residual for regression with least square error."""
    if target.size() != output.size():
        msg = "The shape of target {} should be the same as output {}."
        raise ValueError(msg.format(target.size(), output.size()))

    return target - output
