import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def get_acc(target, output, threshold=0.5):
    return accuracy_score(y_true=target, y_pred=np.where(output >= threshold, 1, 0))


def get_auc(target, output):
    return roc_auc_score(y_true=target, y_score=output)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
