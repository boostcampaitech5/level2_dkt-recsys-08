import os
import tqdm
import numpy as np
from torch import nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.optim import SGD, Adam
from torch.nn import MSELoss
from torch.nn.functional import sigmoid
import torch


def get_metric(targets, preds):
    auc = roc_auc_score(y_true=targets, y_score=preds)
    acc = accuracy_score(y_true=targets, y_pred=np.where(preds >= 0.5, 1, 0))
    return auc, acc


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y) + self.eps)
        return loss


def train(model, dataloader):
    minimum_loss = 999999999
    loss_fn = RMSELoss()
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=0.001)

    for epoch in tqdm.tqdm(range(1)):
        model.train()
        total_loss = 0
        batch = 0
        total_preds = []
        total_targets = []
        for idx, data in enumerate(dataloader["train_dataloader"]):
            x, y = data[0].to("cuda"), data[1].to("cuda")
            y_hat = model(x)
            loss = loss_fn(y.float(), y_hat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch += 1
            y_hat = torch.sigmoid(y_hat)
            total_preds.append(y_hat.detach())
            total_targets.append(y.detach())
        total_preds = torch.concat(total_preds).cpu().numpy()
        total_targets = torch.concat(total_targets).cpu().numpy()
        auc, acc = get_metric(targets=total_targets, preds=total_preds)
        print(f"Train AUC: {auc} ACC: {acc}")
        valid_loss = valid(model, dataloader, loss_fn)
        print(
            f"Epoch: {epoch+1}, Train_loss: {total_loss/batch:.3f}, valid_loss: {valid_loss:.3f}"
        )
        if minimum_loss > valid_loss:
            minimum_loss = valid_loss
            saved_model_path = "./saved_models"
            os.makedirs(saved_model_path, exist_ok=True)
            model_name = "FFM"
            torch.save(model.state_dict(), f"{saved_model_path}/{model_name}_model.pt")
    return model, valid_loss


def valid(model, dataloader, loss_fn):
    model.eval()
    total_loss = 0
    batch = 0
    total_preds = []
    total_targets = []
    for idx, data in enumerate(dataloader["valid_dataloader"]):
        x, y = data[0].to("cuda"), data[1].to("cuda")
        y_hat = model(x)
        loss = loss_fn(y.float(), y_hat)
        total_loss += loss.item()
        batch += 1
        y_hat = torch.sigmoid(y_hat)
        total_preds.append(y_hat.detach())
        total_targets.append(y.detach())
    valid_loss = total_loss / batch
    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()
    auc, acc = get_metric(targets=total_targets, preds=total_preds)
    print(f"VALID AUC: {auc} ACC: {acc}")
    return valid_loss


def test(model, dataloader, use_best_model):
    predicts = list()
    if use_best_model == True:
        model.load_state_dict(torch.load(f"./saved_models/FFM_model.pt"))
    model.eval()

    for idx, data in enumerate(dataloader["test_dataloader"]):
        x = data[0].to("cuda")
        y_hat = model(x)
        list_ = y_hat.tolist()
        predicts.extend(list_)
    return predicts
