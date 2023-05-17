import os
import math
import numpy as np
import torch
import torch.nn as nn
from base import BaseTrainer
from trainer.optimizer import get_optimizer
from trainer.scheduler import get_scheduler
from utils import get_logger, logging_conf
import model.loss as module_loss
import data_loader.data_loaders as module_data_loader
from model.metric import get_acc, get_auc
import wandb

logger = get_logger(logger_conf=logging_conf)


class LSTMTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__()
        args.dataset = getattr(module_data_loader, args.dataset)
        args.train_loader = args.data_loader.get_loader(
            args, args.dataset, args.train_data, True
        )
        args.valid_loader = args.data_loader.get_loader(
            args, args.dataset, args.valid_data, False
        )

        args.total_steps = int(
            math.ceil(len(args.train_loader.dataset) / args.batch_size)
        ) * (args.n_epochs)
        args.warmup_steps = args.total_steps // 10

        args.activation = getattr(torch, args.activation)
        args.criterion = getattr(module_loss, args.loss)
        args.lr_optimizer = get_optimizer(args)
        args.lr_scheduler = get_scheduler(args)

    def run(self, args):
        best_auc = -1
        early_stopping_counter = 0
        for epoch in range(args.n_epochs):
            logger.info("Start Training: Epoch %s", epoch + 1)

            # TRAIN
            train_auc, train_acc, train_loss = self.train(args)

            # VALID
            valid_auc, valid_acc = self.validate(args)

            # wandb.log(dict(epoch=epoch,
            #                train_loss_epoch=train_loss,
            #                train_auc_epoch=train_auc,
            #                train_acc_epoch=train_acc,
            #                valid_auc_epoch=valid_auc,
            #                valid_acc_epoch=valid_acc))

            if valid_auc > best_auc:
                best_auc = valid_auc
                # nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
                model_to_save = (
                    args.model.module if hasattr(args.model, "module") else args.model
                )
                self.save_checkpoint(
                    args,
                    state={
                        "epoch": epoch + 1,
                        "state_dict": model_to_save.state_dict(),
                    },
                )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    logger.info(
                        "EarlyStopping counter: %s out of %s",
                        early_stopping_counter,
                        args.patience,
                    )
                    break

            # scheduler
            if args.scheduler == "plateau":
                args.lr_scheduler.step(best_auc)

    def train(self, args):
        args.model.train()

        total_preds = []
        total_targets = []
        losses = []
        for step, batch in enumerate(args.train_loader):
            batch = {k: v.to(args.device) for k, v in batch.items()}
            preds = args.model(**batch)
            targets = batch["correct"].to(args.device)

            loss = self.compute_loss(args, targets, preds)
            self.update_params(args, loss)

            if step % args.log_steps == 0:
                logger.info("Training steps: %s Loss: %.4f", step, loss.item())

            # predictions
            preds = args.activation(preds[:, -1])
            targets = targets[:, -1]

            total_preds.append(preds.detach())
            total_targets.append(targets.detach())
            losses.append(loss)

        total_preds = torch.concat(total_preds).cpu().numpy()
        total_targets = torch.concat(total_targets).cpu().numpy()

        # Train AUC / ACC
        auc = get_auc(total_targets, total_preds)
        acc = get_acc(total_targets, total_preds)
        loss_avg = sum(losses) / len(losses)
        logger.info("TRAIN AUC : %.4f ACC : %.4f", auc, acc)
        return auc, acc, loss_avg

    def validate(self, args):
        args.model.eval()

        total_preds = []
        total_targets = []
        with torch.no_grad():
            for step, batch in enumerate(args.valid_loader):
                batch = {k: v.to(args.device) for k, v in batch.items()}
                preds = args.model(**batch)
                targets = batch["correct"].to(args.device)

                # predictions
                preds = args.activation(preds[:, -1])
                targets = targets[:, -1]

                total_preds.append(preds.detach())
                total_targets.append(targets.detach())

        total_preds = torch.concat(total_preds).cpu().numpy()
        total_targets = torch.concat(total_targets).cpu().numpy()

        # Train AUC / ACC
        auc = get_auc(total_targets, total_preds)
        acc = get_acc(total_targets, total_preds)
        logger.info("VALID AUC : %.4f ACC : %.4f", auc, acc)
        return auc, acc

    def compute_loss(self, args, targets, preds):
        loss = args.criterion(targets, preds)
        return loss

    def update_params(self, args, loss):
        args.lr_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(args.model.parameters(), args.clip_grad)
        if args.scheduler == "linear_warmup":
            args.lr_scheduler.step()
        args.lr_optimizer.step()

    def save_checkpoint(self, args, state: dict) -> None:
        """Saves checkpoint to a given directory."""
        save_path = os.path.join(args.save_dir, args.model_name)
        logger.info("saving model as %s...", save_path)
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(state, save_path)
