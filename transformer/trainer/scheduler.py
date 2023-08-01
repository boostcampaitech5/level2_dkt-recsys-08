import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup


def get_scheduler(args):
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            args.lr_optimizer,
            patience=args.patience,
            factor=args.factor,
            mode="max",
            verbose=True,
        )
    elif args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            args.lr_optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_steps,
        )
    return scheduler
