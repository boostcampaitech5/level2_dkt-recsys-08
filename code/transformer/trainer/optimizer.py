import torch
from torch.optim import Adam, AdamW


def get_optimizer(args):
    if args.optimizer == "adam":
        optimizer = Adam(
            args.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    elif args.optimizer == "adamW":
        optimizer = AdamW(
            args.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    # 모든 parameter들의 grad값을 0으로 초기화
    return optimizer
