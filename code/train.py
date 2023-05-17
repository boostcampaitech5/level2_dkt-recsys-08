import torch
import data_loader.data_loaders as module_data_loader
import model.model as module_model
import trainer.trainer as module_trainer
from utils import parse_args, set_seeds, get_logger, logging_conf
import wandb

logger = get_logger(logging_conf)


def main(args):
    # wandb.login()
    # wandb.init(project="dkt", config=vars(args))

    set_seeds(args.seed)

    logger.info("Preparing data ...")
    args.data_loader = getattr(module_data_loader, args.data_loader)(args)
    args.train_data, args.valid_data = args.data_loader.split_data(args)

    logger.info("Building Model ...")
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model = getattr(module_model, args.model)(args).to(args.device)

    logger.info("Start Training ...")
    trainer = getattr(module_trainer, args.trainer)(args)
    trainer.run(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
