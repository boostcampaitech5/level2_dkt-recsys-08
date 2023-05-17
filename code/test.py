import os
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data_loader
import model.model as module_model
from utils import parse_args, get_logger, logging_conf

logger = get_logger(logging_conf)


def main(args):
    logger.info("Preparing data ...")
    args.shuffle = False
    args.split_ratio = 0.0
    data_loader = getattr(module_data_loader, args.data_loader)(args, False)
    test_data = data_loader.data
    if args.dataset != "None":
        dataset = getattr(module_data_loader, args.dataset)
        test_data_loader = data_loader.get_loader(args, dataset, test_data, False)

    logger.info("Building Model ...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = getattr(module_model, args.model)(args).to(device)
    save_path = os.path.join(args.save_dir, args.model_name)
    model.load_state_dict(torch.load(save_path)["state_dict"])
    model.eval()

    logger.info("Model Predict ...")
    activation = getattr(torch, args.activation)
    total_outputs = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_data_loader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            output = activation(output[:, -1])
            total_outputs += list(output.cpu().detach().numpy())
    write_path = os.path.join(args.submit_dir, "submission.csv")
    os.makedirs(name=args.submit_dir, exist_ok=True)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_outputs):
            w.write("{},{}\n".format(id, p))
    logger.info("Successfully saved submission as %s", write_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
