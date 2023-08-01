import random
import numpy as np
import torch
from data import context_data_load, last_solved_data_split, context_data_loader
from DeepFM import DEEP_FM_Model
from train import train, test


def main():
    data = context_data_load()
    data = last_solved_data_split(data)
    data = context_data_loader(data)

    model = DEEP_FM_Model(data, 16, (16, 16)).to("cuda")
    model = train(model, data)[0]
    predicts = test(model, data, True)
    submission = data["sub"]
    submission["prediction"] = predicts
    submission.to_csv("DeepFM.csv", index=False)


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
    main()
