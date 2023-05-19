import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")
    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")
    parser.add_argument(
        "--data_dir",
        default="../../data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--file_name", default="train_data.csv", type=str, help="train file name"
    )
    parser.add_argument(
        "--model_dir", default="../model/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="best_model.pt", type=str, help="model file name"
    )
    parser.add_argument(
        "--output_dir", default="../submit/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    ### boosting model 관련 argument ###
    parser.add_argument(
        "--learning_rate", default="0.001", type=float, help="learning rate"
    )
    parser.add_argument("--iterations", default="100", type=int, help="iterations")
    parser.add_argument("--n_estimators", default="100", type=int, help="n_estimators")
    parser.add_argument("--max_depth", default="6", type=int, help="max_depth")
    parser.add_argument("--num_leaves", default="31", type=int, help="num_leaves")

    ### 중요 ###
    parser.add_argument("--model", default="CAT", type=str, help="model type")
    args = parser.parse_args()

    return args
