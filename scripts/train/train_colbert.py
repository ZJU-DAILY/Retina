from pathlib import Path
import os
import sys
import configue
import random
import numpy as np
import torch
import argparse
import typer
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
from LLM4IR.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from LLM4IR.utils.gpu_stats import print_gpu_utilization

def set_random_seed(seed: int = 42) -> None: 
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)  # Python 内置随机数
    np.random.seed(seed)  # NumPy 随机数
    torch.manual_seed(seed)  # PyTorch CPU 随机数
    torch.cuda.manual_seed(seed)  # PyTorch GPU 随机数
    torch.cuda.manual_seed_all(seed)  # PyTorch 所有 GPU 随机数
def main(config_file: Path) -> None:
    set_random_seed(42)
    print_gpu_utilization()
    print("Loading config")
    config = configue.load(config_file, sub_path="config")
    print("Creating Setup")
    if isinstance(config, ColModelTrainingConfig):
        app = ColModelTraining(config)
    else:
        raise ValueError("Config must be of type ColModelTrainingConfig")

    if config.run_train:
        print("Training model")
        app.train()
        app.save(config_file=config_file)
    if config.run_eval:
        print("Running evaluation")
        app.eval()
    print("Done!")
# def parse_args():
#     parser = argparse.ArgumentParser(description="Multimodel training")
#     parser.add_argument("--config", type=str, required=True, help="yaml config file path")
#     return parser.parse_args()

if __name__ == "__main__":
    # args = parse_args()
    # typer.run(main)
    
    # # config_file = './scripts/configs/qwen2/train_biqwen2_hardneg_model.yaml'
    # # config_file = './scripts/configs/qwen2/train_icrr_colqwen2_model.yaml'
    config_file = './scripts/configs/qwen2/train_triple_sparse_qwen2_model.yaml'
    
    main(config_file)
    
