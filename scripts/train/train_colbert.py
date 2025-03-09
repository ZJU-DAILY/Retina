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
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
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

if __name__ == "__main__":
    # typer.run(main)
    
    config_file = './scripts/configs/qwen2/train_point_reranker_qwen2_model.yaml'
    
    main(config_file)
    
