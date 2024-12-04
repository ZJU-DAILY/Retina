from pathlib import Path
import os
import sys
import configue
import typer
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)
from colpali_engine.trainer.colmodel_training import ColModelTraining, ColModelTrainingConfig
from colpali_engine.utils.gpu_stats import print_gpu_utilization

def main(config_file: Path) -> None:

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
    # config_file = './scripts/configs/qwen2/train_biqwen2_hardneg_model.yaml'
    # config_file = './scripts/configs/qwen2/train_icrr_colqwen2_model.yaml'
    config_file = './scripts/configs/qwen2/train_icrr_colqwen2_example_model.yaml'
    # config_file = '/data1/zhh/baselines/mm/icrr/scripts/configs/qwen2/train_icrr_biqwen2_model.yaml'
    main(config_file)
    
