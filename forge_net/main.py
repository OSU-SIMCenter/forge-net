import yaml, argparse
from pathlib import Path
from forge_net.data.process_data import make_dataloaders, make_dataset
# from forge_net.data.process_data_fanglei import make_dataloaders, make_dataset

from forge_net.utils.common import get_project_root
from forge_net.model.trainer import ForgeNetTrainer
from forge_net.eval import evaluate, evaluate_series

def main():

    base_path = get_project_root()

    parser = argparse.ArgumentParser(description="Load experiment configuration from a YAML file.")
    parser.add_argument(
        "--config", 
        type=Path, 
        default=None,
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()
    assert args.config is not None, print("Please provide a valid yaml config file.")
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    
    run_folder = base_path / "runs" / config["run"]["run_name"]
    assert not run_folder.exists(), print("Run already exists")
    config["run"]["run_folder"] = run_folder

    make_dataset(config=config)
    # raise()
    train_loader, test_loader = make_dataloaders(config)
    trainer = ForgeNetTrainer(config, train_loader, test_loader)
    trainer.train()
    config = trainer.config # get any changes the trainer made to the config
    config["run"]["run_folder"] = str(run_folder) #yaml cannot dump Path Objects
    with open(run_folder / "config_out.yml", "w") as file:
        yaml.safe_dump(config, file)
    evaluate(config, trainer)
    evaluate_series(config, trainer,
                    add_chamfer=True, add_hausdorff=False,
                    plot_mode='dist',
                    num_series=1, min_series_length=35, 
                    n_step=15, max_cols=7, save_meshes=False)

if __name__ == "__main__":
    main()    

    