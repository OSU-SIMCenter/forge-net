"""Train ForgeNet against `forge_common.db` data -- same overall flow as
`main.py` (dataset -> dataloaders -> trainer -> train -> evaluate ->
evaluate_series), but wired to `process_data_forge_common.make_dataset`/
`make_dataloaders` instead of `main.py`'s hardcoded `process_data.py`
import. Kept as a SEPARATE entry point rather than editing `main.py`'s
active import, since other configs (`jax_volume.yml`, `experimentB.yml`,
...) still depend on `process_data.py`'s legacy-schema pipeline running
unmodified through `main.py`.

Usage (from `models/forge-net`):
    python -m forge_net.train_forge_common --config forge_net/configs/forge_common_v1.yml
"""

import argparse
from pathlib import Path

import yaml

from forge_net.data.process_data_forge_common import make_dataloaders, make_dataset
from forge_net.eval import evaluate, evaluate_series
from forge_net.model.trainer import ForgeNetTrainer
from forge_net.utils.common import get_project_root


def main():
    base_path = get_project_root()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_folder = base_path / "runs" / config["run"]["run_name"]
    assert not run_folder.exists(), f"Run already exists: {run_folder}"
    config["run"]["run_folder"] = run_folder

    make_dataset(config=config)
    train_loader, test_loader = make_dataloaders(config)
    trainer = ForgeNetTrainer(config, train_loader, test_loader)
    trainer.train()

    config = trainer.config  # get any changes the trainer made (e.g. point_size)
    config["run"]["run_folder"] = str(run_folder)  # yaml cannot dump Path objects
    with open(run_folder / "config_out.yml", "w") as f:
        yaml.safe_dump(config, f)

    evaluate(config, trainer)
    evaluate_series(
        config, trainer,
        num_series=3, min_series_length=15,
        max_cols=7, plot_mode="dist", n_step=3,
        add_mse=True, add_chamfer=False, add_hausdorff=False,
        plot_heatmaps=True, save_meshes=False,
    )


if __name__ == "__main__":
    main()
