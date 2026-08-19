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
    train_loader, test_loader, loss_norm_stats = make_dataloaders(config)
    if config["network"].get("predict_temperature", False):
        # Data-driven normalization constants (see make_dataloaders'/
        # ForgeNetTrainer's docstrings) -- computed fresh from THIS run's
        # actual dataset, not hand-picked, so they always match whatever
        # `--config` points at.
        config["network"]["pos_delta_std"] = loss_norm_stats["pos_delta_std"]
        config["network"]["temp_delta_std"] = loss_norm_stats["temp_delta_std"]
        print(f"Loss normalization stats: {loss_norm_stats}")
    trainer = ForgeNetTrainer(config, train_loader, test_loader)

    # Written HERE (before `trainer.train()`, which can run for hours) so
    # the resolved config -- including anything `_make_network` fills in
    # during `ForgeNetTrainer.__init__` itself, e.g. `point_size`, and the
    # `pos_delta_std`/`temp_delta_std` normalization stats injected above --
    # is available to read WHILE training is still running, not only after
    # it finishes. Re-written again below once training completes, as a
    # safety net in case anything changes later that isn't captured yet.
    def _write_config_out():
        out_config = dict(trainer.config)
        out_config["run"] = dict(out_config["run"])
        out_config["run"]["run_folder"] = str(run_folder)  # yaml cannot dump Path objects
        with open(run_folder / "config_out.yml", "w") as f:
            yaml.safe_dump(out_config, f)

    _write_config_out()
    trainer.train()
    _write_config_out()

    config = trainer.config  # get any changes the trainer made (e.g. point_size)

    if config["network"].get("predict_temperature", False):
        # `evaluate`/`evaluate_series` call `trainer.predict(...)` and
        # divide the raw result by 100.0 -- with `predict_temperature=True`
        # that's now a `(delta_xyz, delta_temp)` tuple, an immediate
        # TypeError (confirmed directly). Neither function knows about the
        # temperature head at all. Skip them here; `eval_thermal.py` is the
        # dedicated replacement for a `predict_temperature` run (mesh +
        # point + prediction + error-over-time GIF).
        print("predict_temperature=True -- skipping legacy evaluate()/evaluate_series() "
              "(see forge_net/eval_thermal.py instead)")
    else:
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
