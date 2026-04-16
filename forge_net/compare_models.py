import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Import your local modules
from forge_net.model.trainer import ForgeNetTrainer
from main import make_dataloaders
# Ensure evaluate_series is imported from wherever you saved it!
# from your_eval_module import evaluate_series 

from pathlib import Path
import torch
from forge_net.utils.common import *
from forge_net.utils.math import *
from forge_net.utils.plotting import * 
from forge_net.invert_deltas import invert_deltas_to_mesh, save_comparison_turntable
# from forge_net.loss.chamfer import chamfer_distance
from pytorch3d.loss import chamfer_distance #original implementation uses a chamfer distance
from tqdm import tqdm 

from forge_net.eval import evaluate_series

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 300 
plt.rcParams['savefig.dpi'] = 300

def plot_model_comparison(stats_chamfer, stats_mse, n_step=2, max_cols=6, fill_variation=True, fig_path=None):
    """
    4-Row Figure:
    Row 0-2: Point clouds starting from Step 1 (Step 0 removed).
    Row 3: Combined line plot showing both MSE and Chamfer metrics for both models.
    """
    def plot_pc(ax, data, color, title=None, s=10, alpha=1.0, fontsize=14, fontweight='bold', keep_axes=False):
        if data is None: return
        ax.scatter(data[:,0], data[:,1], data[:,2], s=s, c=color, alpha=alpha)
        if title: 
            ax.set_title(title, fontsize=fontsize, fontweight=fontweight)
        ax.set_box_aspect([1.1,1,1]) 
        ax.dist = 8
        if not keep_axes:
            ax.set_axis_off()

    # Extract sequence data
    gt_seq = stats_chamfer['all_gt_steps'][-1]
    chamfer_rec_seq = stats_chamfer['all_rec_step_preds'][-1]
    mse_rec_seq = stats_mse['all_rec_step_preds'][-1]
    
    # Determine display indices (Adjusted for no Step 0)
    rec_indices = list(range(n_step - 1, len(chamfer_rec_seq), n_step))
    display_rec_indices = rec_indices[:max_cols - 1] 
    num_cols = 1 + len(display_rec_indices)
    last_step_idx = display_rec_indices[-1] + 1 if display_rec_indices else 1
    
    fig = plt.figure(figsize=(num_cols * 4, 16))
    
    # GridSpec for point clouds and the wide bottom plot
    gs = fig.add_gridspec(4, num_cols, height_ratios=[1.5, 1.5, 1.5, 1.75],
                          wspace=0.05, hspace=0.1)

    # --- ROW 0: Ground Truth (Starting from Step 1) ---
    ax_gt1 = fig.add_subplot(gs[0, 0], projection='3d')
    plot_pc(ax_gt1, gt_seq[1], 'grey', "Step 1", fontsize=16)

    for i, idx in enumerate(display_rec_indices):
        ax = fig.add_subplot(gs[0, 1 + i], projection='3d')
        if idx < len(gt_seq):
            plot_pc(ax, gt_seq[idx], 'grey', f"Step {idx + 1}", fontsize=18)

    # --- ROW 1: Chamfer Model Predictions ---
    ax_c1 = fig.add_subplot(gs[1, 0], projection='3d')
    plot_pc(ax_c1, chamfer_rec_seq[1], 'red')

    for i, idx in enumerate(display_rec_indices):
        ax = fig.add_subplot(gs[1, 1 + i], projection='3d')
        plot_pc(ax, chamfer_rec_seq[idx], 'red')

    # --- ROW 2: MSE Model Predictions (Light Purple) ---
    ax_m1 = fig.add_subplot(gs[2, 0], projection='3d')
    plot_pc(ax_m1, mse_rec_seq[1], 'mediumpurple')

    for i, idx in enumerate(display_rec_indices):
        ax = fig.add_subplot(gs[2, 1 + i], projection='3d')
        plot_pc(ax, mse_rec_seq[idx], 'mediumpurple')

    # Row Annotations on the Right
    x_pos = 0.155
    fig.text(x_pos, 0.720, 'Ground Truth', va='center', rotation=0, fontsize=16, fontweight='bold')
    fig.text(x_pos, 0.5295, 'Chamfer Model', va='center', rotation=0, fontsize=16, fontweight='bold', color='red')
    fig.text(x_pos + 0.0055, 0.342, 'MSE Model', va='center', rotation=0, fontsize=16, fontweight='bold', color='mediumpurple')

    # --- ROW 3: Combined Line Plot (Your Original Version) ---
    import math
    l_col = math.floor((num_cols / 100) * 22.5)
    u_col = math.floor((num_cols / 100) * 82.5)
    ax_loss = fig.add_subplot(gs[3, l_col:u_col+1])

    # Data extraction for lines
    mse_model_mses = np.array(stats_mse['all_rec_step_mses'])[:, :last_step_idx]
    mse_model_chamfers = np.array(stats_mse['all_rec_step_chamfers'])[:, :last_step_idx]
    mse_model_meds = np.array(stats_mse['all_rec_step_dist_means'])[:, :last_step_idx]
    
    chamfer_model_mses = np.array(stats_chamfer['all_rec_step_mses'])[:, :last_step_idx]
    chamfer_model_chamfers = np.array(stats_chamfer['all_rec_step_chamfers'])[:, :last_step_idx]
    chamfer_model_meds = np.array(stats_chamfer['all_rec_step_dist_means'])[:, :last_step_idx]

    steps = np.arange(1, mse_model_mses.shape[1] + 1)

    # Plot lines with your specific colors/styles
    l1 = ax_loss.plot(steps, np.mean(chamfer_model_meds, axis=0), color='red', marker=None, label='Chamfer Model (MED)')
    l2 = ax_loss.plot(steps, np.mean(chamfer_model_chamfers, axis=0), color='darkorange', marker=None, label='Chamfer Model (Chamfer)')
    l3 = ax_loss.plot(steps, np.mean(chamfer_model_mses, axis=0), color='gold', marker=None, label='Chamfer Model (MSE)')
    l4 = ax_loss.plot(steps, np.mean(mse_model_meds, axis=0), color='darkblue', marker=None, label='MSE Model (MED)')
    l5 = ax_loss.plot(steps, np.mean(mse_model_chamfers, axis=0), color='mediumpurple', marker=None, label='MSE Model (Chamfer)')
    l6 = ax_loss.plot(steps, np.mean(mse_model_mses, axis=0), color='lightblue', marker=None, label='MSE Model (MSE)')
    
    

    if fill_variation:
        ax_loss.fill_between(steps, np.mean(mse_model_meds, axis=0) - np.std(mse_model_meds, axis=0), 
                             np.mean(mse_model_meds, axis=0) + np.std(mse_model_meds, axis=0), color='darkblue', alpha=0.1)
        ax_loss.fill_between(steps, np.mean(chamfer_model_meds, axis=0) - np.std(chamfer_model_meds, axis=0), 
                             np.mean(chamfer_model_meds, axis=0) + np.std(chamfer_model_meds, axis=0), color='red', alpha=0.1)

    ax_loss.set_title("Rollout Error Accumulation: MSE vs. Chamfer Trained Models", fontsize=20, fontweight='bold')
    ax_loss.set_xlabel("Step Number", fontweight='bold', fontsize=16)
    ax_loss.set_xticks(steps)
    # ax_loss.grid(False, which='both', alpha=0.3)
    ax_loss.grid(False)
    ax_loss.xaxis.set_major_locator(ticker.MultipleLocator(5))
    ax_loss.tick_params(axis='both', which='major', labelsize=14)
    
    lines = l1 + l2 + l3 + l4 + l5 + l6
    labels = [line.get_label() for line in lines]
    ax_loss.legend(lines, labels, loc='upper left', ncol=1, prop={'weight': 600, 'size': 13})

    # plt.subplots_adjust(left=0.05, right=0.9, top=0.95, bottom=0.05)
    
    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight')

def plot_simplified_comparison(stats_chamfer, stats_mse, fig_path=None):
    """
    Simplified 3x3 Figure:
    Rows: [Ground Truth, Chamfer Model, MSE Model]
    Cols: [Step 0 (Input), Step 1, Last Step]
    """
    def plot_pc(ax, data, color, title=None):
        if data is None: return
        ax.scatter(data[:,0], data[:,1], data[:,2], s=20, c=color, alpha=0.8)
        if title: 
            ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_box_aspect([1,1,1])
        ax.set_axis_off()
        ax.dist = 7

    # Extract sequence data
    gt_seq = stats_chamfer['all_gt_steps'][-1]
    chamfer_seq = stats_chamfer['all_rec_step_preds'][-1]
    mse_seq = stats_mse['all_rec_step_preds'][-1]
    
    last_idx = len(gt_seq) - 1 # The final step in the sequence

    fig = plt.figure(figsize=(18, 15), dpi=300)
    # 3 rows (GT, Chamfer, MSE), 3 columns (0, 1, Final)
    gs = fig.add_gridspec(3, 3, wspace=0.0, hspace=0.05)

    # --- ROW 0: Ground Truth ---
    ax_gt0 = fig.add_subplot(gs[0, 0], projection='3d')
    plot_pc(ax_gt0, gt_seq[0], 'black', "Step 0 (Input)")
    
    ax_gt1 = fig.add_subplot(gs[0, 1], projection='3d')
    plot_pc(ax_gt1, gt_seq[1], 'grey', "Step 1")
    
    ax_gtl = fig.add_subplot(gs[0, 2], projection='3d')
    plot_pc(ax_gtl, gt_seq[last_idx], 'grey', f"Step {last_idx + 1} (Final)")

    # --- ROW 1: Chamfer Model ---
    ax_c0 = fig.add_subplot(gs[1, 0], projection='3d')
    plot_pc(ax_c0, gt_seq[0], 'black')
    
    ax_c1 = fig.add_subplot(gs[1, 1], projection='3d')
    plot_pc(ax_c1, chamfer_seq[1], 'red')
    
    ax_cl = fig.add_subplot(gs[1, 2], projection='3d')
    plot_pc(ax_cl, chamfer_seq[last_idx], 'red')

    # --- ROW 2: MSE Model ---
    ax_m0 = fig.add_subplot(gs[2, 0], projection='3d')
    plot_pc(ax_m0, gt_seq[0], 'black')
    
    ax_m1 = fig.add_subplot(gs[2, 1], projection='3d')
    plot_pc(ax_m1, mse_seq[1], 'mediumpurple')
    
    ax_ml = fig.add_subplot(gs[2, 2], projection='3d')
    plot_pc(ax_ml, mse_seq[last_idx], 'mediumpurple')

    # Add Row Labels on the left
    # Adjusted y-positions for the 3-row layout
    fig.text(0.08, 0.78, 'Ground\nTruth', va='center', ha='center', fontsize=20, fontweight='bold', color='grey')
    fig.text(0.08, 0.50, 'Chamfer\nModel', va='center', ha='center', fontsize=20, fontweight='bold', color='red')
    fig.text(0.08, 0.22, 'MSE\nModel', va='center', ha='center', fontsize=20, fontweight='bold', color='mediumpurple')

    if fig_path:
        plt.savefig(fig_path, bbox_inches='tight')

def load_and_evaluate_model(run_name, config_modifier=None):
    """Helper function to load a model and get its evaluation stats."""
    base_path = get_project_root()
    config_path = base_path / "runs" / run_name / "config_out.yml"
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    if config_modifier:
        config_modifier(config)
        
    train_loader, test_loader = make_dataloaders(config)
    trainer = ForgeNetTrainer(config, train_loader, log_to_tb=False)
    output_folder = Path(config["run"]["run_folder"])
    
    trainer.load(model_path=output_folder / "best_model.pth")
    trainer.net.eval()
    
    print(f"--- Evaluating {run_name} ---")
    stats = evaluate_series(config, trainer,
                            add_mse=False,
                            add_chamfer=True, 
                            add_hausdorff=False,
                            plot_mode='dist',
                            num_series=1, 
                            min_series_length=60, 
                            n_step=10, 
                            max_cols=7, 
                            save_meshes=False)
    return stats, output_folder

if __name__ == "__main__":
    # 1. Load and evaluate the Chamfer model
    run_name_chamfer = "chamfer_1024_unmasked_seeded_tri_ids"
    stats_chamfer, output_folder_chamfer = load_and_evaluate_model(run_name_chamfer)
    
    # 2. Load and evaluate the MSE model 
    # (Replace this with the actual run name for your MSE model)
    run_name_mse = "mse_1024_unmasked_seeded_tri_ids" 
    stats_mse, _ = load_and_evaluate_model(run_name_mse)

    # 3. Plot the comparison
    eval_path = output_folder_chamfer / "eval"
    eval_path.mkdir(exist_ok=True)
    fig_save_path = eval_path / "comparison_eval_series.png"
    fig_save_path = eval_path / "simple_comparison.png"

    
    print("Generating comparison plot...")
    # plot_model_comparison(
    #     stats_chamfer=stats_chamfer,
    #     stats_mse=stats_mse,
    #     n_step=10,
    #     max_cols=7,
    #     # mode='dist',
    #     fill_variation=True,
    #     fig_path=fig_save_path
    # )
    plot_simplified_comparison(stats_chamfer, stats_mse,fig_save_path)
    print(f"Comparison plot saved to {fig_save_path}")