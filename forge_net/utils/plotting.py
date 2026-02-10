import matplotlib.pyplot as plt
import pyvista as pv
pv.start_xvfb()
pv.set_jupyter_backend('static')
import numpy as np
import os
import shutil
        
def clear_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)

def plot_network_weights(state_dict, fig_path=None):
    # Collect all weights into a single array
    all_weights = []
    for param_name, param_tensor in state_dict.items():
        # Flatten the parameter tensor and convert to numpy
        weights = param_tensor.flatten().cpu().numpy()
        all_weights.extend(weights)

    all_weights = np.array(all_weights)

    # Create the histogram
    plt.figure(figsize=(12, 6))
    plt.hist(all_weights, bins=100, edgecolor='black', alpha=0.7)
    plt.xlabel('Parameter Weight Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Model Parameter Weights', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add some statistics as text
    mean_val = np.mean(all_weights)
    std_val = np.std(all_weights)
    median_val = np.median(all_weights)
    min_val = np.min(all_weights)
    max_val = np.max(all_weights)

    stats_text = f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nMedian: {median_val:.4f}\nMin: {min_val:.4f}\nMax: {max_val:.4f}\nTotal params: {len(all_weights):,}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10, family='monospace')

    plt.tight_layout()
    print(f"Histogram saved! Total parameters: {len(all_weights):,}")
    print(f"Weight statistics:")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Std: {std_val:.6f}")
    print(f"  Min: {min_val:.6f}")
    print(f"  Max: {max_val:.6f}")
    if fig_path is not None:
        plt.savefig(fig_path)
    else:
        plt.show()

def deltas_vs_x(pc, delta_x, delta_z, title_prefix=""):
    """
    Create a 2D scatter plot with x-coordinate on x-axis and delta values on y-axis
    
    Args:
        pc: Point cloud coordinates (n_points, 3)
        delta_x: Delta x values (n_points,)
        delta_z: Delta z values (n_points,)
        title_prefix: Optional prefix for title
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Extract x coordinates
    x_coords = pc[:, 0]
    
    # Plot both series
    ax.scatter(x_coords, delta_x, s=20, alpha=0.6, label='Delta X', color='blue')
    ax.scatter(x_coords, delta_z, s=20, alpha=0.6, label='Delta Z', color='red')
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Delta Values', fontsize=12)
    ax.set_title(f'{title_prefix}Delta X and Delta Z vs X Coordinate', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def compare_scatters(pc1, pc2, label_1=None, label_2=None, label_3=None, fig_path=None):
    
    fig = plt.figure(figsize=(20,8))
    
    ax1 = fig.add_subplot(131, projection = '3d')
    ax1.scatter(xs=pc1[:,0],ys=pc1[:,1],zs=pc1[:,2], s=2.2)
    ax1.set_title(label_1)
    
    ax2 = fig.add_subplot(132, projection = '3d')
    ax2.scatter(xs=pc2[:,0],ys=pc2[:,1],zs=pc2[:,2], s=2.2, color='red')
    ax2.set_title(label_2)
    
    ax3 = fig.add_subplot(133, projection = '3d')
    ax3.scatter(xs=pc1[:,0],ys=pc1[:,1],zs=pc1[:,2], s=1.2, color='blue')
    ax3.scatter(xs=pc2[:,0],ys=pc2[:,1],zs=pc2[:,2], s=1.2, color='red')
    ax3.set_title(label_3)
    if fig_path is not None:
        plt.savefig(fig_path)
    else:
        plt.show()

def compare_scatters_w_loss_cont(pc1, pc2, loss_cont, label_1=None, label_2=None, fig_path=None):
    
    fig = plt.figure(figsize=(16,8))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(xs=pc1[:,0], ys=pc1[:,1], zs=pc1[:,2], s=5.0)
    ax1.set_title(label_1)

    ax2 = fig.add_subplot(122, projection='3d')
    scatter = ax2.scatter(xs=pc2[:,0], ys=pc2[:,1], zs=pc2[:,2], s=5.0, c=loss_cont, cmap='jet')
    ax2.set_title(label_2)

    # Create a new axis for the colorbar in the bottom right
    # [left, bottom, width, height] in figure coordinates (0 to 1)
    cbar_ax = fig.add_axes([0.70, 0.85, 0.30, 0.04])  # Adjust these values as needed
    plt.colorbar(scatter, cax=cbar_ax, orientation='horizontal', label='Loss')

    plt.tight_layout()
    if fig_path is not None:
        plt.savefig(fig_path)
    else:
        plt.show()

def visualize_bc_features(pc, bc_mask, distances_to_edge, title_prefix=""):
    """
    Visualize boundary condition features
    
    Args:
        pc: Point cloud coordinates (n_points, 3)
        bc_mask: Binary mask (n_points,) - 1 if in BC, 0 otherwise
        distances_to_edge: Distance to BC edge (n_points,) - negative inside, positive outside
        title_prefix: Optional prefix for titles
    """
    fig = plt.figure(figsize=(20, 8))
    
    # Plot 1: BC mask visualization
    ax1 = fig.add_subplot(131, projection='3d')
    
    # Separate BC and non-BC points
    bc_points = pc[bc_mask == 1]
    non_bc_points = pc[bc_mask == 0]
    
    ax1.scatter(xs=non_bc_points[:, 0], ys=non_bc_points[:, 1], zs=non_bc_points[:, 2], 
                s=2, color='gray', alpha=0.3, label='Non-BC')
    ax1.scatter(xs=bc_points[:, 0], ys=bc_points[:, 1], zs=bc_points[:, 2], 
                s=10, color='red', label='BC Points')
    ax1.set_title(f'{title_prefix}Boundary Condition Mask')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=0, azim=-90, roll=0)
    ax1.legend()
    
    # Plot 2: Distance to BC edge (colored)
    ax2 = fig.add_subplot(132, projection='3d')
    scatter = ax2.scatter(xs=pc[:, 0], ys=pc[:, 1], zs=pc[:, 2], 
                         s=5, c=distances_to_edge, cmap='RdYlBu_r', 
                         vmin=distances_to_edge.min(), vmax=distances_to_edge.max())
    ax2.set_title(f'{title_prefix}Distance to BC Edge\n(negative=inside, positive=outside)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=0, azim=-90, roll=0)

    plt.colorbar(scatter, ax=ax2, label='Distance')
    
    # Plot 3: Combined view with distance colormap on BC region only
    ax3 = fig.add_subplot(133, projection='3d')
    
    # Non-BC points in gray
    ax3.scatter(xs=non_bc_points[:, 0], ys=non_bc_points[:, 1], zs=non_bc_points[:, 2], 
                s=2, color='gray', alpha=0.3)
    ax3.view_init(elev=0, azim=-90, roll=0)
    
    # BC points colored by distance
    if len(bc_points) > 0:
        bc_distances = distances_to_edge[bc_mask == 1]
        scatter3 = ax3.scatter(xs=bc_points[:, 0], ys=bc_points[:, 1], zs=bc_points[:, 2], 
                              s=10, c=bc_distances, cmap='plasma')
        plt.colorbar(scatter3, ax=ax3, label='Distance (BC only)')
    
    ax3.set_title(f'{title_prefix}BC Points Highlighted by Distance')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()

def visualize_point_cloud(pc,a, point_size=5, fig_path=None):
    cloud = pv.PolyData(pc)
    plotter = pv.Plotter()
    plotter.add_points(cloud, point_size=point_size, color="red")
    start = np.array([a[0], 0, pc[:, 2].max() + 1.0])
    # Direction: pointing down in z (negative z direction)
    direction = np.array([0, 0, -1])
    # Add arrow
    plotter.add_arrows(start, direction, mag=1.0, color="blue")
    plotter.show_axes()
    plotter.show_grid()
    if fig_path is not None:
        plotter.screenshot(fig_path)
    else:
        plotter.show()

def visualize_point_diff(pc1, pc2, point_size=5, label=None, fig_path=None):
    
    plotter = pv.Plotter(shape=(1, 2), window_size=(2000,1000), border=False)  # 3 row, 2 columns
    if label is not None:
        plotter.add_title(title=label, font_size=16)

    start = pc1
    direction = pc2 - pc1
    points = pv.PolyData(start)
    points['vectors'] = direction
    arrows = points.glyph(
        orient='vectors',
        scale=False,
        factor=0.5,   
        color_mode='scale',
        geom=pv.Arrow()
    )
    
    plotter.subplot(0, 0)
    plotter.view_xy()
    plotter.add_mesh(
        arrows,
        scalars=None,
        cmap=None,
        # color='blue',
        scalar_bar_args=None
    )
    plotter.reset_camera()
    plotter.show_axes()

    plotter.subplot(0, 1)
    plotter.add_mesh(
        arrows,
        scalars=None,
        cmap=None,
        # color='blue',
        scalar_bar_args=None
    )
    plotter.reset_camera()
    plotter.view_yz()
    plotter.camera.azimuth = -13
    
    plotter.show_axes()
    if fig_path is not None:
        plt.savefig(fig_path)
    else:
        plt.show()

def visualize_vector_diff(pc1, pc2, mesh1=None, mesh2=None, 
                            min_magnitude=2.0, fig_path=None):
    
    plotter = pv.Plotter(shape=(1, 3), window_size=(2000,1000))  # 1 row, 3 columns
    
    start = pc1
    direction = pc2 - pc1
    # Calculate actual magnitude
    actual_mag = np.linalg.norm(direction)
    
    # Normalize and scale to minimum magnitude if needed
    if actual_mag > 0:
        normalized_direction = direction / actual_mag
        display_mag = max(actual_mag, min_magnitude)
        display_direction = normalized_direction * display_mag
    else:
        display_direction = direction
        display_mag = 0
    mag = 1.0
    # View 1: Looking down Z (top view)
    plotter.subplot(0, 0)
    plotter.add_arrows(start, display_direction, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_xy()  # Look down Z axis
    plotter.camera.zoom(1.0)
    plotter.show_grid()
    plotter.add_text("Top View (Z)", font_size=10)
    if mesh1 is not None:
        plotter.add_mesh(mesh1, opacity=0.3, color='green')
    if mesh2 is not None:
        plotter.add_mesh(mesh2, opacity=0.3, color='red')
    
    # View 2: Looking down X (side view)
    plotter.subplot(0, 1)
    plotter.add_arrows(start, display_direction, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_yz()  # Look down X axis
    plotter.camera.zoom(1.3)
    plotter.show_grid()
    plotter.add_text("Side View (X)", font_size=10)
    if mesh1 is not None:
        plotter.add_mesh(mesh1, opacity=0.3, color='green')
    if mesh2 is not None:
        plotter.add_mesh(mesh2, opacity=0.3, color='red')


    # View 3: Looking down Y (front view)
    plotter.subplot(0, 2)
    plotter.add_arrows(start, display_direction, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_xz()  # Look down Y axis
    plotter.camera.zoom(1.0)
    plotter.show_grid()
    plotter.add_text("Front View (Y)", font_size=10)
    if mesh1 is not None:
        plotter.add_mesh(mesh1, opacity=0.3, color='green')
    if mesh2 is not None:
        plotter.add_mesh(mesh2, opacity=0.3, color='red')
    
    if fig_path is not None:
        plotter.savefig(fig_path)
    else:
        plotter.show()

def visualize_vector_diff_w_scale(x_t, x_tp1, x_hat, 
                                    mesh1=None, mesh2=None, 
                                    scale_vectors=False, scale_factor=1, fig_path=None):
    
    plotter = pv.Plotter(shape=(1, 2), window_size=(2000,1000))  # 1 row, 2 columns
    plotter.add_text(text=f"Vector magnitudes scaled by {scale_factor}")
    
    start = x_t
    direction = x_tp1 - x_t
    direction_hat = x_hat - x_t
    direction = x_tp1 - x_t

    points = pv.PolyData(start)
    points['vectors'] = direction
    
    plotter.subplot(0, 0)

    arrows = points.glyph(
        orient='vectors',
        scale=True,  # Don't scale by data
        factor=scale_factor,   # Use fixed magnitude (same as before)
        color_mode='vector',
        geom=pv.Arrow()
    )
    
    plotter.add_mesh(
        arrows,
        scalars=None,
        cmap=None,
        color='blue',
        scalar_bar_args=None
    )
    
    plotter.show_axes()
    plotter.view_xy()  # Look down Z axis
    plotter.camera.zoom(1.0)
    plotter.show_grid()
    plotter.add_text("Actual Vector Field", font_size=10)
    
    plotter.subplot(0, 1)
    points_hat = pv.PolyData(start)
    points_hat['vectors'] = direction_hat
    
    arrows_hat = points_hat.glyph(
        orient='vectors',
        color_mode='vector',
        scale=True,  # Don't scale by data
        factor=scale_factor,   # Use fixed magnitude (same as before)
        geom=pv.Arrow()
    )
    
    plotter.add_mesh(
        arrows_hat,
        cmap='jet',
        # scalar_bar_args={'title': 'Loss'}
    )

    plotter.show_axes()
    plotter.view_xy()  # Look down X axis
    plotter.camera.zoom(1.0)
    plotter.show_grid()
    plotter.add_text("Predicted Vector Field Color by Scale", font_size=10)
    # if mesh1 is not None:
    #     plotter.add_mesh(mesh1, opacity=0.3, color='green')
    # if mesh2 is not None:
    #     plotter.add_mesh(mesh2, opacity=0.3, color='red')
    if fig_path is not None:
        plotter.off_screen = True
        plotter.screenshot(fig_path)
    else:
        plotter.show()

    return plotter

def visualize_vector_diff_w_loss_cont(x_t, x_tp1, x_hat, 
                                      loss_cont, mesh1=None, mesh2=None, 
                                      scale_vectors=False, fig_path=None):
    
    plotter = pv.Plotter(shape=(1, 2), window_size=(2000,1000))  # 1 row, 2 columns
    
    start = x_t
    direction = x_tp1 - x_t
    direction_hat = x_hat - x_t
    direction = x_tp1 - x_t

    points = pv.PolyData(start)
    points['vectors'] = direction
    points['loss'] = loss_cont
    
    plotter.subplot(0, 0)

    arrows = points.glyph(
        orient='vectors',
        scale=False,  # Don't scale by data
        factor=0.5,   # Use fixed magnitude (same as before)
        color_mode='vector',
        geom=pv.Arrow()
    )
    
    plotter.add_mesh(
        arrows,
        scalars=None,
        cmap=None,
        color='blue',
        scalar_bar_args=None
    )
    
    # plotter.show_axes()
    plotter.view_xy()  # Look down Z axis
    plotter.camera.zoom(1.0)
    # plotter.show_grid()
    plotter.add_text("Actual Vector Field", font_size=10)
    
    plotter.subplot(0, 1)
    points_hat = pv.PolyData(start)
    points_hat['vectors'] = direction_hat
    points_hat['loss'] = loss_cont
    
    arrows_hat = points_hat.glyph(
        orient='vectors',
        color_mode='vector',
        scale=False,  # Don't scale by data
        factor=0.5,   # Use fixed magnitude (same as before)
        geom=pv.Arrow()
    )
    
    plotter.add_mesh(
        arrows_hat,
        scalars='loss',
        cmap='jet',
        scalar_bar_args={'title': 'Loss'}
    )

    # plotter.show_axes()
    plotter.view_xy()  # Look down X axis
    plotter.camera.zoom(1.0)
    # plotter.show_grid()
    plotter.add_text("Predicted Vector Field Color by Loss", font_size=10)
    # if mesh1 is not None:
    #     plotter.add_mesh(mesh1, opacity=0.3, color='green')
    # if mesh2 is not None:
    #     plotter.add_mesh(mesh2, opacity=0.3, color='red')
    if fig_path is not None:
        plotter.off_screen = True
        plotter.screenshot(fig_path)
    else:
        plotter.show()

    return plotter

def compare_vector_fields(x_t, x_tp1, x_hat, point_size=5, min_magnitude=2.0, fig_path=None):
    plotter = pv.Plotter(shape=(2, 2), window_size=(2000,1000))  # 3 row, 2 columns
    
    start = x_t
    direction = x_tp1 - x_t
    direction_hat = x_hat - x_t
    
    points = pv.PolyData(start)
    points['vectors'] = direction

    points_hat = pv.PolyData(start)
    points_hat['vectors'] = direction_hat

    arrows = points.glyph(
        orient='vectors',
        scale=False,  # Don't scale by data
        factor=0.33,   # Use fixed magnitude (same as before)
        color_mode='vector',
        geom=pv.Arrow()
    )

    arrows_hat = points_hat.glyph(
        orient='vectors',
        scale=False,  # Don't scale by data
        factor=0.33,   # Use fixed magnitude (same as before)
        color_mode='vector',
        geom=pv.Arrow()
    )
        
    plotter.subplot(0, 0)
    plotter.add_mesh(
        arrows,
        scalars=None,
        cmap=None,
        color='blue',
        scalar_bar_args=None
    )
    plotter.show_axes()
    plotter.view_xy()  # Look down Z axis
    plotter.camera.zoom(1.4)
    plotter.add_text("Top View (Z)", font_size=10)
    
    plotter.subplot(0, 1)
    plotter.add_mesh(
            arrows,
            scalars=None,
            cmap=None,
            color='blue',
            scalar_bar_args=None
        )
    plotter.show_axes()
    plotter.view_yz()  # Look down X axis
    plotter.camera.azimuth = -13  # Rotate around vertical axis
    plotter.camera.zoom(1.75)
    plotter.add_text("Front View (X)", font_size=10)

    # # View 3: Looking down Y (front view)
    # plotter.subplot(0, 2)
    # plotter.add_arrows(start, display_direction, mag=mag, color="blue")
    # plotter.show_axes()
    # plotter.view_xz()  # Look down Y axis
    # plotter.camera.zoom(1.2)
    # plotter.add_text("Front View (Y)", font_size=10)
    
    
    # View 1: Looking down Z (top view)
    plotter.subplot(1, 0)
    plotter.add_mesh(
        arrows_hat,
        scalars=None,
        cmap=None,
        color='blue',
        scalar_bar_args=None
    )
    plotter.show_axes()
    plotter.view_xy()  # Look down Z axis
    plotter.camera.zoom(1.4)
    plotter.add_text("Top View (Z)", font_size=10)

    # View 2: Looking down X (side view)
    plotter.subplot(1, 1)
    plotter.add_mesh(
            arrows_hat,
            scalars=None,
            cmap=None,
            color='blue',
            scalar_bar_args=None
        )
    plotter.show_axes()
    plotter.view_yz()  # Look down X axis
    plotter.camera.azimuth = -13   # Rotate around vertical axis
    plotter.camera.zoom(1.75)
    plotter.add_text("Front View (X)", font_size=10)

    # # View 3: Looking down Y (front view)
    # plotter.subplot(1, 2)
    # plotter.add_arrows(start, display_direction, mag=mag, color="blue")
    # plotter.show_axes()
    # plotter.view_xz()  # Look down Y axis
    # plotter.camera.zoom(1.2)
    # plotter.add_text("Front View (Y)", font_size=10)
    
    if fig_path is not None:
        plotter.off_screen = True
        plotter.screenshot(fig_path)
    else:
        plotter.show()

# Plot losses
def plot_loss(train_loss_list, test_loss_list, write_string, output_folder=None, save_results=True):
            plt.figure(figsize=(10, 6))
            plt.plot(train_loss_list, label="Train", linewidth=2)
            plt.plot(test_loss_list, label="Test", linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            if save_results:
                with open(output_folder / "prints.txt", "a") as file: 
                    file.write(write_string + "\n")
                plt.savefig(output_folder  / "loss", dpi=150, bbox_inches='tight')
            plt.close()

def plot_eval_series(gt_seq, single_preds, rec_preds, losses, 
                     dev_losses, rec_losses, n_step=2, max_cols=6, fig_path=None):
    """
    3-Row Figure where the loss plot is truncated to match the final scatter plot step.
    """
    
    def plot_pc(ax, data, color, title, s=10, alpha=1.0):
        if data is None: return
        ax.scatter(data[:,0], data[:,1], data[:,2], s=s, c=color, alpha=alpha)
        ax.set_title(title)

    # --- 1. Determine Display Indices ---
    rec_indices = list(range(n_step - 1, len(rec_preds), n_step))
    display_rec_indices = rec_indices[:max_cols - 2] 
    num_cols = 2 + len(display_rec_indices)
    
    # Calculate the 'cutoff': the step number of the very last scatter plot
    # If display_rec_indices is [1, 3, 5], the last step is 6 (index 5 + 1)
    if display_rec_indices:
        last_step_idx = display_rec_indices[-1] + 1
    else:
        last_step_idx = 1 # Only the first prediction is shown

    fig = plt.figure(figsize=(num_cols * 4, 12))
    gs = fig.add_gridspec(3, num_cols, height_ratios=[1.5, 1.5, 1])

    # --- ROW 0: Ground Truth ---
    ax_gt0 = fig.add_subplot(gs[0, 0], projection='3d')
    plot_pc(ax_gt0, gt_seq[0], 'black', "GT Start (Step 0)")

    ax_gt1 = fig.add_subplot(gs[0, 1], projection='3d')
    plot_pc(ax_gt1, gt_seq[1] if len(gt_seq) > 1 else gt_seq[0], 'grey', "GT Step 1")

    for i, idx in enumerate(display_rec_indices):
        ax = fig.add_subplot(gs[0, 2 + i], projection='3d')
        if idx < len(gt_seq):
            plot_pc(ax, gt_seq[idx], 'grey', f"GT Step {idx + 1}")

    # --- ROW 1: Model Predictions ---
    ax_m0 = fig.add_subplot(gs[1, 0], projection='3d')
    plot_pc(ax_m0, rec_preds[0], 'black', "Input (Step 0)")

    ax_m1 = fig.add_subplot(gs[1, 1], projection='3d')
    plot_pc(ax_m1, rec_preds[1], 'blue', "Model Pred ($\hat{x}_1$)")

    for i, idx in enumerate(display_rec_indices):
        ax = fig.add_subplot(gs[1, 2 + i], projection='3d')
        plot_pc(ax, rec_preds[idx], 'red', f"Recursive Pred Step {idx + 1}")

    # --- ROW 2: Truncated Loss Curve ---
    ax_loss = fig.add_subplot(gs[2, :])
    
    # Slice the data to stop at the last displayed step
    truncated_losses = losses[:last_step_idx]
    # truncated_dev_losses = dev_losses[:last_step_idx]
    truncated_rec_losses = rec_losses[:last_step_idx]
    steps = np.arange(1, len(truncated_losses) + 1)
    
    ax_loss.plot(steps, truncated_losses, label='Single Step Error', color='blue', marker='o', markersize=4, alpha=0.4)
    # ax_loss.plot(steps, truncated_dev_losses, label='Recursive Accumulation Error', color='cyan', linewidth=2, marker='x', markersize=4)
    ax_loss.plot(steps, truncated_rec_losses, label='Recursive Accumulation Error', color='red', linewidth=2, marker='x', markersize=4)

    ax_loss.set_title(f"Error Accumulation (Truncated at Step {last_step_idx})")
    ax_loss.set_xlabel("Step Number")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_xticks(steps) # Ensure we only show relevant step ticks
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path)

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plotly_eval_series(gt_seq, single_preds, rec_preds, losses, rec_losses, 
                          n_step=2, max_cols=6, mesh_func=None, fig_path=None):
    """
    Interactive Plotly version with synchronized spinning and optional meshing.
    """
    # 1. Determine Display Indices
    rec_indices = list(range(n_step - 1, len(rec_preds), n_step))
    display_rec_indices = rec_indices[:max_cols - 2] 
    num_cols = 2 + len(display_rec_indices)
    last_step_idx = display_rec_indices[-1] + 1 if display_rec_indices else 1
    
    # 2. Initialize Subplots
    # Rows 1 & 2 are 'scene' (3D), Row 3 is 'xy' (2D)
    specs = [[{'type': 'scene'} for _ in range(num_cols)],
             [{'type': 'scene'} for _ in range(num_cols)],
             [{'type': 'xy', 'colspan': num_cols}] + [None]*(num_cols-1)]
    
    fig = make_subplots(
        rows=3, cols=num_cols,
        specs=specs,
        subplot_titles=(["GT Start", "GT Step 1"] + [f"GT Step {i+1}" for i in display_rec_indices] +
                        ["Input", "Pred Step 1"] + [f"Recurse {i+1}" for i in display_rec_indices] +
                        ["Loss Curves"]),
        vertical_spacing=0.2
    )
    
    # Helper to add 3D data (Scatter or Mesh)
    def add_3d_plot(data, row, col, color, name, show_legend=False):
        if data is None: return
        # If mesh_func is provided and returns (vertices, faces)
        if mesh_func is not None:
            verts, faces = mesh_func(data)
            fig.add_trace(go.Mesh3d(
                x=verts[:,0], y=verts[:,1], z=verts[:,2],
                i=faces[:,0], j=faces[:,1], k=faces[:,2],
                color=color, opacity=0.5, name=name, showlegend=show_legend
            ), row=row, col=col)
        else:
            fig.add_trace(go.Scatter3d(
                x=data[:,0], y=data[:,1], z=data[:,2],
                mode='markers', marker=dict(size=2, color=color),
                name=name, showlegend=show_legend
            ), row=row, col=col)
    
    # --- Add 3D Traces ---
    # Row 1: GT
    add_3d_plot(gt_seq[0], 1, 1, 'black', 'GT')
    add_3d_plot(gt_seq[1] if len(gt_seq)>1 else gt_seq[0], 1, 2, 'grey', 'GT')
    for i, idx in enumerate(display_rec_indices):
        add_3d_plot(gt_seq[idx+1], 1, 3+i, 'grey', 'GT')
    
    # Row 2: Predictions
    add_3d_plot(gt_seq[0], 2, 1, 'black', 'Input')
    add_3d_plot(single_preds[0], 2, 2, 'blue', 'Pred')
    for i, idx in enumerate(display_rec_indices):
        add_3d_plot(rec_preds[idx], 2, 3+i, 'red', 'Recursive')
    
    # --- Row 3: Loss (Truncated) ---
    steps = np.arange(1, last_step_idx + 1)
    fig.add_trace(go.Scatter(x=steps, y=losses[:last_step_idx], name='Single Step', line=dict(color='blue')), row=3, col=1)
    fig.add_trace(go.Scatter(x=steps, y=rec_losses[:last_step_idx], name='Recursive', line=dict(color='red')), row=3, col=1)
    # Update all scene aspects to be square and zoom out
    for r in [1, 2]:
        for c in range(1, num_cols + 1):
            scene_idx = (r - 1) * num_cols + c
            scene_name = "scene" if scene_idx == 1 else f"scene{scene_idx}"
            fig.update_layout({
                f"{scene_name}.aspectmode": 'manual',
                f"{scene_name}.aspectratio": dict(x=1.0, y=1, z=0.8),  # Custom aspect ratio (x:y:z)
                f"{scene_name}.camera.eye": dict(x=2.0, y=2.0, z=1.0),
            })
            
    # --- Setup Turntable Animation ---
    # We define a set of frames where the camera angle 'eye' rotates around Z
    frames = []
    for t in np.linspace(0, 2*np.pi, 200):
        # Calculate camera position
        eye_x = 1.5 * np.cos(t)
        eye_y = 1.5 * np.sin(t)
        
        # We need to update every 'scene' in the subplot (rows 1 and 2)
        layout_update = {}
        for r in [1, 2]:
            for c in range(1, num_cols + 1):
                # Calculate linear scene index
                scene_idx = (r - 1) * num_cols + c
                # Plotly uses "scene" for first, "scene2", "scene3", etc. for others
                scene_name = "scene" if scene_idx == 1 else f"scene{scene_idx}"
                # Use nested dictionary structure for layout updates
                layout_update[scene_name] = {
                    'camera': {
                        'eye': dict(x=eye_x, y=eye_y, z=1.3)
                    }
                }
        
        frames.append(go.Frame(layout=layout_update))
    
    fig.frames = frames
    
    # Add Play button for the turntable
    fig.update_layout(
        height=1000,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50),  # Add this line for more padding
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play Turntable", method="animate", args=[None, {"frame": {"duration": 2, "redraw": False}}])]
        )]
    )
    
    fig.write_html(fig_path)

def plotPCbatch(pcArray1, pcArray2, pcArray3, show=True, save=False, name=None, fig_count=9, sizex=12, sizey=4):
    # Select the data from the arrays
    pc1 = pcArray1[0:fig_count]
    pc2 = pcArray2[0:fig_count]
    pc3 = pcArray3[0:fig_count]

    # Create a figure with three rows and fig_count columns
    fig = plt.figure(figsize=(sizex, sizey))
    
    for i in range(fig_count * 3):
        ax = fig.add_subplot(3, fig_count, i + 1, projection='3d')
        
        # Plot data in the first row
        if i < fig_count:
            ax.scatter(pc1[i, :, 0], pc1[i, :, 2], pc1[i, :, 1], c='b', marker='.', alpha=0.3, s=8)
        
        # Plot data in the second row
        elif i < 2 * fig_count:
            ax.scatter(pc2[i - fig_count, :, 0], pc2[i - fig_count, :, 2], pc2[i - fig_count, :, 1], c='r', marker='.', alpha=0.3, s=8)
        
        # Plot data in the third row
        else:
            ax.scatter(pc3[i - 2 * fig_count, :, 0], pc3[i - 2 * fig_count, :, 2], pc3[i - 2 * fig_count, :, 1], c='g', marker='.', alpha=0.3, s=8)

        # Hide the axis
        plt.axis('off')

    # Adjust spacing between plots
    plt.subplots_adjust(wspace=0, hspace=0)

    # Save the figure if save is True
    if save:
        fig.savefig(name)
        plt.close(fig)

    # Show the figure
    if show:
        plt.show()
    else:
        return fig