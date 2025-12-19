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

def plot_network_weights(state_dict):
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
    plt.show()
    print(f"Histogram saved! Total parameters: {len(all_weights):,}")
    print(f"Weight statistics:")
    print(f"  Mean: {mean_val:.6f}")
    print(f"  Std: {std_val:.6f}")
    print(f"  Min: {min_val:.6f}")
    print(f"  Max: {max_val:.6f}")

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

def compare_scatters(pc1, pc2, label_1=None, label_2=None):
    fig = plt.figure(figsize=(20,8))
    ax1 = fig.add_subplot(131, projection = '3d')
    ax1.scatter(xs=pc1[:,0],ys=pc1[:,1],zs=pc1[:,2], s=2)
    ax1.set_title(label_1)
    ax2 = fig.add_subplot(132, projection = '3d')
    ax2.scatter(xs=pc2[:,0],ys=pc2[:,1],zs=pc2[:,2], s=2)
    ax2.set_title(label_2)
    ax3 = fig.add_subplot(133, projection = '3d')

    ax3.scatter(xs=pc1[:,0],ys=pc1[:,1],zs=pc1[:,2], s=1, color='blue')
    ax3.scatter(xs=pc2[:,0],ys=pc2[:,1],zs=pc2[:,2], s=1, color='red')
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

def visualize_point_cloud(pc,a, point_size=5):
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
    plotter.show()

def visualize_point_diff(pc1, pc2, point_size=5):
    # cloud = pv.PolyData(pc)
    plotter = pv.Plotter()
    # plotter.add_points(cloud, point_size=point_size, color="red")
    start = pc1
    direction = pc2 - pc1
    # Add arrow
    plotter.add_arrows(start, direction, mag=1.0, color="blue")
    plotter.show_axes()
    plotter.show()

def visualize_vector_diff(pc1, pc2, mesh1=None, mesh2=None, min_magnitude=2.0):
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
    
    plotter.show()

def compare_vector_fields(x_t, x_tp1, x_hat, point_size=5):
    plotter = pv.Plotter(shape=(2, 3), window_size=(2000,1000))  # 3 row, 2 columns
    
    start = x_t
    direction = x_tp1 - x_t
    mag = 2.0
    # View 1: Looking down Z (top view)
    plotter.subplot(0, 0)
    plotter.add_arrows(start, direction, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_xy()  # Look down Z axis
    plotter.camera.zoom(1.2)
    plotter.add_text("Top View (Z)", font_size=10)
    
    # View 2: Looking down X (side view)
    plotter.subplot(0, 1)
    plotter.add_arrows(start, direction, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_yz()  # Look down X axis
    plotter.camera.zoom(1.2)
    plotter.add_text("Side View (X)", font_size=10)

    # View 3: Looking down Y (front view)
    plotter.subplot(0, 2)
    plotter.add_arrows(start, direction, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_xz()  # Look down Y axis
    plotter.camera.zoom(1.2)
    plotter.add_text("Front View (Y)", font_size=10)

    direction_hat = x_hat - x_t
    # View 1: Looking down Z (top view)
    plotter.subplot(1, 0)
    plotter.add_arrows(start, direction_hat, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_xy()  # Look down Z axis
    plotter.camera.zoom(1.2)
    plotter.add_text("Top View (Z)", font_size=10)

    # View 2: Looking down X (side view)
    plotter.subplot(1, 1)
    plotter.add_arrows(start, direction_hat, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_yz()  # Look down X axis
    plotter.camera.zoom(1.2)
    plotter.add_text("Side View (X)", font_size=10)

    
    # View 3: Looking down Y (front view)
    plotter.subplot(1, 2)
    plotter.add_arrows(start, direction_hat, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_xz()  # Look down Y axis
    plotter.camera.zoom(1.2)
    plotter.add_text("Front View (Y)", font_size=10)
    plotter.show()

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
        fig.savefig(name + '.png')
        plt.close(fig)

    # Show the figure
    if show:
        plt.show()
    else:
        return fig