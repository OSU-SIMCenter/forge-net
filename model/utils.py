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


def compare_scatters(pc1, pc2, label_1=None, label_2=None):
    fig = plt.figure(figsize=(16,10))
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

def visualize_vector_diff(pc1, pc2, point_size=5):
    plotter = pv.Plotter(shape=(3, 1), window_size=(1000,2000))  # 1 row, 3 columns
    
    start = pc1
    direction = pc2 - pc1
    mag = 2.0
    # View 1: Looking down Z (top view)
    plotter.subplot(0, 0)
    plotter.add_arrows(start, direction, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_xy()  # Look down Z axis
    plotter.camera.zoom(1.2)
    plotter.add_text("Top View (Z)", font_size=10)
    
    # View 2: Looking down X (side view)
    plotter.subplot(1, 0)
    plotter.add_arrows(start, direction, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_yz()  # Look down X axis
    plotter.camera.zoom(1.2)
    plotter.add_text("Side View (X)", font_size=10)

    
    # View 3: Looking down Y (front view)
    plotter.subplot(2, 0)
    plotter.add_arrows(start, direction, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_xz()  # Look down Y axis
    plotter.camera.zoom(1.2)
    plotter.add_text("Front View (Y)", font_size=10)
    plotter.show()


def compare_vector_fields(x_t, x_tp1, x_hat, point_size=5):
    plotter = pv.Plotter(shape=(3, 2), window_size=(1000,2000))  # 3 row, 2 columns
    
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
    plotter.subplot(1, 0)
    plotter.add_arrows(start, direction, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_yz()  # Look down X axis
    plotter.camera.zoom(1.2)
    plotter.add_text("Side View (X)", font_size=10)

    # View 3: Looking down Y (front view)
    plotter.subplot(2, 0)
    plotter.add_arrows(start, direction, mag=mag, color="blue")
    plotter.show_axes()
    plotter.view_xz()  # Look down Y axis
    plotter.camera.zoom(1.2)
    plotter.add_text("Front View (Y)", font_size=10)

    direction_hat = x_hat - x_t
    # View 1: Looking down Z (top view)
    plotter.subplot(0, 1)
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
    plotter.subplot(2, 1)
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