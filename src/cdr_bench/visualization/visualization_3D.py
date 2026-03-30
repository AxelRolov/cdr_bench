import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap, Normalize
from plotly.subplots import make_subplots


def plot_3d_with_projections(x, y, z, values, highlight_indices=None):
    """
    Plots a 3D scatter plot along with its 2D projections onto the XY, XZ, and YZ planes.
    Highlights specified points across all plots.

    Parameters:
        x (array-like): X coordinates.
        y (array-like): Y coordinates.
        z (array-like): Z coordinates.
        values (array-like): Values at each coordinate, used for coloring.
        highlight_indices (list): Indices of points to highlight.
    """
    if highlight_indices is None:
        highlight_indices = []

    # Setup markers
    marker_props = dict(size=20 * values / values.max(), color=values, colorscale="Greys", opacity=0.5)

    marker_props_2d = dict(
        size=12,  # * values / values.max(),
        color=values,
        colorscale="Greys",
        opacity=1.0,
    )

    # Create the scatter plots
    traces = {
        "3D": go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=marker_props),
        "XY": go.Scatter(x=x, y=y, mode="markers", marker=marker_props_2d, showlegend=False),
        "XZ": go.Scatter(x=x, y=z, mode="markers", marker=marker_props_2d, showlegend=False),
        "YZ": go.Scatter(x=y, y=z, mode="markers", marker=marker_props_2d, showlegend=False),
    }

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "scatter3d", "colspan": 3}, None, None],
            [{"type": "scatter"}, {"type": "scatter"}, {"type": "scatter"}],
        ],
        subplot_titles=("3D Scatter Plot", "Projection XY", "Projection XZ", "Projection YZ"),
        row_heights=[0.6, 0.4],
    )

    # Add traces to the figure
    fig.add_trace(traces["3D"], row=1, col=1)
    fig.add_trace(traces["XY"], row=2, col=1)
    fig.add_trace(traces["XZ"], row=2, col=2)
    fig.add_trace(traces["YZ"], row=2, col=3)

    # Update layouts and axes for 2D plots
    fig.update_xaxes(title_text="X", row=2, col=1, showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Y", row=2, col=1, showgrid=True, gridcolor="lightgrey")

    fig.update_xaxes(title_text="X", row=2, col=2, showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Z", row=2, col=2, showgrid=True, gridcolor="lightgrey")

    fig.update_xaxes(title_text="Y", row=2, col=3, showgrid=True, gridcolor="lightgrey")
    fig.update_yaxes(title_text="Z", row=2, col=3, showgrid=True, gridcolor="lightgrey")

    # General layout updates
    fig.update_layout(
        height=800, width=800, title_text="3D Plot with 2D Projections", plot_bgcolor="white", paper_bgcolor="white"
    )

    # Show the figure
    fig.show()


def plot_custom_contour_density(
    gtm, grid_dim, values, cmap_start=0, vmin=None, vmax=None, figsize=(8, 6), plot_type="voxels"
):
    # Initialize the plot with custom colormap and normalization
    norm = Normalize(
        vmin=vmin if vmin is not None else np.min(values), vmax=vmax if vmax is not None else np.max(values)
    )
    new_greys = plt.cm.Greys(np.linspace(cmap_start, 1, 256))
    new_greys_cmap = LinearSegmentedColormap.from_list("NewGreys", new_greys)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Setup voxel grid and values
    voxel_values = values.reshape((grid_dim, grid_dim, grid_dim))
    filled = voxel_values > np.min(voxel_values)  # Example condition to fill voxels

    # Voxel coloring based on values
    colors = new_greys_cmap(norm(voxel_values))

    # Plotting voxels
    ax.voxels(filled, facecolors=colors, edgecolor="k", alpha=0.6)
    ax.set_title("Voxel Plot with Custom Coordinates and Colors")
    plt.show()
