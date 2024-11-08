import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
from matplotlib.colors import Normalize, LinearSegmentedColormap

from paper_utils import *
from PIL import Image
import io
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_tangent_planes_3d(tangent_bundle, data_input, angle = [3,3,4], zoom = None):

    data = data_input[:,:3]
    nbrs = NearestNeighbors(n_neighbors = 4, algorithm='auto').fit(data)
    distances, _ = nbrs.kneighbors(data)
    rho = distances[:,1:].mean(axis=1)
    tangent_bundle_rescaled = 0.4*np.median(rho) * tangent_bundle[:,:3]

    n = data.shape[0]

    p1 = data + tangent_bundle_rescaled[:,:,0] + tangent_bundle_rescaled[:,:,1]
    p2 = data + tangent_bundle_rescaled[:,:,1] - tangent_bundle_rescaled[:,:,0]
    p3 = data - tangent_bundle_rescaled[:,:,0] - tangent_bundle_rescaled[:,:,1]
    p4 = data - tangent_bundle_rescaled[:,:,1] + tangent_bundle_rescaled[:,:,0]
    all_p = np.concatenate((p1, p2, p3, p4), axis = 0)

    indices = np.arange(n)
    simplices_1 = np.stack((indices, indices + n, indices + 2*n)).T
    simplices_2 = np.stack((indices, indices + 2*n, indices + 3*n)).T
    simplices = np.concatenate((simplices_1, simplices_2), axis = 0)
    simplices = simplices.reshape(2*n, data.shape[1])

    fig = ff.create_trisurf(x=all_p[:,0], 
                            y=all_p[:,1], 
                            z=all_p[:,2],
                            # colormap=["#166dde", "#d3d3d3", "#e32636"],
                            colormap=["#8ab6ee", "#e53b4a"],
                            simplices=simplices,
                            show_colorbar = False,
                            plot_edges = False,
                            title=None)
    
    angle_a = np.array(angle, dtype=float)
    angle_a /= np.linalg.norm(angle_a)

    scale = np.linalg.norm(data,axis = 1).max()
    if zoom is None:
        zoom = 1/scale
    scale *= zoom
    camera = dict(
        eye=dict(x=6*angle_a[0]/scale, 
                y=6*angle_a[1]/scale, 
                z=8*angle_a[2]/scale),  # Camera position
        center=dict(x=0, 
                    y=0, 
                    z=0),         # Point to look at
        up=dict(x=0, y=0, z=1)              # Up vector
    )
    
    fig.update_layout(width=1000, height=700, margin=dict(l=0, r=0, t=0, b=0))

    fig.update_layout(scene_camera=camera,
                      scene=dict(aspectmode='data',
                                xaxis = dict(visible=False),
                                yaxis = dict(visible=False),
                                zaxis =dict(visible=False)))
    
    # fig.show()

    return fig

def plot_tangent_lines_3d(tangent_bundle, data, angle = [3,3,4], zoom = None):

    nbrs = NearestNeighbors(n_neighbors = 4, algorithm='auto').fit(data)
    distances, _ = nbrs.kneighbors(data)
    rho = distances[:,1:].mean(axis=1)
    tangent_bundle_rescaled = 0.5*np.median(rho) * tangent_bundle

    n = data.shape[0]

    p1 = data + tangent_bundle_rescaled[:,:,0]
    p2 = data - tangent_bundle_rescaled[:,:,0]
    all_p = np.concatenate((p1, p2, p1), axis = 0)

    indices = np.arange(n)
    simplices = np.stack((indices, indices + n, indices + 2*n)).T

    fig = ff.create_trisurf(x=all_p[:,0], 
                            y=all_p[:,1], 
                            z=all_p[:,2],
                            # colormap=["#166dde", "#d3d3d3", "#e32636"],
                            # colormap=["#8ab6ee", "#e53b4a"],
                            simplices=simplices,
                            show_colorbar = False,
                            plot_edges = False,
                            title=None)
    
    angle_a = np.array(angle, dtype=float)
    angle_a /= np.linalg.norm(angle_a)

    scale = np.linalg.norm(data,axis = 1).max()
    if zoom is None:
        zoom = 1/scale
    scale *= zoom
    camera = dict(
        eye=dict(x=6*angle_a[0]/scale, 
                y=6*angle_a[1]/scale, 
                z=8*angle_a[2]/scale),  # Camera position
        center=dict(x=0, 
                    y=0, 
                    z=0),         # Point to look at
        up=dict(x=0, y=0, z=1)              # Up vector
    )
    
    fig.update_layout(width=1000, height=700, margin=dict(l=0, r=0, t=0, b=0))

    fig.update_layout(scene_camera=camera,
                      scene=dict(aspectmode='data',
                                xaxis = dict(visible=False),
                                yaxis = dict(visible=False),
                                zaxis =dict(visible=False)))
    
    # fig.show()

    return fig

def plot_tangent_lines_2d(tangent_bundle, data, xrange = None, yrange = None):

    nbrs = NearestNeighbors(n_neighbors = 4, algorithm='auto').fit(data)
    distances, _ = nbrs.kneighbors(data)
    rho = distances[:,1:].mean(axis=1)
    tangent_bundle_rescaled = 3*np.median(rho) * tangent_bundle[:,:,0]
    
    fig = ff.create_quiver(data[:,0], 
                           data[:,1], 
                           tangent_bundle_rescaled[:,0], 
                           tangent_bundle_rescaled[:,1],
                        scale=0.2,
                        arrow_scale=0.00001,
                        line_width=3,
                        marker=dict(color='black'))
    
    fig2 = ff.create_quiver(data[:,0], 
                           data[:,1], 
                           -tangent_bundle_rescaled[:,0], 
                           -tangent_bundle_rescaled[:,1],
                        scale=0.2,
                        arrow_scale=0.00001,
                        line_width=3,
                        marker=dict(color='black'))

    fig.add_traces(data = fig2.data)

    if xrange is None:
        xrange = [data[:,0].min() - 0.05, data[:,0].max() + 0.05]
    if yrange is None:
        yrange = [data[:,1].min() - 0.05, data[:,1].max() + 0.05]


    fig.update_layout(width=1000, 
                      height=700, 
                      margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=False, 
                    paper_bgcolor='white', 
                    plot_bgcolor='white', 
                    xaxis_visible=False, 
                    yaxis_visible=False,
                    xaxis_range = xrange,
                    yaxis_range = yrange)

    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    
    # fig.show()

    return fig

def plot_curvature_3d(data, S, title, range=None):
    opacity = np.absolute(S)
    if range is None:
        opacity /= opacity.max()
    else:
        opacity /= max(abs(range[0]),abs(range[1]))
    opacity = np.clip(opacity,0,1)
    fig = px.scatter_3d(x = data[:,0],
                y = data[:,1],
                z = data[:,2],
                range_color = range,
                size = opacity,
                title = title,
                color=S)
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.show()

def plot_scalar_3d(data, S, range_col, angle, zoom, centre = [0,0,0]):

    colormap=["#166dde", 
              "#d3d3d3", 
              "#e53b4a"]

    fig = px.scatter_3d(x = data[:,0],
                y = data[:,1],
                z = data[:,2],
                color = S,
                color_continuous_scale = colormap,
                # size = [30]*data.shape[0],
                opacity = 1.,
                range_color = range_col)
    fig.update_traces(marker=dict(line=dict(width=0)))

    angle_a = np.array(angle, dtype=float)
    angle_a /= np.linalg.norm(angle_a)

    scale = np.linalg.norm(data,axis = 1).max()
    if zoom is None:
        zoom = 1/scale
    scale *= zoom
    camera = dict(
        eye=dict(x=6*angle_a[0]/scale, 
                y=6*angle_a[1]/scale, 
                z=8*angle_a[2]/scale),  # Camera position
        center=dict(x=centre[0], 
                    y=centre[1], 
                    z=centre[2]),         # Point to look at
        up=dict(x=0, y=0, z=1)              # Up vector
    )
    fig.update_layout(width=1000, height=800, margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_showscale=False)
    fig.update_layout(scene_camera=camera,
                        scene=dict(aspectmode='data',
                                    xaxis = dict(visible=False),
                                    yaxis = dict(visible=False),
                                    zaxis =dict(visible=False)))
    return fig

def plot_scalar_3d_torus(data, S, angle, zoom, centre = [0,0,0]):

    colormap=["#166dde",
              "#75a0d9",
                "#d3d3d3", 
                "#e53b4a"]

    fig = px.scatter_3d(x = data[:,0],
                y = data[:,1],
                z = data[:,2],
                color = S,
                color_continuous_scale = colormap,
                # size = [30]*data.shape[0],
                opacity = 1.,
                range_color = [-2,1])
    fig.update_traces(marker=dict(line=dict(width=0)))

    angle_a = np.array(angle, dtype=float)
    angle_a /= np.linalg.norm(angle_a)

    scale = np.linalg.norm(data,axis = 1).max()
    if zoom is None:
        zoom = 1/scale
    scale *= zoom
    camera = dict(
        eye=dict(x=6*angle_a[0]/scale, 
                y=6*angle_a[1]/scale, 
                z=8*angle_a[2]/scale),  # Camera position
        center=dict(x=centre[0], 
                    y=centre[1], 
                    z=centre[2]),         # Point to look at
        up=dict(x=0, y=0, z=1)              # Up vector
    )
    fig.update_layout(width=1000, height=800, margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_showscale=False)
    fig.update_layout(scene_camera=camera,
                        scene=dict(aspectmode='data',
                                    xaxis = dict(visible=False),
                                    yaxis = dict(visible=False),
                                    zaxis =dict(visible=False)))
    return fig

def plot_curvature_2d(data, S, title, range=None):
    fig = px.scatter(x = data[:,0],
                y = data[:,1],
                color=S,
                range_color = range,
                title = title
                )
    fig.update_traces(marker=dict(size=20))
    fig.update_layout(
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleratio=1)
    )
    fig.show()

def plot_dims_3d(data, dims, angle, zoom):

    colormap=["black", "#e53b4a", "#8ab6ee", "#d3d3d3"]

    fig = px.scatter_3d(x = data[:,0],
                y = data[:,1],
                z = data[:,2],
                color = dims,
                color_continuous_scale = colormap,
                size = [30]*data.shape[0],
                opacity = 1.,
                range_color = [0,3])
    fig.update_traces(marker=dict(line=dict(width=0)))

    angle_a = np.array(angle, dtype=float)
    angle_a /= np.linalg.norm(angle_a)

    scale = np.linalg.norm(data,axis = 1).max()
    if zoom is None:
        zoom = 1/scale
    scale *= zoom
    camera = dict(
        eye=dict(x=6*angle_a[0]/scale, 
                y=6*angle_a[1]/scale, 
                z=8*angle_a[2]/scale),  # Camera position
        center=dict(x=0, 
                    y=0, 
                    z=0),         # Point to look at
        up=dict(x=0, y=0, z=1)              # Up vector
    )
    fig.update_layout(width=800, height=600, margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_showscale=False)
    fig.update_layout(scene_camera=camera,
                        scene=dict(aspectmode='data',
                                    xaxis = dict(visible=False),
                                    yaxis = dict(visible=False),
                                    zaxis =dict(visible=False)))
    return fig

def plot_dims_2d(data, dims):

    colormap=["black", "#e53b4a", "#8ab6ee", "#d3d3d3"]

    fig = px.scatter(x = data[:,0],
                y = data[:,1],
                color = dims,
                color_continuous_scale = colormap,
                opacity = 1.,
                range_color = [0,3])
    fig.update_traces(marker=dict(line=dict(width=0)))

    fig.update_layout(width=600, height=500, margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_showscale=False)
    fig.update_layout(
    xaxis=dict(scaleanchor='y', scaleratio=1, visible=False),  # Hide x-axis
    yaxis=dict(scaleanchor='x', scaleratio=1, visible=False),  # Hide y-axis
    paper_bgcolor='rgba(0,0,0,0)',  # Hide outer (paper) background
    plot_bgcolor='rgba(0,0,0,0)'    # Hide inner (plot) background
    )
    return fig

def fig_array(fig):
    # fig.show()
    fig_image = fig.to_image(format='png')
    return np.array(Image.open(io.BytesIO(fig_image)))

def plot_ground_truth_torus(r, R, data, S):

    phi = phi_torus(data, R)
    # true_scalar = 2*np.cos(phi)/(r*(R + r*np.cos(phi)))
    true_scalar_sorted = 2*np.cos(np.sort(phi))/(r*(R + r*np.cos(np.sort(phi))))
    range_color=[-2.5, 1.5]


    # Create the scatter plot
    plt.figure(figsize=(10, 6))

    colormap=["#166dde", 
                "#d3d3d3", 
                "#e53b4a"]
    cmap = LinearSegmentedColormap.from_list('c', colormap)
    norm = Normalize(vmin=-max(abs(S)), vmax=max(abs(S)))

    fig = plt.scatter(phi, S, label='Scalar Curvature', 
                # color='#166dde',
                c = S,
                cmap = cmap,
                norm = norm,
                s=20)

    # Create the line plot
    plt.plot(np.sort(phi), true_scalar_sorted, linewidth=2, 
            #  color='#e53b4a',
            color = 'k',
            label='True Scalar')

    # Set the background color to white
    plt.gca().set_facecolor('white')

    # Set the x-axis ticks and labels
    plt.xticks(
        # ticks=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
        ticks=[np.pi, 2*np.pi],
        # labels=[r'0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'],
        labels=[r'$\pi$', r'$2\pi$'],
        color='black'
    )
    plt.xlim(-0, 2 * np.pi + 0.02)

    # Set y-axis ticks and color
    plt.yticks(np.arange(-2, 2, 1), color='black')

    # Position y-ticks so that they align with the x-axis ticks at y=0
    plt.ylim(range_color[0], range_color[1])  # Adjust y-limits as needed

    # Add labels
    plt.xlabel(r'$\phi$ angle')
    plt.ylabel('Scalar curvature')

    # Remove gridlines
    plt.grid(False)

    # Hide the top and right spines (axes)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Move the x-axis to y=0
    plt.gca().spines['bottom'].set_position(('data', 0))

    # Show the plot
    plt.show()

    return fig

def plot_ground_truth_torus_axis(r, R, data, S, ax):

    phi = phi_torus(data, R)
    # true_scalar = 2*np.cos(phi)/(r*(R + r*np.cos(phi)))
    true_scalar_sorted = 2*np.cos(np.sort(phi))/(r*(R + r*np.cos(np.sort(phi))))
    range_color=[-2.5, 1.5]


    # Create the scatter plot
    # plt.figure(figsize=(10, 6))

    colormap=["#166dde",
              "#75a0d9",
                "#d3d3d3", 
                "#e53b4a"]
    cmap = LinearSegmentedColormap.from_list('c', colormap)
    norm = Normalize(vmin=-2, vmax=1)

    ax.scatter(phi, S, label='Scalar Curvature', 
                # color='#166dde',
                c = S,
                cmap = cmap,
                norm = norm,
                s=20)

    # Create the line plot
    ax.plot(np.sort(phi), true_scalar_sorted, linewidth=2, 
            #  color='#e53b4a',
            color = 'k',
            label='True Scalar')

    # Set the background color to white
    # ax.gca().set_facecolor('white')

    # Set the x-axis ticks and labels
    ax.set_xticks(
        # ticks=[0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
        ticks=[np.pi, 2*np.pi],
        # labels=[r'0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'],
        labels=[r'$\pi$', r'$2\pi$'],
        color='black'
    )
    ax.set_xlim(-0, 2 * np.pi + 0.02)

    # Set y-axis ticks and color
    ax.set_yticks(np.arange(-2, 2, 1))

    # Position y-ticks so that they align with the x-axis ticks at y=0
    ax.set_ylim(range_color[0], range_color[1])  # Adjust y-limits as needed

    # Add labels
    ax.set_xlabel(r'$\phi$ angle')
    ax.set_ylabel('Scalar curvature')

    # Remove gridlines
    # ax.grid(False)

    # Hide the top and right spines (axes)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Move the x-axis to y=0
    ax.spines['bottom'].set_position(('data', 0))