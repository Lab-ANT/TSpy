'''
Created by Chengyu on 2021/12/12.
Views defined in StateCorr.
'''

import numpy as np
import matplotlib.pyplot as plt
from TSpy.utils import z_normalize,calculate_density_matrix
# import scipy
# from stview.utils import calculate_scalar_velocity_list, find, normalize, standardize, find, calculate_density_matrix, calculate_velocity_list
# import stview.colors as sc
# from sklearn.cluster import OPTICS
# import matplotlib.gridspec as gridspec
# import seaborn as sns

def embedding_space(embeddings, label=None, alpha=0.8, s=0.1, color='blue', show=False):
    embeddings = np.array(embeddings)
    x = embeddings[:,0]
    y = embeddings[:,1]
    plt.style.use('bmh')
    plt.figure(figsize=(4,4))
    if label:
        pass
    else:
        plt.scatter(x,y,alpha=alpha,s=s)
    if show:
        plt.show()

def density_map_3d(feature_list, n=100, show=False, figsize=(6,6), op=None, t = 1):
    density_matrix,x_s,x_e,y_s,y_e = calculate_density_matrix(feature_list,n)

    density_matrix = z_normalize(density_matrix)

    x, y = np.meshgrid(np.linspace(x_s, x_e, n),
                   np.linspace(y_s, y_e, n))

    if op == 'normalize':
        density_matrix = z_normalize(density_matrix)
    # elif op == 'standardize':
    #     density_matrix = standardize(density_matrix)

    # Plot the surface.
    fig = plt.figure(figsize=figsize)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, density_matrix, cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False, vmax=t)

    # Customize the z axis.
    # ax.set_zlim(-1.01, 1.01)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    if show:
        plt.show()

# color = 'viridis','plasma','inferno','cividis','magma'
# pre-process = 'normalize', 'standardize'
def density_map(feature_list, n=101, show=False, figsize=(4,4), fontsize=10, color = 'plasma', t=1):
    density_matrix,_,_,_,_ = calculate_density_matrix(feature_list,n)

    # normalize
    density_matrix = z_normalize(density_matrix)

    max = np.max(density_matrix)
    min = np.min(density_matrix)
    
    plt.figure(figsize=figsize)
    plt.matshow(density_matrix, cmap=color,vmax=max*t,vmin=min,fignum=0)
    plt.tick_params(labelsize=fontsize)
    cb = plt.colorbar(fraction=0.045)
    cb.ax.tick_params(labelsize=fontsize)
    if show:
        plt.show()
    return density_matrix


# noise = np.random.rand(30000)
# embedding_space(noise.reshape((5000,2)),show=True)
# density_map(noise.reshape((5000,2)),show=True)
# density_map_3d(noise.reshape((10000,3)),show=True)