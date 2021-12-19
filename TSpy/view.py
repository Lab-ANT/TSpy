'''
Created by Chengyu on 2021/12/12.
Views defined in StateCorr.
'''

import numpy as np
import matplotlib.pyplot as plt
from TSpy.utils import z_normalize,calculate_density_matrix, calculate_velocity_list, find
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

# arrow map.
def arrow_map(feature_list, n=100, t=100):
    feature_list = np.array(feature_list)
    x = feature_list[:,0]
    y = feature_list[:,1]
    velocity_list_x, velocity_list_y = calculate_velocity_list(feature_list,interval=t)
    h = w = n
    h_start = np.min(y)
    h_end = np.max(y)
    h_step = (h_end-h_start)/h
    w_start = np.min(x)
    w_end = np.max(x)
    w_step = (w_end-w_start)/w

    row_partition = []
    for i in range(h):
        row_partition.append(find(y,h_start+i*h_step,h_start+(i+1)*h_step))

    U = []
    V = []
    for col_idx in row_partition:
        col = x[col_idx]
        U_col = []
        V_col = []
        for i in range(w):
            idx = find(col,w_start+i*w_step,w_start+(i+1)*w_step)
            x_list = velocity_list_x[idx]
            x_mean = np.mean(x_list)
            y_list = velocity_list_y[idx]
            y_mean = np.mean(y_list)
            U_col.append(x_mean)
            V_col.append(y_mean)
        U.append(np.array(U_col))
        V.append(np.array(V_col))
    U = np.array(U)
    V = np.array(V)
    # U=U.T
    # V=V.T
    U[np.isnan(U)]=0
    V[np.isnan(V)]=0
    # U = normalize(U)
    # V = normalize(V)

    x_, y_ = np.meshgrid(np.linspace(-3, 3, n),
                   np.linspace(-3, 3, n))
    M = np.hypot(U, V)
    fig1, ax1 = plt.subplots()
    ax1.set_title('Arrows scale with plot width, not view')
    Q = ax1.quiver(x_, y_, U, V, M, units='width')

    plt.show()
    # fig3, ax3 = plt.subplots()
    # ax3.set_title("pivot='tip'; scales with x view")
    # M = np.hypot(U, V)
    # Q = ax3.quiver(x_, y_, U, V, M, units='x', pivot='tip', width=0.022,
    #            scale=1 / 0.15)
    # qk = ax3.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
    #                coordinates='figure')
    # ax3.scatter(x_, y_, color='0.5', s=1)

def flow_map(feature_list, n=50, t=100):
    feature_list = np.array(feature_list)
    x = feature_list[:,0]
    y = feature_list[:,1]
    velocity_list_x, velocity_list_y = calculate_velocity_list(feature_list,interval=t)
    h = w = n
    h_start = np.min(y)
    h_end = np.max(y)
    h_step = (h_end-h_start)/h
    w_start = np.min(x)
    w_end = np.max(x)
    w_step = (w_end-w_start)/w

    row_partition = []
    for i in range(h):
        row_partition.append(find(y,h_start+i*h_step,h_start+(i+1)*h_step))
    
    row_partition = list(reversed(row_partition))

    U = []
    V = []
    for col_idx in row_partition:
        col = x[col_idx]
        U_col = []
        V_col = []
        for i in range(w):
            idx = find(col,w_start+i*w_step,w_start+(i+1)*w_step)
            x_list = velocity_list_x[idx]
            x_mean = np.mean(x_list)
            y_list = velocity_list_y[idx]
            y_mean = np.mean(y_list)
            U_col.append(x_mean)
            V_col.append(y_mean)
            # print(U_col)
        U.append(np.array(U_col))
        V.append(np.array(V_col))
    U = np.array(U)
    V = np.array(V)
    U[np.isnan(U)]=0
    V[np.isnan(V)]=0
    # U = normalize(U)
    # V = normalize(V)

    x_, y_ = np.meshgrid(np.linspace(-3, 3, n),
                   np.linspace(-3, 3, n))

    fig0, ax0 = plt.subplots()
    strm = ax0.streamplot(x_, y_, U, V, color=U, linewidth=2, cmap=plt.cm.autumn)
    fig0.colorbar(strm.lines)
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