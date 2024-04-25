import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sbn

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
savefig_opyions = dict(format="png", dpi=300, bbox_inches="tight")


from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

x = np.array([[3,3], [1,1],[2, 2], [2,2], [1,1]])
y = np.array([[2,2],[0,0],[0,0], [3,3], [3,3], [1,1], [0,0]])
def compute_euclidean_distance_matrix(x, y) -> np.array:

    dist = np.zeros((len(y),len(x)))

    for i in range(len(y)):
        for j in range(len(x)):
            dist[i,j] = (x[j,0]-y[i,0])**2

    return dist

def compute_accumulated_cost_matrix(x,y) -> np.array:
    """Compute accumulated cost matrix for warp path using Euclidean distance
    """
    distances = compute_euclidean_distance_matrix(x,y)

    # Initialization
    cost = np.zeros((len(y),len(x)))
    cost[0,0] = distances[0,0]

    for i in range(1,len(y)):
        cost[i,0] =distances[i,0] + cost[i-1,0]

    for j in range(1,len(x)):
        cost[0,j] = distances[0,j] + cost[0,j-1]


    #Accumulating warp path cost

    for i in range(1,len(y)):
        for j in range(1,len(x)):
            cost[i,j] = min( 
                cost[i-1,j],      #insertion
                cost[i, j-1],     #deletion
                cost[i-1, j-1]    #match
                ) + distances[i, j]

    return cost


dtw_distance,warp_path = fastdtw(x,y, dist=euclidean)

cost_matrix = compute_accumulated_cost_matrix(x,y)

fig, ax = plt.subplots(figsize=(12,8))
ax = sbn.heatmap(cost_matrix,annot=None, square=True, linewidth=0.1, cmap="YlGnBu",ax=ax )
ax.invert_yaxis()

# Get warp path in x and y directions
path_x = [p[0] for p in warp_path]
path_y = [p[1] for p in warp_path]

#Align the path from the centre
path_xx = [x+0.5 for x in path_x]
path_yy = [y+0.5 for y in path_y]

ax.plot(path_xx, path_yy, color='blue', linewidth=3, alpha=0.2)

fig.savefig("ex1_heatmap.png")
#fig.savefig("ex1_heatmap.png", **savefig_optional)

#print(np.flipud(cost_matrix))
print("DIST:", dtw_distance)
print("WP", warp_path)

fig, ax = plt.subplots(figsize=(14, 10))

#Remove border/ticks

fig.patch.set_visible(False)
ax.axis('off')

for [map_x, map_y] in warp_path:
    ax.plot([map_x, map_y], [x[map_x], y[map_y]], '--k', linewidth=4)

ax.plot(x, '-ro', label='x', linewidth=4, markersize=20, markerfacecolor='lightcoral',markeredgecolor='lightcoral')
ax.plot(y, '-bo', label='y',linewidth=4, markersize=20, markerfacecolor='skyblue', markeredgecolor='skyblue')
ax.set_title("DTW Distance", fontsize=28, fontweight="bold")

fig.savefig("ex1_dtw_distance.png")
