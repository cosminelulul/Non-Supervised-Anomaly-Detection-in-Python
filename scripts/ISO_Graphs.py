import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n_samples = 150
data = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], n_samples)
x, y = data[:, 0], data[:, 1]

xi_idx = np.argmin(np.linalg.norm(data - [-1, -1.2], axis=1)) # Punct normal
xj_idx = np.argmin(np.linalg.norm(data - [3, 0], axis=1))    # Punct anormal

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

def plot_isolation(ax, target_idx, label, title, num_splits):
    ax.scatter(x, y, c='blue', s=30, edgecolors='none', alpha=0.8)
    
    # Limitele plot-ului
    xmin, xmax, ymin, ymax = -4, 4, -4, 4
    
    # Desenăm tăieturi aleatorii (simulăm procesul de izolare)
    # Pentru punctul normal (xi), avem nevoie de multe tăieturi
    # Pentru cel anormal (xj), avem nevoie de puține
    curr_xmin, curr_xmax = xmin, xmax
    curr_ymin, curr_ymax = ymin, ymax
    
    target_point = data[target_idx]
    
    for i in range(num_splits):
        if i % 2 == 0:
            split = np.random.uniform(curr_xmin + 0.5, curr_xmax - 0.5)
            ax.vlines(split, curr_ymin, curr_ymax, colors='orangered', linewidth=1.5)
            if target_point[0] < split: curr_xmax = split
            else: curr_xmin = split
        else:
            split = np.random.uniform(curr_ymin + 0.5, curr_ymax - 0.5)
            ax.hlines(split, curr_xmin, curr_xmax, colors='orangered', linewidth=1.5)
            if target_point[1] < split: curr_ymax = split
            else: curr_ymin = split

    ax.annotate(label, xy=(target_point[0], target_point[1]), xytext=(target_point[0]-1, target_point[1]-1),
                arrowprops=dict(facecolor='black', arrowstyle='->', connectionstyle="arc3"),
                fontsize=12, fontweight='bold')
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title, y=-0.15)

# Plot Stânga: Punct normal (Multe tăieturi)
plot_isolation(ax1, xi_idx, '$x_i$', 
               "Fig. 2 - un exemplu de izolare a unui punct neanomal\nîntr-o distribuție gaussiană 2D", 
               num_splits=12)

# Plot Dreapta: Punct anormal (Puține tăieturi)
plot_isolation(ax2, xj_idx, '$x_j$', 
               "Fig. 3 - un exemplu de izolare a unui punct anormal\nîntr-o distribuție gaussiană 2D", 
               num_splits=3)

plt.tight_layout()
plt.show()