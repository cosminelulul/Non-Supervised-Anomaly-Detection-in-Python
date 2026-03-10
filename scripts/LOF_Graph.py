import numpy as np
import matplotlib.pyplot as plt

points = {
    'A': np.array([0.6, 0.7]),
    'B': np.array([0.45, 0.4]),
    'C': np.array([0.81, 0.62]),
    'D': np.array([0.85, 0.35])
}


radius_A = 0.35
radius_B = 0.42

fig, ax = plt.subplots(figsize=(8, 8))

circle_A = plt.Circle(points['A'], radius_A, color='black', fill=False, linestyle='--', alpha=0.6)
circle_B = plt.Circle(points['B'], radius_B, color='blue', fill=False, linestyle='--', alpha=0.3)

ax.add_patch(circle_A)
ax.add_patch(circle_B)

for label, coord in points.items():
    color = 'black' if label == 'A' else 'blue'
    ax.scatter(coord[0], coord[1], c=color, s=200, edgecolors='black', zorder=5)
    ax.text(coord[0] - 0.02, coord[1] + 0.04, label, fontsize=14, fontweight='bold')

ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)
ax.set_aspect('equal')
ax.axis('off') 

plt.title("Reachability Distance", fontsize=15)
plt.show()