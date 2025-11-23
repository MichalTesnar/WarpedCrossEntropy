import numpy as np
import os
import matplotlib.pyplot as plt

def plot_vectors():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Origin
    o = [0, 0, 0]

    # 1. Arrows along positive axes (x, y, z) of size 1
    # Quiver arguments: x, y, z, u, v, w
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', label='X')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', label='Y')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', label='Z')

    # 2. Vector exactly between them of size 1
    # The direction is (1, 1, 1). To make size 1, normalize by sqrt(1^2 + 1^2 + 1^2) = sqrt(3)
    val = 1 / np.sqrt(3)
    ax.quiver(0, 0, 0, val, val, val, color='k', label='Between')

    # 3. Arrows along opposite directions (negative x, y, z) of size 1
    ax.quiver(0, 0, 0, -1, 0, 0, color='r', linestyle='dashed', alpha=0.5)
    ax.quiver(0, 0, 0, 0, -1, 0, color='g', linestyle='dashed', alpha=0.5)
    ax.quiver(0, 0, 0, 0, 0, -1, color='b', linestyle='dashed', alpha=0.5)

    # Setting the limits to see everything clearly
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    plt.legend()
    
    # Ensure directory exists
    plt.tight_layout()
    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/vectors.png')
    plt.show()

if __name__ == "__main__":
    plot_vectors()