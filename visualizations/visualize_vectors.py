import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # IMPORT FIX
from warped_cross_entropy import WarpedCrossEntropy, CustomCrossEntropyLoss
from constants import REGIME, Regime
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODEL_CHECKPOINT_FILE = f'checkpoints/model_{REGIME}_epoch_50.pth'
LOSS_CHECKPOINT_FILE = f'checkpoints/loss_{REGIME}_epoch_50.pth'

if REGIME == Regime.NORMAL:
    criterion = CustomCrossEntropyLoss()
elif REGIME == Regime.WARPED:
    criterion = WarpedCrossEntropy(hidden_dimension=3, number_of_classes=10)
    
criterion.load_state_dict(torch.load(LOSS_CHECKPOINT_FILE))


vectors = criterion.class_vectors.detach().cpu().numpy()
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw the unit sphere for reference (wireframe)
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

# Plot vectors
for i, vec in enumerate(vectors):
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color='b', arrow_length_ratio=0.1)
    # Plot numbers 1-10 (indices 0-9 + 1) at the tips
    ax.text(vec[0], vec[1], vec[2], str(i + 1), color='red', fontsize=12)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# plt.savefig("learned_vectors.png")
plt.show()
