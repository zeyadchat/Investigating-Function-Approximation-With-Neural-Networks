"""
===========================================================
Neural Function Approximation Experiment (2D → 1D)
STEP ACTIVATION VERSION
===========================================================

This experiment trains a neural network to approximate:

    f(x, y) = cos(x) * cos(y)

using a STEP activation function.

This results in a piecewise-constant approximation of the
surface, which demonstrates how neural networks can build
discrete partitions of function space.
===========================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
import os

torch.manual_seed(42)

# =========================================================
# Hyperparameters
# =========================================================
HIDDEN_NEURONS = 128
LEARNING_RATE = 1e-4
EPOCHS = 50000
PRINT_INTERVAL = 1000
GIF_DURATION = 0.15

# =========================================================
# Target function
# =========================================================
def surface_exact(x, y):
    return torch.cos(x) * torch.cos(y)

# =========================================================
# STEP activation
# =========================================================
def step(x):
    return (x > 0).float()

# =========================================================
# Neural Network (STEP ACTIVATION)
# =========================================================
class SurfaceNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, HIDDEN_NEURONS)
        self.fc2 = nn.Linear(HIDDEN_NEURONS, 1)

    def forward(self, x):

        # STEP activation (non-smooth, piecewise behavior)
        h = step(self.fc1(x))

        return self.fc2(h)

model = SurfaceNet()

# =========================================================
# Loss + Optimizer
# =========================================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================================================
# Training grid
# =========================================================
x = torch.linspace(-3.14, 3.14, 50)
y = torch.linspace(-3.14, 3.14, 50)

X, Y = torch.meshgrid(x, y, indexing='ij')

XY = torch.stack([X.flatten(), Y.flatten()], dim=1)
Z = surface_exact(X, Y).reshape(-1, 1)

# =========================================================
# Storage
# =========================================================
os.makedirs("frames_surface", exist_ok=True)

loss_history = []

# =========================================================
# Training loop
# =========================================================
for epoch in range(EPOCHS):

    pred = model(XY)
    loss = criterion(pred, Z)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------
    if epoch % PRINT_INTERVAL == 0:

        print(f"[Epoch {epoch:4d}] Loss: {loss.item():.6e}")

        with torch.no_grad():
            Z_pred = model(XY).reshape(50, 50)

        fig = plt.figure(figsize=(8, 4))

        # =========================
        # TRUE SURFACE
        # =========================
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(X.numpy(), Y.numpy(),
                         surface_exact(X, Y).numpy())
        ax1.set_title("True Function f(x, y)")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("f(x,y)")
        ax1.grid(True)

        # =========================
        # STEP Neural Network surface
        # =========================
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X.numpy(), Y.numpy(),
                         Z_pred.numpy())
        ax2.set_title(f"{HIDDEN_NEURONS} Neuron Step Approximation (Epoch {epoch})")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("f(x,y)")
        ax2.grid(True)

        filename = f"frames_surface/frame_{epoch:04d}.png"
        plt.savefig(filename, dpi=120)
        plt.close()

# =========================================================
# GIF CREATION
# =========================================================
print("Creating GIF...")

with imageio.get_writer("surface_step.gif", mode="I", duration=GIF_DURATION) as writer:

    for epoch in range(0, EPOCHS, PRINT_INTERVAL):
        filename = f"frames_surface/frame_{epoch:04d}.png"
        image = imageio.imread(filename)
        writer.append_data(image)

print("GIF saved as surface_step.gif")

# =========================================================
# FINAL EVALUATION
# =========================================================
model.eval()

with torch.no_grad():

    Z_pred = model(XY)
    Z_true = surface_exact(X, Y).reshape(-1, 1)

    final_loss = criterion(Z_pred, Z_true)

Z_pred_grid = Z_pred.reshape(50, 50)

# =========================================================
# FINAL VISUALIZATION
# =========================================================
fig = plt.figure(figsize=(10, 4))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X.numpy(), Y.numpy(), surface_exact(X, Y).numpy())
ax1.set_title("True Function f(x, y)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f(x,y)")
ax1.grid(True)

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X.numpy(), Y.numpy(), Z_pred_grid.numpy())
ax2.set_title("{HIDDEN_NEURONS} Neuron Step Approximation (Final Evaluation)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f(x,y)")
ax2.grid(True)

plt.tight_layout()
plt.savefig("surface_step_final.png", dpi=300)
plt.show()

# =========================================================
# LOSS CURVE
# =========================================================
plt.figure(figsize=(10, 5))

plt.plot(loss_history)
plt.title("Training Loss ({HIDDEN_NEURONS} Neuron Step Surface Model)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("surface_step_loss.png", dpi=300)
plt.show()

# =========================================================
# FINAL METRIC
# =========================================================
print(f"Final MSE: {loss_history[-1]:.6e}")