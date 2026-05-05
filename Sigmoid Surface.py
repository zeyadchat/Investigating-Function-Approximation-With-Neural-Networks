"""
===========================================================
Neural Function Approximation Experiment (2D → 1D)
===========================================================

This experiment extends neural function approximation to
two-dimensional inputs.

The model learns a mapping:

    (x, y) → z

where the target surface is:

    z = cos(x) * cos(y)

This is a smooth periodic surface used to test whether a
neural network can learn spatial structure. Uses sigmoid
activations, which compress hidden activations into
the range (0, 1).
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
HIDDEN_NEURONS = 64
LEARNING_RATE = 1e-5
EPOCHS = 25000
PRINT_INTERVAL = 200
GIF_DURATION = 0.15

# =========================================================
# Target surface function
# =========================================================
def surface_exact(x, y):
    return torch.cos(x) * torch.cos(y)

# =========================================================
# Neural Network Model (SIGMOID ACTIVATION)
# =========================================================
class SurfaceNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, HIDDEN_NEURONS)
        self.fc2 = nn.Linear(HIDDEN_NEURONS, 1)

    def forward(self, x):

        # SIGMOID activation (compresses to 0–1 range)
        h = torch.sigmoid(self.fc1(x))

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
        ax1.set_title("True Surface")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("z")

        # =========================
        # NN SURFACE
        # =========================
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X.numpy(), Y.numpy(),
                         Z_pred.numpy())
        ax2.set_title(f"{HIDDEN_NEURONS} Neuron Sigmoid Approximation (Epoch {epoch})")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")

        filename = f"frames_surface/frame_{epoch:04d}.png"
        plt.savefig(filename, dpi=120)
        plt.close()

# =========================================================
# GIF CREATION
# =========================================================
print("Creating GIF...")

with imageio.get_writer("surface_sigmoid.gif", mode="I", duration=GIF_DURATION) as writer:

    for epoch in range(0, EPOCHS, PRINT_INTERVAL):
        filename = f"frames_surface/frame_{epoch:04d}.png"
        image = imageio.imread(filename)
        writer.append_data(image)

print("GIF saved as surface_sigmoid.gif")

# =========================================================
# FINAL EVALUATION (true model performance on full surface)
# =========================================================
model.eval()

with torch.no_grad():

    # full prediction over grid
    Z_pred = model(XY)

    # true surface values
    Z_true = surface_exact(X, Y).reshape(-1, 1)

    # compute true evaluation loss
    final_loss = criterion(Z_pred, Z_true)

# reshape for plotting
Z_pred_grid = Z_pred.reshape(50, 50)

# =========================================================
# 3D VISUALIZATION (TRUE vs PREDICTED)
# =========================================================
fig = plt.figure(figsize=(10, 4))

# -------------------------
# True surface
# -------------------------
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X.numpy(), Y.numpy(), surface_exact(X, Y).numpy())
ax1.set_title("True Function f(x, y)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("f(x,y)")
ax1.grid(True)

# -------------------------
# Sigmoid Neural Network surface
# -------------------------
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X.numpy(), Y.numpy(), Z_pred_grid.numpy())
ax2.set_title(f"{HIDDEN_NEURONS} Neuron Sigmoid Approximation (Final Evaluation)")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("f(x,y)")
ax2.grid(True)

plt.tight_layout()
plt.savefig("surface_final.png", dpi=300)
plt.show()

# =========================================================
# LOSS CURVE
# =========================================================
plt.figure(figsize=(10, 5))

plt.plot(loss_history)
plt.title(f"Training Loss ({HIDDEN_NEURONS} Neuron Sigmoid Surface Model)")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)

plt.tight_layout()
plt.savefig("surface_sigmoid_loss.png", dpi=300)
plt.show()

# =========================================================
# FINAL METRIC
# =========================================================
print(f"Final MSE: {loss_history[-1]:.6e}")