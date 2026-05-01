"""
===========================================================
Neural Function Approximation Experiment
===========================================================

This experiment extends neural function approximation to a
physical system: the simple pendulum.

The model learns a mapping:

    t → θ(t)

where θ(t) is the angular displacement over time.

The target is the analytical solution under the
small-angle approximation:

    θ(t) = θ₀ cos(ωt)

The target is the analytical solution of the pendulum under
small-angle approximation.
===========================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
import os

# -------------------------
# Hyperparameters
# -------------------------
HIDDEN_NEURONS = 4
LEARNING_RATE = 0.001
EPOCHS = 3000
PRINT_INTERVAL = 20
GIF_DURATION = 0.1

torch.manual_seed(42)

# -------------------------
# Physical system
# -------------------------
g = 9.81
L = 1.0
theta0 = 0.5
omega = (g / L) ** 0.5

def theta_exact(t):
    return theta0 * torch.cos(omega * t)


# -------------------------
# Training data
# -------------------------
# Time samples used for learning dynamics
t_train = torch.linspace(0, 10, 200).unsqueeze(1)
theta_train = theta_exact(t_train)

# Dense evaluation grid for visualization
t_plot = torch.linspace(0, 10, 1000).unsqueeze(1)


# -------------------------
# Model: Sigmoid Network
# -------------------------
# Learns smooth time-dependent mapping
class PendulumNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, HIDDEN_NEURONS)
        self.fc2 = nn.Linear(HIDDEN_NEURONS, 1)

    def forward(self, x):

        # Nonlinear feature extraction over time
        h = torch.sigmoid(self.fc1(x))

        # Map features to angle prediction
        return self.fc2(h)


model = PendulumNet()

# -------------------------
# Loss + Optimizer
# -------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

os.makedirs("frames_pendulum", exist_ok=True)

frames = []
loss_history = []

# -------------------------
# Training loop
# -------------------------
for epoch in range(EPOCHS):

    pred = model(t_train)
    loss = criterion(pred, theta_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    # -------------------------
    # Visualization
    # -------------------------
    if epoch % PRINT_INTERVAL == 0:
        print(f"[Epoch {epoch:4d}] Loss: {loss.item():.6e}")

        with torch.no_grad():
            y_true = theta_exact(t_plot)
            y_pred = model(t_plot)

        plt.figure(figsize=(8, 5))

        plt.plot(t_plot.numpy(), y_true.numpy(), label="True physics solution")
        plt.plot(t_plot.numpy(), y_pred.numpy(), label=f"{HIDDEN_NEURONS} Neuron Pendulum Sigmoid Model")

        plt.title(f"Pendulum Model — Epoch {epoch}")
        plt.legend()
        plt.grid(True)

        plt.text(0.02, 0.95,
                 f"Loss: {loss.item():.6e}",
                 transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        plt.savefig(f"frames_pendulum/{epoch:04d}.png")
        plt.close()

# -------------------------
# GIF generation
# -------------------------
frames = [imageio.v2.imread(f"frames_pendulum/{i:04d}.png")
          for i in range(0, EPOCHS, PRINT_INTERVAL)]

imageio.mimsave("pendulum_sigmoid.gif", frames, duration=GIF_DURATION)

# -------------------------
# Final evaluation
# -------------------------
with torch.no_grad():
    y_final = model(t_plot)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

axes[0].plot(t_plot.numpy(), theta_exact(t_plot).numpy(), label="True physics solution")
axes[0].plot(t_plot.numpy(), y_final.numpy(), label=f"{HIDDEN_NEURONS} Neuron Pendulum Sigmoid Model")
axes[0].set_title("Pendulum Dynamics (Sigmoid Function Model)")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(loss_history)
axes[1].set_title("Training Loss")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("pendulum_final.png")
plt.show()

print(f"Final MSE: {loss_history[-1]:.6e}")