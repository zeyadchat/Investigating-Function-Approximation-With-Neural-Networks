"""
===========================================================
Neural Function Approximation Experiment (Physics System)
===========================================================

This experiment applies neural function approximation to a
physical system: the simple pendulum.

The model learns a mapping:

    t → θ(t)

where θ(t) is the angular displacement over time.

The target is the analytical solution under the
small-angle approximation:

    θ(t) = θ₀ cos(ωt)

This model uses a, STEP FUNCTION representation,
meaning the network approximates dynamics using piecewise
binary transitions.
===========================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# =========================================================
# Hyperparameters
# =========================================================
HIDDEN_NEURONS = 256
LEARNING_RATE = 0.001
EPOCHS = 3000
PRINT_INTERVAL = 20
GIF_DURATION = 0.1

torch.manual_seed(42)

# =========================================================
# Physical system
# =========================================================
g = 9.81
L = 1.0
theta0 = 0.5
omega = (g / L) ** 0.5

def theta_exact(t):
    return theta0 * torch.cos(omega * t)

# =========================================================
# Training data
# =========================================================
t_train = torch.linspace(0, 10, 200).unsqueeze(1)
theta_train = theta_exact(t_train)

t_plot = torch.linspace(0, 10, 1000).unsqueeze(1)

# =========================================================
# STEP FUNCTION NETWORK
# =========================================================
class StepPendulumNet(nn.Module):
    """
    Step-function neural network for time dynamics.

    Each neuron behaves like a binary threshold unit,
    creating piecewise constant approximations.
    """

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, HIDDEN_NEURONS)
        self.fc2 = nn.Linear(HIDDEN_NEURONS, 1)

    def forward(self, x):

        # Step activation (hard threshold approximation)
        h = (torch.sigmoid(100 * self.fc1(x)) > 0.5).float()

        return self.fc2(h)


model = StepPendulumNet()

# =========================================================
# Loss + Optimizer
# =========================================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# =========================================================
# Setup
# =========================================================
os.makedirs("frames_pendulum_step", exist_ok=True)

frames = []
loss_history = []

# =========================================================
# Training loop
# =========================================================
for epoch in range(EPOCHS):

    pred = model(t_train)
    loss = criterion(pred, theta_train)

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
            y_true = theta_exact(t_plot)
            y_pred = model(t_plot)

        plt.figure(figsize=(8, 5))

        plt.plot(t_plot.numpy(), y_true.numpy(),
                 label="True physics solution")

        plt.plot(t_plot.numpy(), y_pred.numpy(),
                 label=f"{HIDDEN_NEURONS} Neuron Step Model")

        plt.title(f"Pendulum Dynamics (Step Model) — Epoch {epoch}")
        plt.xlabel("Time (t)")
        plt.ylabel("Angle θ(t)")
        plt.grid(True)
        plt.legend()

        plt.text(
            0.02, 0.95,
            f"Loss: {loss.item():.6e}",
            transform=plt.gca().transAxes,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

        filename = f"frames_pendulum_step/{epoch:04d}.png"
        plt.savefig(filename, dpi=150)
        plt.close()

# =========================================================
# GIF CREATION
# =========================================================
print("Creating GIF safely...")

with imageio.get_writer(
    "pendulum.gif",
    mode="I",
    duration=GIF_DURATION,
    fps=10,
    quantizer="nq"
) as writer:

    for epoch in range(0, EPOCHS, PRINT_INTERVAL):

        filename = f"frames_pendulum/frame_{epoch:04d}.png"

        if os.path.exists(filename):
            image = imageio.imread(filename)

            image = image[::2, ::2]

            writer.append_data(image)

print("GIF saved as pendulum_step.gif")

# =========================================================
# FINAL EVALUATION
# =========================================================
with torch.no_grad():
    y_final = model(t_plot)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

# -------------------------
# Function approximation
# -------------------------
axes[0].plot(t_plot.numpy(), theta_exact(t_plot).numpy(),
             label="True physics solution")

axes[0].plot(t_plot.numpy(), y_final.numpy(),
             label=f"{HIDDEN_NEURONS} Neuron Step Model")

axes[0].set_title("Pendulum Dynamics (Step Function Model)")
axes[0].set_xlabel("Time (t)")
axes[0].set_ylabel("Angle θ(t)")
axes[0].grid(True)
axes[0].legend()

# -------------------------
# Loss curve
# -------------------------
axes[1].plot(loss_history)
axes[1].set_title("Training Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("MSE Loss")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("pendulum_step_final.png")
plt.show()

print(f"Final MSE: {loss_history[-1]:.6e}")