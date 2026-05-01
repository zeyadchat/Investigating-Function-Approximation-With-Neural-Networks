"""
===========================================================
Neural Function Approximation Experiment
===========================================================

This experiment compares different neural architectures
for function approximation.

This model uses a piecewise constant representation of a
function using learned breakpoints.

The network partitions the input space into intervals and
assigns a learned value to each region.

This connects to the idea that neural networks approximate
functions through space partitioning.
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
# Number of breakpoints controls resolution of approximation
# More breakpoints → finer partitioning of input space
NUM_BREAKPOINTS = 64

LEARNING_RATE = 0.002
EPOCHS = 3000
PRINT_INTERVAL = 20
GIF_DURATION = 0.3

torch.manual_seed(42)

# -------------------------
# Training data
# -------------------------
# Target function: x^2 (standard benchmark for approximation)
x_train = torch.linspace(-1, 1, 600).unsqueeze(1)
y_train = x_train ** 2

# Dense grid for visualization only (not used in training)
x_plot = torch.linspace(-1, 1, 2000).unsqueeze(1)
y_true = x_plot ** 2


# -------------------------
# Model: Step Function Network
# -------------------------
# Idea: split input space into regions and assign constant values
class StepNet(nn.Module):
    def __init__(self, n):
        super().__init__()

        # Fixed partition points (not learned)
        self.breakpoints = torch.linspace(-1, 1, n)

        # Learned output value for each region
        self.values = nn.Parameter(torch.randn(n + 1, 1))

    def forward(self, x):

        # Determine which side of each breakpoint x lies on
        h = (x > self.breakpoints).float()

        # Convert cumulative step pattern into region indicators
        left = 1 - h[:, 0:1]
        middle = h[:, :-1] - h[:, 1:]
        right = h[:, -1:]

        # Combine all regions into one representation
        regions = torch.cat([left, middle, right], dim=1)

        # Each region contributes a learned constant value
        return regions @ self.values


model = StepNet(NUM_BREAKPOINTS)

# -------------------------
# Loss + Optimizer
# -------------------------
criterion = nn.MSELoss()

# Only region values are trained; structure is fixed
optimizer = optim.Adam([model.values], lr=LEARNING_RATE)

# -------------------------
# Setup for visualization
# -------------------------
os.makedirs("frames_step", exist_ok=True)

frames = []
loss_history = []

# -------------------------
# Training loop
# -------------------------
for epoch in range(EPOCHS):

    # Forward pass: predict function values
    pred = model(x_train)

    # Measure approximation error
    loss = criterion(pred, y_train)

    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    # -------------------------
    # Visualization
    # -------------------------
    # Track how approximation improves over training
    if epoch % PRINT_INTERVAL == 0:
        print(f"[Epoch {epoch:4d}] Loss: {loss.item():.6f}")

        with torch.no_grad():
            y_pred = model(x_plot)

        plt.figure(figsize=(8, 5))

        plt.plot(x_plot.numpy(), y_true.numpy(), label="True function $x^2$")
        plt.plot(x_plot.numpy(), y_pred.numpy(), label=f"{NUM_BREAKPOINTS} Neuron Step Model")

        plt.title(f"Step Model — Epoch {epoch}")
        plt.legend()
        plt.grid(True)

        # Show current training error
        plt.text(0.02, 0.95,
                 f"Loss: {loss.item():.6f}",
                 transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        plt.savefig(f"frames_step/{epoch:04d}.png")
        plt.close()

# -------------------------
# Create training GIF
# -------------------------
frames = [imageio.v2.imread(f"frames_step/{i:04d}.png")
          for i in range(0, EPOCHS, PRINT_INTERVAL)]

imageio.mimsave("step.gif", frames, duration=GIF_DURATION)

# -------------------------
# Final evaluation
# -------------------------
with torch.no_grad():
    y_final = model(x_plot)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

# Function approximation quality
axes[0].plot(x_plot.numpy(), y_true.numpy(), label="True function $x^2$")
axes[0].plot(x_plot.numpy(), y_final.numpy(), label=f"{NUM_BREAKPOINTS} Neuron Step Model")
axes[0].set_title("Function Approximation")
axes[0].grid(True)
axes[0].legend()

# Training convergence
axes[1].plot(loss_history)
axes[1].set_title("Training Loss")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("step_final.png")
plt.show()

print(f"Final MSE: {loss_history[-1]:.6e}")