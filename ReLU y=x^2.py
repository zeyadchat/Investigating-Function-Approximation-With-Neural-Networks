"""
===========================================================
Neural Function Approximation Experiment
===========================================================

This experiment compares different neural architectures
for function approximation.

This model uses a ReLU neural network to approximate a
nonlinear function (x^2).

ReLU networks create piecewise linear representations of
functions by combining linear regions.
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
# Number of hidden neurons controls number of linear regions
HIDDEN_NEURONS = 8

LEARNING_RATE = 0.001
EPOCHS = 3000
PRINT_INTERVAL = 20
GIF_DURATION = 0.3

torch.manual_seed(42)

# -------------------------
# Training data
# -------------------------
# Target function: x^2
# Used as a standard benchmark for nonlinear approximation
x_train = torch.linspace(-1, 1, 600).unsqueeze(1)
y_train = x_train ** 2

# Dense evaluation grid for visualization only
x_plot = torch.linspace(-1, 1, 1000).unsqueeze(1)
y_true = x_plot ** 2


# -------------------------
# Model: ReLU Neural Network
# -------------------------
# ReLU creates piecewise linear behavior in the function space
class ReLUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Fully connected hidden layer
        self.fc1 = nn.Linear(1, HIDDEN_NEURONS)

        # Output layer maps features to scalar prediction
        self.fc2 = nn.Linear(HIDDEN_NEURONS, 1)

    def forward(self, x):

        # ReLU introduces nonlinearity and piecewise linear regions
        h = torch.relu(self.fc1(x))

        # Linear combination of activated features
        return self.fc2(h)


model = ReLUNet()

# -------------------------
# Loss + Optimizer
# -------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

os.makedirs("frames_relu", exist_ok=True)

frames = []
loss_history = []

# -------------------------
# Training loop
# -------------------------
# Model learns weights that shape piecewise linear segments
for epoch in range(EPOCHS):

    pred = model(x_train)
    loss = criterion(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    # -------------------------
    # Visualization
    # -------------------------
    if epoch % PRINT_INTERVAL == 0:
        print(f"[Epoch {epoch:4d}] Loss: {loss.item():.6f}")

        with torch.no_grad():
            y_pred = model(x_plot)

        plt.figure(figsize=(8, 5))

        plt.plot(x_plot.numpy(), y_true.numpy(), label="True function $x^2$")
        plt.plot(x_plot.numpy(), y_pred.numpy(), label=f"{HIDDEN_NEURONS} Neuron ReLU Model")

        plt.title(f"ReLU Model — Epoch {epoch}")
        plt.legend()
        plt.grid(True)

        plt.text(0.02, 0.95,
                 f"Loss: {loss.item():.6f}",
                 transform=plt.gca().transAxes,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        plt.savefig(f"frames_relu/{epoch:04d}.png")
        plt.close()

# -------------------------
# GIF generation
# -------------------------
frames = [imageio.v2.imread(f"frames_relu/{i:04d}.png")
          for i in range(0, EPOCHS, PRINT_INTERVAL)]

imageio.mimsave("relu.gif", frames, duration=GIF_DURATION)

# -------------------------
# Final evaluation
# -------------------------
with torch.no_grad():
    y_final = model(x_plot)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

axes[0].plot(x_plot.numpy(), y_true.numpy(), label="True function $x^2$")
axes[0].plot(x_plot.numpy(), y_final.numpy(), label=f"{HIDDEN_NEURONS} Neuron ReLU Model")
axes[0].set_title("Function Approximation")
axes[0].grid(True)
axes[0].legend()

axes[1].plot(loss_history)
axes[1].set_title("Training Loss")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("relu_final.png")
plt.show()

print(f"Final MSE: {loss_history[-1]:.6e}")