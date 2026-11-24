import matplotlib.pyplot as plt
import numpy as np

from machine_learning_from_scratch.logistic import sigmoid

# 1. The sigmoid at a few values
print("sigmoid(z) for a few values of z")
for z in (-10, -4, -2, -1, 0, 1, 2, 4, 10):
    print(f"  z = {z:3d}  ->  g(z) = {sigmoid(z):.5f}")

# 2. Always above 0.5 when z > 0, below when z < 0
z = np.linspace(-10, 10, 1001)
g = sigmoid(z)
print(f"\nrange of g over z in [-10, 10]: {g.min():.5f} to {g.max():.5f}")
print(f"g(z) >= 0.5 exactly when z >= 0: {np.array_equal(g >= 0.5, z >= 0)}")

# 3. Plot
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(z, g, color="tab:blue", label="g(z)")
ax.axhline(0.5, color="gray", linestyle=":")
ax.axvline(0, color="gray", linestyle=":")
ax.scatter([0], [0.5], color="tab:red", zorder=3, label="g(0) = 0.5")
ax.set_xlabel("z")
ax.set_ylabel("g(z)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/sigmoid.png")
