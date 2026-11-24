import matplotlib.pyplot as plt
import numpy as np

from machine_learning_from_scratch.data_classification import make_late_deliveries
from machine_learning_from_scratch.logistic import compute_cost_logistic, predict_proba

# 1. The loss of one example: -log(f) if y = 1, -log(1 - f) if y = 0
print("loss of one example")
print("   f    y = 1    y = 0")
for f in (0.99, 0.9, 0.5, 0.1, 0.01):
    print(f"  {f:4.2f}   {-np.log(f):6.3f}   {-np.log(1 - f):6.3f}")

# 2. The cost of the small problem, with two sets of parameters
X_small = np.array([[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [3.0, 0.5], [2.0, 2.0], [1.0, 2.5]])
y_small = np.array([0, 0, 0, 1, 1, 1])
print("\ncost of a small problem (6 examples), w = [1, 1]")
for b in (-3.0, -4.0):
    print(f"  b = {b:4.1f}: J = {compute_cost_logistic(X_small, y_small, np.array([1.0, 1.0]), b):.4f}")

# 3. The plain formula breaks for a confident wrong answer, the one in the library does not
X_one, y_one = np.array([[1.0]]), np.array([0])
w_one = np.array([40.0])  # z = 40: f is 1.0 in floating point, for a label 0
f = predict_proba(X_one, w_one, 0.0)
with np.errstate(all="ignore"):
    plain = np.mean(-y_one * np.log(f) - (1 - y_one) * np.log(1 - f))
print(f"\none example with y = 0 and z = 40 (f = {f[0]}, so 1 - f = {1 - f[0]})")
print(f"  plain formula:         {plain}   (log of 0)")
print(f"  compute_cost_logistic: {compute_cost_logistic(X_one, y_one, w_one, 0.0)}")

# 4. Why not the squared error with the sigmoid: the cost along w, with b fixed
X, y = make_late_deliveries()
km = X[:, :1] / 1000
ws = np.linspace(-3, 6, 181)
squared = np.array([np.mean((predict_proba(km, np.array([w]), -5.0) - y) ** 2) / 2 for w in ws])
logistic = np.array([compute_cost_logistic(km, y, np.array([w]), -5.0) for w in ws])
print("\ncost along w (distance only, b = -5)")
flat = squared[0]
print(f"  squared error: lowest {squared.min():.3f} at w = {ws[squared.argmin()]:.2f}, flat at {flat:.3f} for w < -1")
print(f"  logistic:      lowest {logistic.min():.3f} at w = {ws[logistic.argmin()]:.2f}, {logistic[0]:.1f} at w = -3")
bends_squared = np.sum(np.diff(squared, 2) < -1e-9)
bends_logistic = np.sum(np.diff(logistic, 2) < -1e-9)
print(f"  places where the curve bends the wrong way: squared {bends_squared}, logistic {bends_logistic}")

# 5. Plot
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
p = np.linspace(0.001, 0.999, 300)
axes[0].plot(p, -np.log(p), label="y = 1:  -log(f)")
axes[0].plot(p, -np.log(1 - p), label="y = 0:  -log(1 - f)")
axes[0].set_ylim(0, 5)
axes[0].set_xlabel("f (predicted probability)")
axes[0].set_ylabel("loss of one example")
axes[0].legend()
axes[1].plot(ws, squared, color="tab:red")
axes[1].set_title("squared error with the sigmoid")
axes[1].set_xlabel("w")
axes[1].set_ylabel("J")
axes[2].plot(ws, logistic, color="tab:green")
axes[2].set_title("logistic cost")
axes[2].set_xlabel("w")
axes[2].set_ylabel("J")
fig.tight_layout()
fig.savefig("figures/logistic_cost.png")
