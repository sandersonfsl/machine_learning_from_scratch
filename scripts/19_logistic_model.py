import matplotlib.pyplot as plt
import numpy as np

from machine_learning_from_scratch.data_classification import make_late_deliveries
from machine_learning_from_scratch.logistic import predict_class, predict_proba

X, y = make_late_deliveries()
X = X * [0.001, 1.0]  # distance in km, items unchanged: columns are [km, items]

# Parameters chosen by hand, not trained: z = 0.9*km + 0.36*items - 7.5
w = np.array([0.9, 0.36])
b = -7.5

# 1. Probabilities for a few deliveries
examples = np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 10.0]])
print("probability of being late, w = [0.9, 0.36], b = -7.5")
for (km, items), p in zip(examples, predict_proba(examples, w, b)):
    print(f"  {km:.0f} km, {items:2.0f} items  ->  {p:.3f}")

# 2. Decision: late when the probability reaches 0.5, which is the same as z >= 0
predicted = predict_class(X, w, b)
z = np.dot(X, w) + b
print(f"\nsame decision from 'probability >= 0.5' and 'z >= 0': {np.array_equal(predicted, (z >= 0).astype(int))}")
print(f"accuracy on the 100 deliveries: {np.mean(predicted == y):.0%}")
print(f"predicted late: {predicted.sum()}, really late: {y.sum()}")

# 3. The decision boundary is the line where z = 0, so items = (7.5 - 0.9*km) / 0.36
print("\non the boundary line, z = 0 and the probability is exactly 0.5:")
for km in (4.0, 6.0):
    items = -(w[0] * km + b) / w[1]
    print(f"  {km:.0f} km, {items:5.2f} items  ->  {predict_proba(np.array([km, items]), w, b):.3f}")

# 4. Plot
km_grid, items_grid = np.meshgrid(np.linspace(0, 10.5, 200), np.linspace(0, 13, 200))
grid = np.column_stack([km_grid.ravel(), items_grid.ravel()])
prob = predict_proba(grid, w, b).reshape(km_grid.shape)

fig, ax = plt.subplots(figsize=(7, 5))
shade = ax.contourf(km_grid, items_grid, prob, levels=np.linspace(0, 1, 11), cmap="RdYlBu_r", alpha=0.35)
ax.contour(km_grid, items_grid, prob, levels=[0.5], colors="black", linewidths=2)
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=18, color="tab:blue", label="on time (0)")
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=18, color="tab:red", marker="x", label="late (1)")
ax.plot([], [], color="black", linewidth=2, label="decision boundary (p = 0.5)")
fig.colorbar(shade, label="probability of being late")
ax.set_xlabel("distance (km)")
ax.set_ylabel("items")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
fig.tight_layout()
fig.savefig("figures/logistic_model.png")
