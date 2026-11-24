import matplotlib.pyplot as plt
import numpy as np

from machine_learning_from_scratch.data_classification import make_late_deliveries
from machine_learning_from_scratch.gradient_descent import gradient_descent
from machine_learning_from_scratch.logistic import (
    accuracy,
    compute_cost_logistic,
    compute_gradient_logistic,
    predict_proba,
)
from machine_learning_from_scratch.scaling import apply_zscore, unscale_parameters, zscore_normalize

# 1. The gradient of the small problem, checked against a numerical derivative
X_small = np.array([[0.5, 1.5], [1.0, 1.0], [1.5, 0.5], [3.0, 0.5], [2.0, 2.0], [1.0, 2.5]])
y_small = np.array([0, 0, 0, 1, 1, 1])
w0, b0 = np.array([2.0, 3.0]), 1.0
dj_dw, dj_db = compute_gradient_logistic(X_small, y_small, w0, b0)
h = 1e-6
numeric_db = (compute_cost_logistic(X_small, y_small, w0, b0 + h) - compute_cost_logistic(X_small, y_small, w0, b0 - h)) / (2 * h)
print("gradient of the small problem at w = [2, 3], b = 1")
print(f"  dj_dw = {dj_dw.round(5)}, dj_db = {dj_db:.5f}")
print(f"  numerical derivative in b: {numeric_db:.5f}")

# 2. Train on the deliveries: columns [km, items], scaled, as in Step 15
X, y = make_late_deliveries()
X = X * [0.001, 1.0]
X_scaled, mu, sigma = zscore_normalize(X)
w_s, b_s, costs = gradient_descent(
    X_scaled, y, np.zeros(2), 0.0, 0.5, 1000, compute_cost_logistic, compute_gradient_logistic
)
w, b = unscale_parameters(w_s, b_s, mu, sigma)

print("\nlogistic regression trained with gradient descent, alpha = 0.5")
for i in (0, 9, 99, 999):
    print(f"  iteration {i + 1:4d}: J = {costs[i]:.4f}")
print(f"  w = {w.round(3)}, b = {b:.2f}   (per km and per item)")
print(f"  accuracy = {accuracy(X, y, w, b):.0%}")

hand = compute_cost_logistic(X, y, np.array([0.9, 0.36]), -7.5)
print(f"  cost: trained {costs[-1]:.4f}, Step 19's hand-picked parameters {hand:.4f}")

# 3. Scaling matters here too
_, _, raw_costs = gradient_descent(X, y, np.zeros(2), 0.0, 0.1, 1000, compute_cost_logistic, compute_gradient_logistic)
print(f"\nwithout scaling, alpha = 0.1, 1000 iterations: J = {raw_costs[-1]:.4f} (scaled: {costs[-1]:.4f})")

# 4. A new delivery: 6 km, 7 items
new = np.array([6.0, 7.0])
p = predict_proba(apply_zscore(new, mu, sigma), w_s, b_s)
print(f"\nnew delivery, 6 km and 7 items: probability of being late = {p:.3f}")
print(f"  same probability with the unscaled parameters: {predict_proba(new, w, b):.3f}")

# 5. Plot
fig, (ax_cost, ax_map) = plt.subplots(1, 2, figsize=(12, 4.8))
ax_cost.plot(range(1, 301), costs[:300])
ax_cost.set_xlabel("iteration")
ax_cost.set_ylabel("cost J")
ax_cost.set_title("cost during training")

km_grid, items_grid = np.meshgrid(np.linspace(0, 10.5, 200), np.linspace(0, 13, 200))
grid = np.column_stack([km_grid.ravel(), items_grid.ravel()])
prob = predict_proba(grid, w, b).reshape(km_grid.shape)
shade = ax_map.contourf(km_grid, items_grid, prob, levels=np.linspace(0, 1, 11), cmap="RdYlBu_r", alpha=0.35)
ax_map.contour(km_grid, items_grid, prob, levels=[0.5], colors="black", linewidths=2)
ax_map.scatter(X[y == 0, 0], X[y == 0, 1], s=18, color="tab:blue", label="on time (0)")
ax_map.scatter(X[y == 1, 0], X[y == 1, 1], s=18, color="tab:red", marker="x", label="late (1)")
ax_map.plot([], [], color="black", linewidth=2, label="decision boundary")
fig.colorbar(shade, ax=ax_map, label="probability of being late")
ax_map.set_xlabel("distance (km)")
ax_map.set_ylabel("items")
ax_map.set_title("trained model")
ax_map.legend(loc="lower center", bbox_to_anchor=(0.5, 1.07), ncol=3, frameon=False)
fig.tight_layout()
fig.savefig("figures/logistic_gradient_descent.png")
