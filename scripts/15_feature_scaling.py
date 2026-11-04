import matplotlib.pyplot as plt
import numpy as np

from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.data_multiple import FEATURES, TRUE_B, TRUE_W, make_deliveries
from machine_learning_from_scratch.gradient_descent import gradient_descent
from machine_learning_from_scratch.model import predict
from machine_learning_from_scratch.scaling import apply_zscore, unscale_parameters, zscore_normalize


def largest_stable_alpha(X):
    """Gradient descent diverges above this learning rate.

    It is 2 divided by the largest curvature of the cost function.
    """
    A = np.column_stack([X, np.ones(len(X))])
    curvature = np.linalg.eigvalsh(A.T @ A / len(A))
    return 2 / curvature.max()


X, y = make_deliveries()
m, n = X.shape
X_scaled, mu, sigma = zscore_normalize(X)

# 1. What scaling does to the features
print(f"features: {FEATURES}")
print("range of each feature (max - min)")
print("  raw:   ", np.ptp(X, axis=0))
print("  scaled:", np.ptp(X_scaled, axis=0).round(2))
print("mean of the scaled features:", np.abs(X_scaled.mean(axis=0)).round(6))
print("std of the scaled features: ", X_scaled.std(axis=0).round(6))

# 2. What scaling does to the learning rate
print(f"\nlargest stable alpha, raw features:    {largest_stable_alpha(X):.1e}")
print(f"largest stable alpha, scaled features: {largest_stable_alpha(X_scaled):.2f}")

# 3. Train on the scaled features, with a learning rate far larger than before
A = np.column_stack([X, np.ones(m)])
solution = np.linalg.lstsq(A, y, rcond=None)[0]
w_best, b_best = solution[:-1], solution[-1]
cost_best = compute_cost(X, y, w_best, b_best)

alpha, iterations = 0.1, 1000
w_scaled, b_scaled, costs = gradient_descent(X_scaled, y, np.zeros(n), 0.0, alpha, iterations)
close = next(i for i, cost in enumerate(costs) if cost - cost_best < 1e-3)
print(f"\nalpha = {alpha}, {iterations:,} iterations on the scaled features")
print(f"  cost within 0.001 of the minimum ({cost_best:.3f}) after {close + 1} iterations")
print(f"  scaled parameters: w = {w_scaled.round(3)}, b = {b_scaled:.2f}")

# 4. Back to the original units
w, b = unscale_parameters(w_scaled, b_scaled, mu, sigma)
print("\nparameters in the original units")
print(f"  learned: w = {w.round(4)}, b = {b:.2f}")
print(f"  exact:   w = {w_best.round(4)}, b = {b_best:.2f}")
print(f"  rule:    w = {TRUE_W}, b = {TRUE_B}")

# 5. A new delivery: 6500 m, 4 items, rush hour
x_new = np.array([6500.0, 4.0, 1.0])
from_scaled = predict(apply_zscore(x_new, mu, sigma), w_scaled, b_scaled)
from_original = predict(x_new, w, b)
forgot_to_scale = predict(x_new, w_scaled, b_scaled)
print(f"\nnew delivery {x_new}")
print(f"  scaled with the training mean and std: {from_scaled:.2f} min")
print(f"  original parameters, no scaling:       {from_original:.2f} min")
print(f"  scaled parameters on the raw example:  {forgot_to_scale:.0f} min  <- mistake")

# 6. The same two features before and after scaling, on equal axes
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
left.scatter(X[:, 0], X[:, 1], s=12, color="tab:blue")
left.set_title("Raw features")
left.set_xlabel("distance_m")
left.set_ylabel("items")
left.axis("equal")
right.scatter(X_scaled[:, 0], X_scaled[:, 1], s=12, color="tab:blue")
right.set_title("Scaled features (z-score)")
right.set_xlabel("distance_m")
right.set_ylabel("items")
right.axis("equal")
fig.tight_layout()
fig.savefig("figures/feature_scaling.png")
