import matplotlib.pyplot as plt
import numpy as np

from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.data_curved import make_curved_deliveries
from machine_learning_from_scratch.data_multiple import make_deliveries
from machine_learning_from_scratch.features import add_interaction, polynomial_features
from machine_learning_from_scratch.gradient_descent import gradient_descent
from machine_learning_from_scratch.scaling import unscale_parameters, zscore_normalize


def fit(X, y, alpha=0.1, num_iters=5000):
    """Scale the features, train with gradient descent and return w and b in the original units."""
    X_scaled, mu, sigma = zscore_normalize(X)
    w_s, b_s, _ = gradient_descent(X_scaled, y, np.zeros(X.shape[1]), 0.0, alpha, num_iters)
    return unscale_parameters(w_s, b_s, mu, sigma)


# 1. Feature engineering: in the rush hour every meter takes longer
X, y = make_deliveries(rush_slowdown=0.002)
X_new = add_interaction(X, 0, 2)  # distance_m * rush_hour

print("deliveries where rush hour adds 0.002 min per meter")
for name, features in [("3 features", X), ("3 features + distance*rush_hour", X_new)]:
    w, b = fit(features, y)
    print(f"  {name:32s} cost = {compute_cost(features, y, w, b):.3f}")
print(f"  noise floor (std 3.0 -> J about {3.0**2 / 2:.1f})")

# 2. Polynomial regression: the time curves with the distance
km, y_curve = make_curved_deliveries()
print("\ncurved deliveries, time against distance in km")
fits = {}
for degree in (1, 2, 3):
    features = polynomial_features(km, degree)
    w, b = fit(features, y_curve)
    fits[degree] = (w, b)
    print(f"  degree {degree}: cost = {compute_cost(features, y_curve, w, b):6.3f}   w = {w.round(4)}, b = {b:.2f}")
print("  rule: 10 + 5*km - 0.25*km**2")

# 3. Why the polynomial columns need scaling too
features = polynomial_features(km, 3)
print(f"\nrange of the columns km, km**2, km**3: {np.ptp(features, axis=0).round(0)}")
with np.errstate(all="ignore"):
    _, _, costs_big = gradient_descent(features, y_curve, np.zeros(3), 0.0, 1e-4, 100)
    _, _, costs_small = gradient_descent(features, y_curve, np.zeros(3), 0.0, 1e-6, 5000)
print(f"  raw columns, alpha = 1e-4:                  cost = {costs_big[-1]:.3g}  (diverges)")
print(f"  raw columns, alpha = 1e-6, 5000 iterations: cost = {costs_small[-1]:.2f}")
_, _, costs_scaled = gradient_descent(zscore_normalize(features)[0], y_curve, np.zeros(3), 0.0, 0.1, 5000)
print(f"  scaled columns, alpha = 0.1, 5000 iterations: cost = {costs_scaled[-1]:.3f}")

# 4. Plot
grid = np.linspace(0.5, 10, 200)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(km, y_curve, s=14, color="tab:blue", label="deliveries")
for degree in (1, 2, 3):
    w, b = fits[degree]
    ax.plot(grid, polynomial_features(grid, degree) @ w + b, label=f"degree {degree}")
ax.set_xlabel("distance (km)")
ax.set_ylabel("time (min)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/polynomial_regression.png")
