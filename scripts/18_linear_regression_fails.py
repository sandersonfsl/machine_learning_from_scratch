import matplotlib.pyplot as plt
import numpy as np

from machine_learning_from_scratch.data_classification import make_late_deliveries

X, y = make_late_deliveries()
km = X[:, 0] / 1000  # distance only, so the problem fits in a 2D plot


def fit_line(km, y):
    """Least-squares line through the 0/1 labels, and the distance where it crosses 0.5."""
    slope, intercept = np.polyfit(km, y, 1)
    return slope, intercept, (0.5 - intercept) / slope


def accuracy(km, y, slope, intercept):
    """Share of deliveries where 'line >= 0.5' matches the label."""
    return np.mean((slope * km + intercept >= 0.5) == y)


# 1. A straight line through the 0/1 labels
slope, intercept, cut = fit_line(km, y)
print("late (1) or on time (0), predicted by a straight line on the distance")
print(f"  line: {slope:.3f} * km + ({intercept:.3f})")
print(f"  at 0 km the line says {intercept:.2f}, at 10 km it says {slope * 10 + intercept:.2f}  (not between 0 and 1)")
print(f"  crosses 0.5 at {cut:.2f} km, accuracy = {accuracy(km, y, slope, intercept):.0%}")

# 2. Add a few very long deliveries: all of them late, so nothing should change
km_far = np.array([20.0, 25.0, 30.0, 35.0, 40.0])
km_all, y_all = np.concatenate([km, km_far]), np.concatenate([y, np.ones(len(km_far), dtype=int)])
slope2, intercept2, cut2 = fit_line(km_all, y_all)
print("\nthe same line after adding 5 deliveries of 20 to 40 km (all late)")
print(f"  crosses 0.5 at {cut2:.2f} km  (moved by {cut2 - cut:+.2f} km)")
print(f"  accuracy on the original deliveries = {accuracy(km, y, slope2, intercept2):.0%}")

# 3. Plot
grid = np.linspace(0, 40, 200)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.scatter(km, y, s=14, color="tab:blue", label="deliveries")
ax.scatter(km_far, np.ones(len(km_far)), s=30, color="tab:red", label="added long deliveries")
ax.plot(grid, slope * grid + intercept, color="tab:green", label="line, original data")
ax.plot(grid, slope2 * grid + intercept2, color="tab:orange", label="line, with long deliveries")
ax.axhline(0.5, color="gray", linestyle=":")
ax.axvline(cut, color="tab:green", linestyle="--", alpha=0.6)
ax.axvline(cut2, color="tab:orange", linestyle="--", alpha=0.6)
ax.set_ylim(-0.5, 1.5)
ax.set_xlabel("distance (km)")
ax.set_ylabel("late (1) or on time (0)")
ax.legend()
fig.tight_layout()
fig.savefig("figures/linear_regression_fails.png")
