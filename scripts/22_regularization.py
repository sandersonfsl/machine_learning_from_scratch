from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from machine_learning_from_scratch.data_classification import make_late_deliveries
from machine_learning_from_scratch.features import polynomial_features_2d
from machine_learning_from_scratch.gradient_descent import gradient_descent
from machine_learning_from_scratch.logistic import (
    accuracy,
    compute_cost_logistic,
    compute_gradient_logistic,
    predict_proba,
)
from machine_learning_from_scratch.scaling import apply_zscore, zscore_normalize

DEGREE = 6
LAMBDAS = (0.0, 0.1, 1.0, 10.0)


def to_km(X):
    return X * [0.001, 1.0]  # columns [km, items]


# A small training set, and a large test set that the training never sees
X_train, y_train = make_late_deliveries(m=30, seed=0)
X_test, y_test = make_late_deliveries(m=1000, seed=1)
X_train, X_test = to_km(X_train), to_km(X_test)

# 27 columns: every product of km and items up to degree 6, scaled with the training set
features_train = polynomial_features_2d(X_train, DEGREE)
Z_train, mu, sigma = zscore_normalize(features_train)
Z_test = apply_zscore(polynomial_features_2d(X_test, DEGREE), mu, sigma)
print(f"{len(y_train)} training deliveries, {len(y_test)} test deliveries, {Z_train.shape[1]} polynomial features")

# 1. Train with several values of lambda
models = {}
print("\nlogistic regression, 20,000 iterations")
print("  lambda   train acc   test acc   train cost   test cost   sum of |w|")
for lambda_ in LAMBDAS:
    w, b, _ = gradient_descent(
        Z_train,
        y_train,
        np.zeros(Z_train.shape[1]),
        0.0,
        0.1,
        20_000,
        partial(compute_cost_logistic, lambda_=lambda_),
        partial(compute_gradient_logistic, lambda_=lambda_),
    )
    models[lambda_] = (w, b)
    print(
        f"  {lambda_:6g}   {accuracy(Z_train, y_train, w, b):9.0%}   {accuracy(Z_test, y_test, w, b):8.1%}"
        f"   {compute_cost_logistic(Z_train, y_train, w, b):10.3f}   {compute_cost_logistic(Z_test, y_test, w, b):9.3f}"
        f"   {np.abs(w).sum():10.1f}"
    )

# 2. The straight-line boundary of Step 21, on the same data, for comparison
Z1_train, mu1, sigma1 = zscore_normalize(X_train)
w1, b1, _ = gradient_descent(
    Z1_train, y_train, np.zeros(2), 0.0, 0.5, 2000, compute_cost_logistic, compute_gradient_logistic
)
print(
    f"\nstraight boundary (degree 1): train {accuracy(Z1_train, y_train, w1, b1):.0%}, "
    f"test {accuracy(apply_zscore(X_test, mu1, sigma1), y_test, w1, b1):.1%}"
)

# 3. Plot the boundary for each lambda
km_grid, items_grid = np.meshgrid(np.linspace(0, 10.5, 200), np.linspace(0, 13, 200))
grid = np.column_stack([km_grid.ravel(), items_grid.ravel()])
Z_grid = apply_zscore(polynomial_features_2d(grid, DEGREE), mu, sigma)

fig, axes = plt.subplots(1, len(LAMBDAS), figsize=(16, 4.2), sharey=True)
for ax, lambda_ in zip(axes, LAMBDAS):
    w, b = models[lambda_]
    prob = predict_proba(Z_grid, w, b).reshape(km_grid.shape)
    ax.contourf(km_grid, items_grid, prob, levels=[0, 0.5, 1], colors=["tab:blue", "tab:red"], alpha=0.15)
    ax.contour(km_grid, items_grid, prob, levels=[0.5], colors="black", linewidths=2)
    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], s=22, color="tab:blue")
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], s=22, color="tab:red", marker="x")
    ax.set_title(f"lambda = {lambda_:g}\ntest accuracy {accuracy(Z_test, y_test, w, b):.1%}")
    ax.set_xlabel("distance (km)")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 13)
axes[0].set_ylabel("items")
fig.tight_layout()
fig.savefig("figures/regularization.png")
