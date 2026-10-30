import numpy as np

from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.data_multiple import TRUE_B, TRUE_W, make_deliveries
from machine_learning_from_scratch.gradient_descent import gradient_descent

X, y = make_deliveries()
m, n = X.shape

# Reference: the exact best parameters, solved directly by linear algebra
# (the "normal equation"). Gradient descent should end up close to this.
A = np.column_stack([X, np.ones(m)])
solution = np.linalg.lstsq(A, y, rcond=None)[0]
w_best, b_best = solution[:-1], solution[-1]
print(f"true rule:      w = {TRUE_W}, b = {TRUE_B}")
print(f"exact solution: w = {w_best.round(4)}, b = {b_best:.2f}")
print(f"lowest possible cost: {compute_cost(X, y, w_best, b_best):.3f}")

# Gradient descent on the raw features, starting from w = 0 and b = 0
for alpha, iterations in [(1e-7, 50), (1e-8, 100_000)]:
    w, b, costs = gradient_descent(X, y, np.zeros(n), 0.0, alpha, iterations)
    print(f"\nalpha = {alpha}, {iterations:,} iterations")
    print(f"  cost: {costs[0]:.3e} at the start -> {costs[-1]:.3e} at the end")
    print(f"  w = {w.round(4)}, b = {b:.2f}")
