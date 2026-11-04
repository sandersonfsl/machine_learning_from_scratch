import matplotlib.pyplot as plt
import numpy as np

from machine_learning_from_scratch.convergence import converged_at, is_decreasing
from machine_learning_from_scratch.data_multiple import make_deliveries
from machine_learning_from_scratch.gradient_descent import gradient_descent
from machine_learning_from_scratch.scaling import zscore_normalize

X, y = make_deliveries()
X_scaled, _, _ = zscore_normalize(X)
iterations = 200
epsilon = 1e-3

# Try alpha in steps of about 3x, from far too small to too large
alphas = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 1.9, 2.0]

print(f"{iterations} iterations on the scaled features, epsilon = {epsilon}")
print(f"{'alpha':>7} {'final cost':>12} {"never up":>12} {'converged at':>13}")
curves = {}
for alpha in alphas:
    with np.errstate(all="ignore"):
        _, _, costs = gradient_descent(X_scaled, y, np.zeros(3), 0.0, alpha, iterations)
    curves[alpha] = costs
    stop = converged_at(costs, epsilon)
    print(f"{alpha:>7} {costs[-1]:>12.4g} {str(is_decreasing(costs)):>12} {str(stop):>13}")

fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
for alpha in [0.001, 0.01, 0.1, 1.0]:
    left.plot(curves[alpha], label=f"alpha = {alpha}")
left.set_title("Learning curves")
left.set_xlabel("iteration")
left.set_ylabel("cost J")
left.set_yscale("log")
left.legend()
for alpha in [1.9, 2.0]:
    right.plot(curves[alpha][:100], label=f"alpha = {alpha}")
right.set_title("Just above the limit (1.87)")
right.set_xlabel("iteration")
right.set_ylabel("cost J")
right.set_yscale("log")
right.legend()
fig.tight_layout()
fig.savefig("figures/learning_curve.png")
