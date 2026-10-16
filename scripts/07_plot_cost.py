import matplotlib.pyplot as plt
import numpy as np

from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.data import x, y

b = 12  # keep b fixed so the cost depends on w alone

# Try many values of w, from 0 to 6, and compute the cost of each
ws = np.linspace(0, 6, 100)
costs = [compute_cost(x, y, w, b) for w in ws]

best_w = ws[np.argmin(costs)]
print(f"lowest cost with b={b}: w = {best_w:.2f}")

plt.plot(ws, costs, color="blue")
plt.scatter(best_w, min(costs), color="red", zorder=3, label=f"minimum, w = {best_w:.2f}")
plt.title(f"Cost J(w) with b = {b}")
plt.xlabel("w (minutes per km)")
plt.ylabel("Cost J")
plt.legend()
plt.savefig("figures/plot_cost.png")
