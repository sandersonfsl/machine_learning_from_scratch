import matplotlib.pyplot as plt

from machine_learning_from_scratch.data import x, y
from machine_learning_from_scratch.gradient_descent import gradient_descent
from machine_learning_from_scratch.model import predict

w, b, costs = gradient_descent(x, y, w=0.0, b=0.0, alpha=0.01, num_iters=10_000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: the learned line over the training set
ax1.scatter(x, y, marker="x", color="red", label="training set")
ax1.plot(x, predict(x, w, b), color="blue", label=f"learned: w={w:.2f}, b={b:.2f}")
ax1.scatter(6.5, predict(6.5, w, b), color="green", zorder=3, label="6.5 km prediction")
ax1.set_title("The learned model")
ax1.set_xlabel("Distance (km)")
ax1.set_ylabel("Delivery time (min)")
ax1.legend()

# Right: the cost after each iteration of gradient descent
ax2.plot(range(1, len(costs) + 1), costs, color="blue")  # iterations 1, 2, 3, ...
ax2.set_title("Cost during training")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Cost J")
ax2.set_xscale("log")  # log scale: most of the drop happens in the first steps

fig.tight_layout()
fig.savefig("figures/plot_result.png")
