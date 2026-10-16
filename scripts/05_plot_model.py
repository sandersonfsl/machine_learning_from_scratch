import matplotlib.pyplot as plt

from machine_learning_from_scratch.data import x, y
from machine_learning_from_scratch.model import predict

plt.scatter(x, y, marker="x", color="red", label="training set")

# Draw the line of each model: one prediction for every distance in x
plt.plot(x, predict(x, w=3, b=12), color="blue", label="w=3, b=12")
plt.plot(x, predict(x, w=1, b=5), color="green", label="w=1, b=5")

plt.title("Two different models")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery time (min)")
plt.legend()
plt.savefig("figures/plot_model.png")
