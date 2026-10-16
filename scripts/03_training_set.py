import matplotlib.pyplot as plt

from machine_learning_from_scratch.data import m, x, y

print("m =", m)

# The first training example is at position 0 in Python
print("first example:", x[0], "km ->", y[0], "min")

# Plot: each cross is one delivery (one row of the training set)
plt.scatter(x, y, marker="x", color="red")
plt.title("Delivery time vs distance")
plt.xlabel("Distance (km)")
plt.ylabel("Delivery time (min)")
plt.savefig("figures/training_set.png")
