from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.data import x, y
from machine_learning_from_scratch.model import predict

# Step by step on the first delivery, with w=3 and b=12
prediction = predict(x[0], w=3, b=12)
error = prediction - y[0]
print(f"prediction: {prediction:.1f} min, real: {y[0]} min")
print(f"error: {error:.1f}, squared error: {error**2:.2f}")

# The cost of the whole training set, for different lines
print()
print("w=3, b=12:", round(compute_cost(x, y, w=3, b=12), 2))
print("w=1, b=5: ", round(compute_cost(x, y, w=1, b=5), 2))
print("w=0, b=0: ", round(compute_cost(x, y, w=0, b=0), 2))
