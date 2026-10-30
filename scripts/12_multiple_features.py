import numpy as np

from machine_learning_from_scratch.data_multiple import FEATURES, make_deliveries
from machine_learning_from_scratch.model import predict, predict_loop

X, y = make_deliveries()
m, n = X.shape
print(f"m = {m} examples, n = {n} features: {FEATURES}")
print("X shape:", X.shape, "| y shape:", y.shape)

# x^(1): the first example is a list with one value per feature
print("\nx^(1) =", X[0], "-> y^(1) =", y[0], "min")
# x^(1)_1: feature 1 of example 1 (Python: row 0, column 0)
print("x^(1)_1 =", X[0, 0], "m")

# A model with hand-picked parameters: one weight per feature, plus b
w = np.array([0.003, 1.0, 7.0])
b = 10.0

# f(x) = w1*x1 + w2*x2 + w3*x3 + b, one term at a time
print(f"\nprediction, term by term: {predict_loop(X[0], w, b):.2f} min")
# f(x) = w . x + b, the same thing with the dot product
print(f"prediction, dot product:  {predict(X[0], w, b):.2f} min")
print(f"real time:                {y[0]:.2f} min")
