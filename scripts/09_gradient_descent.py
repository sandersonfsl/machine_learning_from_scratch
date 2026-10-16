from machine_learning_from_scratch.data import x, y
from machine_learning_from_scratch.gradient_descent import gradient_descent
from machine_learning_from_scratch.model import predict

w, b, costs = gradient_descent(x, y, w=0.0, b=0.0, alpha=0.01, num_iters=10_000)

for i in range(0, 10_000, 1000):
    print(f"iteration {i:5d}: cost={costs[i]:.3f}")

print(f"\nlearned: w={w:.2f}, b={b:.2f}")
print(f"prediction for 6.5 km: {predict(6.5, w, b):.1f} min")
