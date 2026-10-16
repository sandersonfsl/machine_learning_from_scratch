from machine_learning_from_scratch.cost import compute_cost
from machine_learning_from_scratch.data import x, y
from machine_learning_from_scratch.gradient import compute_gradient

w, b = 0.0, 0.0  # starting guess
alpha = 0.01  # learning rate: how big each step is

print(f"start: w={w}, b={b}, cost={compute_cost(x, y, w, b):.2f}")

dj_dw, dj_db = compute_gradient(x, y, w, b)
print(f"derivatives: dj_dw={dj_dw:.2f}, dj_db={dj_db:.2f}")

# One step downhill. Both new values use the OLD w and b (simultaneous update)
new_w = w - alpha * dj_dw
new_b = b - alpha * dj_db
w, b = new_w, new_b

print(f"after 1 step: w={w:.2f}, b={b:.2f}, cost={compute_cost(x, y, w, b):.2f}")
