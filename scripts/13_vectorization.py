import time

import numpy as np

from machine_learning_from_scratch.model import predict, predict_loop


def best_time(function, repeats=20):
    """Smallest time, in milliseconds, over several runs.

    The first calls are not counted: they pay a start-up cost (memory, threads)
    that has nothing to do with the computation being measured.
    """
    for _ in range(3):
        function()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return min(times) * 1000


def update_loop(w, d, alpha):
    """w_j := w_j - alpha * d_j, one parameter at a time."""
    w = w.copy()
    for j in range(len(w)):
        w[j] = w[j] - alpha * d[j]
    return w


def update_vectorized(w, d, alpha):
    """The same update for every parameter at once."""
    return w - alpha * d


rng = np.random.default_rng(0)

# 1. The prediction f(x) = w . x + b: loop vs dot product
print("prediction")
print(f"{'n':>10} {'loop (ms)':>12} {'np.dot (ms)':>12} {'speedup':>9}")
for n in (16, 1_000, 100_000, 1_000_000):
    w, x = rng.random(n), rng.random(n)
    loop = best_time(lambda: predict_loop(x, w, 0.0))
    dot = best_time(lambda: predict(x, w, 0.0))
    print(f"{n:>10,} {loop:>12.4f} {dot:>12.4f} {loop / dot:>8.0f}x")

# 2. The update w := w - alpha * d for 16 parameters: same result, two ways
w, d = rng.random(16), rng.random(16)
same = np.allclose(update_loop(w, d, 0.1), update_vectorized(w, d, 0.1))
print("\nupdate of 16 parameters, same result:", same)

# 3. The same update with many parameters
print("\nupdate w := w - alpha * d")
print(f"{'n':>10} {'loop (ms)':>12} {'vector (ms)':>12} {'speedup':>9}")
for n in (16, 1_000, 100_000, 1_000_000):
    w, d = rng.random(n), rng.random(n)
    loop = best_time(lambda: update_loop(w, d, 0.1))
    vec = best_time(lambda: update_vectorized(w, d, 0.1))
    print(f"{n:>10,} {loop:>12.4f} {vec:>12.4f} {loop / vec:>8.0f}x")
