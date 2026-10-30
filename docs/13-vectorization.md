# Step 13: Vectorization

The prediction `f(x) = w · x + b` can be computed in three ways:

```python
# 1. Each term written by hand: only practical for a handful of features
f = w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + b

# 2. A loop over the features (predict_loop)
f = 0.0
for j in range(n):
    f += w[j] * x[j]
f += b

# 3. The dot product (predict)
f = np.dot(w, x) + b
```

The third form is **vectorized**, and it has two benefits: the code is shorter,
and it runs much faster.

The speed comes from how the work is done:

- The **loop** calculates one product at a time, in sequence: first index 0,
  then index 1, then index 2, and so on.
- The **dot product** multiplies every pair of numbers at the same time, in
  parallel on the hardware (a CPU, or a GPU), and then adds the products with
  specialized hardware, instead of one addition after another.

The same idea applies to the gradient descent update `w_j := w_j - α · d_j`,
which will be needed for every parameter at each step:

```python
def update_loop(w, d, alpha):
    w = w.copy()
    for j in range(len(w)):
        w[j] = w[j] - alpha * d[j]      # one parameter at a time
    return w


def update_vectorized(w, d, alpha):
    return w - alpha * d                # all parameters at once
```

`scripts/13_vectorization.py` runs both versions and measures the time.

```bash
uv run scripts/13_vectorization.py
```

```
prediction
         n    loop (ms)  np.dot (ms)   speedup
        16       0.0054       0.0012        5x
     1,000       0.3103       0.0014      226x
   100,000      22.0566       0.0068     3229x
 1,000,000     218.5620       0.0285     7661x

update of 16 parameters, same result: True

update w := w - alpha * d
         n    loop (ms)  vector (ms)   speedup
        16       0.0066       0.0016        4x
     1,000       0.3790       0.0023      164x
   100,000      27.8438       0.0432      645x
 1,000,000     277.0212       3.5047       79x
```

The exact times depend on the machine, and can vary from one run to the next;
the shape of the table does not.

- With **16 features** the difference is small, about 4 to 5 times.
- With **many features** it grows enormously: for a million features the loop
  takes 218 ms and the dot product takes 0.03 ms. With thousands of features and
  large training sets, this can be the difference between a training run that
  finishes in a minute or two and one that takes hours.
- The two versions of the update give **the same result**; the vectorized one is
  a single line.

The helper `best_time` takes the smallest of 20 runs and ignores the first three
calls. The first calls pay a start-up cost (memory allocation, threads) that
would otherwise distort the measurement.

---

[← Step 12: Multiple features](12-multiple-features.md) · [Step 14: Gradient descent with multiple features →](14-multiple-gradient-descent.md)
