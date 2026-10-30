# Step 9: Gradient descent in a loop

Repeating the step many times is a `for` loop. In `src/machine_learning_from_scratch/gradient_descent.py`:

```python
def gradient_descent(x, y, w, b, alpha, num_iters):
    costs = []
    for _ in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w, b = w - alpha * dj_dw, b - alpha * dj_db
        costs.append(compute_cost(x, y, w, b))  # keep the cost of each step
    return w, b, costs
```

The loop starts from `w = 0, b = 0`, with `alpha = 0.01` and 10,000 iterations
(`compute_gradient` is the function from [Step 8](08-gradient-step.md)).

```bash
uv run scripts/09_gradient_descent.py
```

```
iteration     0: cost=186.487
iteration  1000: cost=4.533
iteration  2000: cost=4.321
iteration  3000: cost=4.319
iteration  4000: cost=4.318
...
learned: w=2.85, b=12.85
prediction for 6.5 km: 31.4 min
```

What happened:

- The cost fell fast at first (418 → 4.5 in the first thousand steps) and then
  more and more slowly. Steps get smaller as the slope flattens near the
  bottom of the bowl.
- After about 4,000 iterations nothing changes anymore. This is
  **convergence**: the bottom of the bowl has been reached.
- The model learned `w ≈ 2.85` and `b ≈ 12.85`. The data was generated from
  3 and 12 plus random noise, so the learned values are close but not
  identical. The noise in the 50 deliveries pulls them slightly. NumPy's exact
  least squares fit (`np.polyfit(x, y, 1)`) gives the same numbers, which
  confirms gradient descent found the best line for this data.

This is a complete learning algorithm: **linear regression trained with
gradient descent**. It answers the original question: a delivery 6.5 km away
should be promised in about **31 minutes**.

---

[← Step 8: Gradient descent, one step](08-gradient-step.md) · [Step 10: Visualizing the result →](10-plot-result.md)
