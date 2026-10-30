# Step 6: The cost function

To compare lines with a number, the cost function measures how far the
predictions are from the right answers. It is built in small pieces:

1. **Error** of one example: prediction minus the right answer,
   $\hat{y}^{(i)} - y^{(i)}$.
2. **Squared error**: $(\hat{y}^{(i)} - y^{(i)})^2$. Squaring removes the sign
   (being 7 minutes early or late is equally wrong) and punishes big errors
   more than small ones.
3. **Sum** over all $m$ examples.
4. **Average** by dividing by $m$, so the number does not grow just because
   there is more data.
5. **Divide by 2** as well. It changes nothing about which line is best and
   makes the math cleaner in gradient descent ([Step 8](08-gradient-step.md)).

The result is the **squared error cost function**:

$$J(w,b) = \frac{1}{2m}\sum_{i=1}^{m}\left(f_{w,b}(x^{(i)}) - y^{(i)}\right)^2$$

In `src/machine_learning_from_scratch/cost.py`, each line of the function is one piece:

```python
def compute_cost(x, y, w, b):
    m = len(x)
    predictions = predict(x, w, b)
    errors = predictions - y
    squared_errors = errors**2
    return squared_errors.sum() / (2 * m)
```

```bash
uv run scripts/06_cost.py
```

```
prediction: 13.5 min, real: 20.5 min
error: -7.0, squared error: 49.00

w=3, b=12: 4.41
w=1, b=5:  170.23
w=0, b=0:  418.37
```

The first delivery was predicted 7 minutes too short (error −7, squared 49).
Over the whole training set, the blue line from the last step costs 4.41,
the green one 170.23, and a flat line at zero costs 418.37. **The smaller the
cost, the better the line fits.** Learning means finding the `w` and `b` that
make $J(w,b)$ as small as possible.

---

[← Step 5: Drawing the model](05-plot-model.md) · [Step 7: The shape of the cost →](07-plot-cost.md)
