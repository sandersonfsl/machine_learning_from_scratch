# Step 11: Automated tests

A test is a small program that checks that the code gives the right answer, and
reports only "passed" or "failed". Instead of running a script and judging the
output by eye, the check is written once and repeated with a single command.

The tool is [pytest](https://docs.pytest.org/). It is added as a **development
dependency**, one that is only needed to work on the project, not to use it:

```bash
uv add --dev pytest
```

This adds a `[dependency-groups]` section to `pyproject.toml` (shown in [Step 2](02-workspace.md)).

pytest looks for files named `test_*.py` inside `tests/`, and runs every
function whose name starts with `test_`. A test uses `assert`: a statement
that must be true, otherwise the test fails. In `tests/test_model.py`:

```python
from machine_learning_from_scratch.model import predict


def test_predict_applies_the_line():
    # With w=3 and b=12, a 2 km delivery takes 3 * 2 + 12 = 18 minutes
    assert predict(2.0, w=3, b=12) == 18.0
```

The expected value (18) is computed by hand, independently of the code. That is
what makes the test meaningful.

```bash
uv run pytest
```

```
tests/test_model.py .                                                    [100%]

============================== 1 passed in 0.01s ===============================
```

Each `.` is a test that passed. If the code is changed so that `predict`
returns `w * x - b`, the same command reports exactly what went wrong:

```
    def test_predict_applies_the_line():
        # With w=3 and b=12, a 2 km delivery takes 3 * 2 + 12 = 18 minutes
>       assert predict(2.0, w=3, b=12) == 18.0
E       assert -6.0 == 18.0
E        +  where -6.0 = predict(2.0, w=3, b=12)

FAILED tests/test_model.py::test_predict_applies_the_line - assert -6.0 == 18.0
```

The test points to the failing line, the value expected (`18.0`) and the value
obtained (`-6.0`). A mistake that would otherwise silently produce a wrong
model is caught immediately.

## Testing the rest of the library

Each module gets its own test file, named after it. Every test follows the same
pattern: a small input, an expected value computed by hand, and an `assert`.

| File | Test | What it protects |
|---|---|---|
| `test_cost.py` | cost is zero when the line fits perfectly | the sign and the baseline of the cost |
| `test_cost.py` | cost matches a hand calculation | the formula, including the `2m` in the denominator |
| `test_gradient.py` | gradient matches a hand calculation | the derivative formulas |
| `test_gradient.py` | gradient is zero at the best line | the minimum is where the slope is flat |
| `test_gradient.py` | gradient agrees with nudging the parameters | see below |
| `test_gradient_descent.py` | recovers `w=3, b=12` from data with no noise | the algorithm finds the right answer |
| `test_gradient_descent.py` | cost never increases | every step goes downhill |
| `test_gradient_descent.py` | returns one cost per iteration | the shape of what is returned |
| `test_gradient_descent.py` | one step updates `w` and `b` simultaneously | the detail from [Step 8](08-gradient-step.md) |

**Gradient checking.** The derivative can be measured without any formula: nudge
a parameter slightly up and down, and see how much the cost changes. That slope
must match what `compute_gradient` returns. If the formula had a mistake, the
two numbers would disagree. It is a standard technique for checking gradients in
machine learning code.

```python
nudge_w = (compute_cost(x, y, w + h, b) - compute_cost(x, y, w - h, b)) / (2 * h)
assert dj_dw == pytest.approx(nudge_w, rel=1e-4)
```

`pytest.approx` compares decimal numbers with a tolerance, since results such as
`14 / 3` cannot be stored exactly.

```bash
uv run pytest
```

```
============================== 10 passed in 0.24s ===============================
```

**Do the tests catch mistakes?** A test that never fails proves nothing, so it is
worth breaking the code on purpose and checking that a test complains:

| Bug introduced | Tests that fail |
|---|---|
| gradient forgets to multiply the error by `x` | 4 |
| cost divides by `m` instead of `2m` | 2 |
| `b` updated using the already-updated `w` | 1 (`test_one_step_updates_w_and_b_simultaneously`) |

The last bug is the instructive one. Gradient descent with a non-simultaneous
update still converges to nearly the same line, so tests that only look at the
final result pass. Only a test that follows **a single step** and compares it with
a hand calculation exposes it. The lesson: test the behavior of the algorithm
in small pieces, not just where it ends.

---

[← Step 10: Visualizing the result](10-plot-result.md) · [Step 12: Multiple features →](12-multiple-features.md)
