# lutInterp

Generate C lookup-table interpolation code from Python callables — using integer-only arithmetic (no floating point at runtime) suitable for embedded targets (32-bit MCUs).

Complex math functions sometimes consume too much runtime on MCUs, need extra libraries
and extra engineering. It's much simpler to do the math in python and produce a lookup
table which can be interpolated, all with fix point arithmetic.

## Installation

```bash
pip install lutinterp
```

## What it does

Given a Python function, `CInterpolator` produces:

- A **C coefficient array** (`int8_t`, `int16_t`, or `int32_t`) scaled to maximally fill the chosen integer type
- A **C function** that approximates the original using only integer arithmetic and bit-shifts — no division, no floating point

The x-axis is parameterized with power-of-2 exponents so that table lookup is a single right-shift instruction.

## Usage

```python
import numpy as np
from lutinterp import CInterpolator

# Approximate sqrt over [0, 4096] with a linear LUT
lut = CInterpolator(
    np.sqrt,
    typ="linear",
    coefResExp=16,       # store coefficients as int16_t
    xSupportPointsExp=4, # 2^4 = 16 segments
    xRangeExp=12,        # input range [0, 2^12] = [0, 4096]
    xRangeSign="pos",
)

print(lut)              # prints C code to stdout
lut.print_to_file("sqrt_lut.c")  # or save to a file
```

### Cubic spline

```python
lut = CInterpolator(
    np.sqrt,
    typ="cubic",
    coefResExp=16,
    xSupportPointsExp=3,
    xRangeExp=12,
    xRangeSign="pos",
)
```

### Visualise error

```python
lut.plot_error()   # relative error vs exact function
lut.plot_func()    # overlay: linear / cubic / pchip / exact
```

## Key parameters

| Parameter | Description |
|-----------|-------------|
| `typ` | `'linear'` or `'cubic'` |
| `coefResExp` | Bit-width of coefficients (8, 16, or 32) |
| `xSupportPointsExp` | `log2` of the number of LUT segments |
| `xRangeExp` | `log2` of the input range |
| `xRangeSign` | `'pos'` → [0, 2^n], `'neg'` → [2^n, 0], `'center'` → [-2^n, +2^n] |

## License

MIT — see [LICENSE](LICENSE).
