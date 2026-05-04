# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`lutInterp` is a Python package that generates C lookup-table interpolation code. Given a Python callable, it produces a C coefficient array and a C function that approximates the callable using integer-only arithmetic (no floating point) — suitable for embedded targets.

## Commands

```bash
# Install dependencies (including dev extras)
uv sync --group dev

# Run tests
uv run pytest

# Run a script that uses the package
uv run python my_script.py
```

## Project structure

```
src/lutinterp/
    __init__.py        # exports CInterpolator
    interpolater.py    # contains CInterpolator class
tests/
    test_interpolater.py
pyproject.toml
```

The package is `lutinterp`; the public API is a single class `CInterpolator` (note correct spelling — the file is named `interpolater.py` but the class is `CInterpolator`).

## Architecture

**Core data flow:**

1. `__init__` → `_create_interp_points()` builds x/y sample arrays plus a `CubicSpline` and `PchipInterpolator` from scipy.
2. `_scale_coef()` scales coefficients to maximally fill the signed integer type chosen by `coefResExp` (e.g. 16 → int16_t), storing per-column shift amounts in `_coefShift`.
3. `__repr__` / `print_to_file` assembles the C output by calling `_c_header()`, `_c_array()`, `_c_func_std()`, and either `_c_formula_linear()` or `_c_formula_cubic()`.

**Key design constraint — power-of-2 arithmetic:**

All x-axis parameters are expressed as binary exponents (`xRangeExp`, `xSupportPointsExp`). The spacing between support points is always a power of 2, so the generated C function can locate the correct table segment with a single right-shift (`val >> xSupRangeExp`) instead of a division. The fractional position within the segment (`x = val & mask`) is also extracted with a bitmask. The `_coefShift` values compensate for the integer scaling so the final result stays in range.

**Linear vs. cubic output:**

- Linear: `_coef` is a 1-D array of y values; the C function interpolates between adjacent entries.
- Cubic: `_coef` is a 2-D array (`CubicSpline.c.T`), shape `(n_segments, 4)` — one row of `[c3, c2, c1, c0]` per segment. The generated C code evaluates the polynomial with bit-shift scaling to avoid overflow.

**`xRangeSign`:**

Controls the sign/centering of the input domain: `'pos'` → `[0, 2^xRangeExp]`, `'neg'` → `[2^xRangeExp, 0]` (reversed), `'center'` → `[-2^xRangeExp, +2^xRangeExp]`. This affects both the index offset in `_c_func_std` and the choice of `uint32_t` vs. `int32_t` for the C function argument. Invalid values raise `ValueError`.

**`scaleCoef` parameter:**

When `False`, skips `_scale_coef()` and sets all shifts to 0. Useful for direct-lookup tables or testing.

**`fargs` parameter:**

Extra positional arguments forwarded to `func` when sampling x values. Allows parameterized callables without wrapping them in a lambda.

**Plotting helpers:**

`plot_error()` and `plot_func()` display matplotlib figures comparing linear, cubic, pchip, and exact results. These require `plt.show()` to block and are for interactive use only.
