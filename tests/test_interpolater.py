import numpy as np
import pytest

from lutinterp import CInterpolator


def linear_fn(x):
    return x + 100


# --- Step 1: smoke ---


def test_constructs_and_renders():
    ci = CInterpolator(np.sin, name="sine", xRangeExp=10, xSupportPointsExp=3)
    out = str(ci)
    assert "sine_linear_interpolation" in out


# --- Step 2: structural ---


def test_linear_coef_is_1d():
    ci = CInterpolator(linear_fn, typ="linear", xRangeExp=10, xSupportPointsExp=3)
    assert ci._coef.ndim == 1


def test_cubic_coef_is_2d_with_4_columns():
    ci = CInterpolator(np.sin, typ="cubic", xRangeExp=10, xSupportPointsExp=3)
    assert ci._coef.ndim == 2
    assert ci._coef.shape[1] == 4


def test_support_point_count():
    ci = CInterpolator(linear_fn, xSupportPointsExp=3, xRangeExp=10)
    assert len(ci._x) == 2**3 + 1


@pytest.mark.parametrize(
    "xRangeSign,expected_x0,expected_xlast",
    [
        ("pos", 0.0, 2**8),
        ("neg", 2**8, 0.0),
        ("center", -(2**8), 2**8),
    ],
)
def test_x_range_endpoints(xRangeSign, expected_x0, expected_xlast):
    func = np.sin if xRangeSign == "center" else linear_fn
    ci = CInterpolator(func, xRangeExp=8, xSupportPointsExp=2, xRangeSign=xRangeSign)
    assert ci._x[0] == expected_x0
    assert ci._x[-1] == expected_xlast


@pytest.mark.parametrize(
    "coefResExp,expected_dtype",
    [
        (8, "int8_t"),
        (16, "int16_t"),
        (32, "int32_t"),
    ],
)
def test_c_array_dtype(coefResExp, expected_dtype):
    ci = CInterpolator(
        linear_fn, name="f", coefResExp=coefResExp, xRangeExp=10, xSupportPointsExp=3
    )
    assert expected_dtype in str(ci)


@pytest.mark.parametrize(
    "xRangeSign,expected_input_type",
    [
        ("pos", "uint32_t"),
        ("center", "int32_t"),
    ],
)
def test_c_function_input_type(xRangeSign, expected_input_type):
    func = np.sin if xRangeSign == "center" else linear_fn
    ci = CInterpolator(
        func, name="f", xRangeSign=xRangeSign, xRangeExp=10, xSupportPointsExp=3
    )
    assert expected_input_type in str(ci)


# --- Step 3: numerical round-trip ---


def _eval_linear_formula(ci: CInterpolator, val: int) -> int:
    """Reproduce the generated C linear interpolation formula in Python.

    Mirrors _c_formula_linear exactly: same integer types, same bit-shifts.
    The coef array is cast to the same integer type the C code uses so that
    truncation behaviour matches.
    """
    if ci._coefResExp <= 8:
        coef = [int(c) for c in ci._coef.astype(np.int8)]
    elif ci._coefResExp <= 16:
        coef = [int(c) for c in ci._coef.astype(np.int16)]
    else:
        coef = [int(c) for c in ci._coef.astype(np.int32)]

    shift = ci._coefShift[0]
    xSupRangeExp = ci._xSupRangeExp
    mask = (1 << xSupRangeExp) - 1

    p = val >> xSupRangeExp
    x = val & mask

    if shift >= 0:
        shift1 = xSupRangeExp
        const = coef[p]
    else:
        shift1 = xSupRangeExp + shift   # may be negative
        const = coef[p] << int(abs(shift))

    diff = coef[p + 1] - coef[p]
    if shift1 > 0:
        linear_term = (diff * x) >> shift1
    elif shift1 == 0:
        linear_term = diff * x
    else:
        linear_term = (diff * x) << int(abs(shift1))

    result = linear_term + const
    if shift > 0:
        result = result >> shift
    return result


def test_linear_roundtrip_accuracy():
    """C formula evaluated in Python must approximate func to within ~2 output units."""

    def quad(x):
        return 0.001 * x**2

    ci = CInterpolator(
        quad,
        name="q",
        typ="linear",
        coefResExp=16,
        xRangeExp=10,
        xSupportPointsExp=4,   # 16 segments → fine enough for a quadratic
        xRangeSign="pos",
    )

    # 50 evenly-spaced integer inputs; stay below 2**xRangeExp so p stays in range
    vals = np.arange(10, 2**10, 2**10 // 50, dtype=int)

    errors = [abs(_eval_linear_formula(ci, int(v)) - quad(v)) for v in vals]
    assert max(errors) < 3, f"max absolute error = {max(errors):.2f}"
