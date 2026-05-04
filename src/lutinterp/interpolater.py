# -*- coding: utf-8 -*-
"""Create lookup tables for interpolation in C."""

import copy
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interp
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator


def _build_xs(
    xSign: str,
    xRangeExp: int,
    xSupPointsExp: int,
    xStart: int,
    xEnd: int | None,
) -> NDArray[np.float64]:
    """Return the cropped x support-point array."""
    n = (2**xSupPointsExp) + 1
    r = 2**xRangeExp
    if xSign == "pos":
        x = np.linspace(0, r, n)
    elif xSign == "neg":
        x = np.linspace(r, 0, n)
    else:  # "center"
        x = np.linspace(-r, r, n)
    return x[xStart:xEnd]


def _sample_with_sim(
    func: Callable[..., NDArray[np.float64]],
    fargs: tuple,
    fkwargs: dict,
    xs: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return (ys, xsim, ysim)."""
    ys = func(xs, *fargs, **fkwargs)
    xsim = np.linspace(xs[0], xs[-1], 10000)[1:-1]
    ysim = func(xsim, *fargs, **fkwargs)
    return ys, xsim, ysim


def _fit_splines(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    xsim: NDArray[np.float64],
    ysim: NDArray[np.float64],
) -> tuple[interp.CubicSpline, PchipInterpolator]:
    """Return (pp, pchip) fitted to xs/ys with boundary slopes from xsim/ysim."""
    slopeStart = (ysim[1] - ysim[0]) / (xsim[1] - xsim[0])
    slopeEnd = (ysim[-1] - ysim[-2]) / (xsim[-1] - xsim[-2])
    bc = ((1, slopeStart), (1, slopeEnd))
    # CubicSpline and PchipInterpolator require strictly increasing x.
    # For 'neg' mode xs is decreasing, so sort before passing to scipy.
    if xs[0] > xs[-1]:
        xsAsc, ysAsc = xs[::-1], ys[::-1]
        bcAsc = ((1, slopeEnd), (1, slopeStart))
    else:
        xsAsc, ysAsc = xs, ys
        bcAsc = bc
    pp = interp.CubicSpline(xsAsc, ysAsc, bc_type=bcAsc)  # type: ignore[arg-type]
    pchip = PchipInterpolator(xsAsc, ysAsc)
    return pp, pchip


def _extract_coef(
    typ: str,
    pp: interp.CubicSpline,
    ys: NDArray[np.float64],
    xs: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the coefficient array for the given interpolation type."""
    if typ == "cubic":
        coef = copy.copy(pp.c.T)
        return coef[::-1] if xs[0] > xs[-1] else coef
    else:  # "linear"
        return np.array(ys)


class CInterpolator(object):
    """Class to produce interpolation tables and the according C-function

    The function that is produced can be either linear or a cubic spline.
    Note that the cubic spline needs four times more memory for the same
    amount of supporting points.

    Parameters
    ----------
    func : callable
        The function that is interpolated.
    fargs : tuple
        Extra positional arguments forwarded to `func` when sampling x values.
    fkwargs : dict or None
        Extra keyword arguments forwarded to `func` when sampling x values.
    name : str
        A string used to name the printed function and the coefficient table.
        If left None, `func.__name__` is used instead.
    typ : str
        Either 'linear' or 'cubic'. Defines what will be printed into the
        output file.
    coefResExp : int
        2**x exponent defining the resolution of the coefficients. Typically
        one chooses 8, 16 or 32.
    xSupportPointsExp : int
        2**x exponent that defines the number of supporting points.
    xRangeExp : int
        2**x exponent that defines the input range. The input range is
        devided into slices by the number of supporting points (2**nExp).
    xRangeSign : str
        Sign of the input range, 'pos', 'neg' or 'center'
    xStartIdx, xEndIdx : int or None
        List indexes to crop the input range, producing a smaller output table.
    scaleCoef: Bool
        Usually the coeficients are scalled to get the best resolution.
        It can be disabled for test purpose or to get the plain table for
        direct lookup.
    """

    def __init__(
        self,
        func: Callable[..., NDArray[np.float64]],
        fargs: tuple = (),
        fkwargs: dict | None = None,
        name: str | None = None,
        typ: str = "linear",
        coefResExp: int = 16,
        xSupportPointsExp: int = 3,
        xRangeExp: int = 12,
        xRangeSign: str = "pos",
        xStartIdx: int = 0,
        xEndIdx: int | None = None,
        scaleCoef: bool = True,
    ) -> None:
        self._func = func
        self._funcName = func.__name__ if name is None else name
        self._fargs = fargs
        self._fkwargs = {} if fkwargs is None else fkwargs
        if typ not in ["linear", "cubic"]:
            raise ValueError(f"typ must be 'linear' or 'cubic', got {typ!r}")
        self._type = typ
        self._coefResExp = np.clip(coefResExp, 1, 32)
        self._xSupPointsExp = xSupportPointsExp
        self._xRangeExp = xRangeExp
        if xRangeSign not in ["pos", "neg", "center"]:
            raise ValueError(
                f"xRangeSign must be 'pos', 'neg', or 'center', got {xRangeSign!r}"
            )
        self._xSign = xRangeSign
        if xRangeSign in ["pos", "neg"]:
            self._xSupRangeExp = int(xRangeExp - xSupportPointsExp)
        else:
            self._xSupRangeExp = int(xRangeExp + 1 - xSupportPointsExp)
        self._xStart = xStartIdx
        self._xEnd = xEndIdx

        self._coefShift: list[int] = []
        self._x = _build_xs(
            self._xSign, self._xRangeExp, self._xSupPointsExp, self._xStart, self._xEnd
        )
        self._y, self._xsim, self._ysim = _sample_with_sim(
            self._func, self._fargs, self._fkwargs, self._x
        )
        self._pp, self._pchip = _fit_splines(self._x, self._y, self._xsim, self._ysim)
        self._coef = _extract_coef(self._type, self._pp, self._y, self._x)
        if scaleCoef:
            self._scale_coef()
        else:
            self._coefShift = [0, 0]
        if self._type == "cubic":
            self._check_overflow()
        self._coefDataType = None

    def __repr__(self):
        s = []
        s.append(self._c_header())
        s.append(self._c_array())
        s.append("\n")
        s.append(self._c_func_std())
        if self._type == "cubic":
            y = self._c_formula_cubic()
            # s.append(self._c_formula_cubic())
        elif self._type == "linear":
            y = self._c_formula_linear()
            # s.append(self._c_formula_linear())
        s.append(" " * 4 + f"return {y};" + "\n}\n")
        s.append(f"/*{'-' * 75}*/\n")
        return "".join(s)

    def plot_error(self):
        """Generate an error plot comparing results from interpolation with
        selected method against user supplied function.
        """
        plt.figure(65)
        ylin = np.interp(self._xsim, self._x, self._y)
        y = 100 * (self._ysim - ylin) / self._ysim
        plt.plot(self._xsim, y, label="linear")
        y = 100 * (self._ysim - self._pp(self._xsim)) / self._ysim
        plt.plot(self._xsim, y, label="cubic")
        y = 100 * (self._ysim - self._pchip(self._xsim)) / self._ysim
        plt.plot(self._xsim, y, label="pchip")
        plt.title(f"error [%] ({self._funcName})")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_func(self):
        """Plot function values from linear, cubic and exact solution."""
        plt.figure(66)
        ylin = np.interp(self._xsim, self._x, self._y)
        plt.plot(self._xsim, ylin, label="linear")
        plt.plot(self._xsim, self._pp(self._xsim), label="cubic")
        plt.plot(self._xsim, self._pchip(self._xsim), label="pchip")
        plt.plot(self._xsim, self._ysim, label="exact")
        plt.title(f"function ({self._funcName})")
        plt.grid(True)
        plt.legend()
        plt.show()

    def print_to_file(self, filename: str | Path = "output.txt"):
        """Print the C function and coefficient array to a file."""
        Path(filename).parent.mkdir(parents=True, exist_ok=True)
        with Path(filename).open("w") as f:
            f.write(str(self))

    def _c_array(self):
        """Generate the C array string from numpy array."""
        if self._coefResExp <= 8:
            dataType = "int8_t"
            coef = self._coef.astype(np.int8)
        elif self._coefResExp <= 16:
            dataType = "int16_t"
            coef = self._coef.astype(np.int16)
        elif self._coefResExp <= 32:
            dataType = "int32_t"
            coef = self._coef.astype(np.int32)
        self._coefDataType = dataType
        s = "const {} {}Coef" + "[{}]" * len(coef.shape) + " =\n"
        s = s.format(dataType, self._funcName, *coef.shape)
        tmp = np.array2string(coef, separator=", ").replace("[", "{")
        tmp = tmp.replace("]", "}")
        s += tmp + ";\n"
        return s

    def _c_header(self):
        """Generate a header string."""
        s = []
        s.append("/**")
        s.append(f" * {self._funcName}")
        s.append(" *")
        s.append(f" * Input range: [{int(self._x[0])} : {int(self._x[-1] - 1)}]")
        s.append(f" * Output range: [{int(self._y[0])} : {int(self._y[-1])}]")
        s.append(" */")
        return "\n".join(s) + "\n"

    def _c_func_std(self, indent=4):
        """funfact: because coef is signed, x sould be signed too
        otherwise 64x64 multiplications are used which is much slower!"""
        assert self._coefDataType is not None
        indent = " " * indent
        inputType = "uint32_t" if self._xSign == "pos" else "int32_t"
        outputType = "uint32_t" if min(self._ysim) >= 0 else "int32_t"
        s = []
        tmp = f"{outputType} {self._funcName}_{self._type}_interpolation({inputType}"
        s.append("extern " + tmp + " val);")
        s.append(tmp + " val) {")
        n = self._xSupPointsExp
        if self._xSign == "pos":
            offset = -self._xStart
        elif self._xSign == "neg":
            offset = 2**n - 1 - self._xStart
        elif self._xSign == "center":
            offset = 2 ** (n - 1) - 1 - self._xStart
        tmp = "val >> {:d}".format(self._xSupRangeExp)
        if offset < 0:
            p = "uint32_t p = ({}) - {:d};".format(tmp, int(np.abs(offset)))
        elif offset > 0:
            p = "uint32_t p = ({}) + {:d};".format(tmp, offset)
        else:
            p = "uint32_t p = {};".format(tmp)
        s.append(indent + p)
        s.append(indent + f"if (p > {int(len(self._x) - 2):d}) while (1);")
        cDType = f"const {self._coefDataType} *"
        if self._type == "cubic":
            s.append(indent + cDType + f"c = {self._funcName}Coef[p];")
        else:
            s.append(indent + cDType + f"c = {self._funcName}Coef;")
        s.append(indent + f"int32_t x = val & {(2**self._xSupRangeExp) - 1:#x};")
        return "\n".join(s) + "\n"

    def _c_formula_cubic(self, indent=4):
        """ """
        shift = self._coefShift
        xExp = self._xSupRangeExp
        #
        if shift[3] > 0:
            shift[0] -= shift[3]
            shift[1] -= shift[3]
            shift[2] -= shift[3]
        # cubic part
        shift1 = 0
        shift2 = 0
        if xExp > int(31 / 2):  # make sure no overflow with x**3
            tmp = int(xExp - int(31 / 2))
            shift1 = 3 * tmp
            cubic = "((x>>{:d})*(x>>{:d}))*(x>>{:d})".format(*([tmp] * 3))
        else:
            cubic = "(x*x)*x"
        if 3 * xExp > 31:  # make sure no overflow
            shift2 = int((3 * xExp) - 31)
            s = "(int32_t) (((int64_t) {}) >> {:d})"
            cubic = s.format(cubic, shift2)
        if shift[0] - shift1 - shift2 > 63:
            print("warning, the cubic part will always be 0!")
        if xExp * 3 + self._coefResExp > 31:
            s = "((int32_t) (((int64_t) ({}) * c[0]) >> {:d}))"
        else:
            s = "(((int32_t) ({}) * c[0]) >> {:d})"
        cubic = s.format(cubic, shift[0] - shift1 - shift2)
        # quadratic part
        shift1 = 0
        shift2 = 0
        if xExp > 31:  # make sure no overflow with x**2 (64-bit result)
            tmp = int(xExp - 31)
            shift1 = 3 * tmp
            quadr = "(x>>{:d})*(x>>{:d})".format(tmp, tmp)
        else:
            quadr = "x*x"
        if 2 * xExp > 31:  # make sure no overflow
            shift2 = int((2 * xExp) - 31)
            s = "(int32_t) (((int64_t) {}) >> {:d})"
            quadr = s.format(quadr, shift2)
        if shift[1] - shift1 - shift2 > 63:
            print("warning, the quadratic part will always be 0!")
        if xExp * 2 + self._coefResExp > 31:
            s = "((int32_t) (((int64_t) ({}) * c[1]) >> {:d}))"
        else:
            s = "(((int32_t) ({}) * c[1]) >> {:d})"
        quadr = s.format(quadr, shift[1] - shift1 - shift2)
        # linear
        if xExp + self._coefResExp > 31:
            s = "((int32_t) (((int64_t) x * c[2]) >> {:d}))"
        else:
            s = "(((int32_t) x * c[2]) >> {:d})"
        linear = s.format(shift[2])
        if shift[2] > 63:
            print("warning, the linear part will always be 0!")
        # const
        if shift[3] >= 0:
            const = "c[3]"
        elif shift[3] < 0:
            const = f"((int32_t) c[3] << {int(np.abs(shift[3]))})"
        indent = " " * 4
        s = (
            f"{cubic}\n"
            + indent * 3
            + f"+ {quadr}\n"
            + indent * 3
            + f"+ {linear}\n"
            + indent * 3
            + f"+ {const}"
        )
        if shift[3] > 0:
            s = f"({s}) >> {shift[3]}"
        return s

    def _c_formula_linear(self):
        """ """
        shift = self._coefShift[0]
        if shift >= 0:
            shift1 = self._xSupRangeExp
            const = "c[p]"
        else:
            shift1 = self._xSupRangeExp + shift
            const = f"((int32_t) c[p] << {int(np.abs(shift))})"
        if shift1 > 0:
            linear = f"(((int32_t) (c[p+1] - c[p]) * x) >> {shift1:d})"
        elif shift1 == 0:
            linear = "((int32_t) (c[p+1] - c[p]) * x)"
        else:
            linear = f"(((int32_t) (c[p+1] - c[p]) * x) << {np.abs(shift1):d})"
        s = f"{linear} + {const}"
        if shift > 0:
            s = f"({s}) >> {shift}"
        return s

    def _scale_coef(self):
        """Scale the coefficients.

        The coefficients are scalled to maximize the user supplied
        `_coefResExp`. Since the coefficients are stored in signed integers,
        the maximum is calculated from 2**(_coefResExp - 1).
        """
        res = self._coefResExp
        coef = self._coef

        # linear funcion with 1-dim coefs.
        if len(np.shape(coef)) == 1:
            f = 2 ** (res - 1) / max(np.abs(coef))
            shift = int(np.log2(f) - 1) if np.log2(f) < 0 else int(np.log2(f))
            coef = coef * 2**shift
            if max(coef) == 2 ** (res - 1):
                shift -= 1
                coef = coef * 2 ** (-1)
            self._coefShift.append(shift)

        # cubic funcion with 2-dim coefs.
        else:
            for col in range(len(coef[0, :])):
                f = 2 ** (res - 1) / max(np.abs(coef[:, col]))
                shift = int(np.log2(f) - 1) if np.log2(f) < 0 else int(np.log2(f))
                coef[:, col] = coef[:, col] * 2**shift
                if max(coef[:, col]) == 2 ** (res - 1):
                    shift -= 1
                    coef[:, col] = coef[:, col] * 2 ** (-1)
                self._coefShift.append(shift)
        self._coef = coef

    def _check_overflow(self):
        """Print some warnings if an overflow will/could occur."""
        coef = self._coef
        for col in range(len(coef[0, :]) - 1):
            exp = len(coef[0, :]) - 1 - col
            x = np.log2(
                max(np.abs(coef[:, col]))
                * (2 ** (self._xSupPointsExp * exp - self._coefShift[col]))
            )
            if x > 31:
                print(
                    f"The x**{exp} part results in an overflow. "
                    f"Use int64 intermediate result type"
                )
            elif x > 29:
                print(
                    f"The x**{exp} part is rather high. "
                    f"Be carefull when building the sum."
                    f"Eventuelly use int64 intermediate result type"
                )
