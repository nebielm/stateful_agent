import ast

import pytest

from app.services.tools import calculator
from app.utils import math_tools


@pytest.mark.parametrize("expression,expected", [
    ("2 + 3 * 4", 14), ("2**10", 1024), ("9**0.5", 3.0),
    ("2**-3", 0.125), ("(-2)**3", -8), ("10 / 4", 2.5), ("2**1000", 2**1000),
])
def test_bounded_calculator_preserves_ordinary_arithmetic(expression, expected):
    assert math_tools.safe_eval(expression) == expected


@pytest.mark.parametrize("expression", [
    "9**999999999", "9**9**9", "(2**1000)**1000", "2**-1001",
    "1e309", "1e308*1e308", "(-1)**0.5", "True+1", "1+" * 26 + "1",
])
def test_calculator_rejects_pathological_or_non_real_results(expression):
    with pytest.raises((ValueError, OverflowError)):
        math_tools.safe_eval(expression)
    assert isinstance(calculator.invoke({"expression": expression}), str)


def test_excessive_power_is_rejected_before_power_operation(monkeypatch):
    def forbidden(*args):
        pytest.fail("Pathological exponentiation must not be evaluated")

    monkeypatch.setitem(math_tools.ALLOWED_OPERATORS, ast.Pow, forbidden)
    with pytest.raises(ValueError, match="limit"):
        math_tools.safe_eval("9**999999999")
