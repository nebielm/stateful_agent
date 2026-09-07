import ast
import math
import operator as op


ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg,
}

MAX_EXPRESSION_LENGTH = 50
MAX_EXPONENT = 1000
MAX_INTEGER_BITS = 4096


def safe_eval(expr: str):
    if len(expr) > MAX_EXPRESSION_LENGTH:
        raise ValueError("Expression must be at most 50 characters")

    def checked(value):
        if type(value) not in (int, float):
            raise ValueError("Only real numbers are allowed")
        if isinstance(value, int) and value.bit_length() > MAX_INTEGER_BITS:
            raise ValueError("Integer result exceeds calculator limit")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Non-finite result is not allowed")
        return value

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return checked(node.value)

        if isinstance(node, ast.BinOp):
            if type(node.op) not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed")
            left, right = _eval(node.left), _eval(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > MAX_EXPONENT:
                    raise ValueError("Exponent exceeds calculator limit")
                # Bound integer allocation before performing exponentiation.
                if isinstance(left, int) and isinstance(right, int) and right > 0:
                    if max(1, abs(left).bit_length()) * right > MAX_INTEGER_BITS:
                        raise ValueError("Power exceeds calculator size limit")
            return checked(ALLOWED_OPERATORS[type(node.op)](left, right))

        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed")
            return checked(ALLOWED_OPERATORS[type(node.op)](_eval(node.operand)))

        raise ValueError("Invalid expression")

    node = ast.parse(expr, mode="eval").body
    return _eval(node)
