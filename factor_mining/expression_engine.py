"""Recursive descent parser and evaluator for formulaic alpha expressions.

Parses expressions like:
    cs_rank(ts_corr(close, volume, 10)) - ts_lag(cs_rank(close), 5)

into an AST, then evaluates against a data panel to produce (M, T) signal arrays.
"""

import re
import logging
from typing import Any, Optional

import numpy as np

from .operators import OPERATOR_REGISTRY

logger = logging.getLogger(__name__)

MAX_RECURSION_DEPTH = 20


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

TOKEN_PATTERNS = [
    ("NUMBER", r"-?\d+\.?\d*"),
    ("IDENT", r"[a-zA-Z_][a-zA-Z0-9_]*"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("PLUS", r"\+"),
    ("MINUS", r"-"),
    ("STAR", r"\*"),
    ("SLASH", r"/"),
    ("SKIP", r"\s+"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in TOKEN_PATTERNS))


class Token:
    __slots__ = ("type", "value")

    def __init__(self, type_: str, value: str):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


def tokenize(expr: str) -> list[Token]:
    tokens = []
    for m in _TOKEN_RE.finditer(expr):
        kind = m.lastgroup
        value = m.group()
        if kind == "SKIP":
            continue
        tokens.append(Token(kind, value))
    return tokens


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------

class ASTNode:
    pass


class NumberNode(ASTNode):
    def __init__(self, value: float):
        self.value = value

    def __repr__(self):
        return f"Num({self.value})"


class FieldNode(ASTNode):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return f"Field({self.name})"


class FuncCallNode(ASTNode):
    def __init__(self, name: str, args: list[ASTNode]):
        self.name = name
        self.args = args

    def __repr__(self):
        return f"Call({self.name}, {self.args})"


class BinOpNode(ASTNode):
    def __init__(self, op: str, left: ASTNode, right: ASTNode):
        self.op = op
        self.left = left
        self.right = right

    def __repr__(self):
        return f"BinOp({self.op}, {self.left}, {self.right})"


class UnaryMinusNode(ASTNode):
    def __init__(self, operand: ASTNode):
        self.operand = operand

    def __repr__(self):
        return f"Neg({self.operand})"


# ---------------------------------------------------------------------------
# Recursive Descent Parser
# ---------------------------------------------------------------------------

class Parser:
    """Parses a tokenized expression into an AST.

    Grammar:
        expr      -> term (('+' | '-') term)*
        term      -> unary (('*' | '/') unary)*
        unary     -> '-' unary | primary
        primary   -> NUMBER | func_call | IDENT | '(' expr ')'
        func_call -> IDENT '(' arg_list ')'
        arg_list  -> expr (',' expr)*
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, expected_type: Optional[str] = None) -> Token:
        tok = self.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of expression")
        if expected_type and tok.type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {tok.type} ({tok.value!r})")
        self.pos += 1
        return tok

    def parse(self) -> ASTNode:
        node = self.parse_expr()
        if self.pos < len(self.tokens):
            raise SyntaxError(f"Unexpected token: {self.tokens[self.pos]}")
        return node

    def parse_expr(self) -> ASTNode:
        node = self.parse_term()
        while self.peek() and self.peek().type in ("PLUS", "MINUS"):
            op = self.consume().value
            right = self.parse_term()
            node = BinOpNode(op, node, right)
        return node

    def parse_term(self) -> ASTNode:
        node = self.parse_unary()
        while self.peek() and self.peek().type in ("STAR", "SLASH"):
            op = self.consume().value
            right = self.parse_unary()
            node = BinOpNode(op, node, right)
        return node

    def parse_unary(self) -> ASTNode:
        if self.peek() and self.peek().type == "MINUS":
            self.consume()
            operand = self.parse_unary()
            return UnaryMinusNode(operand)
        return self.parse_primary()

    def parse_primary(self) -> ASTNode:
        tok = self.peek()
        if tok is None:
            raise SyntaxError("Unexpected end of expression")

        if tok.type == "NUMBER":
            self.consume()
            return NumberNode(float(tok.value))

        if tok.type == "IDENT":
            name = self.consume().value
            if self.peek() and self.peek().type == "LPAREN":
                self.consume("LPAREN")
                args = self.parse_arg_list()
                self.consume("RPAREN")
                return FuncCallNode(name, args)
            return FieldNode(name)

        if tok.type == "LPAREN":
            self.consume("LPAREN")
            node = self.parse_expr()
            self.consume("RPAREN")
            return node

        raise SyntaxError(f"Unexpected token: {tok}")

    def parse_arg_list(self) -> list[ASTNode]:
        args = []
        if self.peek() and self.peek().type == "RPAREN":
            return args
        args.append(self.parse_expr())
        while self.peek() and self.peek().type == "COMMA":
            self.consume("COMMA")
            args.append(self.parse_expr())
        return args


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ExpressionEngine:
    """Parse and evaluate formulaic alpha expressions against a data panel."""

    INFIX_TO_FUNC = {
        "+": "add",
        "-": "sub",
        "*": "mul",
        "/": "div",
    }

    _ast_cache: dict[str, ASTNode] = {}

    def __init__(self, data_panel: dict[str, np.ndarray]):
        """
        Args:
            data_panel: Dict from build_market_panel(). Keys include feature names
                        (open, high, low, close, volume, amount, vwap, returns)
                        plus metadata (codes, dates).
        """
        self.data = data_panel
        self._depth = 0

    def evaluate(self, expression: str) -> np.ndarray:
        """Parse and evaluate an expression string.

        Returns:
            2D signal array (M, T).
        """
        self._depth = 0
        if expression not in self._ast_cache:
            tokens = tokenize(expression)
            parser = Parser(tokens)
            self._ast_cache[expression] = parser.parse()
        ast = self._ast_cache[expression]
        result = self._eval_node(ast)

        if isinstance(result, (int, float)):
            M, T = self.data["close"].shape
            result = np.full((M, T), float(result))

        result = np.where(np.isfinite(result), result, np.nan)
        return result

    def _eval_node(self, node: ASTNode) -> Any:
        self._depth += 1
        if self._depth > MAX_RECURSION_DEPTH:
            raise RecursionError(f"Expression exceeds max depth {MAX_RECURSION_DEPTH}")

        try:
            if isinstance(node, NumberNode):
                return node.value

            if isinstance(node, FieldNode):
                return self._resolve_field(node.name)

            if isinstance(node, UnaryMinusNode):
                val = self._eval_node(node.operand)
                if isinstance(val, (int, float)):
                    return -val
                return -val

            if isinstance(node, BinOpNode):
                return self._eval_binop(node)

            if isinstance(node, FuncCallNode):
                return self._eval_func(node)

            raise ValueError(f"Unknown AST node: {type(node)}")
        finally:
            self._depth -= 1

    def _resolve_field(self, name: str) -> np.ndarray:
        if name in self.data and isinstance(self.data[name], np.ndarray):
            return self.data[name]
        raise ValueError(f"Unknown field: {name}")

    def _eval_binop(self, node: BinOpNode) -> np.ndarray:
        left = self._eval_node(node.left)
        right = self._eval_node(node.right)
        left = self._ensure_array(left)
        right = self._ensure_array(right)

        func_name = self.INFIX_TO_FUNC.get(node.op)
        if func_name and func_name in OPERATOR_REGISTRY:
            func, _ = OPERATOR_REGISTRY[func_name]
            return func(left, right)

        raise ValueError(f"Unknown binary operator: {node.op}")

    def _eval_func(self, node: FuncCallNode) -> np.ndarray:
        name = node.name
        if name not in OPERATOR_REGISTRY:
            raise ValueError(f"Unknown operator: {name}")

        func, arity = OPERATOR_REGISTRY[name]
        args = [self._eval_node(a) for a in node.args]

        if arity == "ts":
            if len(args) != 2:
                raise ValueError(f"{name} expects (array, window), got {len(args)} args")
            arr = self._ensure_array(args[0])
            d = int(args[1]) if isinstance(args[1], (int, float)) else int(args[1].flat[0])
            d = max(2, min(d, arr.shape[1]))
            return func(arr, d)

        if arity == "ts2":
            if len(args) != 3:
                raise ValueError(f"{name} expects (array, array, window), got {len(args)} args")
            a = self._ensure_array(args[0])
            b = self._ensure_array(args[1])
            d = int(args[2]) if isinstance(args[2], (int, float)) else int(args[2].flat[0])
            d = max(2, min(d, a.shape[1]))
            return func(a, b, d)

        if arity == "cs":
            if len(args) != 1:
                raise ValueError(f"{name} expects (array,), got {len(args)} args")
            return func(self._ensure_array(args[0]))

        if arity == "unary":
            if len(args) != 1:
                raise ValueError(f"{name} expects (array,), got {len(args)} args")
            return func(self._ensure_array(args[0]))

        if arity == "binary":
            if len(args) != 2:
                raise ValueError(f"{name} expects (array, array), got {len(args)} args")
            return func(self._ensure_array(args[0]), self._ensure_array(args[1]))

        if arity == "power":
            if len(args) != 2:
                raise ValueError(f"{name} expects (array, number), got {len(args)} args")
            arr = self._ensure_array(args[0])
            n = float(args[1]) if isinstance(args[1], (int, float)) else float(args[1].flat[0])
            return func(arr, n)

        raise ValueError(f"Unknown arity type: {arity}")

    def _ensure_array(self, val: Any) -> np.ndarray:
        if isinstance(val, np.ndarray):
            return val
        M, T = self.data["close"].shape
        return np.full((M, T), float(val))


# ---------------------------------------------------------------------------
# Validation utility
# ---------------------------------------------------------------------------

def validate_expression(expr: str) -> tuple[bool, str]:
    """Validate expression syntax without evaluating.

    Returns:
        (is_valid, error_message)
    """
    try:
        tokens = tokenize(expr)
        if not tokens:
            return False, "Empty expression"
        parser = Parser(tokens)
        ast = parser.parse()
        _check_operators(ast)
        return True, ""
    except Exception as e:
        return False, str(e)


def _check_operators(node: ASTNode):
    """Recursively check that all function calls reference known operators."""
    if isinstance(node, FuncCallNode):
        if node.name not in OPERATOR_REGISTRY:
            raise ValueError(f"Unknown operator: {node.name}")
        for arg in node.args:
            _check_operators(arg)
    elif isinstance(node, BinOpNode):
        _check_operators(node.left)
        _check_operators(node.right)
    elif isinstance(node, UnaryMinusNode):
        _check_operators(node.operand)
