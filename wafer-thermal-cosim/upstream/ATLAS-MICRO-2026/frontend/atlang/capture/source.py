"""Source loading and runtime scope for atlang AST replay."""

from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
from typing import Any, Callable


# Source and environment handling

class FunctionSource:
    def __init__(self, start_line: int, tree: ast.Module) -> None:
        self.start_line = start_line
        self.tree = tree

    def line_for(self, node: ast.AST) -> int | None:
        line_number = getattr(node, "lineno", None)
        if line_number is None:
            return None
        return self.start_line + line_number - 1


class RuntimeScope:
    def __init__(self, base_environment: dict[str, Any]) -> None:
        self.scopes = [base_environment]

    def get(self, name: str) -> Any:
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]
        if hasattr(builtins, name):
            return getattr(builtins, name)
        raise NameError(f"Name {name!r} is not available during atlang AST replay.")

    def set(self, name: str, value: Any) -> None:
        self.scopes[-1][name] = value

    def push(self, values: dict[str, Any] | None = None) -> None:
        self.scopes.append({} if values is None else dict(values))

    def pop(self) -> None:
        if len(self.scopes) == 1:
            raise RuntimeError("Cannot pop the base atlang replay scope.")
        self.scopes.pop()


def load_function_source(func: Callable[..., Any]) -> FunctionSource:
    source_lines, start_line = inspect.getsourcelines(func)
    source = textwrap.dedent("".join(source_lines))
    return FunctionSource(start_line=start_line, tree=ast.parse(source))


def find_function_node(tree: ast.Module, function_name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return node
    raise ValueError(f"Could not find function definition for {function_name!r}.")


def function_environment(func: Callable[..., Any], caller_locals: dict[str, Any]) -> dict[str, Any]:
    environment = dict(func.__globals__)
    environment.update(caller_locals)
    closure = inspect.getclosurevars(func)
    environment.update(closure.globals)
    environment.update(closure.nonlocals)
    environment.update(closure.builtins)
    return environment


__all__ = ["FunctionSource", "RuntimeScope", "find_function_node", "function_environment", "load_function_source"]
