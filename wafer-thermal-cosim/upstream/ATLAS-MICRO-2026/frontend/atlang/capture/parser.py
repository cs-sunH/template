"""Public parser entry for atlang main functions."""

from __future__ import annotations

from typing import Any, Callable

from .frame import FrameBuilder
from .replay import AstReplayer
from .source import RuntimeScope, find_function_node, function_environment, load_function_source
from ..ir.kernel import AtlangKernel


def parse_main_function(func: Callable[..., Any], caller_locals: dict[str, Any] | None = None) -> AtlangKernel:
    function_source = load_function_source(func)
    function_node = find_function_node(function_source.tree, func.__name__)
    runtime_scope = RuntimeScope(function_environment(func, caller_locals or {}))
    builder = FrameBuilder(func, runtime_scope)
    builder.register_parameters()
    AstReplayer(builder, function_source).execute_block(function_node.body)
    return builder.to_kernel()


__all__ = ["parse_main_function"]
