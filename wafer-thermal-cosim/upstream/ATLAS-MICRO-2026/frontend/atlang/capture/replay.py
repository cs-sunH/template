"""Python AST replay for atlang source kernels."""

from __future__ import annotations

import ast
import builtins
import inspect
from typing import Any

from .frame import (
    FrameBuilder,
    PythonRange,
    _MISSING,
    _compare_values,
    _make_python_range,
    _single_target_name,
    _target_names,
)
from .source import FunctionSource, find_function_node, load_function_source
from ..ir.expr import deferred_getitem, is_expr, make_binary_expr, make_call_expr, make_unary_expr
from ..language import Buffer, CoreArray, Kernel, MPMD, ProcessStyleCall, SPMD, SerialRange, Tensor


# AST replay

class AstReplayer:
    def __init__(self, builder: FrameBuilder, function_source: FunctionSource) -> None:
        self.builder = builder
        self.function_source = function_source
        self.runtime_scope = builder.runtime_scope

    def execute_block(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.execute_statement(statement)

    def execute_statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.With):
            self._execute_with(statement)
        elif isinstance(statement, ast.For):
            self._execute_for(statement)
        elif isinstance(statement, ast.If):
            self._execute_if(statement)
        elif isinstance(statement, ast.Assign):
            self._execute_assign(statement)
        elif isinstance(statement, ast.AnnAssign):
            self._execute_ann_assign(statement)
        elif isinstance(statement, ast.Expr):
            self._execute_expr_statement(statement)
        elif isinstance(statement, ast.Return):
            raise ReturnValue(None if statement.value is None else self.eval_expr(statement.value))
        elif isinstance(statement, ast.Pass):
            return
        else:
            raise NotImplementedError(f"Unsupported atlang statement {type(statement).__name__}.")

    def eval_expr(self, expression: ast.AST) -> Any:
        if isinstance(expression, ast.Constant):
            return expression.value
        if isinstance(expression, ast.Name):
            return self.runtime_scope.get(expression.id)
        if isinstance(expression, ast.Attribute):
            return getattr(self.eval_expr(expression.value), expression.attr)
        if isinstance(expression, ast.Tuple):
            return tuple(self.eval_expr(element) for element in expression.elts)
        if isinstance(expression, ast.List):
            return [self.eval_expr(element) for element in expression.elts]
        if isinstance(expression, ast.ListComp):
            return self._eval_list_comp(expression)
        if isinstance(expression, ast.Dict):
            return {self.eval_expr(key): self.eval_expr(value) for key, value in zip(expression.keys, expression.values)}
        if isinstance(expression, ast.Subscript):
            container = self.eval_expr(expression.value)
            key = self._eval_slice(expression.slice)
            return deferred_getitem(container, key)
        if isinstance(expression, ast.BinOp):
            return self._eval_bin_op(expression)
        if isinstance(expression, ast.UnaryOp):
            return self._eval_unary_op(expression)
        if isinstance(expression, ast.BoolOp):
            return self._eval_bool_op(expression)
        if isinstance(expression, ast.Compare):
            return self._eval_compare(expression)
        if isinstance(expression, ast.Call):
            return self._eval_call(expression)
        if isinstance(expression, ast.IfExp):
            condition = self.eval_expr(expression.test)
            if is_expr(condition):
                return make_call_expr("if_then_else", condition, self.eval_expr(expression.body), self.eval_expr(expression.orelse))
            return self.eval_expr(expression.body if builtins.bool(condition) else expression.orelse)
        raise NotImplementedError(f"Unsupported atlang expression {type(expression).__name__}.")

    def _execute_with(self, statement: ast.With) -> None:
        if len(statement.items) != 1:
            self._execute_nested_with_items(statement.items, statement.body)
            return
        item = statement.items[0]
        context_value = self.eval_expr(item.context_expr)
        if isinstance(context_value, CoreArray):
            self.builder.enter_core_array(context_value, lambda: self.execute_block(statement.body))
        elif isinstance(context_value, SPMD):
            self.builder.enter_spmd_region(
                context_value,
                item.optional_vars,
                self.bind_target,
                lambda: self.execute_block(statement.body),
            )
        elif isinstance(context_value, MPMD):
            self.builder.enter_mpmd_region(
                context_value,
                item.optional_vars,
                self.bind_target,
                lambda: self.execute_block(statement.body),
            )
        elif isinstance(context_value, Kernel):
            self.builder.enter_kernel_region(
                context_value,
                item.optional_vars,
                self.bind_target,
                lambda: self.execute_block(statement.body),
            )
        else:
            raise TypeError(f"Unsupported atlang context manager {context_value!r}.")

    def _execute_nested_with_items(self, items: list[ast.withitem], body: list[ast.stmt]) -> None:
        nested = ast.With(items=items[1:], body=body, type_comment=None)
        ast.copy_location(nested, items[0].context_expr)
        outer = ast.With(items=[items[0]], body=[nested], type_comment=None)
        ast.copy_location(outer, items[0].context_expr)
        self._execute_with(outer)

    def _execute_for(self, statement: ast.For) -> None:
        iterable = self.eval_expr(statement.iter)
        target_name = _single_target_name(statement.target)
        if isinstance(iterable, SerialRange):
            self.builder.enter_serial_loop(target_name, iterable, lambda: self.execute_block(statement.body))
            return
        if isinstance(iterable, PythonRange):
            self.builder.enter_python_range_loop(target_name, iterable, lambda: self.execute_block(statement.body))
            return
        if isinstance(iterable, builtins.range):
            for value in iterable:
                self.bind_target(statement.target, value)
                self.execute_block(statement.body)
            return
        raise TypeError("atlang for-loops must iterate over A.Serial(...) or ordinary range(...).")

    def _execute_if(self, statement: ast.If) -> None:
        condition = self.eval_expr(statement.test)
        if not is_expr(condition):
            self.execute_block(statement.body if builtins.bool(condition) else statement.orelse)
            return
        self.builder.push_condition(condition)
        try:
            self.execute_block(statement.body)
        finally:
            self.builder.pop_condition()
        self.builder.push_condition(make_unary_expr("not", condition))
        try:
            self.execute_block(statement.orelse)
        finally:
            self.builder.pop_condition()

    def _execute_assign(self, statement: ast.Assign) -> None:
        value = self.eval_expr(statement.value)
        for target in statement.targets:
            self.bind_target(target, value, source_line=self.function_source.line_for(statement))

    def _execute_ann_assign(self, statement: ast.AnnAssign) -> None:
        if statement.value is None:
            return
        value = self.eval_expr(statement.value)
        self.bind_target(statement.target, value, source_line=self.function_source.line_for(statement))

    def _execute_expr_statement(self, statement: ast.Expr) -> None:
        if isinstance(statement.value, ast.Call) and self._try_inline_helper(statement.value):
            return
        value = self.eval_expr(statement.value)
        if isinstance(value, ProcessStyleCall):
            self.builder.record_process_call(value, self.function_source.line_for(statement))

    def bind_target(self, target: ast.AST, value: Any, source_line: int | None = None) -> None:
        if isinstance(target, ast.Name):
            if isinstance(value, Tensor):
                value = Tensor(
                    shape=value.shape,
                    dtype=value.dtype,
                    data=value.data,
                    strides=value.strides,
                    name=target.id,
                )
                self.runtime_scope.set(target.id, value)
                self.builder.register_tensor_assignment(target.id, value, source_line)
                return
            if isinstance(value, Buffer):
                value = Buffer(shape=value.shape, dtype=value.dtype, name=target.id, attrs=dict(value.attrs))
                self.runtime_scope.set(target.id, value)
                return
            self.runtime_scope.set(target.id, value)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            values = tuple(value)
            if len(values) != len(target.elts):
                raise ValueError("Destructuring assignment arity does not match captured values.")
            for child_target, child_value in zip(target.elts, values):
                self.bind_target(child_target, child_value, source_line=source_line)
            return
        raise NotImplementedError(f"Unsupported assignment target {type(target).__name__}.")

    def _eval_slice(self, slice_node: ast.AST) -> Any:
        if isinstance(slice_node, ast.Slice):
            return slice(
                None if slice_node.lower is None else self.eval_expr(slice_node.lower),
                None if slice_node.upper is None else self.eval_expr(slice_node.upper),
                None if slice_node.step is None else self.eval_expr(slice_node.step),
            )
        if isinstance(slice_node, ast.Tuple):
            return tuple(self._eval_slice(element) for element in slice_node.elts)
        return self.eval_expr(slice_node)

    def _eval_bin_op(self, expression: ast.BinOp) -> Any:
        lhs = self.eval_expr(expression.left)
        rhs = self.eval_expr(expression.right)
        if isinstance(expression.op, ast.Add):
            return lhs + rhs
        if isinstance(expression.op, ast.Sub):
            return lhs - rhs
        if isinstance(expression.op, ast.Mult):
            return lhs * rhs
        if isinstance(expression.op, ast.Div):
            return lhs / rhs
        if isinstance(expression.op, ast.FloorDiv):
            return lhs // rhs
        if isinstance(expression.op, ast.Mod):
            return lhs % rhs
        raise NotImplementedError(f"Unsupported binary operator {type(expression.op).__name__}.")

    def _eval_unary_op(self, expression: ast.UnaryOp) -> Any:
        value = self.eval_expr(expression.operand)
        if isinstance(expression.op, ast.USub):
            return -value
        if isinstance(expression.op, ast.UAdd):
            return +value
        if isinstance(expression.op, ast.Not):
            return make_unary_expr("not", value)
        raise NotImplementedError(f"Unsupported unary operator {type(expression.op).__name__}.")

    def _eval_bool_op(self, expression: ast.BoolOp) -> Any:
        values = [self.eval_expr(value) for value in expression.values]
        op_name = "and" if isinstance(expression.op, ast.And) else "or"
        current = values[0]
        for value in values[1:]:
            current = make_binary_expr(op_name, current, value)
        return current

    def _eval_compare(self, expression: ast.Compare) -> Any:
        lhs = self.eval_expr(expression.left)
        comparisons = []
        for op, comparator in zip(expression.ops, expression.comparators):
            rhs = self.eval_expr(comparator)
            comparisons.append(_compare_values(op, lhs, rhs))
            lhs = rhs
        current = comparisons[0]
        for comparison in comparisons[1:]:
            current = make_binary_expr("and", current, comparison)
        return current

    def _eval_call(self, expression: ast.Call) -> Any:
        func = self.eval_expr(expression.func)
        args = [self.eval_expr(arg) for arg in expression.args]
        kwargs: dict[str, Any] = {}
        for keyword in expression.keywords:
            if keyword.arg is None:
                kwargs.update(self.eval_expr(keyword.value))
            else:
                kwargs[keyword.arg] = self.eval_expr(keyword.value)
        if func is builtins.len:
            if kwargs or len(args) != 1:
                raise TypeError("len() in atlang kernels expects exactly one positional argument.")
            if is_expr(args[0]):
                return make_call_expr("len", args[0])
            return builtins.len(args[0])
        if func is builtins.range:
            if kwargs:
                raise TypeError("range() in atlang kernels does not accept keyword arguments.")
            host_range = _make_python_range(args)
            if any(is_expr(value) for value in (host_range.start, host_range.stop, host_range.step)):
                return host_range
            return builtins.range(int(host_range.start), int(host_range.stop), int(host_range.step))
        return func(*args, **kwargs)

    def _eval_list_comp(self, expression: ast.ListComp) -> list[Any]:
        if any(generator.is_async for generator in expression.generators):
            raise NotImplementedError("Async comprehensions are not supported in atlang kernels.")
        target_names = [
            target_name
            for generator in expression.generators
            for target_name in _target_names(generator.target)
        ]
        saved_values = {
            target_name: self.runtime_scope.get(target_name) if self.builder._name_exists(target_name) else _MISSING
            for target_name in target_names
        }
        result: list[Any] = []

        def visit_generator(generator_index: int) -> None:
            if generator_index == len(expression.generators):
                result.append(self.eval_expr(expression.elt))
                return
            generator = expression.generators[generator_index]
            iterable = self.eval_expr(generator.iter)
            for value in iterable:
                self.bind_target(generator.target, value)
                should_visit = True
                for condition_expr in generator.ifs:
                    condition = self.eval_expr(condition_expr)
                    if is_expr(condition):
                        raise ValueError("List comprehension filters must resolve to static predicates.")
                    if not builtins.bool(condition):
                        should_visit = False
                        break
                if should_visit:
                    visit_generator(generator_index + 1)

        try:
            visit_generator(0)
        finally:
            for target_name, saved_value in saved_values.items():
                if saved_value is _MISSING:
                    self.runtime_scope.scopes[-1].pop(target_name, None)
                else:
                    self.runtime_scope.set(target_name, saved_value)
        return result

    def _try_inline_helper(self, expression: ast.Call) -> bool:
        func = self.eval_expr(expression.func)
        module_name = getattr(func, "__module__", "") or ""
        if not inspect.isfunction(func) or module_name.startswith("frontend.atlang"):
            return False
        args = [self.eval_expr(arg) for arg in expression.args]
        kwargs = {keyword.arg: self.eval_expr(keyword.value) for keyword in expression.keywords if keyword.arg is not None}
        signature = inspect.signature(func)
        bound_arguments = signature.bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        helper_source = load_function_source(func)
        helper_node = find_function_node(helper_source.tree, func.__name__)
        self.runtime_scope.push(dict(bound_arguments.arguments))
        try:
            AstReplayer(self.builder, helper_source).execute_block(helper_node.body)
        except ReturnValue:
            pass
        finally:
            self.runtime_scope.pop()
        return True


class ReturnValue(Exception):
    def __init__(self, value: Any) -> None:
        super().__init__("atlang helper returned")
        self.value = value

__all__ = ["AstReplayer", "ReturnValue"]
