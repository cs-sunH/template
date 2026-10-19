# TileLang and ATLang Simulator Programming Interface

## 1. Scope

TileLang and ATLang expose the same ATLAS simulator programming model. This
document describes only the common operator-programming surface required by
ATLAS; TileLang parameters used exclusively for GPU lowering or runtime code
generation are intentionally excluded.

Examples are available in `frontend/ops/tilelang/` and
`frontend/ops/atlang/`. The API prefix is `T` for
`tilelang.language` and `A` for `frontend.atlang.language`.

## 2. Entry Points

TileLang wraps the outer Python function with `@tilelang.jit(simulator=True)`
and defines the captured kernel with `@T.prim_func`:

```python
@tilelang.jit(simulator=True)
def operator(system_config, intermediate_result_dir=""):
    @T.prim_func
    def kernel(input_tensor: T.Tensor, output_tensor: T.Tensor):
        ...

    return kernel
```

ATLang is simulator-only and uses `@A.main` on the captured function:

```python
def operator(system_config, intermediate_result_dir=""):
    @A.main
    def kernel(input_tensor: A.Tensor, output_tensor: A.Tensor):
        ...

    return kernel
```

Cross-operator tensors must appear in the captured function's parameter list.
Each parameter must receive exactly one body-side `Tensor` layout declaration.
Specialized `gemm` and `decode-attention` extraction also reads the outer
function's `operator_dict` and `dtype`. General SPMD/MPMD programs describe
their complete work in the DSL and do not require `operator_dict`.

## 3. Execution Structure

### `CoreArray(shape, **kwargs)`

`CoreArray` is the top-level hardware execution context. `shape` is a non-empty
tuple of logical core-array dimensions; core IDs use row-major order.

The ATLAS extraction path consumes these keyword arguments:

| Parameter | Purpose |
| --- | --- |
| `system_config` | Parsed `CloudSystemConfig` or `EdgeSystemConfig`; required. |
| `dram_row_size` | DRAM row size in bytes used for placement and task generation. |
| `flit_size` | NoC flit size in bytes; required by cloud communication. |
| `min_tM`, `min_tK`, `min_tN` | Minimum/fixed GEMM tile factors. |
| `min_tS` | Decode-attention context tile factor. |
| `inter_chip_communication_list` | Model-level collectives finalized outside the chip simulator. |
| `num_layers` | Transformer-layer multiplier; defaults to `1`. |
| `num_workers` | Worker count for tiling exploration and validation. |
| `intermediate_result_dir` | Destination for generated YAML, logs, and intermediate results. |
| `gemm_tiling_cache_dir` | Reusable GEMM tiling-cache directory. |
| `general_autotune_min_tile_size` | Lower bound for general-kernel autotune candidates. |
| `general_autotune_max_candidates_per_tunable` | Search cap for each tunable tile dimension. |

The current edge softmax path additionally uses
`edge_softmax_attention_operator_name`, `edge_softmax_batch_size`,
`edge_softmax_context_length`, `edge_softmax_kv_group_num`, and
`edge_softmax_kv_head_num`. Custom edge kernels that emit the softmax
placeholder must provide all five fields.

ATLang derives element byte size from tensor dtype. Do not add a separate
`element_size` argument to ATLang source kernels.

### `SPMD(name, type, **kwargs)`

`SPMD` defines one operator executed with a common program across the core
array.

| `type` | Required metadata | Bound values |
| --- | --- | --- |
| `gemm` | `gemm_shape=(M,K,N)`, `gemm_b`, `core_dim_mapping` | `(core_M, core_K, core_N)` |
| `decode-attention` | `attention_shape`, `context_slot_mapping` | `(core_R, core_KV, core_N, core_S, core_H)` |
| `general` | No operator-specific metadata | None |

`core_dim_mapping` is ordered `(M, K, N)`. Each entry is `None`, one core-array
axis, or a tuple of axes. Splitting uses ceiling division over the product of
the selected axis extents.

`attention_shape` is
`(request_count, kv_head_count, q_heads_per_kv_head,
max_context_length_per_core, head_dim)`.

### `MPMD(name, type, **kwargs)`

`MPMD` defines operator regions in which different static core groups can run
different kernels. It returns symbolic `CORE_ID`, which may be used in address
and communication expressions. Current types are `communication` and
`general`.

General MPMD kernels must provide a non-empty, non-overlapping `core_list`.
Every listed ID must satisfy `0 <= core_id < CoreArray.core_num`; omitted cores
represent empty tasks.

### `Kernel(*blocks, autotune=None, core_list=None)`

`blocks` are static expressions, normally `ceildiv(problem_size, tile_size)`.
The context returns one symbolic block variable per expression. The extraction
path evaluates the expression and uses the divisor as the fixed tile size or as
the initial autotune description.

- `autotune=True` explores legal tile factors.
- `autotune=False` keeps the source tiling fixed.
- `core_list` selects the participating cores of an MPMD kernel.
- General SPMD kernels omit `core_list`; general MPMD kernels require it.

## 4. Tensors and Control Flow

### `Tensor(shape, strides, dtype)`

`shape` and `strides` define the logical storage layout consumed by placement
generation. Use explicit strides for simulator programs. `dtype` determines
element byte size and operation typing; ATLAS examples use `float16`.

### `alloc(shape, dtype)`

Allocates a kernel-local scratch buffer. These buffers model local SRAM or
temporary compute storage and can be used by copy, GEMM, vector, reduction,
send, and receive operations.

### `Serial(start, stop=None, step=None)`

Defines simulator-visible sequential iteration. Nested `Serial` loops multiply
execution counts and sibling loops retain source order. Ordinary host Python
loops should only construct metadata; use `Serial` for modeled work.

### `ceildiv(lhs, rhs)`

Builds a symbolic ceiling-division expression. It is commonly used for kernel
grid and tile calculations.

## 5. Modeled Operations

| Interface | Simulator meaning |
| --- | --- |
| `copy(src, dst)` | Tensor/local-buffer data movement. Tensor-to-local models a DRAM read; local-to-tensor models a DRAM write. |
| `fill(buffer, value)`, `clear(buffer)` | Local initialization. |
| `gemm(A, B, C, transpose_A=False, transpose_B=False)` | Matrix computation inferred from buffer regions. |
| `add`, `sub`, `mul`, `div`, and other vector helpers | Element-wise vector computation when an output buffer is supplied. |
| `reduce_*`, `cumsum` | Reduction/vector operations with `dim` and output buffer metadata. |
| `send(src_core, dst_core, buffer)` | NoC transfer issued by the source core. |
| `recv(src_core, dst_core, buffer)` | Matching NoC receive into the destination buffer. |

This table intentionally summarizes the commonly used modeled operations. The
complete public ATLang interface is listed in
`frontend/atlang/language/__init__.py` under `__all__`. Implementations of the
omitted operations are organized by category in
`frontend/atlang/language/memory.py`, `gemm.py`, `vector.py`, `reduction.py`,
`communication.py`, and `math.py`; structural interfaces are defined in
`objects.py` and `control.py` in the same directory.

Predicated Python `if` statements inside captured kernels are retained as
conditions on modeled actions. Communication schedules must issue matching
send/receive pairs with the same payload shape.

## 6. Minimal General SPMD Example

The example below uses ATLang spelling. Replace `A` with `T`, use
`@T.prim_func`, and add the TileLang outer JIT wrapper for the TileLang form.

```python
@A.main
def kernel(input_tensor: A.Tensor, weight: A.Tensor, output: A.Tensor):
    with A.CoreArray(
        shape=(4, 4),
        system_config=cloud_config,
        dram_row_size=128 * 1024,
        flit_size=cloud_config.chip_config.noc_config.flit_size,
        inter_chip_communication_list=[],
        num_workers=num_workers,
        intermediate_result_dir=output_dir,
        gemm_tiling_cache_dir=cache_dir,
    ):
        input_tensor = A.Tensor(shape=(16, 1024), strides=(1024, 1), dtype=A.float16)
        weight = A.Tensor(shape=(1024, 1024), strides=(1, 1024), dtype=A.float16)
        output = A.Tensor(shape=(16, 1024), strides=(1024, 1), dtype=A.float16)

        with A.SPMD(name="custom_gemm", type="general"):
            with A.Kernel(
                A.ceildiv(16, 4),
                A.ceildiv(1024, 32),
                A.ceildiv(1024, 32),
                autotune=False,
            ) as (bM, bK, bN):
                tM = A.ceildiv(16, bM)
                tK = A.ceildiv(1024, bK)
                tN = A.ceildiv(1024, bN)
                input_tile = A.alloc((tM, tK), A.float16)
                weight_tile = A.alloc((tK, tN), A.float16)
                partial = A.alloc((tM, tN), A.float16)
                accumulator = A.alloc((tM, tN), A.float16)

                for m in A.Serial(bM):
                    for n in A.Serial(bN):
                        A.clear(accumulator)
                        for k in A.Serial(bK):
                            A.copy(input_tensor[m * tM, k * tK], input_tile)
                            A.copy(weight[k * tK, n * tN], weight_tile)
                            A.gemm(input_tile, weight_tile, partial)
                            A.add(accumulator, partial, accumulator)
                        A.copy(accumulator, output[m * tM, n * tN])
```

For full cloud, edge, SPMD, and MPMD examples, use the implementations under
`frontend/ops/tilelang/` and `frontend/ops/atlang/` as the source of truth.
