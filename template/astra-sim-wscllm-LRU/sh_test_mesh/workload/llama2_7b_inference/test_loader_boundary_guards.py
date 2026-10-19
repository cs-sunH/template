#!/usr/bin/env python3
"""test_loader_boundary_guards.py -- 2026-10-01 深挖批次 loader 加固的
钉死测试（generate_trace.load_request_queue 五个新门 + 严格数字词法）。

被钉死的门（当批全部为行为变更，此前无任何用例喂这些形状）：
  1. _require_ascii_digits：拒 '+'/' '/'_'/Unicode 数字（Python int() 的
     宽松面），config 侧 parse_int/parse_nonnegative_int/parse_rank_id 同门；
  2. BOM 拒绝：utf-8 open + 首列名 BOM 前缀即拒（C++ 按原始字节比对
     表头，BOM 文件在 C++ 启动期才炸——loader 侧近端拒绝）；
  3. 表头逐列 strip（与 C++ trim_header_name 语义等价）：列名带空格可
     过、列序错即拒；
  4. 空 request_id 拒绝（不再合成 f"request_{len}"）；
  5. '#' 前缀行拒绝（本 loader 是三端口径中最紧的一端）。
"""

import sys
import tempfile
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
SH_TEST_DIR = MODULE_DIR.parents[1]
for _p in (str(MODULE_DIR), str(SH_TEST_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_trace import (  # noqa: E402
    parse_int,
    parse_nonnegative_int,
    parse_rank_id,
    load_request_queue,
)

HEADER = ("session_id,turn_index,request_id,prefill_length,decode_length,"
          "session_arrival_time_ns,inter_request_interval_ns,description")
ROW_T0 = "session_0,0,r0,100,50,1000,,d"
ROW_T1 = "session_0,1,r1,100,50,,500000,d"


def _queue_file(text: str) -> Path:
    # mkdtemp 而非 TemporaryDirectory：后者的 finalizer 在 _queue_file
    # 返回、对象出作用域时立即清目录（pytest 下先于 load 触发）。
    directory = tempfile.mkdtemp(prefix="loader_guard_")
    path = Path(directory) / "q_request_queue_recompute.csv"
    path.write_text(text, encoding="utf-8", newline="")
    import atexit
    import shutil
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    return path


def test_ascii_digits_lexicon():
    for bad in ("+1", " 1", "1_0", "١٢٣", "", "-1", "1.0", "1e3"):
        for fn in (parse_int, parse_nonnegative_int):
            try:
                fn(bad, "k")
            except ValueError:
                pass
            else:
                raise AssertionError(f"{fn.__name__} accepted {bad!r}")
    try:
        parse_rank_id("+1")
    except ValueError:
        pass
    else:
        raise AssertionError("parse_rank_id accepted '+1'")
    assert parse_int("2", "k") == 2
    assert parse_nonnegative_int("0", "k") == 0
    assert parse_rank_id("0") == 0


def test_bom_rejected():
    path = _queue_file("﻿" + HEADER + "\n" + ROW_T0 + "\n")
    try:
        load_request_queue(path)
    except ValueError as exc:
        assert "BOM" in str(exc)
    else:
        raise AssertionError("BOM header was not rejected")


def test_header_trim_and_order():
    # 列名带空格：与 C++ trim_header_name 同语义，可过。
    spaced = HEADER.replace(",", " , ")
    path = _queue_file(spaced + "\n" + ROW_T0 + "\n")
    specs = load_request_queue(path)
    assert len(specs) == 1 and specs[0].request_id == "r0"
    # 列序错：拒。
    wrong = HEADER.replace("turn_index,request_id", "request_id,turn_index")
    path2 = _queue_file(wrong + "\n" + ROW_T0 + "\n")
    try:
        load_request_queue(path2)
    except ValueError:
        pass
    else:
        raise AssertionError("misordered header was accepted")


def test_empty_request_id_rejected():
    path = _queue_file(HEADER + "\n" + "session_0,0,,100,50,1000,,d\n")
    try:
        load_request_queue(path)
    except ValueError as exc:
        assert "request_id" in str(exc)
    else:
        raise AssertionError("empty request_id was not rejected")


def test_hash_comment_row_rejected():
    path = _queue_file(HEADER + "\n" + ROW_T0 + "\n"
                       + "session_0,1,#note,100,50,,500,d\n")
    try:
        load_request_queue(path)
    except ValueError as exc:
        assert "comment" in str(exc) or "#" in str(exc)
    else:
        raise AssertionError("'#'-prefixed request_id was not rejected")


def test_wellformed_two_turn_queue_loads():
    path = _queue_file(HEADER + "\n" + ROW_T0 + "\n" + ROW_T1 + "\n")
    specs = load_request_queue(path)
    assert [s.request_id for s in specs] == ["r0", "r1"]


def _run_all() -> None:
    test_ascii_digits_lexicon()
    test_bom_rejected()
    test_header_trim_and_order()
    test_empty_request_id_rejected()
    test_hash_comment_row_rejected()
    test_wellformed_two_turn_queue_loads()
    print("[loader-boundary-guards] all 6 cases PASS")


if __name__ == "__main__":
    _run_all()
