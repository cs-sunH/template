"""阶段 7 §10.3:checkpointing 机制单测(临时文件+原子替换+有界保留)。"""

import json
import os

import pytest

from online.checkpointing import (
    CheckpointError,
    cleanup_tmp,
    read_checkpoint,
    write_checkpoint,
)


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_write_read_roundtrip(tmp_path):
    path = str(tmp_path / "ckpt.json")
    write_checkpoint(path, {"seq": 42, "window": [1, 2, 3]}, keep=3)
    assert read_checkpoint(path) == {"seq": 42, "window": [1, 2, 3]}
    # 无 .tmp 残留
    assert not os.path.exists(path + ".tmp")


def test_bounded_retention_keeps_latest_keep_versions(tmp_path):
    path = str(tmp_path / "ckpt.json")
    for i in range(1, 6):
        write_checkpoint(path, {"version": i}, keep=3)
    # 保留 path(5)、path.1(4)、path.2(3);更旧(1、2)已删除
    assert read_checkpoint(path) == {"version": 5}
    assert read_checkpoint(path + ".1") == {"version": 4}
    assert read_checkpoint(path + ".2") == {"version": 3}
    assert not os.path.exists(path + ".3")
    # 轮转后最旧版本不再存在
    assert read_checkpoint(path + ".3") is None


def test_atomic_replace_never_exposes_tmp_as_checkpoint(tmp_path):
    path = str(tmp_path / "ckpt.json")
    write_checkpoint(path, {"version": 1}, keep=3)
    # 模拟"原子替换前崩溃"残留:一个半截 .tmp 文件。
    _write(path + ".tmp", '{"version": 9, "truncat')
    # 下一次写入覆盖残留,不影响正式检查点。
    write_checkpoint(path, {"version": 2}, keep=3)
    assert read_checkpoint(path) == {"version": 2}
    assert not os.path.exists(path + ".tmp")


def test_cleanup_tmp_removes_stale_tmp(tmp_path):
    path = str(tmp_path / "ckpt.json")
    _write(path + ".tmp", "junk")
    cleanup_tmp(path)
    assert not os.path.exists(path + ".tmp")
    # 无残留时静默
    cleanup_tmp(path)


def test_read_missing_returns_none(tmp_path):
    assert read_checkpoint(str(tmp_path / "nope.json")) is None


def test_read_corrupt_raises(tmp_path):
    path = str(tmp_path / "ckpt.json")
    _write(path, "not json {{{")
    with pytest.raises(CheckpointError):
        read_checkpoint(path)


def test_keep_lower_bound_never_errors(tmp_path):
    path = str(tmp_path / "ckpt.json")
    write_checkpoint(path, {"v": 1}, keep=0)  # 非法 keep 回落为 3
    assert read_checkpoint(path) == {"v": 1}


def test_single_version_rotation(tmp_path):
    path = str(tmp_path / "ckpt.json")
    write_checkpoint(path, {"v": 1}, keep=1)
    write_checkpoint(path, {"v": 2}, keep=1)
    assert read_checkpoint(path) == {"v": 2}
    # keep=1:旧版直接删除,不留 path.1
    assert not os.path.exists(path + ".1")
