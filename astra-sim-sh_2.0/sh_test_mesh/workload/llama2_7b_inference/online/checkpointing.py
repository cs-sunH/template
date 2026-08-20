"""阶段 7 §10.3:检查点持久化机制(临时文件 + 原子替换 + 有界保留)。

总体方案 §5.6:检查点只保存恢复所需的窗口位置、活动账本、KV 元数据、
ingress sequence 和最近的幂等状态,采用临时文件加原子替换,只保留有界
数量的检查点。本模块是 sh_2.0 的落地件:

- `write_checkpoint(path, payload, keep)`:把 `payload`(可 JSON 序列化的
  dict)原子写入 `path`——先写 `<path>.tmp` 再 `os.replace`(读者永远看
  不到半截文件,崩溃留下的 `.tmp` 是垃圾文件,下次写入会覆盖它),并做
  有界保留:旧版轮转为 `<path>.1`、`<path>.2`……最多保留 `keep` 份
  (含当前版),超出删除。
- `read_checkpoint(path)`:读回 dict;文件不存在返回 None,损坏(半截/
  非法 JSON)抛 `CheckpointError`(fail-closed:恢复路径必须显式处理,
  不能静默当作空状态)。
- `cleanup_tmp(path)`:删除 `<path>.tmp` 垃圾(原子替换前崩溃的残留)。

写入与读取都是进程内同步操作(单写者);跨进程并发读不是本机制的目标
(检查点在进程退出/恢复边界使用,不是热路径同步通道)。幂等性由调用方
保证(恢复后重放最近幂等状态),本模块只保证文件级原子性与有界保留。
"""

import json
import os


class CheckpointError(Exception):
    """检查点文件损坏或不可读。"""


def _tmp_path(path: str) -> str:
    return path + ".tmp"


def write_checkpoint(path: str, payload: dict, keep: int = 3) -> str:
    """原子写检查点并做有界轮转保留,返回最终路径。

    - 先写 `<path>.tmp`,fsync 后 `os.replace` 到 `path`(原子替换,读者
      永远不会读到半截内容)。
    - 旧版轮转:`path` -> `path.1` -> `path.2` …… 最多保留 `keep` 份
      (含当前版),超出删除最旧。
    - `keep` 必须是 >= 1 的整数(防御:非法值回落为 3)。
    """
    if keep < 1:
        keep = 3
    tmp = _tmp_path(path)
    with open(tmp, "w", encoding="utf-8") as target:
        json.dump(payload, target, sort_keys=True)
        target.write("\n")
        target.flush()
        os.fsync(target.fileno())
    # 有界保留:先把旧版依次后移,再原子替换当前版。后移顺序从最旧开始,
    # 避免覆盖尚未移动的版本。keep=1 不轮转,直接删除旧版。
    if keep == 1:
        if os.path.exists(path):
            os.remove(path)
        os.replace(tmp, path)
        return path
    for i in range(keep - 1, 0, -1):
        older = "{}.{}".format(path, i)
        if i == keep - 1:
            # 最旧位置:先删除(它将被上一级的后移覆盖)。
            if os.path.exists(older):
                os.remove(older)
        else:
            newer = "{}.{}".format(path, i + 1)
            if os.path.exists(older):
                os.replace(older, newer)
    if os.path.exists(path):
        os.replace(path, path + ".1")
    os.replace(tmp, path)
    return path


def read_checkpoint(path: str):
    """读回 `write_checkpoint` 写出的 dict;文件不存在返回 None。

    损坏(读失败/非法 JSON)抛 `CheckpointError`——恢复路径必须显式
    处理,不允许静默把损坏状态当作空状态继续。
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, ValueError) as exc:
        raise CheckpointError("bad checkpoint {}: {}".format(path, exc)) from exc
    return payload


def cleanup_tmp(path: str) -> None:
    """删除 `<path>.tmp` 垃圾(原子替换前崩溃的残留)。"""
    tmp = _tmp_path(path)
    try:
        os.remove(tmp)
    except OSError:
        pass
