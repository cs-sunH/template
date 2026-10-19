"""NoC task helpers shared by atlang YAML materializers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def assign_general_execution_noc_flit_ids(
    execution_by_core: list[list[dict[str, Any]]],
    next_flit_by_src: dict[int, int] | None = None,
) -> None:
    if next_flit_by_src is None:
        next_flit_by_src = defaultdict(int)
    iteration_count = max((len(execution) for execution in execution_by_core), default=0)
    for iteration_index in range(iteration_count):
        iteration_tasks_list = [
            execution[iteration_index]
            for execution in execution_by_core
            if iteration_index < len(execution)
        ]
        pending_recv: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for iteration_tasks in iteration_tasks_list:
            for recv_task in iteration_tasks["noc_rx"]:
                key = (int(recv_task["src"]), int(recv_task["dst"]), int(recv_task["flit_num"]))
                pending_recv[key].append(recv_task)

        for iteration_tasks in iteration_tasks_list:
            for send_task in iteration_tasks["noc_tx"]:
                key = (int(send_task["src"]), int(send_task["dst"]), int(send_task["flit_num"]))
                recv_candidates = pending_recv.get(key)
                if not recv_candidates:
                    raise ValueError(f"NoC send {key} at iteration {iteration_index} has no matching recv.")
                recv_task = recv_candidates.pop(0)
                init_flit_id = int(next_flit_by_src.get(key[0], 0))
                send_task["init_flit_id"] = init_flit_id
                recv_task["init_flit_id"] = init_flit_id
                next_flit_by_src[key[0]] = init_flit_id + key[2]

        leftover_recv = [key for key, recv_tasks in pending_recv.items() if recv_tasks]
        if leftover_recv:
            raise ValueError(f"NoC recv tasks at iteration {iteration_index} have no matching sends: {leftover_recv}.")


def assign_task_description_noc_flit_ids(task_description_list: list[Any]) -> None:
    next_flit_by_src: dict[int, int] = defaultdict(int)
    for task_description in task_description_list:
        task_kind = task_description[0]
        if task_kind == "communication":
            _assign_communication_payload_noc_flit_ids(task_description[1], next_flit_by_src)
        elif task_kind == "general":
            task_payload = task_description[1]
            if "execution" in task_payload:
                assign_general_execution_noc_flit_ids([task_payload["execution"]], next_flit_by_src)
            else:
                assign_general_execution_noc_flit_ids(task_payload["per_core_execution"], next_flit_by_src)
        elif task_kind in ("computation", "gemm"):
            pass
        else:
            raise ValueError(f"Unsupported task kind {task_kind!r} while assigning NoC flit ids.")


def _assign_communication_payload_noc_flit_ids(
    communication_description_list: list[tuple[dict[str, Any], list[dict[str, Any]]]],
    next_flit_by_src: dict[int, int],
) -> None:
    for _, per_core_description_list in communication_description_list:
        if not per_core_description_list:
            continue
        iteration_count = max(
            len(per_core_description["communication"]["on_chip"])
            for per_core_description in per_core_description_list
        )
        for iteration_index in range(iteration_count):
            pending_recv: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
            for per_core_description in per_core_description_list:
                on_chip = per_core_description["communication"]["on_chip"]
                if iteration_index >= len(on_chip):
                    continue
                for recv_task in on_chip[iteration_index]["rx"]["noc"]:
                    pending_recv[(int(recv_task["src"]), int(recv_task["dst"]))].append(recv_task)

            for per_core_description in per_core_description_list:
                on_chip = per_core_description["communication"]["on_chip"]
                if iteration_index >= len(on_chip):
                    continue
                for send_task in on_chip[iteration_index]["tx"]["noc"]:
                    key = (int(send_task["src"]), int(send_task["dst"]))
                    recv_candidates = pending_recv.get(key)
                    if not recv_candidates:
                        raise ValueError(
                            f"Missing matching noc_recv for communication pair {key} at final bundle iteration "
                            f"{iteration_index}."
                        )
                    recv_task = recv_candidates.pop(0)
                    if int(send_task["flit_num"]) != int(recv_task["flit_num"]):
                        raise ValueError(
                            f"Communication pair {key} at final bundle iteration {iteration_index} has mismatched "
                            f"flit counts: {send_task['flit_num']} vs {recv_task['flit_num']}."
                        )
                    init_flit_id = int(next_flit_by_src.get(key[0], 0))
                    send_task["init_flit_id"] = init_flit_id
                    recv_task["init_flit_id"] = init_flit_id
                    next_flit_by_src[key[0]] = init_flit_id + int(send_task["flit_num"])

            leftover_recv = [key for key, recv_tasks in pending_recv.items() if recv_tasks]
            if leftover_recv:
                raise ValueError(
                    f"Unpaired noc_recv tasks remain at final bundle iteration {iteration_index}: {leftover_recv}."
                )


__all__ = ["assign_general_execution_noc_flit_ids", "assign_task_description_noc_flit_ids"]
