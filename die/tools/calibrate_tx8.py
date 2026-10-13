#!/usr/bin/env python3
"""Audit the TX8 XLSX XML and reproduce the conditional tx8_reference_v1 model.

This tool deliberately treats the workbook's 13 group times as cached source
values with mixed evidence, not as 13 measurements. It uses only the Python
standard library and writes the calibration, sample ledger, report, and CSV
under the repository's die/ tree.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import posixpath
import zipfile
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET


NS = {
    "m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}
DEFAULT_WORKBOOK = "TX8_单die建模_精简版.xlsx"
TOLERANCE_US = 1e-6

# Stable, normalized keys are independent of workbook display labels and C++
# naming. In particular AV (workbook) == AV/PV (C++) == AV (math-node sheet).
GROUPS = [
    ("norm_1", "RMSNorm_1", "RMSNorm_1", "vector", ["rmsnorm_1"]),
    ("q_proj", "Q_proj", "Q_proj", "matrix", ["q_proj"]),
    ("k_proj", "K_proj", "K_proj", "matrix", ["k_proj"]),
    ("v_proj", "V_proj", "V_proj", "matrix", ["v_proj"]),
    ("rope_qk", "RoPE_QK", "RoPE_QK", "vector", ["rope_q", "rope_k"]),
    ("qk_matmul", "QKT", "QKT", "matrix", ["qkt"]),
    ("score_mul_softmax", "score_mul_softmax", "score_mul_softmax", "vector", ["score_scale", "softmax"]),
    ("av_pv", "AV", "AV/PV", "matrix", ["av"]),
    ("o_proj", "O_proj", "O_proj", "matrix", ["o_proj"]),
    ("residual_1", "res_add_1", "res_add_1", "vector", ["res_add_1"]),
    ("norm_2", "RMSNorm_2", "RMSNorm_2", "vector", ["rmsnorm_2"]),
    ("ffn_fused", "FFN", "FFN", "matrix", ["ffn_gate", "ffn_up", "silu", "ffn_gate_mul", "ffn_down"]),
    ("residual_2", "res_add_2", "res_add_2", "vector", ["res_add_2"]),
]

NODE_KEYS = {
    "RMSNorm_1": "rmsnorm_1", "Q_proj": "q_proj", "K_proj": "k_proj",
    "V_proj": "v_proj", "RoPE_Q": "rope_q", "RoPE_K": "rope_k",
    "QKT": "qkt", "score_scale": "score_scale", "softmax": "softmax",
    "AV": "av", "O_proj": "o_proj", "res_add_1": "res_add_1",
    "RMSNorm_2": "rmsnorm_2", "gate_proj": "ffn_gate",
    "up_proj": "ffn_up", "SiLU": "silu", "gate_mul": "ffn_gate_mul",
    "down_proj": "ffn_down", "res_add_2": "res_add_2",
}


def load_xlsx(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        rels = {e.attrib["Id"]: e.attrib["Target"] for e in rels_root}
        shared: list[str] = []
        if "xl/sharedStrings.xml" in names:
            ss_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for si in ss_root.findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.findall(".//m:t", NS)))

        sheets: dict[str, dict[str, dict[str, Any]]] = {}
        sheet_paths: dict[str, str] = {}
        for sheet in workbook.findall("m:sheets/m:sheet", NS):
            title = sheet.attrib["name"]
            target = rels[sheet.attrib[f"{{{NS['r']}}}id"]]
            sheet_path = posixpath.normpath(posixpath.join("xl", target.lstrip("/")))
            if target.startswith("/"):
                sheet_path = target.lstrip("/")
            sheet_paths[title] = sheet_path
            root = ET.fromstring(archive.read(sheet_path))
            cells: dict[str, dict[str, Any]] = {}
            for c in root.findall(".//m:sheetData/m:row/m:c", NS):
                ref = c.attrib["r"]
                f = c.find("m:f", NS)
                v = c.find("m:v", NS)
                inline = c.find("m:is", NS)
                raw_v = v.text if v is not None else None
                typ = c.attrib.get("t")
                value: Any = raw_v
                if typ == "s" and raw_v is not None:
                    value = shared[int(raw_v)]
                elif typ == "inlineStr" and inline is not None:
                    value = "".join(t.text or "" for t in inline.findall(".//m:t", NS))
                cells[ref] = {
                    "type": typ or "n",
                    "value": value,
                    "raw_v": raw_v,
                    "formula": f.text if f is not None else None,
                }
            sheets[title] = cells

    formula_cells = []
    for sheet_name, cells in sheets.items():
        for ref, cell in cells.items():
            if cell["formula"] is not None:
                formula_cells.append({
                    "sheet": sheet_name,
                    "cell": ref,
                    "formula": cell["formula"],
                    "cached_value_xml": cell["raw_v"],
                })
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "archive_entries": names,
        "core_properties_present": any(n.startswith("docProps/") for n in names),
        "sheet_paths": sheet_paths,
        "sheets": sheets,
        "formula_cells": formula_cells,
    }


def val(book: dict[str, Any], sheet: str, ref: str) -> Any:
    try:
        return book["sheets"][sheet][ref]["value"]
    except KeyError as exc:
        raise ValueError(f"Missing workbook cell {sheet}!{ref}") from exc


def num(book: dict[str, Any], sheet: str, ref: str) -> float:
    raw = val(book, sheet, ref)
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric {sheet}!{ref}, got {raw!r}") from exc


def raw_num(book: dict[str, Any], sheet: str, ref: str) -> str:
    raw = book["sheets"][sheet][ref]["raw_v"]
    if raw is None:
        raise ValueError(f"Missing numeric XML cache in {sheet}!{ref}")
    return raw


def text(book: dict[str, Any], sheet: str, ref: str) -> str:
    value = val(book, sheet, ref)
    return "" if value is None else str(value)


def fmt_us(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.9f}"


def derive(workbook_path: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], str, list[dict[str, Any]]]:
    book = load_xlsx(workbook_path)
    overview = "总览_大小算子数据"
    hw = "硬件配置"
    dims = "维度定义"
    nodes_sheet = "数学节点19行"
    timing = "计时组13行"
    hw_infer = "参数及推断依据"

    shape = {
        "B": int(num(book, dims, "B2")),
        "S_q": int(num(book, dims, "B3")),
        "S_k": int(num(book, dims, "B3")),
        "H": int(num(book, dims, "B4")),
        "F": int(num(book, dims, "B5")),
        "H_q": int(num(book, dims, "B6")),
        "H_kv": int(num(book, dims, "B7")),
        "head_dim": int(num(book, dims, "B8")),
        "G": int(num(book, dims, "B9")),
        "F_tile": int(num(book, dims, "B10")),
    }
    M = shape["B"] * shape["S_q"]
    J = shape["H_kv"] * shape["head_dim"]
    A = shape["B"] * shape["H_q"] * shape["S_q"] * shape["S_k"]
    H, F, D = shape["H"], shape["F"], shape["head_dim"]

    matrix_peak_raw = num(book, hw, "B4") * 1e12
    vector_peak_raw = num(book, hw, "B5") * 1e12
    dram_peak = num(book, hw, "B3") * 1e9
    matrix_bw = num(book, hw, "B7") * 1e9
    norm_source_us = num(book, timing, "B2")
    norm_bytes = 4 * M * H + 2 * H
    vector_bw = norm_bytes / (norm_source_us * 1e-6)
    eta_vector_memory = vector_bw / dram_peak
    ffn_total_us = num(book, timing, "B13")
    eta_compute = (6 * M * H * F / matrix_peak_raw + 16 * M * F / vector_peak_raw) / (ffn_total_us / 1e6)
    matrix_peak_effective = matrix_peak_raw * eta_compute
    vector_peak_effective = vector_peak_raw * eta_compute

    # Integer operation and byte accounting from design doc §9.4 and the
    # corresponding independent implementation in Tx8Reference.cc.
    A_q = shape["B"] * shape["H_q"] * shape["S_q"] * shape["S_k"]
    groups_raw = [
        (0, 6 * M * H, 4 * M * H + 2 * H),
        (2 * M * H * H, 0, 2 * (M * H + H * H + M * H)),
        (2 * M * H * J, 0, 2 * (M * H + H * J + M * J)),
        (2 * M * H * J, 0, 2 * (M * H + H * J + M * J)),
        (0, 3 * M * (H + J), 4 * M * (H + J) + 4 * shape["S_q"] * D),
        (2 * A_q * D, 0, 2 * shape["B"] * shape["H_q"] * shape["S_q"] * D + 2 * shape["B"] * shape["H_kv"] * shape["S_k"] * D + 4 * A_q),
        (0, 12 * A_q, 8 * A_q),
        (2 * A_q * D, A_q, 4 * A_q + 2 * shape["B"] * shape["H_kv"] * shape["S_k"] * D + 2 * shape["B"] * shape["H_q"] * shape["S_q"] * D),
        (2 * M * H * H, 0, 2 * (M * H + H * H + M * H)),
        (0, M * H, 6 * M * H),
        (0, 6 * M * H, 4 * M * H + 2 * H),
        (6 * M * H * F, 16 * M * F, 6 * H * F + 4 * M * H),
        (0, M * H, 6 * M * H),
    ]

    groups: list[dict[str, Any]] = []
    source_time_rows = []
    for idx, (key, source_id, cpp_name, mem_class, node_keys) in enumerate(GROUPS):
        row = idx + 2
        source_time = num(book, timing, f"B{row}")
        source_label = text(book, timing, f"A{row}")
        source_kind = "reported_anchor" if idx == 0 else (
            "reported_anchor_reused" if idx == 10 else (
                "conditional" if idx == 11 else "derived_engineering_estimate"
            )
        )
        fop, eop, dbytes = groups_raw[idx]
        bandwidth = matrix_bw if mem_class == "matrix" else vector_bw
        compute_s = fop / matrix_peak_effective + eop / vector_peak_effective
        memory_s = dbytes / bandwidth
        prediction_s = max(compute_s, memory_s)
        source_time_rows.append(source_time)
        groups.append({
            "source_key": key,
            "source_group_id": source_id,
            "workbook_label": source_label,
            "cpp_group_name": cpp_name,
            "node_keys": node_keys,
            "evidence_status": source_kind,
            "source_time_cell": f"{timing}!B{row}",
            "source_time_us_xml": raw_num(book, timing, f"B{row}"),
            "source_time_us": source_time,
            "Fop": int(fop),
            "Eop": int(eop),
            "D_bytes": int(dbytes),
            "memory_class": mem_class,
            "compute_us": compute_s * 1e6,
            "memory_us": memory_s * 1e6,
            "reference_prediction_us": prediction_s * 1e6,
            "delta_prediction_minus_source_us": prediction_s * 1e6 - source_time,
            "prediction_error_abs_us": abs(prediction_s * 1e6 - source_time),
            "prediction_rule": "max(Fop/peak + Eop/peak, D_bytes/BW_class); compute and memory start together",
        })

    source_formula_sum_us = sum(source_time_rows)
    source_block_us = num(book, timing, "B15")
    formula_cell = book["sheets"][timing]["B16"]
    formula_cache_us = num(book, timing, "B16")
    formula_sum_error_us = abs(source_formula_sum_us - formula_cache_us)
    formula_block_error_us = abs(source_block_us - formula_cache_us)

    attention_keys = {"q_proj", "k_proj", "v_proj", "rope_qk", "qk_matmul", "score_mul_softmax", "av_pv", "o_proj"}
    attention_us = sum(g["reference_prediction_us"] for g in groups if g["source_key"] in attention_keys)
    attention_source_sum_us = sum(g["source_time_us"] for g in groups if g["source_key"] in attention_keys)
    block_prediction_us = sum(g["reference_prediction_us"] for g in groups)
    group_by_id = {g["source_group_id"]: g for g in groups}
    group_by_key = {g["source_key"]: g for g in groups}

    p = matrix_peak_effective
    v = vector_peak_effective
    ffn_parts = [
        ("ffn_gate", (6 * M * H * F / 3) / p),
        ("ffn_up", (6 * M * H * F / 3) / p),
        ("silu", (15 * M * F) / v),
        ("ffn_gate_mul", (M * F) / v),
        ("ffn_down", (6 * M * H * F / 3) / p),
    ]
    ffn_components = [{"node_key": k, "component_budget_us": seconds * 1e6, "evidence_status": "derived_shared_budget_attribution"} for k, seconds in ffn_parts]
    ffn_component_sum = sum(item["component_budget_us"] for item in ffn_components)

    node_records: list[dict[str, Any]] = []
    seen_owner: set[str] = set()
    # workbook math node rows 2..20; preserve display text and shape strings.
    for row in range(2, 21):
        display_name = text(book, nodes_sheet, f"A{row}")
        source_group_id = text(book, nodes_sheet, f"B{row}")
        node_key = NODE_KEYS[display_name]
        group_id = {"AV": "AV", "FFN": "FFN"}.get(source_group_id, source_group_id)
        group = group_by_id[group_id]
        owner = group["source_key"] not in seen_owner
        seen_owner.add(group["source_key"])
        component = next((x["component_budget_us"] for x in ffn_components if x["node_key"] == node_key), None)
        node_records.append({
            "node_key": node_key,
            "workbook_math_node": display_name,
            "math_node_row": f"{nodes_sheet}!A{row}",
            "source_group_id": source_group_id,
            "source_key": group["source_key"],
            "dependency": text(book, nodes_sheet, f"C{row}"),
            "input_shape_source": text(book, nodes_sheet, f"D{row}"),
            "second_input_shape_source": text(book, nodes_sheet, f"E{row}"),
            "output_shape_source": text(book, nodes_sheet, f"F{row}"),
            "execution_note": text(book, nodes_sheet, f"G{row}"),
            "is_group_timing_owner": owner,
            "group_total_counted_once": owner,
            "source_group_time_us": group["source_time_us"] if owner else None,
            "reference_group_prediction_us": group["reference_prediction_us"] if owner else None,
            "reference_component_attribution_us": component,
            "time_status": group["evidence_status"] if owner else "shared_group_no_independent_node_delay",
        })

    cache_rows = []
    for row in range(2, 15):
        cache_rows.append({
            "cell": f"{timing}!B{row}",
            "group_id": text(book, timing, f"A{row}"),
            "cached_numeric_xml": raw_num(book, timing, f"B{row}"),
            "interpretation": next(g["evidence_status"] for g in groups if g["source_group_id"] == text(book, timing, f"A{row}")),
        })
    cache_ledger = {
        "group_time_cells": cache_rows,
        "hardcoded_total": {
            "cell": f"{timing}!B15", "cached_numeric_xml": raw_num(book, timing, "B15"),
            "value_us": source_block_us, "role": "hardcoded total; not independently measured",
        },
        "formula_and_cache": {
            "cell": f"{timing}!B16", "formula": formula_cell["formula"],
            "cached_numeric_xml": raw_num(book, timing, "B16"), "cached_value_us": formula_cache_us,
            "recomputed_sum_of_B2_B14_us": source_formula_sum_us,
            "recomputed_minus_cache_us": source_formula_sum_us - formula_cache_us,
            "cache_roundtrip_exact": formula_cell["raw_v"] == raw_num(book, timing, "B16"),
        },
    }

    cache_reconciled = formula_sum_error_us < TOLERANCE_US and formula_block_error_us < TOLERANCE_US
    reference_reproduced = all(g["prediction_error_abs_us"] < TOLERANCE_US for g in groups) and abs(attention_us - attention_source_sum_us) < TOLERANCE_US and abs(block_prediction_us - formula_cache_us) < TOLERANCE_US and abs(ffn_component_sum - ffn_total_us) < TOLERANCE_US

    source_book = {
        "filename": workbook_path.name,
        "sha256": book["sha256"],
        "archive_entries": book["archive_entries"],
        "core_or_custom_properties_present": book["core_properties_present"],
        "worksheet_names_and_xml": book["sheet_paths"],
        "formula_cells": book["formula_cells"],
        "source_identity_note": text(book, overview, "A2"),
        "source_timing_note": text(book, overview, "A12"),
        "cache_ledger": cache_ledger,
        "cache_reconciliation_status": "passed" if cache_reconciled else "failed",
        "cache_reconciliation_tolerance_us": TOLERANCE_US,
    }

    sample = {
        "schema_version": 1,
        "sample_set": "tx8_reference_reconstruction_v1",
        "created_by": "die/tools/calibrate_tx8.py",
        "workbook": source_book,
        "source_data_identity": {
            "original_measurements_available_in_workbook": False,
            "workbook_statement": "A2 says the original nine records were removed; the current values are reconstructed engineering calibration data.",
            "raw_logs_or_repetitions_present": False,
            "246us": {"cell": f"{timing}!B2", "classification": "reported_anchor", "interpretation": "one complete RMSNorm based on the accompanying explanation; no raw log or repetitions"},
            "3363us": {"cell": f"{timing}!B13", "classification": "conditional", "interpretation": "original SiLU label reinterpreted as entire fused FFN; optional assumption, not independent PE/Vec measurement"},
            "derived_timing_groups": "all other group times are model estimates, a reused Norm anchor, or attribution values and have zero independent empirical fit weight",
            "block_total": "B15 hardcoded value and B16 formula cache are workbook accounting, not an independent layer measurement",
        },
        "shape": {
            **shape,
            "M": M,
            "J": J,
            "A": A_q,
            "shape_definition_source": f"{dims}!B2:B10",
            "stored_dtypes": {"input": "BF16", "weights": "BF16", "projection_output": "BF16", "attention_score_and_probability": "FP32", "accumulator": "FP32"},
            "arithmetic_precision_note": "Ops and physical byte counts below use integer formulas and the explicit reference dtypes; no source latency is inferred from a string-only size field.",
        },
        "conditional_reference_fit": {
            "profile": "tx8_reference_v1",
            "identifiability": "two conditional reference effective quantities only; not independently validated",
            "fit_sample_count": 2,
            "fit_sample_weights": {"RMSNorm_246us": "fixed-byte vector DDR inversion, conditional reference only", "FFN_3363us": "conditional fused-FFN interpretation, with matrix/vector computation efficiency bound together"},
            "eta_compute_reference": eta_compute,
            "eta_pe_reference": eta_compute,
            "eta_vec_reference": eta_compute,
            "eta_compute_binding": "eta_pe_reference == eta_vec_reference; one shared parameter, not two identified efficiencies",
            "eta_vector_memory_reference": eta_vector_memory,
            "bw_vector_reference_bytes_per_s": vector_bw,
            "bw_matrix_reference_bytes_per_s": matrix_bw,
            "matrix_peak_raw_flop_per_s": matrix_peak_raw,
            "vector_peak_raw_eqop_per_s": vector_peak_raw,
            "matrix_peak_effective_flop_per_s": matrix_peak_effective,
            "vector_peak_effective_eqop_per_s": vector_peak_effective,
            "dram_peak_bytes_per_s": dram_peak,
            "matrix_bw_source": f"{hw}!B7, selected 70% of 200 GB/s; engineering setting rather than independent measurement",
            "vector_bw_inversion": "(4*M*H+2*H)/246us, fixed BF16 input/output and one gamma read assumption",
            "compute_efficiency_inversion": "(6*M*H*F/128e12 + 16*M*F/1e12)/3363us, conditional FFN interpretation",
            "target_int4_transfer_allowed": False,
            "independent_validation_set": False,
        },
        "groups": groups,
        "aggregates": {
            "attention_core_source_group_sum_us": attention_source_sum_us,
            "attention_core_reference_prediction_us": attention_us,
            "attention_group_keys_inclusive": sorted(attention_keys),
            "ffn_group_source_us": ffn_total_us,
            "ffn_group_reference_prediction_us": group_by_key["ffn_fused"]["reference_prediction_us"],
            "ffn_compute_component_attribution": ffn_components,
            "ffn_component_sum_us": ffn_component_sum,
            "ffn_memory_service_budget_us": group_by_key["ffn_fused"]["memory_us"],
            "block_source_sum_B2_B14_us": source_formula_sum_us,
            "block_hardcoded_B15_us": source_block_us,
            "block_formula_cache_B16_us": formula_cache_us,
            "block_reference_prediction_us": block_prediction_us,
            "block_prediction_minus_formula_cache_us": block_prediction_us - formula_cache_us,
        },
        "math_nodes": node_records,
        "validation_statuses": {
            "xlsx_cache_reconcile": "passed" if cache_reconciled else "failed",
            "tx8_reference_reproduction": "passed" if reference_reproduced else "failed",
            "coarse_library_execution": "not_run_cabi_not_required_for_xlsx_and_reference_reconstruction",
            "heldout_hardware_validation": "not_available_no_auditable_measurements_or_holdout",
        },
        "limitations": [
            "The shared reference efficiency is conditional on a single FFN explanation and fixed peak assumptions.",
            "The two reference quantities do not identify target INT4 PE, target Vec/SFU, target HBM, NoC, SRAM, launch, or pipeline efficiency.",
            "The 19 math nodes are not 19 independent timing samples; RoPE_Q/K and score_scale/softmax share parent group budgets.",
            "The AV workbook key maps once to stable source_key av_pv (C++ label AV/PV); group totals are counted once.",
            "Reference prediction agreement with the workbook's reconstructed values is reproduction, not prediction accuracy or hardware validation.",
        ],
    }

    config = {
        "schema_version": 1,
        "config_name": "calibration_tx8",
        "source_profile": "tx8_reference_v1",
        "calibration_status": "conditional_reference_reconstruction_not_hardware_calibrated",
        "target_profile_status": "assumed_not_calibrated",
        "source_workbook": {
            "filename": workbook_path.name,
            "sha256": book["sha256"],
            "source_measurement_rows_removed": True,
        },
        "reference_profile": {
            "scope": "TX8 BF16/FP32 reference formulas only; no target architecture use",
            "shape": {"B": shape["B"], "S_q": shape["S_q"], "S_k": shape["S_k"], "H": H, "F": F, "H_q": shape["H_q"], "H_kv": shape["H_kv"], "head_dim": D, "M": M, "J": J, "A": A_q},
            "dtype_path": "BF16 inputs/weights/projection outputs; FP32 attention score/probability and accumulation",
            "nominal_matrix_peak_flop_per_s": matrix_peak_raw,
            "nominal_vector_peak_eqop_per_s": vector_peak_raw,
            "dram_peak_bytes_per_s": dram_peak,
            "matrix_effective_bw_bytes_per_s": matrix_bw,
            "matrix_effective_bw_status": "engineering_setting_70_percent_of_peak",
            "matrix_effective_flop_per_s": matrix_peak_effective,
            "vector_effective_eqop_per_s": vector_peak_effective,
            "vector_effective_bw_bytes_per_s": vector_bw,
            "eta_compute_reference": eta_compute,
            "eta_pe_reference": eta_compute,
            "eta_vec_reference": eta_compute,
            "eta_vector_memory_reference": eta_vector_memory,
            "fit_parameters": {
                "eta_compute_reference": {"value": eta_compute, "unit": "fraction", "range": "(0,1]", "source": "conditional 3363us fused-FFN anchor plus fixed nominal peaks", "fit_sample": f"{timing}!B13", "applies_to": "tx8_reference_v1 only", "identifiability": "one shared matrix/vector effective compute efficiency"},
                "eta_vector_memory_reference": {"value": eta_vector_memory, "unit": "fraction of 200 GB/s", "range": "(0,1]", "source": "reported 246us Norm anchor with assumed fixed BF16 bytes", "fit_sample": f"{timing}!B2", "applies_to": "tx8_reference_v1 vector DDR path only", "identifiability": "conditional on fixed traffic assumption"},
            },
            "launch_s": 0.0,
            "memory_setup_s": 0.0,
            "full_score_materialization": True,
            "skip_fully_masked_tiles": False,
            "source_keys": {g[0]: {"workbook": g[1], "cpp": g[2]} for g in GROUPS},
            "av_pv_mapping": {"stable_source_key": "av_pv", "workbook_id": "AV", "cpp_group_name": "AV/PV", "count_once": True},
            "block_total_policy": "sum only the 13 canonical group rows; never add BLOCK_TOTAL as a fourteenth component",
        },
        "target_calibration": {
            "profile": "coarse_die_v2",
            "status": "assumed_not_calibrated",
            "tx8_reference_parameter_transfer": "prohibited",
            "parameters": {
                "eta_pe": {"value": 0.90, "unit": "fraction", "range": "(0,1]", "source": "engineering_prior", "status": "assumed", "note": "target INT4 matrix effective rate; TX8 BF16 shared efficiency is not a target initialization or fit"},
                "eta_vec": {"value": 0.90, "unit": "fraction", "range": "(0,1]", "source": "engineering_prior", "status": "assumed"},
                "eta_hbm_matrix": {"value": 0.70, "unit": "fraction", "range": "(0,1]", "source": "engineering_prior", "status": "assumed"},
                "eta_hbm_vector": {"value": 0.70, "unit": "fraction", "range": "(0,1]", "source": "engineering_prior", "status": "assumed", "binding": "share matrix prior unless independent target probes exist"},
                "eta_sram": {"value": 1.0, "unit": "fraction", "range": "(0,1]", "source": "engineering_prior", "status": "assumed"},
                "eta_local": {"value": 0.70, "unit": "fraction", "range": "(0,1]", "source": "engineering_prior", "status": "assumed"},
                "eta_mesh": {"value": 0.70, "unit": "fraction", "range": "(0,1]", "source": "engineering_prior", "status": "assumed"},
                "eta_inject": {"value": 0.70, "unit": "fraction", "range": "(0,1]", "source": "engineering_prior", "status": "assumed"},
                "rho_sram": {"value": 0.90, "unit": "fraction", "range": "(0,1]", "source": "engineering_prior", "status": "assumed"},
                "launch_ns_per_macro": {"value": 1000, "unit": "ns/macro", "range": "[0,infinity)", "source": "engineering_prior", "status": "assumed"},
                "memory_setup_ns_per_macro": {"value": 0, "unit": "ns/macro", "range": "[0,infinity)", "source": "engineering_prior", "status": "assumed"},
                "kappa_quant_eqops": {"value": 4, "unit": "eq-op/converted operand", "range": "[0,infinity)", "source": "engineering_prior", "status": "assumed"},
                "kappa_acc_convert_eqops": {"value": 2, "unit": "eq-op/converted accumulator", "range": "[0,infinity)", "source": "engineering_prior", "status": "assumed"},
            },
        },
        "evidence_statuses": sample["validation_statuses"],
    }

    # One CSV row per math node. Group timing fields appear on only one owner
    # row per canonical group; shared children have blank timing cells.
    csv_rows: list[dict[str, Any]] = []
    for node in node_records:
        group = group_by_key[node["source_key"]]
        csv_rows.append({
            "source_key": node["source_key"],
            "node_key": node["node_key"],
            "workbook_math_node": node["workbook_math_node"],
            "source_group_id": node["source_group_id"],
            "cpp_group_name": group["cpp_group_name"],
            "source_cell": group["source_time_cell"] if node["is_group_timing_owner"] else "",
            "source_group_time_us_count_once": fmt_us(node["source_group_time_us"]),
            "source_evidence_status": node["time_status"],
            "reference_group_prediction_us_count_once": fmt_us(node["reference_group_prediction_us"]),
            "Fop": group["Fop"],
            "Eop": group["Eop"],
            "D_bytes": group["D_bytes"],
            "memory_class": group["memory_class"],
            "reference_compute_us_group": fmt_us(group["compute_us"]) if node["is_group_timing_owner"] else "",
            "reference_memory_us_group": fmt_us(group["memory_us"]) if node["is_group_timing_owner"] else "",
            "ffn_component_attribution_us": fmt_us(node["reference_component_attribution_us"]),
            "counting_note": ("group delay belongs to shared parent; child has no independent delay" if not node["is_group_timing_owner"] else (
                "FFN component attribution is a split of the one fused group compute budget; do not add it to group total" if group["source_key"] == "ffn_fused" else "canonical group delay counted once"
            )),
        })

    attention_err = max(g["prediction_error_abs_us"] for g in groups)
    report_lines = [
        "# TX8 workbook audit and conditional reference reconstruction",
        "",
        f"Source workbook: `{workbook_path.name}` (SHA-256 `{book['sha256']}`). The XLSX is parsed directly from ZIP/XML with Python's standard library.",
        "",
        "The workbook overview states that the original nine records were removed. Its 13 group values are a mixture of one reported anchor, a reused Norm anchor, one conditional reinterpretation, and model estimates/attributions. The 19 math-node rows do not provide 19 independent timing samples. No raw timing logs or repeat distributions are present.",
        "",
        "## Source cache audit",
        "",
        f"`{timing}!B16` contains formula `{formula_cell['formula']}` and cached XML value `{raw_num(book, timing, 'B16')}` ({formula_cache_us:.15f} μs). The tool recomputes B2:B14 as {source_formula_sum_us:.15f} μs, a difference of {source_formula_sum_us - formula_cache_us:+.3g} μs. `{timing}!B15` is a separate hardcoded value `{raw_num(book, timing, 'B15')}` ({source_block_us:.15f} μs), differing from the formula cache by {source_block_us - formula_cache_us:+.3g} μs. The formula cell and its cache are preserved verbatim in `tx8_samples.json`.",
        "",
        f"Cache reconciliation: **{'passed' if cache_reconciled else 'failed'}** (required absolute difference < {TOLERANCE_US:g} μs). This checks workbook accounting, not timing accuracy.",
        "",
        "## Boundary records and identifiable reference quantities",
        "",
        f"- 246 μs at `{timing}!B2`: `reported_anchor`, interpreted as one full RMSNorm by the workbook explanation; the raw record and repeats are absent. Under the fixed BF16 traffic count `{norm_bytes}` bytes, this gives reference vector DDR service {vector_bw/1e9:.12f} GB/s and η_vector_memory={eta_vector_memory:.15f} relative to 200 GB/s.",
        f"- 3363 μs at `{timing}!B13`: `conditional`; the original SiLU label is reinterpreted as the whole fused FFN. Under 128 TFLOP/s matrix peak, 1 T eq-op/s vector peak, and a single shared matrix/vector compute efficiency, this gives η_compute_reference={eta_compute:.16f}. This does not separately identify PE and Vec efficiency.",
        "- The 140 GB/s matrix memory service rate is the workbook's selected 70% setting, not an independent measurement.",
        "- These two inversions apply only to `tx8_reference_v1`. They do not calibrate or initialize target INT4 PE, target Vec/SFU, HBM, NoC, SRAM, launch, or pipeline parameters.",
        "",
        "## Canonical 13-group reconstruction",
        "",
        "All operation counts and physical byte counts are integer-valued and use the workbook shape B=1, Sq=Sk=1024, H=4096, F=14336, Hq=32, Hkv=8, d=128; BF16 inputs/weights/projection outputs; FP32 score/probability and accumulation. Each group starts compute and memory together and completes at their maximum. The AV workbook timing group maps once to the stable `av_pv` key and C++ `AV/PV` label.",
        "",
        "| Key | Group | Fop | Eop | Bytes | Compute μs | Memory μs | Predicted μs | XLSX μs | Evidence |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for g in groups:
        report_lines.append(
            f"| `{g['source_key']}` | {g['cpp_group_name']} | {g['Fop']} | {g['Eop']} | {g['D_bytes']} | {g['compute_us']:.6f} | {g['memory_us']:.6f} | {g['reference_prediction_us']:.6f} | {g['source_time_us']:.6f} | {g['evidence_status']} |"
        )
    report_lines.extend([
        "",
        f"Maximum absolute group difference between the reference equations and the workbook cached values: {max(g['prediction_error_abs_us'] for g in groups):.3g} μs. Reference formula reproduction: **{'passed' if reference_reproduced else 'failed'}** against the <{TOLERANCE_US:g} μs threshold.",
        "",
        f"The attention-core sum counts Q/K/V projections, RoPE, QK, scale+softmax, AV/PV, and O projection exactly once: source group sum {attention_source_sum_us:.9f} μs; reference reconstruction {attention_us:.9f} μs. The workbook's B6 attention subtotal is an engineering estimate and is not included as another component.",
        "",
        f"FFN is one fused 3363 μs group. Its compute-budget attribution is gate {ffn_components[0]['component_budget_us']:.9f}, up {ffn_components[1]['component_budget_us']:.9f}, SiLU {ffn_components[2]['component_budget_us']:.9f}, gate-mul {ffn_components[3]['component_budget_us']:.9f}, and down {ffn_components[4]['component_budget_us']:.9f} μs; these sum to {ffn_component_sum:.9f} μs and are not five independently measured kernel times. Its memory service budget is {group_by_key['ffn_fused']['memory_us']:.9f} μs.",
        "",
        f"The 13 group reference predictions sum to {block_prediction_us:.12f} μs. The attention and FFN parent values are reporting subtotals only; neither is added again to the block total. The formula cache B16 is the authoritative workbook sum cache; B15 remains separately identified as a hardcoded total.",
        "",
        "## Evidence state",
        "",
        "| Layer | Status | Meaning |",
        "|---|---|---|",
        f"| XLSX cache | {'passed' if cache_reconciled else 'failed'} | Exact XML formula/cache extraction; recomputed sum agrees within 1e-6 μs; says nothing about prediction accuracy. |",
        f"| TX8 reference | {'passed' if reference_reproduced else 'failed'} | Independent 13-group equations and attention/FFN/block accounting reproduce the reconstructed workbook values. |",
        "| New coarse-library execution | not_run | This task only reconstructs workbook/reference behavior; C ABI is not needed. |",
        "| Independent held-out hardware validation | not_available | No auditable raw measurements or holdout set exist in this workbook. |",
        "",
        "The reference agreement is **reproduction**, not prediction accuracy. Since the source values were themselves reconstructed, no inference about real TX8 hardware accuracy follows.",
        "",
    ])
    return config, sample, csv_rows, "\n".join(report_lines), groups


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    script = Path(__file__).resolve()
    repo = script.parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=repo / DEFAULT_WORKBOOK)
    parser.add_argument("--output-root", type=Path, default=repo / "die")
    args = parser.parse_args(argv)
    workbook_path = args.workbook.resolve()
    output_root = args.output_root.resolve()
    config, sample, rows, report, groups = derive(workbook_path)

    write_json(output_root / "configs" / "calibration_tx8.json", config)
    write_json(output_root / "calibration" / "tx8_samples.json", sample)
    (output_root / "calibration" / "report.md").write_text(report, encoding="utf-8")
    csv_path = output_root / "calibration" / "per_operator.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output_root / 'configs' / 'calibration_tx8.json'}")
    print(f"Wrote {output_root / 'calibration' / 'tx8_samples.json'}")
    print(f"Wrote {output_root / 'calibration' / 'report.md'}")
    print(f"Wrote {csv_path}")
    print(f"Reference group count: {len(groups)}; formula cache {sample['validation_statuses']['xlsx_cache_reconcile']}; reference {sample['validation_statuses']['tx8_reference_reproduction']}")
    if sample["validation_statuses"]["xlsx_cache_reconcile"] != "passed" or sample["validation_statuses"]["tx8_reference_reproduction"] != "passed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
