#!/usr/bin/env python3
"""Check the fixed Core resource fixture against the authoritative DOCX.

This is an offline source audit. It neither searches for hardware nor changes
runtime inputs. Only Python's standard library is required.
"""
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

DIE = Path(__file__).resolve().parents[1]
NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
HEADING = re.compile(r"NoC 位宽 B = (64|128|256|512) bit：Core 的 PE/Vec 安装数量与算力")


def text(node):
    return "".join(t.text or "" for t in node.findall(".//w:t", NS))


def read_rows(document):
    with zipfile.ZipFile(document) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    rows, active_bits, widths = [], None, []
    for node in root.find("w:body", NS):
        heading = HEADING.fullmatch(text(node))
        if heading:
            if active_bits is not None:
                raise ValueError("NoC heading has no corresponding resource table")
            active_bits = int(heading[1])
        if node.tag != "{" + NS["w"] + "}tbl" or active_bits is None:
            continue
        table = node.findall("./w:tr", NS)
        if len(table) != 9:
            raise ValueError("each NoC resource table must have eight data rows")
        widths.append(active_bits)
        for row in table[1:]:
            cells = [text(cell) for cell in row.findall("./w:tc", NS)]
            if len(cells) != 8:
                raise ValueError("unexpected DOCX Core table shape")
            rows.append({"sram_mib_per_core": float(cells[0]),
                         "noc_width_bits": active_bits,
                         "n_vec": int(cells[1]), "n_pe": int(cells[2]),
                         "l0_mm_display": float(cells[5]) / 1000})
        active_bits = None
    if widths != [64, 128, 256, 512] or len(rows) != 32:
        raise ValueError("DOCX must contain exactly four ordered resource tables")
    return rows


def main():
    fixture = json.loads((DIE / "configs/core32.json").read_text())
    source = fixture["source_document"]
    document = DIE.parent / source["path"]
    if hashlib.sha256(document.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("DOCX changed: review hardware authority before updating source identity")
    rows = read_rows(document)
    if rows != fixture["rows"]:
        raise ValueError("Core fixture differs from the actual DOCX tables")
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != fixture["rows_sha256"] or fixture["table_identity"] != {
            "id": "core32_v1", "rows_sha256": digest}:
        raise ValueError("Core fixture identity does not match the source rows")
    print(json.dumps({"status": "passed", "document": source,
                      "row_count": len(rows), "rows_sha256": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
