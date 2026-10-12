#!/usr/bin/env python3
"""test_generate_trace_lexical_identity.py -- D3 (2026-09-28) two-end
lexical/identity alignment between the Python loader
(generate_trace.load_request_queue) and the C++ WindowedTraceReader.

Contracts under test:
  - identity gate mirrors the C++ C3 gate: an EMPTY request_id is rejected
    (the loader previously synthesized f"request_{len(requests)}" -- a
    two-end identity fork) and a CROSS-ROW DUPLICATE request_id is rejected;
  - '#' comment rows no longer exist on the Python side: a request_id
    starting with '#' is rejected (the C++ reader has no comment syntax and
    treats such a row as a malformed data row; skipping it here shifted
    every later request's queue_index against the C++ data-row numbering);
  - the numeric lexicon is plain ASCII digits, matching the C++ strict
    lexer (parse_u64_field): '+1' / ' 1' / '1_0' / Unicode digits / '5x'
    are rejected at parse time instead of failing only at runtime on the
    C++ side;
  - zero-length prefill/decode stays rejected (the loader's long-standing
    positive-length gate; the derive materializer and the C++ reader now
    reject the same shape);
  - TraceBuilder.arm_timer_gate / TraceBuilder.timer_gate are GONE (D3
    dead-code removal; the offline static-ET pipeline that timed arrivals
    with timer nodes was deleted 2026-08-18 and nothing referenced either
    method) -- the assertions pin the removal against resurrection.
"""

import sys
import tempfile
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
for _p in (str(MODULE_DIR), str(MODULE_DIR.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_trace import (  # noqa: E402
    RequestSpec,
    TraceBuilder,
    load_request_queue,
    parse_int,
    parse_nonnegative_int,
    parse_rank_id,
)

QUEUE_HEADER = (
    "session_id,turn_index,request_id,prefill_length,decode_length,"
    "session_arrival_time_ns,inter_request_interval_ns,description"
)


def _queue(rows):
    handle = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    handle.write(QUEUE_HEADER + "\n" + "\n".join(rows) + "\n")
    handle.close()
    return Path(handle.name)


def _spec(request_id="s0_r0", **overrides):
    fields = dict(
        session_id="s0", turn_index=0, request_id=request_id,
        prefill_length=100, decode_length=10,
        session_arrival_time_ns=1000, inter_request_interval_ns=None)
    fields.update(overrides)
    return RequestSpec(**fields)


class LoaderIdentityGateTests(unittest.TestCase):
    def test_happy_path_still_loads(self):
        specs = load_request_queue(_queue([
            "s0,0,s0_r0,100,10,1000,,d",
            "s0,1,s0_r1,100,10,,2000,d",
        ]))
        self.assertEqual([s.request_id for s in specs], ["s0_r0", "s0_r1"])

    def test_empty_request_id_rejected_not_synthesized(self):
        with self.assertRaises(ValueError) as caught:
            load_request_queue(_queue(["s0,0,,100,10,1000,,d"]))
        self.assertIn("missing request_id", str(caught.exception))
        self.assertIn("no longer synthesizes", str(caught.exception))

    def test_duplicate_request_id_rejected(self):
        with self.assertRaises(ValueError) as caught:
            load_request_queue(_queue([
                "s0,0,dup,100,10,1000,,d",
                "s0,1,dup,100,10,,2000,d",
            ]))
        self.assertIn("duplicates", str(caught.exception))
        self.assertIn("mirrors the C++ identity gate", str(caught.exception))

    def test_hash_request_id_rejected_as_comment_shape(self):
        with self.assertRaises(ValueError) as caught:
            load_request_queue(_queue([
                "s0,0,s0_r0,100,10,1000,,d",
                "s0,1,#note,100,10,,2000,d",
            ]))
        self.assertIn("no comment syntax", str(caught.exception))

    def test_whole_line_comment_row_still_fail_closed(self):
        with self.assertRaises(ValueError):
            load_request_queue(_queue([
                "s0,0,s0_r0,100,10,1000,,d",
                "# annotator note,,,,,,",
            ]))

    def test_duplicate_session_turn_pin_still_rejected(self):
        with self.assertRaises(ValueError) as caught:
            load_request_queue(_queue([
                "s0,0,a,100,10,1000,,d",
                "s0,0,b,100,10,1000,,d",
            ]))
        self.assertIn("must be unique", str(caught.exception))

    def test_zero_length_prefill_still_rejected(self):
        with self.assertRaises(ValueError) as caught:
            load_request_queue(_queue(["s0,0,s0_r0,0,10,1000,,d"]))
        self.assertIn("positive", str(caught.exception))


class StrictAsciiLexemeTests(unittest.TestCase):
    def test_loose_shapes_rejected_everywhere(self):
        for parser, value in (
            (parse_nonnegative_int, "+5"),
            (parse_nonnegative_int, " 5"),
            (parse_nonnegative_int, "1_0"),
            (parse_nonnegative_int, "١٢٣"),   # Unicode digits: int() took these
            (parse_nonnegative_int, "5x"),
            (parse_nonnegative_int, ""),
            (parse_int, "-3"),
        ):
            with self.subTest(parser=parser.__name__, value=value):
                with self.assertRaises(ValueError) as caught:
                    parser(value, "field")
                self.assertIn("ASCII digits", str(caught.exception))
        for value in ("0x2", "+1", " 2", "٣"):
            with self.subTest(parser="parse_rank_id", value=value):
                with self.assertRaises(ValueError) as caught:
                    parse_rank_id(value)
                self.assertIn("ASCII digits", str(caught.exception))

    def test_plain_digits_still_accepted(self):
        self.assertEqual(parse_nonnegative_int("0", "field"), 0)
        self.assertEqual(parse_int("7", "field"), 7)
        self.assertEqual(parse_rank_id("12"), 12)


class TimerGateRemovalPins(unittest.TestCase):
    def test_trace_builder_timer_gate_methods_are_gone(self):
        self.assertFalse(hasattr(TraceBuilder, "timer_gate"))
        self.assertFalse(hasattr(TraceBuilder, "arm_timer_gate"))

    def test_trace_builder_still_emits_serial_chains(self):
        builder = TraceBuilder(remote_operand_loads=False)
        builder.comp("first", num_ops=1, tensor_size=1)
        builder.all_reduce("ar", comm_size=8, pg_name="1")
        self.assertEqual(builder.node_count, 2)
        self.assertEqual(tuple(builder.nodes[1].data_deps),
                         (builder.nodes[0].id,))


if __name__ == "__main__":
    unittest.main()
