#!/usr/bin/env python3
"""Focused E0 candidate-face arbitration tests."""

import os
import sys
import unittest
from dataclasses import replace

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _path in (_HERE, _PARENT):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from joint.joint_cost_model import (  # noqa: E402
    ACTION_COPY,
    ACTION_RECOMPUTE,
    ACTION_STAY,
    ActionCandidate,
    ActionCostBreakdown,
)
from joint.joint_scheduler import _projection_arbitrate  # noqa: E402
from joint.prediction_config import (  # noqa: E402
    JointPredictionConfig,
    PredictionParams,
)


class _Context:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _candidate(instance, action, cost, *, quality=None, continuation=False):
    context = _Context()
    breakdown = ActionCostBreakdown(
        target_wait_ns=0, history_prep_ns=0, eviction_wait_ns=0,
        compute_ns=cost, remote_read_ns=0, merge_ns=0,
        contention_divisor=1, hops=0,
        predicted_service_done=cost, predicted_merge_done=cost,
        projection_stats={"native_kernel": dict(quality or {})},
    )
    return ActionCandidate(
        instance_index=instance, action=action, applicable=True,
        inapplicable_reason=None, cost_ns=cost, breakdown=breakdown,
        is_continuation=continuation, projection_context=context,
    )


class _CostModel:
    def __init__(self, exact_costs):
        self.prediction = JointPredictionConfig(params=PredictionParams(
            window_refine_margin_ppm=20_000))
        self.exact_costs = exact_costs
        self.repriced = []

    def _resolved_prediction_config(self):
        return self.prediction

    def reprice_candidate_exact(self, candidate):
        candidate_id = (
            f"{candidate.instance_index}:{candidate.action}")
        self.repriced.append(candidate_id)
        exact_cost = self.exact_costs[candidate_id]
        return replace(
            candidate,
            cost_ns=exact_cost,
            breakdown=replace(
                candidate.breakdown,
                predicted_service_done=exact_cost,
                predicted_merge_done=exact_cost,
                compute_ns=exact_cost,
                projection_stats={"native_kernel": {
                    "approximation_profile": "exact",
                    "window_count": 0,
                    "refinement_count": 0,
                    "fully_exact": True,
                    "unsupported_reason": None,
                }},
            ),
            projection_precision="exact",
        )


class WindowCandidateArbitrationTests(unittest.TestCase):
    def test_sensitive_margin_reprices_every_applicable_candidate(self):
        candidates = [
            _candidate(0, ACTION_STAY, 100, quality={
                "approximation_profile": "native_die_timer_window_v1",
                "window_count": 1, "fully_exact": False,
            }),
            _candidate(0, ACTION_RECOMPUTE, 101, quality={
                "approximation_profile": "native_die_timer_window_v1",
                "window_count": 1, "fully_exact": False,
            }),
            _candidate(1, ACTION_COPY, 500, quality={
                "approximation_profile": "native_die_timer_window_v1",
                "window_count": 1, "fully_exact": False,
            }),
            ActionCandidate(
                instance_index=1, action="remote-read", applicable=False,
                inapplicable_reason="fixture inapplicable", cost_ns=None),
        ]
        model = _CostModel({
            "0:stay": 103, "0:recompute": 99, "1:copy": 150,
        })

        final, arbitration = _projection_arbitrate(
            model, candidates, boundary=False)

        self.assertEqual(model.repriced,
                         ["0:stay", "0:recompute", "1:copy"])
        self.assertEqual(arbitration["trigger_reason"], "sensitive_margin")
        self.assertEqual(arbitration["approximate_selected_id"], "0:stay")
        self.assertEqual(arbitration["exact_selected_id"], "0:recompute")
        self.assertEqual(arbitration["exact_recomputed_ids"], model.repriced)
        self.assertEqual(
            [row["cost_ns"] for row in arbitration["approx_results"]],
            [100, 101, 500, None])
        self.assertEqual(
            [row["cost_ns"] for row in arbitration["exact_results"]],
            [103, 99, 150, None])
        self.assertEqual(final[1].projection_precision, "exact")
        self.assertTrue(all(item.projection_context.closed
                            for item in candidates[:3]))

    def test_window_anomaly_triggers_full_face_even_outside_margin(self):
        candidates = [
            _candidate(0, ACTION_STAY, 100, quality={
                "approximation_profile": "exact", "window_count": 0,
                "fully_exact": True,
            }),
            _candidate(0, ACTION_RECOMPUTE, 500, quality={
                "approximation_profile": "exact", "window_count": 0,
                "fully_exact": True, "unsupported_reason": "window input",
            }),
        ]
        model = _CostModel({"0:stay": 100, "0:recompute": 490})

        _final, arbitration = _projection_arbitrate(
            model, candidates, boundary=False)

        self.assertEqual(arbitration["trigger_reason"], "unsupported_window")
        self.assertEqual(model.repriced, ["0:stay", "0:recompute"])

    def test_boundary_equal_cost_keeps_continuation_tie_order(self):
        candidates = [
            _candidate(0, ACTION_RECOMPUTE, 100, continuation=False),
            _candidate(1, ACTION_STAY, 100, continuation=True),
        ]
        model = _CostModel({"0:recompute": 100, "1:stay": 100})

        final, arbitration = _projection_arbitrate(
            model, candidates, boundary=True)

        self.assertEqual(arbitration["trigger_reason"], "sensitive_margin")
        self.assertEqual(arbitration["approximate_selected_id"], "1:stay")
        self.assertEqual(arbitration["exact_selected_id"], "1:stay")
        self.assertEqual(
            min((item for item in final if item.applicable),
                key=lambda item: (item.cost_ns, item.boundary_order_key())).action,
            ACTION_STAY)

    def test_single_exact_candidate_skips_reprice_but_releases_context(self):
        candidate = _candidate(0, ACTION_STAY, 10, quality={
            "approximation_profile": "exact", "window_count": 0,
            "fully_exact": True,
        })
        model = _CostModel({"0:stay": 10})

        final, arbitration = _projection_arbitrate(
            model, [candidate], boundary=False)

        self.assertEqual(model.repriced, [])
        self.assertIsNone(arbitration["trigger_reason"])
        self.assertEqual(arbitration["exact_results"], [])
        self.assertEqual(final[0].cost_ns, 10)
        self.assertTrue(candidate.projection_context.closed)


if __name__ == "__main__":
    unittest.main()
