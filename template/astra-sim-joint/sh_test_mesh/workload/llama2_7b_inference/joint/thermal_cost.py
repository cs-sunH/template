#!/usr/bin/env python3
"""thermal_cost.py -- feed a thermal resource envelope into joint cost estimates.

A candidate's projected completion time must be computed under the *same*
resource epoch the executor will use.  This module converts a
:class:`ResourceEnvelope` into the multipliers the cost/projection paths apply,
and defines the cache key that binds a cached cost to that epoch so a stale
peak-rate cost can never survive a cap change.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from wafercosim.contracts import ContractError, ResourceEnvelope


@dataclass(frozen=True)
class ThermalCostView:
    """One frozen, consistent view of the thermal caps for one decision.

    A single decision evaluates all candidates against one ``resource_epoch``;
    the view is validated before commit so a mid-decision epoch change cannot
    mix plans computed under different caps.
    """

    resource_epoch: int
    as_of_ns: int
    state_version: int
    hardware_hash: str
    mapping_epoch: str
    domain_fractions: dict[str, float]

    @staticmethod
    def from_envelope(envelope: ResourceEnvelope, *, state_version: int,
                      hardware_hash: str, mapping_epoch: str) -> "ThermalCostView":
        fractions = {row["domain"]: row["compute_throughput_fraction"]
                     for row in envelope.domains}
        for domain, fraction in fractions.items():
            if not 0.0 < fraction <= 1.0:
                raise ContractError(
                    f"domain {domain!r} has an unbalanced throughput fraction "
                    f"{fraction!r}; zero-rate domains must be handled by the "
                    "scheduler pause/terminate path, not filled into a solver")
        return ThermalCostView(
            resource_epoch=envelope.resource_epoch,
            as_of_ns=envelope.as_of_ns,
            state_version=state_version,
            hardware_hash=hardware_hash,
            mapping_epoch=mapping_epoch,
            domain_fractions=fractions,
        )

    def compute_scale(self, domain: str | None = None) -> float:
        """Completion-time multiplier; >= 1.0 when a domain is de-rated.

        With the fixed-voltage/frequency-only model, halving the compute
        throughput doubles the compute-bound completion time.  A domain absent
        from the envelope is unconstrained here (scale 1.0).
        """
        if domain is None:
            worst = min(self.domain_fractions.values()) if self.domain_fractions else 1.0
        else:
            worst = self.domain_fractions.get(domain, 1.0)
        if worst <= 0.0:
            raise ContractError(
                f"compute scale for domain {domain!r} is undefined at a zero rate")
        return 1.0 / worst

    def cache_key(self, *, horizon_ns: int, prediction_config: str = "") -> tuple:
        """Cost-cache key including hardware, mapping and thermal epoch.

        A key that omits the thermal epoch would let a peak-rate cost survive a
        de-rate.  Temperature-dependent horizons additionally bind the thermal
        state version and horizon.
        """
        return (
            "thermal-cost-v1",
            self.hardware_hash,
            self.mapping_epoch,
            self.resource_epoch,
            self.state_version,
            int(horizon_ns),
            prediction_config,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "resource_epoch": self.resource_epoch,
            "as_of_ns": self.as_of_ns,
            "state_version": self.state_version,
            "hardware_hash": self.hardware_hash,
            "mapping_epoch": self.mapping_epoch,
            "domain_fractions": dict(self.domain_fractions),
        }


__all__ = ["ThermalCostView"]
