#!/usr/bin/env python3
"""thermal_policy.py -- joint-side thermal policy producing a ResourceEnvelope.

The thermal core returns temperatures and model domains; *throttling policy* is
owned here, never by the thermal configuration (contract: the thermal profile
describes geometry/materials/cooling only).  Thresholds, frequency steps,
recovery hysteresis, minimum dwell and the sampling step live in this policy so
a thermal run cannot smuggle scheduling parameters back into the thermal input.

The first version controls the coarse ``COMPUTE_GATE`` (normalized
``second_of_service/s``); the mapped fraction is the execution-time compute
de-rate, approximated as ``f / f_base`` and declared as such in the manifest.
It is not a voltage/OPP-level DVFS model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from wafercosim.contracts import ContractError, ResourceEnvelope

DEFAULT_DOMAIN_STEPS = (1.0, 0.75, 0.5)
OPERATING_FRACTION_MODEL = "compute_gate_fraction_as_frequency_ratio_approx"


@dataclass(frozen=True)
class DomainPolicy:
    """Per-control-domain thermal thresholds and frequency ladder."""

    domain: str
    threshold_k: float
    resume_k: float
    steps: tuple[float, ...] = DEFAULT_DOMAIN_STEPS
    min_dwell_ns: int = 0
    sampling_ns: int = 1_000_000
    source: str = "configured"

    def __post_init__(self) -> None:
        if not self.domain or not isinstance(self.domain, str):
            raise ContractError("domain policy requires a non-empty domain name")
        if not self.threshold_k > self.resume_k:
            raise ContractError(
                f"domain {self.domain!r} requires threshold_k > resume_k "
                "(recovery must be hysteretic)")
        if not self.steps or any(not (0.0 <= s <= 1.0) for s in self.steps):
            raise ContractError(f"domain {self.domain!r} steps must be within [0,1]")
        if list(self.steps) != sorted(self.steps, reverse=True):
            raise ContractError(f"domain {self.domain!r} steps must be descending")


@dataclass(frozen=True)
class ThermalPolicyConfig:
    policy_version: str
    base_frequency_hz: float
    domains: tuple[DomainPolicy, ...]
    minimum_fraction: float = 0.5

    @staticmethod
    def from_json(payload: Mapping[str, Any]) -> "ThermalPolicyConfig":
        if not isinstance(payload, Mapping):
            raise ContractError("thermal policy config must be an object")
        domains = tuple(
            DomainPolicy(
                domain=row["domain"],
                threshold_k=float(row["threshold_k"]),
                resume_k=float(row["resume_k"]),
                steps=tuple(float(s) for s in row.get("steps", DEFAULT_DOMAIN_STEPS)),
                min_dwell_ns=int(row.get("min_dwell_ns", 0)),
                sampling_ns=int(row.get("sampling_ns", 1_000_000)),
                source=str(row.get("source", "configured")),
            )
            for row in payload.get("domains", [])
        )
        if not domains:
            raise ContractError("thermal policy config requires at least one domain")
        return ThermalPolicyConfig(
            policy_version=str(payload["policy_version"]),
            base_frequency_hz=float(payload["base_frequency_hz"]),
            domains=domains,
            minimum_fraction=float(payload.get("minimum_fraction", 0.5)),
        )

    def step_floor(self, domain: str, requested: float) -> float:
        """Clamp a requested fraction to the ladder floor and hardware peak."""
        for policy in self.domains:
            if policy.domain == domain:
                floor = min(policy.steps) if policy.steps else self.minimum_fraction
                return max(requested, floor, self.minimum_fraction)
        return max(requested, self.minimum_fraction)


class ThermalPolicy:
    """Discrete thermal controller with per-domain hysteresis and minimum dwell.

    Operating semantics (plan §8.1): the feedback temperature is computed from
    already-committed activity, and the resulting cap applies from
    ``effective_from_ns`` onward.  A temperature overrun discovered at the end of
    a window is therefore *not* claimed to have been throttled inside that
    window -- that detection latency is a declared property of the discrete
    model.
    """

    def __init__(self, config: ThermalPolicyConfig):
        self.config = config
        self._state: dict[str, dict[str, Any]] = {
            policy.domain: {"step": 0, "since_ns": 0, "reason": "init"}
            for policy in config.domains
        }

    def _die_from_domain(self, domain: str) -> int:
        head = domain.split(".")[0]
        if not head.startswith("die"):
            raise ContractError(
                f"control domain {domain!r} is not a 'die<N>' domain")
        try:
            return int(head[3:])
        except ValueError as exc:
            raise ContractError(f"control domain {domain!r} has a bad die index") from exc

    def compute_envelope(self, feedback: Any, *, resource_epoch: int,
                         effective_from_ns: int, as_of_ns: int) -> ResourceEnvelope:
        if feedback.resource_epoch < 0:
            raise ContractError("feedback resource epoch must be non-negative")
        rows: list[dict[str, Any]] = []
        for policy in self.config.domains:
            die = self._die_from_domain(policy.domain)
            temperature = feedback.die_block_max_k.get(die)
            if temperature is None:
                raise ContractError(
                    f"thermal feedback carries no temperature for die {die} "
                    f"(domain {policy.domain!r})")
            state = self._state[policy.domain]
            step = state["step"]
            reason = state["reason"]
            if temperature >= policy.threshold_k:
                if step < len(policy.steps) - 1:
                    step += 1
                    state["since_ns"] = as_of_ns
                    reason = (
                        f"overrun: die{die} {temperature:.2f}K >= "
                        f"threshold {policy.threshold_k:.2f}K")
                else:
                    reason = (
                        f"held at floor: die{die} {temperature:.2f}K >= "
                        f"threshold {policy.threshold_k:.2f}K")
            elif temperature <= policy.resume_k and step > 0:
                dwell_ok = as_of_ns - int(state["since_ns"]) >= policy.min_dwell_ns
                if dwell_ok:
                    step -= 1
                    state["since_ns"] = as_of_ns
                    reason = (
                        f"recovery: die{die} {temperature:.2f}K <= "
                        f"resume {policy.resume_k:.2f}K")
                else:
                    reason = (
                        f"dwell block: die{die} {temperature:.2f}K <= "
                        f"resume {policy.resume_k:.2f}K but min dwell pending")
            else:
                reason = f"stable: die{die} {temperature:.2f}K"
            state["step"] = step
            state["reason"] = reason
            fraction = policy.steps[step]
            rows.append({
                "domain": policy.domain,
                "step": step,
                "compute_throughput_fraction": fraction,
                "resource_ceiling": policy.steps[0],
                "temperature_margin_k": policy.resume_k - temperature,
                "reason": reason,
                "applied": False,
                "as_of_ns": as_of_ns,
            })
        return ResourceEnvelope.build(
            policy_version=self.config.policy_version,
            resource_epoch=resource_epoch,
            effective_from_ns=effective_from_ns,
            as_of_ns=as_of_ns,
            domains=rows,
        )

    def manifest(self) -> dict[str, Any]:
        return {
            "policy_version": self.config.policy_version,
            "base_frequency_hz": self.config.base_frequency_hz,
            "minimum_fraction": self.config.minimum_fraction,
            "fraction_model": OPERATING_FRACTION_MODEL,
            "fraction_model_note": (
                "compute-throughput fraction is approximated as f/f_base; it is a "
                "compute-stage de-rate, not a voltage/OPP DVFS model"),
            "domains": [
                {
                    "domain": p.domain,
                    "threshold_k": p.threshold_k,
                    "resume_k": p.resume_k,
                    "steps": list(p.steps),
                    "min_dwell_ns": p.min_dwell_ns,
                    "sampling_ns": p.sampling_ns,
                    "source": p.source,
                }
                for p in self.config.domains
            ],
        }


__all__ = [
    "DomainPolicy", "ThermalPolicyConfig", "ThermalPolicy",
    "DEFAULT_DOMAIN_STEPS", "OPERATING_FRACTION_MODEL",
]
