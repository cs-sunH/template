"""Run-scoped allocation of immutable pipeline descriptor identities.

This module allocates descriptor segment ordinals only.  Physical staging
allocation IDs are owned by ``KVCacheManager`` and passed in explicitly.
Candidate prediction calls :meth:`preview_batch`; accepted graph emission
calls :meth:`commit_batch` after its enclosing batch has passed validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

from joint.pipeline_descriptor import PipelineIdentity, U64_MAX


class PipelineIdentityError(ValueError):
    pass


@dataclass(frozen=True)
class PipelineIdentitySeed:
    request_id: str
    stage: str
    generation: int
    stage_epoch: int
    rank: int
    staging_allocation_id: int


def _uint(value, name, *, positive=False):
    if type(value) is not int or not 0 <= value <= U64_MAX:
        raise PipelineIdentityError(f"{name} must be uint64")
    if positive and value == 0:
        raise PipelineIdentityError(f"{name} must be positive")
    return value


class PipelineIdentityAllocator:
    """Preview or commit segment ordinals in structural emission order.

    Ordinals are local to ``(request, stage, generation, stage_epoch, rank)``.
    This allocator has no clock/train/delivery fallback and never allocates a
    staging domain.  Multiple alternatives can preview the same next ordinal;
    only the selected/finalized batch advances the counters.
    """

    def __init__(self):
        self._next_segment = {}

    @staticmethod
    def _validate_seed(seed):
        if not isinstance(seed, PipelineIdentitySeed):
            raise PipelineIdentityError("expected PipelineIdentitySeed")
        if not isinstance(seed.request_id, str) or not seed.request_id:
            raise PipelineIdentityError("request_id must be a non-empty string")
        if seed.stage not in ("prefill", "decode"):
            raise PipelineIdentityError("stage must be prefill or decode")
        generation = _uint(seed.generation, "generation")
        expected_generation = 0 if seed.stage == "prefill" else 1
        if generation != expected_generation:
            raise PipelineIdentityError("generation must match the stage context")
        _uint(seed.stage_epoch, "stage_epoch")
        _uint(seed.rank, "rank")
        _uint(seed.staging_allocation_id, "staging_allocation_id", positive=True)
        return (seed.request_id, seed.stage, generation, seed.stage_epoch,
                seed.rank)

    @staticmethod
    def _identity(seed, segment_ordinal):
        request_hex = seed.request_id.encode("utf-8").hex()
        segment_id = (
            f"pp1/{request_hex}/{seed.stage}/{seed.generation}/"
            f"{seed.stage_epoch}/{seed.rank}/{segment_ordinal}")
        buffer_domain_id = (
            f"pb1/{request_hex}/{seed.rank}/"
            f"{seed.staging_allocation_id}")
        return PipelineIdentity(
            request_id=seed.request_id,
            stage=seed.stage,
            generation=seed.generation,
            stage_epoch=seed.stage_epoch,
            segment_ordinal=segment_ordinal,
            rank=seed.rank,
            staging_allocation_id=seed.staging_allocation_id,
            segment_id=segment_id,
            buffer_domain_id=buffer_domain_id)

    def preview_batch(self, seeds: Iterable[PipelineIdentitySeed]
                      ) -> Tuple[PipelineIdentity, ...]:
        """Return prospective identities without changing any counter."""
        next_by_key = dict(self._next_segment)
        result = []
        for seed in seeds:
            key = self._validate_seed(seed)
            ordinal = next_by_key.get(key, 0)
            _uint(ordinal, "segment_ordinal")
            result.append(self._identity(seed, ordinal))
            if ordinal == U64_MAX:
                raise PipelineIdentityError("segment ordinal exhausted uint64")
            next_by_key[key] = ordinal + 1
        return tuple(result)

    def commit_batch(self, seeds: Iterable[PipelineIdentitySeed]
                     ) -> Tuple[PipelineIdentity, ...]:
        """Allocate ordinals atomically after validating the whole batch."""
        seeds = tuple(seeds)
        identities = self.preview_batch(seeds)
        next_by_key = dict(self._next_segment)
        for seed, identity in zip(seeds, identities):
            key = self._validate_seed(seed)
            ordinal = next_by_key.get(key, 0)
            if ordinal != identity.segment_ordinal:
                raise PipelineIdentityError(
                    "segment ordinal state changed during batch allocation")
            next_by_key[key] = ordinal + 1
        self._next_segment = next_by_key
        return identities

    def next_ordinal(self, seed: PipelineIdentitySeed) -> int:
        """Return the next ordinal for inspection without reserving it."""
        key = self._validate_seed(seed)
        return self._next_segment.get(key, 0)

    def snapshot_state(self) -> dict:
        """Return deterministic JSON state for run/replay restoration."""
        rows = [
            {"request_id": key[0], "stage": key[1],
             "generation": key[2], "stage_epoch": key[3],
             "rank": key[4], "next_segment_ordinal": value}
            for key, value in sorted(self._next_segment.items())]
        return {"version": 1, "next_segment_ordinals": rows}

    def restore_state(self, state, *, committed_identities=()) -> None:
        """Restore committed counters and merge descriptor identities safely.

        A journal snapshot may lag the descriptor ledger.  In that case each
        already-committed identity raises its key's next ordinal to at least
        ``segment_ordinal + 1``; replay can never rename a live descriptor or
        reuse a committed ordinal.
        """
        if not isinstance(state, dict) or set(state) != {
                "version", "next_segment_ordinals"}:
            raise PipelineIdentityError("invalid pipeline identity state keys")
        if type(state["version"]) is not int or state["version"] != 1:
            raise PipelineIdentityError("unsupported pipeline identity state version")
        rows = state["next_segment_ordinals"]
        if not isinstance(rows, (list, tuple)):
            raise PipelineIdentityError("next_segment_ordinals must be an array")
        restored = {}
        prior_key = None
        expected_fields = {
            "request_id", "stage", "generation", "stage_epoch", "rank",
            "next_segment_ordinal"}
        for row in rows:
            if not isinstance(row, dict) or set(row) != expected_fields:
                raise PipelineIdentityError("invalid segment ordinal state row")
            seed = PipelineIdentitySeed(
                request_id=row["request_id"], stage=row["stage"],
                generation=row["generation"], stage_epoch=row["stage_epoch"],
                rank=row["rank"], staging_allocation_id=1)
            key = self._validate_seed(seed)
            next_ordinal = _uint(
                row["next_segment_ordinal"], "next_segment_ordinal")
            if next_ordinal == 0:
                raise PipelineIdentityError("committed next ordinal must be positive")
            if prior_key is not None and key <= prior_key:
                raise PipelineIdentityError("segment ordinal rows must be uniquely sorted")
            prior_key = key
            restored[key] = next_ordinal
        for identity in committed_identities:
            if not isinstance(identity, PipelineIdentity):
                raise PipelineIdentityError("committed identity has wrong type")
            seed = PipelineIdentitySeed(
                request_id=identity.request_id, stage=identity.stage,
                generation=identity.generation, stage_epoch=identity.stage_epoch,
                rank=identity.rank,
                staging_allocation_id=identity.staging_allocation_id)
            key = self._validate_seed(seed)
            lower_bound = _uint(identity.segment_ordinal,
                                 "committed segment_ordinal") + 1
            if lower_bound > U64_MAX:
                raise PipelineIdentityError("committed segment ordinal exhausted uint64")
            restored[key] = max(restored.get(key, 0), lower_bound)
        self._next_segment = restored
