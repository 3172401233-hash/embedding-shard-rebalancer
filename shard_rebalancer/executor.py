"""Simulated executor that walks a Plan phase by phase.

It tracks, per cluster, the set of shards holding a *readable* replica and
asserts the core safety invariant: at every point in the migration each
cluster has at least one readable replica.  A move can be forced to fail to
exercise the rollback path.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from .model import MoveOp, MovePhase, MoveStatus, Plan, Workload


class ReplicaInvariantError(AssertionError):
    pass


class Executor:
    def __init__(self, workload: Workload) -> None:
        self.workload = workload
        # cluster_id -> set of shards with a readable replica.
        self.replicas: Dict[str, Set[str]] = {
            cid: {sid} for cid, sid in workload.current_assignment.items()
        }
        # Cluster with no current placement is treated as not yet loaded.
        self.routing: Dict[str, str] = dict(workload.current_assignment)
        self.log: List[str] = []

    def _check_invariant(self) -> None:
        for cid in self.workload.clusters:
            if cid in self.replicas and not self.replicas[cid]:
                raise ReplicaInvariantError(
                    f"cluster {cid} has no readable replica"
                )

    def run(self, plan: Plan, fail_move: Optional[str] = None) -> Plan:
        """Execute moves in order.  ``fail_move`` names a cluster whose move
        fails during SYNC, exercising the rollback path."""
        for move in plan.moves:
            self._apply_move(move, fail=move.cluster_id == fail_move)
            self._check_invariant()
        return plan

    def _apply_move(self, move: MoveOp, fail: bool) -> None:
        cid = move.cluster_id
        self.replicas.setdefault(cid, {move.source})
        if fail:
            move.fail()
            # Rollback: discard any partial target copy, keep the source.
            self.replicas[cid].discard(move.target)
            self.replicas[cid].add(move.source)
            self.log.append(f"{cid}: rollback, source {move.source} kept")
            return
        for phase in move.phases:
            if phase is MovePhase.COPY:
                # Target now holds a full copy but is not yet serving.
                self.replicas[cid].add(move.target)
            elif phase is MovePhase.SYNC:
                pass  # delta sync + checksum; both replicas readable
            elif phase is MovePhase.CUTOVER:
                self.routing[cid] = move.target
            elif phase is MovePhase.CLEANUP:
                # Only now may the source copy go away.
                if self.routing.get(cid) == move.target:
                    self.replicas[cid].discard(move.source)
            self._check_invariant()
        move.status = MoveStatus.DONE
        self.log.append(f"{cid}: {move.source} -> {move.target} done")
