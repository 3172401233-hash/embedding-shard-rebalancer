"""Greedy + local-search shard planner.

Approximation strategy (the joint balance / co-location problem is NP-hard):

1.  Seed the assignment with the current placement so unchanged clusters
    never move.
2.  Repair capacity/empty-shard violations greedily: repeatedly move the
    cheapest cluster off the most overloaded shard onto the shard with the
    most free room and highest affinity to what the cluster talks to.
3.  Refine with a bounded local search that tries single-cluster moves and
    accepts them when a weighted cost (load imbalance + cross-shard affinity
    cut + movement) improves.
4.  Emit a phased move plan (copy -> sync -> cutover -> cleanup) so the
    source replica keeps serving reads until the target is verified.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .model import MoveOp, Plan, Workload


class PlanningError(Exception):
    """Raised when no feasible placement exists (e.g. capacity too small)."""


class ShardPlanner:
    def __init__(
        self,
        imbalance_weight: float = 1.0,
        affinity_weight: float = 1.0,
        movement_weight: float = 0.05,
        max_refine_rounds: int = 50,
    ) -> None:
        self.imbalance_weight = imbalance_weight
        self.affinity_weight = affinity_weight
        self.movement_weight = movement_weight
        self.max_refine_rounds = max_refine_rounds

    # ------------------------------------------------------------------ API

    def plan(self, workload: Workload) -> Plan:
        self._validate(workload)
        assignment = dict(workload.current_assignment)
        for cid in workload.clusters:
            assignment.setdefault(cid, self._least_loaded_shard(workload, assignment))

        assignment = self._repair_capacity(workload, assignment)
        assignment = self._refine(workload, assignment)
        return self._build_plan(workload, assignment)

    def replan_after_failure(self, workload: Workload, plan: Plan,
                             failed: MoveOp) -> Plan:
        """Re-plan treating the failed move's target as unusable for that
        cluster.  The source replica is untouched by the rollback, so reads
        were never interrupted; we simply route the cluster elsewhere."""
        remaining = dict(plan.assignment)
        remaining[failed.cluster_id] = failed.source
        # Try every shard except the failed target, pick the cheapest legal
        # destination by incremental cost.
        best: Optional[Tuple[float, str]] = None
        for sid in workload.shards:
            if sid == failed.target or sid == failed.source:
                continue
            candidate = dict(remaining)
            candidate[failed.cluster_id] = sid
            if not self._fits(workload, candidate):
                continue
            cost = self._cost(workload, candidate, workload.current_assignment)
            if best is None or cost < best[0]:
                best = (cost, sid)
        if best is None:
            # No alternative: keep the cluster where it is.
            new_assignment = remaining
        else:
            new_assignment = dict(remaining)
            new_assignment[failed.cluster_id] = best[1]
        new_assignment = self._repair_capacity(workload, new_assignment)
        new_assignment = self._refine(workload, new_assignment)
        return self._build_plan(workload, new_assignment)

    # ------------------------------------------------------------- validate

    def _validate(self, workload: Workload) -> None:
        if not workload.shards:
            raise PlanningError("no shards provided")
        total_cap = sum(s.capacity for s in workload.shards.values())
        total_need = sum(c.storage_weight for c in workload.clusters.values())
        if total_need > total_cap:
            raise PlanningError(
                f"total storage {total_need:.0f} exceeds capacity {total_cap:.0f}"
            )
        for cid, sid in workload.current_assignment.items():
            if cid not in workload.clusters:
                raise PlanningError(f"assignment references unknown cluster {cid}")
            if sid not in workload.shards:
                raise PlanningError(f"assignment references unknown shard {sid}")

    # ----------------------------------------------------------- load utils

    def _loads(self, workload: Workload,
               assignment: Dict[str, str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        storage = {sid: 0.0 for sid in workload.shards}
        qps = {sid: 0.0 for sid in workload.shards}
        for cid, sid in assignment.items():
            c = workload.clusters[cid]
            storage[sid] += c.storage_weight
            qps[sid] += c.query_rate
        return storage, qps

    def _fits(self, workload: Workload, assignment: Dict[str, str]) -> bool:
        storage, _ = self._loads(workload, assignment)
        return all(
            storage[sid] <= workload.shards[sid].capacity + 1e-9
            for sid in storage
        )

    def _least_loaded_shard(self, workload: Workload,
                            assignment: Dict[str, str]) -> str:
        storage, _ = self._loads(workload, assignment)
        return min(
            workload.shards,
            key=lambda s: storage[s] / max(workload.shards[s].capacity, 1e-9),
        )

    # ---------------------------------------------------- capacity repair

    def _repair_capacity(self, workload: Workload,
                         assignment: Dict[str, str]) -> Dict[str, str]:
        assignment = dict(assignment)
        for _ in range(10 * max(1, len(assignment))):
            storage, _ = self._loads(workload, assignment)
            over = [
                sid for sid in workload.shards
                if storage[sid] > workload.shards[sid].capacity + 1e-9
            ]
            if not over:
                return assignment
            src = max(over, key=lambda s: storage[s] - workload.shards[s].capacity)
            # Candidate clusters on the overloaded shard, cheapest to move
            # first (small clusters disturb less data).
            members = [cid for cid, sid in assignment.items() if sid == src]
            members.sort(key=lambda c: workload.clusters[c].storage_weight)
            moved = False
            for cid in members:
                target = self._best_target(workload, assignment, cid, exclude=src)
                if target is not None:
                    assignment[cid] = target
                    moved = True
                    break
            if not moved:
                raise PlanningError(
                    f"cannot relieve overloaded shard {src}: no legal target"
                )
        raise PlanningError("capacity repair did not converge")

    def _best_target(self, workload: Workload, assignment: Dict[str, str],
                     cid: str, exclude: Optional[str] = None) -> Optional[str]:
        cluster = workload.clusters[cid]
        storage, qps = self._loads(workload, assignment)
        best: Optional[Tuple[float, str]] = None
        for sid, shard in workload.shards.items():
            if sid == exclude or sid == assignment.get(cid):
                continue
            if storage[sid] + cluster.storage_weight > shard.capacity + 1e-9:
                continue
            if qps[sid] + cluster.query_rate > shard.query_capacity + 1e-9:
                continue
            # Prefer shards that already hold clusters this cluster is
            # affinity-linked to, then shards with more free room.
            co_located = sum(
                workload.affinity_between(cid, other)
                for other, osid in assignment.items()
                if osid == sid and other != cid
            )
            free_ratio = 1.0 - storage[sid] / max(shard.capacity, 1e-9)
            score = co_located + 0.1 * free_ratio
            if best is None or score > best[0]:
                best = (score, sid)
        return best[1] if best else None

    # --------------------------------------------------------- local search

    def _cost(self, workload: Workload, assignment: Dict[str, str],
              reference: Dict[str, str]) -> float:
        storage, qps = self._loads(workload, assignment)
        # Normalised imbalance: variance of utilisation across shards.  A
        # variance term (unlike max-minus-mean) strictly prefers spreading
        # load onto empty/new nodes.
        utils = [
            storage[sid] / max(workload.shards[sid].capacity, 1e-9)
            for sid in workload.shards
        ]
        mean_u = sum(utils) / len(utils)
        imbalance = sum((u - mean_u) ** 2 for u in utils) / len(utils)
        # Query hot-spotting.
        qutils = [
            qps[sid] / max(workload.shards[sid].query_capacity, 1e-9)
            for sid in workload.shards
            if math.isfinite(workload.shards[sid].query_capacity)
        ]
        if qutils:
            mean_q = sum(qutils) / len(qutils)
            imbalance += sum((q - mean_q) ** 2 for q in qutils) / len(qutils)
        # Cross-shard affinity cut (query amplification proxy).
        cut = 0.0
        for a, row in workload.affinity.items():
            for b, w in row.items():
                if a in assignment and b in assignment and assignment[a] != assignment[b]:
                    cut += w
        # Movement churn vs the reference placement.
        moved = sum(
            workload.clusters[cid].storage_weight
            for cid in assignment
            if reference.get(cid) != assignment[cid]
        )
        total_storage = max(
            sum(c.storage_weight for c in workload.clusters.values()), 1e-9
        )
        return (
            self.imbalance_weight * imbalance
            + self.affinity_weight * cut
            + self.movement_weight * moved / total_storage
        )

    def _refine(self, workload: Workload,
                assignment: Dict[str, str]) -> Dict[str, str]:
        assignment = dict(assignment)
        reference = dict(workload.current_assignment)
        best_cost = self._cost(workload, assignment, reference)
        cids = sorted(workload.clusters)
        for _ in range(self.max_refine_rounds):
            improved = False
            for cid in cids:
                src = assignment[cid]
                for sid in workload.shards:
                    if sid == src:
                        continue
                    candidate = dict(assignment)
                    candidate[cid] = sid
                    if not self._fits(workload, candidate):
                        continue
                    cost = self._cost(workload, candidate, reference)
                    if cost < best_cost - 1e-9:
                        assignment = candidate
                        best_cost = cost
                        improved = True
                        src = sid
            if not improved:
                break
        return assignment

    # ---------------------------------------------------------- plan output

    def _build_plan(self, workload: Workload,
                    assignment: Dict[str, str]) -> Plan:
        moves: List[MoveOp] = []
        for cid in sorted(assignment):
            src = workload.current_assignment.get(cid)
            dst = assignment[cid]
            if src is not None and src != dst:
                moves.append(
                    MoveOp(
                        cluster_id=cid,
                        source=src,
                        target=dst,
                        bytes_moved=workload.clusters[cid].storage_weight,
                    )
                )
        # Order moves so that shards receiving data free up first: emit moves
        # out of the most overloaded shards before moves into them.
        storage, qps = self._loads(workload, workload.current_assignment)
        moves.sort(
            key=lambda m: storage.get(m.source, 0.0)
            / max(workload.shards[m.source].capacity, 1e-9),
            reverse=True,
        )
        final_storage, final_qps = self._loads(workload, assignment)
        return Plan(
            assignment=assignment,
            moves=moves,
            query_amplification=self.estimate_query_amplification(workload, assignment),
            movement_bytes=sum(m.bytes_moved for m in moves),
            moved_clusters=len(moves),
            shard_loads=final_storage,
            shard_query_loads=final_qps,
        )

    # ------------------------------------------------------------- metrics

    def estimate_query_amplification(self, workload: Workload,
                                     assignment: Dict[str, str]) -> float:
        """Expected number of shards a query must fan out to.

        A query lands on the shard of its nearest centroid (1 shard), plus
        every *other* shard holding clusters that carry a meaningful share of
        that cluster's k-NN edges.
        """
        total_qps = sum(c.query_rate for c in workload.clusters.values())
        if total_qps <= 0:
            total_qps = float(len(workload.clusters)) or 1.0
            weights = {cid: 1.0 for cid in workload.clusters}
        else:
            weights = {cid: c.query_rate for cid, c in workload.clusters.items()}
        expected = 0.0
        for cid, cluster in workload.clusters.items():
            home = assignment.get(cid)
            touched = {home} if home else set()
            for other, w in workload.affinity.get(cid, {}).items():
                if w > 0 and assignment.get(other) not in (None, home):
                    touched.add(assignment[other])
            expected += (weights[cid] / total_qps) * max(1, len(touched))
        return expected
