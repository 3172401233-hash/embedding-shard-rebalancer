"""Data model for the vector shard rebalancer.

The planner works on *clusters* of vectors rather than individual vectors.
A cluster groups vectors that belong to the same centroid; the neighbor
distribution describes, per cluster, how the k-NN graph of its vectors is
spread across other clusters.  Keeping high-affinity clusters on the same
shard reduces cross-shard fan-out at query time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple


@dataclass
class Cluster:
    """A group of vectors assigned to one centroid."""

    cluster_id: str
    centroid: Tuple[float, ...]
    vector_count: int
    # Average bytes per vector (payload + index overhead).  Storage weight of
    # the cluster is vector_count * bytes_per_vector.
    bytes_per_vector: float = 1.0
    # Queries per second whose nearest centroid is this cluster.
    query_rate: float = 0.0

    @property
    def storage_weight(self) -> float:
        return self.vector_count * self.bytes_per_vector


@dataclass
class Shard:
    """A physical shard (node) holding a set of clusters."""

    shard_id: str
    # Maximum storage weight the shard can hold.
    capacity: float
    # Maximum query rate the shard can serve before being "hot".
    query_capacity: float = float("inf")


@dataclass
class Workload:
    """Full planner input."""

    clusters: Dict[str, Cluster]
    shards: Dict[str, Shard]
    # affinity[a][b] = fraction of cluster a's k-NN edges landing in cluster b.
    # Rows need not sum to 1 (self edges may be omitted); treated as weights.
    affinity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Current placement: cluster_id -> shard_id.
    current_assignment: Dict[str, str] = field(default_factory=dict)

    def affinity_between(self, a: str, b: str) -> float:
        w = 0.0
        w += self.affinity.get(a, {}).get(b, 0.0)
        w += self.affinity.get(b, {}).get(a, 0.0)
        return w


class MovePhase(str, Enum):
    """Phases of a single cluster migration.

    The source replica is only deleted in CLEANUP, after CUTOVER has flipped
    routing to the target.  Until then the source keeps serving reads, so
    every vector always has at least one readable replica.
    """

    COPY = "copy"            # target pulls a snapshot; source still serves
    SYNC = "sync"            # delta sync + checksum verification
    CUTOVER = "cutover"      # routing table flips to the target replica
    CLEANUP = "cleanup"      # source replica dropped after grace period
    ROLLBACK = "rollback"    # failure: drop partial target copy, keep source


class MoveStatus(str, Enum):
    PLANNED = "planned"
    DONE = "done"
    FAILED = "failed"


@dataclass
class MoveOp:
    """Migration of one cluster from one shard to another."""

    cluster_id: str
    source: str
    target: str
    phases: List[MovePhase] = field(
        default_factory=lambda: [
            MovePhase.COPY,
            MovePhase.SYNC,
            MovePhase.CUTOVER,
            MovePhase.CLEANUP,
        ]
    )
    status: MoveStatus = MoveStatus.PLANNED
    # Bytes moved (vector_count * bytes_per_vector of the cluster).
    bytes_moved: float = 0.0

    def fail(self) -> None:
        """Mark the migration failed; the safe rollback keeps the source."""
        self.status = MoveStatus.FAILED
        self.phases = [MovePhase.ROLLBACK]


@dataclass
class Plan:
    """Planner output."""

    assignment: Dict[str, str]                 # cluster_id -> target shard_id
    moves: List[MoveOp]                        # ordered migration steps
    query_amplification: float                 # est. shards touched per query
    movement_bytes: float                      # total bytes to copy
    moved_clusters: int
    shard_loads: Dict[str, float] = field(default_factory=dict)
    shard_query_loads: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"moves={len(self.moves)} bytes={self.movement_bytes:.0f} "
            f"query_amplification={self.query_amplification:.3f}"
        ]
        for sid in sorted(self.shard_loads):
            lines.append(
                f"  {sid}: storage={self.shard_loads[sid]:.0f} "
                f"qps={self.shard_query_loads.get(sid, 0.0):.1f}"
            )
        return "\n".join(lines)
