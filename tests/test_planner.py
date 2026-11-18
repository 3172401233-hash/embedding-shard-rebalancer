import unittest

from shard_rebalancer import (
    Cluster,
    Executor,
    MovePhase,
    MoveStatus,
    PlanningError,
    Shard,
    ShardPlanner,
    Workload,
)


def make_cluster(cid, count=100, qps=10.0, bpv=1.0):
    return Cluster(
        cluster_id=cid,
        centroid=(0.0, 0.0),
        vector_count=count,
        bytes_per_vector=bpv,
        query_rate=qps,
    )


def make_workload(cluster_specs, shard_specs, affinity=None, assignment=None):
    clusters = {cid: make_cluster(cid, **spec) for cid, spec in cluster_specs.items()}
    shards = {sid: Shard(shard_id=sid, **spec) for sid, spec in shard_specs.items()}
    return Workload(
        clusters=clusters,
        shards=shards,
        affinity=affinity or {},
        current_assignment=assignment or {},
    )


class HotShardTest(unittest.TestCase):
    """One shard exceeds capacity and query budget; planner must drain it."""

    def test_hot_shard_is_drained(self):
        wl = make_workload(
            cluster_specs={
                "c1": {"count": 400, "qps": 80.0},
                "c2": {"count": 400, "qps": 70.0},
                "c3": {"count": 100, "qps": 10.0},
                "c4": {"count": 100, "qps": 10.0},
            },
            shard_specs={
                "s1": {"capacity": 600, "query_capacity": 100.0},
                "s2": {"capacity": 600, "query_capacity": 100.0},
            },
            assignment={"c1": "s1", "c2": "s1", "c3": "s1", "c4": "s2"},
        )
        plan = ShardPlanner().plan(wl)
        for sid, load in plan.shard_loads.items():
            self.assertLessEqual(load, wl.shards[sid].capacity + 1e-9)
        for sid, qps in plan.shard_query_loads.items():
            self.assertLessEqual(qps, wl.shards[sid].query_capacity + 1e-9)
        self.assertGreater(plan.moved_clusters, 0)
        self.assertGreater(plan.movement_bytes, 0)

    def test_infeasible_capacity_raises(self):
        wl = make_workload(
            cluster_specs={"c1": {"count": 1000}},
            shard_specs={"s1": {"capacity": 100}},
            assignment={"c1": "s1"},
        )
        with self.assertRaises(PlanningError):
            ShardPlanner().plan(wl)


class EmptyShardTest(unittest.TestCase):
    """An empty shard should absorb load from overloaded peers."""

    def test_empty_shard_gets_filled(self):
        wl = make_workload(
            cluster_specs={
                "c1": {"count": 300, "qps": 5.0},
                "c2": {"count": 300, "qps": 5.0},
                "c3": {"count": 300, "qps": 5.0},
            },
            shard_specs={
                "s1": {"capacity": 600},
                "s2": {"capacity": 600},
            },
            assignment={"c1": "s1", "c2": "s1", "c3": "s1"},
        )
        plan = ShardPlanner().plan(wl)
        self.assertGreater(plan.shard_loads["s2"], 0)
        self.assertLessEqual(plan.shard_loads["s1"], 600)
        # Exactly the overflow (300) needs to move, not more.
        self.assertEqual(plan.movement_bytes, 300)


class NewNodeTest(unittest.TestCase):
    """A newly joined node starts empty and should take over hot clusters."""

    def test_new_node_absorbs_load(self):
        wl = make_workload(
            cluster_specs={
                f"c{i}": {"count": 100, "qps": 50.0} for i in range(6)
            },
            shard_specs={
                "s1": {"capacity": 400, "query_capacity": 150.0},
                "s2": {"capacity": 400, "query_capacity": 150.0},
                "s3": {"capacity": 400, "query_capacity": 150.0},  # new node
            },
            assignment={f"c{i}": ("s1" if i < 3 else "s2") for i in range(6)},
        )
        plan = ShardPlanner().plan(wl)
        self.assertGreater(plan.shard_loads["s3"], 0)
        for sid, qps in plan.shard_query_loads.items():
            self.assertLessEqual(qps, wl.shards[sid].query_capacity + 1e-9)


class CentroidDriftTest(unittest.TestCase):
    """After centroids drift, affinity changes and the plan should re-cluster."""

    def test_drift_triggers_regrouping(self):
        # Initially c1<->c2 are neighbours on s1, c3<->c4 on s2.
        assignment = {"c1": "s1", "c2": "s1", "c3": "s2", "c4": "s2"}
        shards = {"s1": {"capacity": 500}, "s2": {"capacity": 500}}
        specs = {f"c{i}": {"count": 100, "qps": 1.0} for i in range(1, 5)}
        before = make_workload(
            specs, shards,
            affinity={"c1": {"c2": 1.0}, "c3": {"c4": 1.0}},
            assignment=assignment,
        )
        plan_before = ShardPlanner().plan(before)
        # Nothing should move: neighbours are already co-located.
        self.assertEqual(plan_before.moved_clusters, 0)

        # Drift: c2 is now nearest to c3's centroid region.
        after = make_workload(
            specs, shards,
            affinity={"c1": {"c4": 1.0}, "c2": {"c3": 1.0}},
            assignment=assignment,
        )
        plan_after = ShardPlanner(affinity_weight=5.0).plan(after)
        self.assertGreater(plan_after.moved_clusters, 0)
        self.assertLessEqual(
            plan_after.query_amplification,
            ShardPlanner().estimate_query_amplification(after, assignment),
        )


class FailedMigrationTest(unittest.TestCase):
    """A failed move must roll back without losing the readable replica."""

    def _workload(self):
        return make_workload(
            cluster_specs={
                "c1": {"count": 300, "qps": 5.0},
                "c2": {"count": 300, "qps": 5.0},
            },
            shard_specs={
                "s1": {"capacity": 400},
                "s2": {"capacity": 400},
            },
            assignment={"c1": "s1", "c2": "s1"},
        )

    def test_rollback_keeps_source_replica(self):
        wl = self._workload()
        planner = ShardPlanner()
        plan = planner.plan(wl)
        self.assertEqual(len(plan.moves), 1)
        move = plan.moves[0]
        ex = Executor(wl)
        ex.run(plan, fail_move=move.cluster_id)
        self.assertEqual(move.status, MoveStatus.FAILED)
        self.assertEqual(move.phases, [MovePhase.ROLLBACK])
        # Source replica is still the only readable copy.
        self.assertEqual(ex.replicas[move.cluster_id], {move.source})
        self.assertEqual(ex.routing[move.cluster_id], move.source)

    def test_replan_after_failure_reroutes(self):
        wl = make_workload(
            cluster_specs={
                "c1": {"count": 300, "qps": 5.0},
                "c2": {"count": 300, "qps": 5.0},
            },
            shard_specs={
                "s1": {"capacity": 400},
                "s2": {"capacity": 400},
                "s3": {"capacity": 400},
            },
            assignment={"c1": "s1", "c2": "s1"},
        )
        planner = ShardPlanner()
        plan = planner.plan(wl)
        move = plan.moves[0]
        move.fail()
        new_plan = planner.replan_after_failure(wl, plan, move)
        # The failed cluster must not be sent back to the failed target.
        for m in new_plan.moves:
            if m.cluster_id == move.cluster_id:
                self.assertNotEqual(m.target, move.target)
        for sid, load in new_plan.shard_loads.items():
            self.assertLessEqual(load, wl.shards[sid].capacity + 1e-9)


class ReadabilityInvariantTest(unittest.TestCase):
    """Throughout a normal migration every cluster keeps a readable replica."""

    def test_invariant_holds_during_cutover(self):
        wl = make_workload(
            cluster_specs={f"c{i}": {"count": 100, "qps": 2.0} for i in range(8)},
            shard_specs={
                "s1": {"capacity": 500},
                "s2": {"capacity": 500},
                "s3": {"capacity": 500},
            },
            assignment={f"c{i}": "s1" for i in range(8)},
        )
        plan = ShardPlanner().plan(wl)
        ex = Executor(wl)
        ex.run(plan)  # raises ReplicaInvariantError if violated
        for move in plan.moves:
            self.assertEqual(move.status, MoveStatus.DONE)
            self.assertEqual(
                move.phases,
                [MovePhase.COPY, MovePhase.SYNC, MovePhase.CUTOVER, MovePhase.CLEANUP],
            )
        # Routing converged to the planned assignment.
        for cid, sid in plan.assignment.items():
            self.assertEqual(ex.routing[cid], sid)


class QueryAmplificationTest(unittest.TestCase):
    """Co-locating neighbour clusters lowers the fan-out estimate."""

    def test_colocation_reduces_amplification(self):
        specs = {f"c{i}": {"count": 100, "qps": 10.0} for i in range(4)}
        shards = {f"s{i}": {"capacity": 300} for i in range(2)}
        affinity = {"c0": {"c1": 0.9}, "c2": {"c3": 0.9}}
        scattered = {"c0": "s0", "c1": "s1", "c2": "s0", "c3": "s1"}
        grouped = {"c0": "s0", "c1": "s0", "c2": "s1", "c3": "s1"}
        wl = make_workload(specs, shards, affinity=affinity, assignment=scattered)
        planner = ShardPlanner()
        amp_scattered = planner.estimate_query_amplification(wl, scattered)
        amp_grouped = planner.estimate_query_amplification(wl, grouped)
        self.assertGreater(amp_scattered, amp_grouped)
        self.assertAlmostEqual(amp_grouped, 1.0)


if __name__ == "__main__":
    unittest.main()
