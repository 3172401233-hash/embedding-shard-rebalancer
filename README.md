# embedding-shard-rebalancer

Python 向量分片再平衡库：输入 shard 容量、中心向量（聚簇）、近邻分布、查询负载和当前副本放置，生成分片迁移计划。目标是在均衡存储/查询负载的同时尽量保留近邻聚簇共置，并保证迁移过程中每个向量始终至少有一个可读副本。

## 快速开始

```python
from shard_rebalancer import Cluster, Shard, ShardPlanner, Workload, Executor

workload = Workload(
    clusters={
        "c1": Cluster("c1", centroid=(0.1, 0.2), vector_count=400, query_rate=80.0),
        "c2": Cluster("c2", centroid=(0.9, 0.8), vector_count=300, query_rate=20.0),
    },
    shards={
        "s1": Shard("s1", capacity=500, query_capacity=100.0),
        "s2": Shard("s2", capacity=500, query_capacity=100.0),
    },
    affinity={"c1": {"c2": 0.6}},          # c1 的 k-NN 有 60% 落在 c2
    current_assignment={"c1": "s1", "c2": "s1"},
)

plan = ShardPlanner().plan(workload)
print(plan.summary())        # 移动量、查询放大估算、各 shard 终态负载
for move in plan.moves:
    print(move.cluster_id, move.source, "->", move.target, move.phases)

Executor(workload).run(plan)  # 模拟执行，校验副本不变量
```

运行测试：

```bash
python -m unittest
```

## 近似步骤

联合优化（负载均衡 + 近邻共置 + 最小迁移量）是 NP-hard，规划器采用贪心 + 局部搜索的近似：

1. **保留现状**：以当前放置为初始解，未受影响的聚簇不产生迁移。
2. **容量修复**：反复从利用率最高的超载 shard 上，把迁移代价最小（存储权重最小）的聚簇移到"与其近邻共置度最高且仍有剩余容量/查询余量"的 shard，直到无超载；无法修复时抛出 `PlanningError`。
3. **局部搜索精炼**：在有限轮数内尝试单聚簇移动，按加权代价函数择优接受：
   `代价 = w1·利用率方差(存储+QPS) + w2·跨 shard 近邻割边 + w3·迁移字节占比`。
   方差项保证空 shard / 新节点会被均匀填充；割边项是查询放大的代理。
4. **生成分阶段迁移**：每个移动按 `copy → sync → cutover → cleanup` 排序执行——目标副本校验通过、路由切换后才删除源副本，因此任何时刻每个向量至少有一个可读副本；失败时走 `rollback`，丢弃目标半成品、保留源副本，并可用 `replan_after_failure` 将该聚簇改投其他 shard。

## 输出指标

- **移动量**：`plan.movement_bytes`（总迁移字节）与 `plan.moved_clusters`。
- **查询放大估算**：`plan.query_amplification`，按查询速率加权的"每次查询期望触及的 shard 数"（主聚簇所在 shard + 承载其近邻边的其他 shard）。
- **切换阶段**：每个 `MoveOp.phases` 列出 copy/sync/cutover/cleanup（或 rollback）。

## 测试覆盖

`tests/test_planner.py` 覆盖：热点 shard 排空、空 shard 填充、新节点加入、中心漂移后重聚簇、失败迁移回滚与重规划、迁移全程可读副本不变量、共置降低查询放大。
