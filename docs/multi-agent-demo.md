# 多代理演示

以下记录来自一次真实运行（模型 `deepseek-v4.1-flash`，2026-10-16），不是脚本模拟。复现时结果会因模型而异；运行时保证的是投递、调度、隔离和限制，协作方式由代理自己决定。

## 准备

一个有两个缺陷的统计模块：`median` 对偶数长度取错值，`variance` 运算优先级错误且应为样本方差。`SPEC.md` 给出要求，`test_stats.py` 有 2 个失败用例。

```sh
cd demo-workspace
agent "/env create stats-team --template multi-agent --scope workspace"
agent /env limits max_activations_per_agent=5 max_tool_calls_per_agent=30 token_budget_per_agent=250000 --env stats-team
agent --environment stats-team --yes \
  "stats.py 的测试失败了。请团队自主决定如何调查、讨论和分工：找出根因，按 SPEC.md 修复，运行测试验证，并在交付前做一次独立评审。"
```

环境来自 `multi-agent` 模板：researcher（只读 + 联网搜索）、engineer（读写与执行）、critic（只读），以及 6 个内置协作技能。

## 第一轮：分工、依赖与独立评审

主代理复现失败、读取 `team-collaboration` 与 `task-delegation` 技能，权衡后认为任务较小，没有召集 researcher，也没有使用正式协议，而是创建了两个有依赖关系的任务。时间线摘自 `/env log`：

```text
+  34.0s main      task_created         t1 → engineer：修复 stats.py 使其符合 SPEC.md 并通过测试
+  34.0s engineer  activation_started   t1
+  38.3s main      task_created         t2 → critic：独立评审修复，depends_on=[t1]
+  65.0s engineer  task_updated         t1 completed（显式完成）
+  65.0s critic    activation_started   t2（依赖完成的同一时刻被唤醒）
+ 139.3s critic    task_updated         t2 completed：“同意交付”，列出 2 个残余项
+ 158.1s main      decision_recorded    R1：n=1 时 variance 行为保持现状，交由用户决定
```

- engineer 先复现、修改、再用 4000 组随机数据与标准库 `statistics` 对比验证，只改动 `stats.py`。
- critic 只有只读工具，进行了 22 次检查后给出评审结论，并指出自己无法运行测试这一验证缺口。
- 主代理没有直接采信报告：重新运行测试并做了 5000 组差分验证，记录团队决策，最后把唯一需要产品判断的事项（单元素输入的方差）交给用户。

结果：`stats.py` 两处修复，4/4 测试通过；`SPEC.md` 与测试文件未改动。三个代理各自独立调用模型：engineer 1 次激活、14 次工具调用，critic 1 次激活、22 次工具调用。

该轮命令以退出码 1 结束：原有的验收启发式把最终答复中“critic 无法执行测试”一句判定为“受阻”。这是 `task_acceptance.py` 的误判，与团队运行时无关。

## 第二轮：恢复会话后的直接求助与正式会议

新进程中通过标准输入继续同一会话（团队状态跨进程保留，第 1 轮的 t1、t2 和决策仍在）：

```sh
printf '%s\n' "/resume latest" \
  "关于上一轮留下的 R1（variance 在 n=1 时的行为）：请团队权衡可选方案（抛 ValueError、返回 0.0、返回 NaN 等），参考 Python statistics 模块与常见表格软件的做法，形成团队决定并记录理由，然后实现、补充测试并验证。" \
  "/env status" "/env log 80" | agent
```

提示词没有要求使用任何协议。实际过程：

```text
+  40.6s main        task_created        t3 → researcher：取证 n=1 时的先例行为
+ 117.4s researcher  message_sent        request → engineer：自己不能 exec，请对方运行探针
+ 128.1s engineer    message_sent        reply_to=m1：原样贴回探针输出
+ 191.9s researcher  message_sent        request → engineer：追加请求，用 curl 获取官方文档
+ 232.9s engineer    message_sent        reply_to=m3：网络可用，5 个文档地址的结果
+ 248.0s researcher  budget_exhausted    本轮 token 预算耗尽
+ 304.0s researcher  message_sent        （运行时代答）暂时无法回答，请求保留到下一轮
+ 387.8s engineer    activation_started  meeting p1 第 1 轮（main 召集 engineer 与 critic）
+ 387.8s critic      activation_started  meeting p1 第 1 轮（两人并行、独立）
+ 405.9s engineer    budget_exhausted    第 2 轮记为缺席，会议不被阻塞
+ 417.2s critic      activation_finished 第 2 轮：与 engineer 立场一致，可以收敛
+ 442.8s main        decision_recorded   R1：少于两个数据点时抛 ValueError
+ 505.9s main        task_created        t5 → critic：独立复核实现
+ 532.3s critic      task_updated        t5 completed：同意交付
```

- researcher 根据成员的工具差异主动找 engineer 帮忙，整个往来不经过主代理。
- 主代理在需要权衡时召集了正式会议；两人第 1 轮各自给出带证据的立场，第 2 轮 engineer 因预算缺席，会议照常收敛，主持人（main）记录 5 条决策并生成会议产物。
- engineer 预算耗尽后由 main 实现（`stats.py` 守卫、`SPEC.md` 契约一行、3 个新测试），critic 只读复核后同意交付；最终 7/7 测试通过。

本次演示把 `token_budget_per_agent` 设为 25 万，researcher 与 engineer 都触发了限制，运行时自动替它们答复，没有代理无限等待。预算按每次请求的完整输入（含缓存命中部分）加输出计算，因此默认值已提高到 100 万。

## 演示暴露并已修复的问题

真实运行发现了脚本测试没有覆盖的两个问题，均已修复并补充回归测试：

1. 成员把任务标记为 `in_progress` 被当作“已处理”，之后激活失败时任务停留在 `in_progress`，第一轮的调查报告丢失。现在只有结果性状态才视为已处理，失败时的答复会附上该成员最近一次报告。
2. 主代理通知 engineer 任务已作废，却没有取消任务；未关闭的任务会在下一轮再次唤醒成员。主代理提示与 `task-delegation` 技能已说明应使用 `update_task status=cancelled`。
