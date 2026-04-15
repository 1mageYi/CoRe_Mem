# V2.1 Long-Run Plan

## 目标

这轮 `v2.1` 长跑不再解决“有没有完整闭环”，而是解决：

- 系统/模型/latent 本体是否真的更强
- learned path 是否真正进入在线主链
- 提升是否在多个 benchmark 上稳定成立
- 整体框架是否更 robust、更可扩展、更可解释

当前主方向是：**更加注重 learned model、本体智能和更好的 latent**，而不是继续把 rule-based reader/writer 微调当作主收益来源。

## 硬约束

- 不做任何兜底/fallback/benchmark-specific shortcut
- benchmark 只作为验证和诊断信号，不作为投机优化对象
- retained 收益必须落在 `online memory -> belief -> answer` 主链
- 文档、verifier、artifact 必须同步更新

## 成功标准

至少同时满足这些条件：

- learned memory 在线主链在 `PersonaMem` 和 `LongMemEval-S` 上都有 fresh current-head 证据
- learned path 的证据不只存在于 `1` 样本，而是扩到 `64/128` 级切片
- non-tiny 训练对在线表现有可证明贡献
- local intrinsic 指标和 online canary 方向一致
- failure analysis 能稳定归因到 `parser / retrieval / belief / answer / provider` 某一层
- 不新增 fallback 或 benchmark-specific shortcut

## 阶段 A：把 learned path 从“存在”推进到“可测”

- 跑 current-head learned-mode `PersonaMem 64`
- 跑 current-head learned-mode `LongMemEval-S 64`
- 固化 learned-mode artifact 命名、summary 和 analysis 输出
- 让 “fresh learned-mode evidence” 变成 verifier 硬门槛

里程碑：

- `PersonaMem 64 learned-mode` fresh artifact
- `LongMemEval-S 64 learned-mode` fresh artifact
- 两者都有 analysis artifact

## 阶段 B：让训练真正服务在线链路

- 优先 learned 化：
  - `query encoder`
  - `slot encoder`
  - retrieval / rerank
  - belief selection / composition
- 避免继续把“复述 JSON”当成唯一训练目标
- 训练配置显式区分 `online_aligned`
- checkpoint 必须能被在线系统加载

里程碑：

- 至少 1 个 learned 子模块进入在线默认主链
- 至少 1 组训练前后 online canary 对比为正
- `trained_eval` 不只是非零，而是和 online 提升相关

## 阶段 C：专项提升 LongMemEval-S

- 做分层 failure analysis
- 查 retrieval miss、belief confusion、latent composition 失效点
- 修对 `LongMemEval-S` 最关键的 `1-2` 个错误源
- 所有改动都必须在 `PersonaMem` 上不过度退化

里程碑：

- `LongMemEval-S` provider/local 指标显著高于当前基线
- analysis 报告稳定指出主错误簇
- 改动在两 benchmark 上都可解释

## 阶段 D：扩大样本，验证 robustness

- `PersonaMem` 从 `64` 扩到 `128`
- `LongMemEval-S` 视预算扩到更大切片
- 对比 current-head 与历史 best
- 验证 learned-memory 提升不是 canary 偶然现象

里程碑：

- `PersonaMem 128 learned-mode` 不明显退化
- `LongMemEval-S` 扩样后趋势保持
- failure taxonomy 在扩大样本后仍成立

## 阶段 E：收口成 v2.1

- 固化默认训练配方
- 固化默认 learned-mode benchmark 配方
- 固化 verifier 套件
- 更新文档为 “robust learned-memory v2.1”
- 明确列出允许和禁止的策略

里程碑：

- 一个默认训练命令
- 一个默认 learned-mode benchmark 命令
- 一套 `v2.1` verifier
- 一份方法说明文档

## 主指标

- 本体指标：
  - `stage2_latent_core_quality_score`
  - `stage2_v21_learned_memory_score`
- 长跑主指标：
  - `stage2_v21_longrun_score`
- 在线指标：
  - learned-mode `PersonaMem 64/128`
  - learned-mode `LongMemEval-S 64`

## 守卫

- 不新增 fallback
- 不新增 benchmark-specific shortcut
- `run_experiment.py --verify-only` 持续通过
- stage-2 tests 持续通过
- current-head 文档与 verifier 同步
- 如果 online 提升但 local/core 指标恶化，不能直接保留

## 自动迭代闭环

每轮 autoresearch 都按这个闭环：

1. 跑 verifier 和 fresh canary
2. 做 failure analysis
3. 只改 `system/model/latent` 主链相关部分
4. 重跑训练或 canary
5. 比较前后结果
6. 只有在“无 fallback/shortcut 且可复现增益”时才 keep
7. 每 `3-5` 轮做一次 milestone review，必要时 pivot
