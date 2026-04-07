# Current Status

## 当前结论

- 第一阶段方法已锁定为 `vanilla CoRe Memory`，第一阶段 benchmark 已锁定为：
  - 主：`PersonaMem`
  - 次：`LongMemEval-S`
  - 未来扩展：`LoCoMo`
- 第二阶段方法已锁定为 `V2.0 structured latent-slot memory`。
- 第一阶段与第二阶段当前并行存在：
  - 第一阶段承担平台、benchmark 与可复现实验闭环
  - 第二阶段承担 latent memory 方法升级与本地训练/评估主线

## 当前状态

- 文档状态：stage-1 truth 已同步；stage-2 真源、设计文档与状态文档已锁定并同步
- 第一阶段代码状态：platform ready + acceptance verifier landed + resumable benchmark runner landed + embedding-backed memory interface landed + Gemini-compatible retry/backoff / pacing / supervisor path landed + real 1-sample benchmark runs verified
- 第一阶段 memory 接入状态：`VanillaCoreMemory` 最小原型已实现并经单测验证，但当前正式 benchmark runner 仍采用 full-context prompt 主链路，尚未把 memory write/read 接入正式推理闭环
- 第一阶段 benchmark 适配状态：PersonaMem / LongMemEval-S 已分别完成默认 provider 路径与 Gemini 路径下的真实 1-sample run；正式 Gemini run 当前停在 PersonaMem `22/589`、LongMemEval-S `19/500`，且当前 key 继续触发 `HTTP 429`
- 第二阶段设计状态：`V2.0` 方法主线、训练数据策略、训练任务、intrinsic metrics、canary 协议、默认 backbone 与输出目录均已在真源文档中锁定
- 第二阶段实现状态：尚未开始代码实现；当前处于“设计完成，待实现 observation / belief / data pipeline”的状态
- 环境状态：repo 内 project-local conda env 与缓存痕迹已清理；环境复现真相为 `environment.yaml` + 默认 conda named env `core_mem`
- 测试状态：当前 `pytest` 通过（31 tests），`scripts/run_experiment.py --verify-only` 输出 `36`，`scripts/verify_stage1_acceptance.py` 返回 `5/7`

## 当前最重要的下一步

- 第一阶段 formal benchmark 仍受外部 provider `HTTP 429` 阻断，但第二阶段不依赖该 blocker 才能继续推进。当前全局 top next action 已切换为：启动第二阶段 `V2.0` 的 observation normalization、belief JSON 目标生成和公开数据集接入骨架。

## 关键约束

- 真源文档为 `docs/requirements.md`
- 第一阶段与第二阶段都必须遵守真源中的分阶段边界
- `benchmark` 是 evaluation source，不是第二阶段 primary training source
- 第二阶段训练策略为 `public-datasets-first, synthetic-minimal`
- 第一阶段 v1 应尽量保留为 naive baseline 参考

## 风险与待定项

- 阿里云兼容接口当前仍返回 `HTTP 403 AllocationQuota.FreeTierOnly`
- 用户批准的 Gemini 路径已实测可用，但当前 key 在 formal benchmark 负载下连续触发 `HTTP 429`
- PersonaMem formal run 当前停在 `22/589`，LongMemEval-S formal run 当前停在 `19/500`
- 第二阶段 `DuLeMon / REALTALK / MS-TOD` 是否纳入主训练集仍待后续实现时根据可用性与清洗成本判断
- 第二阶段 `Optimus-like` 路线当前只锁定为 ablation / 潜在升级，不是主线
- `sentence-transformers` 依赖已经进入 `core_mem` 环境，但真正首次模型权重下载仍需遵守“外部目录修改先确认”的规则来控制缓存位置
