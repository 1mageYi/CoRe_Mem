# Graph Memory TODO

## Phase 0: 文档与基线准备
- [x] 固化 benchmark 顺序：PERMA -> PersonaMem（最终） -> LoCoMo（可选）
- [x] 写清 baseline 对照：semantic-only、旧 core/residual、mem0-like
- [x] 定义统一指标看板（准确率、延迟、记忆覆盖、图统计）

## Phase 1: Graph MVP（PERMA）
- [x] 定义 node/edge schema 与存储接口
- [x] 实现 non-LLM extractor（规则主干）
- [x] 实现 Add 流程（dedup + create + merge）
- [x] 实现三类 edge 构建（semantic/temporal/co-usage）
- [x] 实现 Search 流程（seed + expand + rerank）
- [x] 实现 prompt builder（可解释 evidence 输出）
- [x] 增加 PERMA 风格 smoke runner（最小可运行）
- [x] 增加真实 PERMA mini eval（graph-full vs semantic-only）
- [x] 增加 mini e2e 回归测试文件（真实 PERMA 样本，limit=2）

## Phase 2: Core/Residual 自分化
- [ ] 引入 PageRank/入度/使用频次计算 core_score
- [ ] 实现 core prior 融合打分
- [ ] 完成 lambda_core 网格搜索
- [ ] 对比“无 prior vs 有 prior”

## Phase 3: PERMA 消融与稳健性
- [ ] edge 消融（semantic-only / +temporal / +co-usage / full）
- [ ] merge 策略消融（latest-wins vs multi-version）
- [ ] co-usage 污染控制（最小共现阈值 + 衰减）
- [ ] temporal probing 分段评测（早期/中期/后期）
- [ ] 失败剖析：解释 user109 上 graph-full 相比 semantic-only 的退化来源
- [ ] 加入 per-sample debug 导出（seed/expanded/top evidence/score 分解）

## Phase 4: PersonaMem 最终迁移验证
- [ ] 适配 PersonaMem 数据入口
- [ ] 复用同一 memory graph 流程执行评测
- [ ] 对比旧方案与图方案增益/退化点
- [ ] 输出最终对照报告

## Phase 5: 可选 LoCoMo 压测
- [ ] 超长上下文下时延与内存占用评估
- [ ] 时序/因果相关任务的图扩展收益分析

## 交付物检查
- [ ] docs 完整（requirements / implementation_plan / todo / current_status）
- [ ] 实验配置可复现（参数、随机种子、run 命名）
- [ ] 结论含失败分析，不只报最好结果
