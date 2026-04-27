# Project Structure (Living Doc)

## 目的
这是项目的“整体框架总览文档”。  
要求：一目了然、少细节、可持续更新。  
当实现方案变化时，优先更新本文件，让代码结构与系统流程保持一致。

## 1) 端到端主链路
1. `Raw Conversation` 输入
2. `Extractor` 抽取结构化记忆片段
3. `Embedding` 生成结构化片段向量
4. `Node Builder` 判定 merge/new，写入 memory node
5. `Edge Builder` 构建 semantic / temporal / co-usage 边
6. `Retriever` 用 query 召回 + 图扩展 + 融合重排
7. `Prompt Builder` 组装最终 memory context
8. `LLM Answer` 输出回答
9. `Post-Query Update` 更新 co-usage 边与统计信息

## 2) 模块框架（高层）
- `Extractor`：把对话转成结构化事件/偏好记录（非 LLM）
- `Embedder`：对结构化记录做 embedding
- `Graph Store`：存 node/edge/state/version
- `Add Pipeline`：写入流程（dedup、merge、new、edge维护）
- `Search Pipeline`：检索流程（seed、expand、rerank）
- `Core/Residual Layering`：基于中心性动态分层
- `Evaluation`：PERMA 主测，PersonaMem 最终验证
  - `PERMA Loader`：真实数据读取与 task/eval 对齐
  - `Eval Runner`：graph-full vs semantic-only 对照评测
  - `Mini E2E Test`：真实 PERMA 小样本回归守门

## 2.1 代码入口（当前）
- 训练/评测入口：
  - `train/perma/eval_perma_graph_vs_semantic.py`
  - `train/perma/smoke_graph_mvp.py`
- 可复用评测核心：
  - `src/graph_mem/perma_eval.py`
- 测试入口：
  - `tests/graph_mem/test_smoke_pipeline.py`
  - `tests/graph_mem/test_e2e_perma_mini.py`

## 3) Node / Edge（抽象定义）
### Node（记忆节点）
- 结构化文本内容
- embedding
- 时间与来源信息
- 状态信息（active/inactive/superseded/conflicted）

### Edge（三类）
- `semantic`：语义相似，双向
- `temporal`：时间前后，单向（old -> new）
- `co-usage`：同次 query 共用且原无边，双向

## 4) Retrieval 总览
- Step A：query 语义召回 seed nodes
- Step B：图扩展补全（按 edge 类型控制范围）
- Step C：融合打分（semantic + centrality + temporal/edge evidence）
- Step D：选 top evidence，构建 prompt

## 5) 当前留空位（待实现后填充）
- Extractor 具体实现方案：`RuleExtractor (MVP)`，后续补轻量模型兜底
- Embedding 模型最终选型与版本：`BAAI/bge-m3`（已定）
- 冲突解算的最终规则（v1）：`Hybrid`（compatible merge, revision new-version）
- 参数搜索后的稳定配置：`TBD`

## 6) 更新规则
- 架构路径变化：必须更新本文件。
- 模块新增/合并：必须更新模块框架与主链路。
- 保持“高层可读”，不要塞实现细节和实验日志。
