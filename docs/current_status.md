# Current Status

## 当前结论

- Write path 核心模块已全部实现并通过测试。
- 合并机制确定为在线质心（Online Centroid），recency_weight=1.5。
- Embedding model 确定为 `sentence-transformers/all-MiniLM-L6-v2`（384 维）。
- Reader 和 Writer fact 提取逻辑尚未实现。

## 当前状态

- 文档状态：done
- 目录与工程骨架：done
- 环境依赖文件：done（`pyproject.toml` + `requirements.txt`，venv Python 3.10）
- 代码状态：
  - `slot.py`：MemorySlot 数据结构 + 向量工具函数 ✓
  - `embedding.py`：MiniLM-L6-v2 全局单例封装 ✓
  - `config.py`：EmbeddingSettings + MemorySettings（含 merge/promotion 参数）✓
  - `residual_manager.py`：residual 存储、合并、驱逐、promotion 判断 ✓
  - `core_updater.py`：core 存储、合并、强制合并驱逐 ✓
  - `writer.py`：write path 入口（text delta → encode → residual → core promotion）✓
  - `reader.py`：skeleton only（未实现）
  - `benchmarks/*`：skeleton only（未实现）
  - `evaluation/*`：skeleton only（未实现）
- benchmark 适配状态：not started
- 测试状态：28 个 unit tests 全部通过
- 持久化：safetensors + JSON，save/load 已实现并通过测试

## 当前最重要的下一步

- Writer fact 提取逻辑（`TD-012`）：当前 writer 接收已有 text/embedding，后续需接入 LLM 做 prompt-based fact extraction。
- Reader 实现（`TD-013`）：query → top-k core + top-j residual 检索 + vec2text 解码。

## 已完成的设计决策

| 决策 | 结论 |
|------|------|
| Embedding model | `all-MiniLM-L6-v2`（384 维） |
| Core 结构 | 多 slot（32 个） |
| Residual 结构 | 64 个 slot |
| 合并机制 | 在线质心（Running Mean），recency_weight=1.5 |
| Residual 驱逐 | merge_count 最低优先 |
| Core 驱逐 | 强制合并进 cos_sim 最近的 core slot |
| Promotion 条件 | merge_count >= 5 且 stability_window=3 内变化 < epsilon=0.05 |
| 持久化格式 | safetensors (embeddings) + JSON (metadata) |
| Text shadow | 不存储（后续用 vec2text 解码） |

## 关键约束

- 真源文档为 `docs/requirements.md`
- 第一阶段不跑 baseline
- 第一阶段接受 negative result
- 第一阶段重点是系统完整、实验可运行、结论可信

## 风险与待定项

- Writer fact extraction 方法待实现（LLM prompt-based）
- Reader + vec2text 设计待定
- PersonaMem / LongMemEval-S benchmark adapter 未开始
