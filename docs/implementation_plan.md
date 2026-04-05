# Implementation Plan

## 第一阶段总策略

第一阶段只实现 `vanilla CoRe Memory` 的最小完整闭环，并优先确保：

1. 文档系统可恢复
2. 环境固定
3. Benchmark adapter 可运行
4. 方法主干可运行
5. 评测流程可运行
6. 输出结果可追溯
7. 测试可防回归

## 阶段拆分

### 阶段 A：项目初始化

- 建立 `AGENTS.md` 与文档系统
- 固定目录结构
- 固定环境约束

### 阶段 B：基础工程骨架

- 环境创建与依赖管理
- 配置系统
- API provider adapter
- 输出目录与结果记录规范

### 阶段 C：Benchmark adapter

- PersonaMem 32k 主任务接入
- PersonaMem 128k / 1M 启动开关预留
- LongMemEval-S 接入
- 官方协议对齐

### 阶段 D：CoRe Memory 方法实现

- writer
- core updater
- residual manager
- reader
- evaluation / output pipeline

### 阶段 E：测试与复现

- 核心模块 unit tests
- 最小 E2E smoke test
- 总入口与分步骤脚本

### 阶段 F：正式运行与结果整理

- PersonaMem 正式结果
- LongMemEval-S 正式结果
- 表格与结果数据沉淀

## 第一阶段不做

- baseline 整合
- ablation
- spectral extension
- 重型训练
- 报告写作打磨
