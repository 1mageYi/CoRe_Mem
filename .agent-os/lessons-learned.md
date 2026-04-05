# Lessons Learned

## Purpose

本文件用于记录失败探索、陷阱、未采用方案及其重试条件。

## Entries

- 2026-04-05:
  - `conda` 在当前机器上直接运行会触发 plugin / CUDA virtual package 的权限异常；后续应优先尝试 `--no-plugins` 或 `CONDA_NO_PLUGINS=true`。
  - Windows 下超大 `apply_patch` 可能触发 `CreateProcessAsUserW failed: 206`；需要将补丁拆成更小批次提交。
  - 在当前机器上，named env 注册依赖用户目录写权限，容易失败；项目内 prefix 环境 `.conda_envs/core_mem` 是更稳定的落地方案。
  - 如果目标执行平台最终是 WSL，就不要继续把 bootstrap 逻辑写死到 Windows；应尽快把仓库收敛成平台中立状态后迁移。
