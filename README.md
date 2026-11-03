# FLARE 备份

这是 Jingzhou Jiang 的 **FLARE 私有备份仓库**，保存论文、实验材料与可重新运行的代码。
备份日期：2026-11-03。目标是恢复完整研究流程，不要求恢复原参数或逐位一致的结果。

## 保存内容

| 位置 | 内容 |
| --- | --- |
| `code/` | 修复后的完整源码、配置、依赖版本、复现说明与小规模验证记录 |
| `metadata/` | 备份说明、逐文件来源/日期/大小/SHA-256、模型来源清单、修复补丁及新聊天交接说明 |
| [Release 备份附件](https://github.com/JoeJingzhou/FLARE-backup/releases/tag/backup-2026-11-03) | FLARE 主备份、修复版代码压缩包，以及两个附件的 SHA-256 校验文件 |

主备份 `FLARE_latest_minimal.tar.gz` 为 **237,471,374 字节（226.47 MiB）**，包含：

- 2026-05-11 的发布代码及 Git 历史。
- 最新现存论文源码、参考文献、图表及论文 Git 历史（来源 `flow-mi-v5/latex`，2026-04-20）。
- 实验结果、统计分析、消融、敏感性分析、汇总表和必要基线材料。
- 原实验辅助脚本与配置、数据划分、标签、标准化参数、逐样本 NLL。

修复版 `FLARE_reproducible_20261103.tar.gz` 为 **54,749 字节**。
原始发布代码基准提交：`b49944e66892e11d24f81059fc1a87b70c3c45b8`。

LRD 分出的资料包继续单独保存在本地，**本仓库未上传 LRD 资料包，也不是完整 LRD 项目备份**。
本地完整目录：`D:\OneDrive\Desktop\FLARE-backup-20261103`。

## 恢复与使用

1. 从 Release 下载两个备份压缩包和 `FLARE_SHA256SUMS.txt`，核对 SHA-256。
2. 查看 [`code/REPRODUCE.md`](code/REPRODUCE.md) 与 [`code/修复与验证说明.txt`](code/修复与验证说明.txt)。
3. 优先使用 `code/` 或修复版压缩包中的代码；主备份用于恢复论文、结果与原始辅助材料。
4. 建立 Python 3.10 环境，按 `code/requirements-repro.txt` 安装依赖；重新获取公开模型和原数据。
5. 按数据集配置生成对应向量，再运行 `scripts/run_flow_pipeline.py`。

小规模流程验证：

```bash
cd code
python -m unittest discover -s tests -v
python scripts/smoke_workflow.py --output-dir /tmp/flare-smoke-new
python scripts/smoke_workflow.py --output-dir /tmp/flare-text-smoke-new --with-embeddings
```

新聊天接手时，先阅读 [`metadata/新聊天交接说明.txt`](metadata/新聊天交接说明.txt)。

## 验证范围

- 原始备份及修复包已做解压和逐文件 SHA-256 验证。
- 5 项回归检查、64 个合成样本的训练与评分、64 条真实文本的完整 CPU 流程通过。
- 文本流程采用同一 MiniLM 模型的 mean/CLS 两种池化；每个方向有 13 个测试样本。
- 数据顺序错误拒绝检查、MI bounds 入口、editable 安装和 `pip check` 通过。
- 未重跑论文全量数据、其余 embedding 模型和全部基线。完整新环境依赖安装尚未验证。

没有保存公开预训练模型缓存、流权重、检查点及可重新生成的 embedding/X/Y 数组。
因此仍需获取原始数据与公开模型，并准备计算资源；这不是离线镜像。
