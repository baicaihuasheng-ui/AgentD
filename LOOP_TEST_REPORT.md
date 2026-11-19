# 本机小批量闭环验收报告

## 最终结果

真实闭环通过，退出码为 0。目标为人 BCL-2，UniProt P10415，疾病背景为慢性淋巴细胞白血病。ChEMBL 提取的起始分子为 Venetoclax，分子式 C45H50ClN7O7S，RDKit 分子量为 868.457。

REINVENT 和 Mol2Mol 各产生 2 个有效分子，合并后为 4 个不同的规范化分子。4 个候选均完成结构修改、再次亲和力和 DeepPK 预测，最终对照表共 4 行、337 列。候选 ID 唯一且全程对应，所有亲和力及 RDKit 性质为有限数值。

完整批次标识为 `BCL2_loop_20261105_170300_a4fe00`，产物保存在仓库的 `runs` 目录下。

## 预测变化

| candidate_id | 优化前 pKd | 优化后 pKd | pKd 变化 | 优化前 QED | 优化后 QED |
|---|---|---|---|---|---|
| c0001-93282888 | 5.359160 | 4.878193 | -0.480968 | 0.494415 | 0.828142 |
| c0002-4f102a81 | 5.990246 | 5.591798 | -0.398448 | 0.227808 | 0.239490 |
| c0003-c502db4a | 6.128785 | 6.173669 | +0.044884 | 0.584099 | 0.551105 |
| c0004-3d6fe511 | 6.375962 | 6.214812 | -0.161150 | 0.530149 | 0.475085 |

1 个分子的预测亲和力提高，3 个降低。2 个分子的 QED 提高，2 个降低。完整 ADMET、置信信息、修改目标和理由均保存在逐候选文件中。本次验收条件为生成、有效结构修改和再次预测完成。

## 接口与费用

- 模型使用 `deepseek-flash`，关闭思考模式和接口自动重试。
- 对话 16 次，Search 1 次，Fetch 0 次。
- 对话输入共 45,291 tokens，输出共 4,892 tokens。
- 按高峰未缓存输入和输出价格核算的费用上界为 0.134718 元，预留余额为 0。实际扣费以服务端账单为准。
- 原始预算为 1 元、20 次对话、6 次 Search 和 3 次 Fetch。请求计数在恢复过程中持续累计。
- 原始付费批次总耗时约 22.86 分钟，截止时间在恢复过程中持续有效。
- DeepPK 基线任务为 `admet_1793870341.199011`，修改后任务为 `admet_1793870566.3376489`。两个任务均已完成，每组输入各提交一次。

价格按执行当日官方页面核对，输入为 2 元/百万 tokens，输出为 8 元/百万 tokens，Search 与 Fetch 各为 0.005 元/次。[官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)

## 启动故障及恢复记录

最早一次启动 `BCL2_loop_20261105_165808_a1d73b` 加载了用户目录依赖，付费请求数为 0。已修正 MCP 环境变量传递，强制 `PYTHONNOUSERSITE=1`。

付费批次中修复了 Windows 子进程标准输入继承，以及 LLM 生成 TOML 中当前 REINVENT 版本不接受的 `tb_logdir` 字段。子进程标准流已隔离，采样参数已接入固定版本的 schema 校验。恢复复用了成功的需求解析、TOML 和 4 个生成候选，原预算、截止时间及已成功的 8 次对话持续保留。先前启动尝试和中断的 worker 状态保存在日志及 acceptance.json 中。

另修复了旧的不存在模块 `retrieval_new` 导入、Windows worker 退出码记录、规范化分子去重、基线失败的逐候选状态及限时中断的状态输出。原有工作副本保留为运行源码。

## 离线与交付验证

76 passed, 7 warnings, 4 subtests passed in 30.96s

`pip check` 和 `git diff --check` 通过。测试覆盖乱序合并、重复身份、预测缺失、非有限亲和力、异常优化格式、无效结构、结构未变化复用、修改后 worker 失败、基线合并失败、子进程标准流、跨进程预算竞争、请求用量和失败状态。新增源码使用原环境的已锁定依赖。

对两个实际运行目录的 64 个文本产物进行了精确密钥暴露检查，通过。源码哈希清单为 `reports/loop-source-manifest.json`。

## 产物位置

- `iter_00/property/comparison.csv` 为完整前后对照。
- `iter_00/property/affinity_before.csv` 和 `affinity_after.csv` 为亲和力预测。
- `iter_00/property/admet_before.csv` 和 `admet_after.csv` 为完整 ADMET 预测。
- `iter_00/refinement/optimization_input.csv` 为实际优化输入。
- `iter_00/refinement/refinement.csv` 为修改结构、目标、理由和状态。
- `usage.json` 为跨进程调用和费用记录。
- `summary.json` 与 `acceptance.json` 为流水线及启动器验收记录。

## Linux 部署包与 OSS 后续步骤

1. 基于当前源码制作 Linux 部署包，保留配置模板，将环境名和模型路径设置为 DSW 的 `agentd311` 及 `/mnt/workspace` 目录。
2. 打包固定快照的 E5、ProtT5、MoLFormer 与预测头，准备 Python 3.11 Linux 的 SentencePiece、Accelerate 和 psutil 增量 wheels，并核对已有 DSW 依赖。
3. 对源码、模型及增量 wheels 分别生成 SHA-256 和大小清单，按单个对象低于 5 GB 分包。包内密钥仅提供空模板。
4. 打包完成后报告实际字节数、OSS 对象路径、存储及请求费用，取得上传确认，再上传并用 stat 核对。之后在 A10、32 GB 内存的 DSW 上进行同样的验收。
