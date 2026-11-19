# 本机小批量生成与优化闭环

## 运行方式

现有环境为 `D:\anaconda\envs\agentd-deepseek`。配置检查默认只显示密钥是否存在及测试限额。

```powershell
Set-Location 'D:\CodexData\Downloads\AgentD-deepseek-local'
$env:TEMP='D:\CodexData\Temp'
$env:TMP=$env:TEMP
$env:PYTHONNOUSERSITE='1'
$env:HF_HOME='D:\CodexData\Temp\hf-cache-agentd-local'
$env:HF_MODULES_CACHE='D:\CodexData\Temp\huggingface\modules'
$env:AGENTD_ENV_FILE=Join-Path $PWD '.env.deepseek'
& 'D:\anaconda\envs\agentd-deepseek\python.exe' scripts/check_pipeline.py
```

完成费用确认后，真实批次使用以下命令。每次会创建独立的 run ID 和预算文件。

```powershell
& 'D:\anaconda\envs\agentd-deepseek\python.exe' scripts/check_pipeline.py --live --confirmed-budget-rmb 1
```

本阶段固定 BCL-2、慢性淋巴细胞白血病、两种生成器各采样 2 个分子、1 轮优化和 `deepseek-flash` 非思考模式。任务由原 YAML 和 `run_agentd.py` 入口执行，保留终端问答入口。

## 预算与时限

- 所有进程共享 `usage.json`，使用 OS 文件锁防止并发突破限额。
- 对话最多 20 次，每次最多输出 2048 tokens。序列化请求限制为 32 KiB，费用预留采用每输入字节 2 tokens 加 512 tokens 余量。
- Search 最多 6 次，每次最多返回 3 条。普通搜索向代理提供每条最多 1500 字符的摘录。Fetch 最多 3 次。
- 预算为 1 元。每次调用前预留按高峰未缓存价格计算的费用，取得用量后按同一保守价格结算。无法取得用量的失败请求保留预留费用，服务器账单可能更低。
- 对话和 Search/Fetch 的 SDK 自动重试关闭，预算不足时在请求前停止。流式调用在受限批次中被禁止。
- 总截止时间为创建预算后 30 分钟，启动器到期清理所跟踪的进程树。每个 DeepPK 任务轮询窗口为 10 分钟，并保存已有任务 ID。

本次核对的官方价格为输入 2 元/百万 tokens、输出 8 元/百万 tokens，Search 与 Fetch 各 0.005 元/次。价格来源保存在测试报告中，后续真实批次执行前应重新核对。[官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)

## 数据衔接

生成结果经过 RDKit 校验和规范化分子去重，随后分配稳定的 `candidate_id`。预测表按 ID 对齐，同时校验对应 SMILES，阻止缺行、重复 ID、错配和无效数值。

优化输入包含候选、亲和力和 ADMET 的实际预测值及置信信息。每个候选记录原分子、修改后分子、目标性质、理由、状态和错误。修改后的有效结构再次预测，规范化后结构相同的候选复用基线预测。

对照表保留所有候选，包含亲和力、ADMET、QED、分子量、LogP 的前后值和可计算的变化量。原始预测、理由和来源分别保存。基线预测失败、合并失败或修改后预测失败均写入候选状态，并使最终验收失败。至少一个候选须完成结构修改和再次预测。

## 输出

每次任务位于 `runs/<run_id>`。

| 文件或目录 | 内容 |
|---|---|
| `extraction.json` | 需求解析、靶点序列及起始分子 |
| `pool/` | 两个生成器的输出及合并候选 |
| `iter_00/refinement/optimization_input.csv` | 实际传给优化步骤的预测表 |
| `iter_00/refinement/refinement.csv` | 全部候选的修改记录与状态 |
| `iter_00/property/` | 前后预测、对照表、DeepPK 状态及资源记录 |
| `usage.json` | 请求计数、token 用量及保守费用 |
| `summary.json` | 流水线结果和产物路径 |
| `acceptance.json` | 启动器退出状态、耗时和先前启动尝试 |

受控恢复支持生成阶段和第一轮基线预测阶段。恢复前需结束该批旧进程并核对状态，使用原 run ID、原预算和原截止时间。已有需求解析、TOML 及生成结果优先复用，DeepPK 使用相同输入对应的既有任务 ID。恢复命令为 `scripts/check_pipeline.py --live --confirmed-budget-rmb 1 --resume-run <run_id>`。

后续 Linux 包制作及 OSS 上传按实际产物大小另行安排。
