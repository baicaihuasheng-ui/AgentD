# 本机亲和力模型与 DeepPK 配置

## 已安装组件

环境为 `D:\anaconda\envs\agentd-deepseek`，Python 3.11.18。

| 组件 | 版本或快照 |
|---|---|
| ProtT5 | `94a6abc029ae13029317b140b7424e012bf8dfbf` |
| MoLFormer compat-v4 | `7b12d946c181a37f6012b9dc3b002275de070314` |
| BAPULM | AgentD `2ecae86` 自带预测头，严格加载 |
| Transformers | 4.41.0 |
| PyTorch | 2.6.0+cu124 |
| SentencePiece | 0.2.0 |
| Accelerate | 0.31.0 |
| psutil | 7.2.2 |

模型位于 `D:\CodexData\Downloads\agentd-models`，两个快照合计 2,604,035,998 字节。各文件来源、大小、SHA-256 及发布方 LFS 哈希记录在 `environment/affinity-models.lock.json`，模型目录也各自保存 `manifest.json`。MoLFormer 的本地自定义代码已检查导入和文件操作，记录在 `reports/molformer-code-audit.json`。

ProtT5 使用 GPU FP16，MoLFormer 使用 GPU FP32，两者顺序加载和释放。BAPULM 使用 CPU FP32。保留原有包含特殊 token 的平均池化和结果缩放，随机种子为 2102。MoLFormer 每次独立设种子，使蛋白缓存命中与否不会改变其随机特征初始化。

## 运行配置

现有 `runtime.local.json` 已补入以下字段。它们也支持对应的 `AGENTD_` 大写环境变量，并由任务配置传给子进程。密钥继续独立存放。

| 配置字段 | 用途 |
|---|---|
| `protein_model_path` | ProtT5 完整快照目录 |
| `molecule_model_path` | MoLFormer compat-v4 完整快照目录 |
| `affinity_checkpoint_path` | BAPULM 预测头，默认使用仓库自带文件 |
| `affinity_cache_path` | CPU 蛋白特征缓存目录 |
| `affinity_device` | `auto`、`cpu` 或 `cuda:0` |
| `affinity_precision` | GPU ProtT5 的 `float16` 或 `float32` |

CPU ProtT5 始终采用 FP32。GPU 显存不足时仅对蛋白编码切换 CPU 重试一次。缓存键包含序列、模型清单哈希、预处理版本、精度、设备和种子。后续 DSW 部署时将这几个目录改为 Linux 路径。

## 验收结果

完整离线测试为 58 项通过、4 个子测试通过，另有 7 条依赖弃用警告。`pip check` 和 `git diff --check` 通过。新增依赖后的 E5/FAISS 检索、CUDA 计算、REINVENT 和 Mol2Mol GPU 采样均通过，两种生成模式各得到 2 个有效 SMILES。

`reports/affinity-acceptance/acceptance-first-cold.json` 保存首次完整冷启动结果，`acceptance.json` 保存修正验收脚本后的再次运行结果。

- 网络连接被主动拦截，两个本地编码器成功推理，特征维度分别为 1024 和 768。
- 官方短序列 `PRTEINO` 通过。BCL-2 使用 UniProt P10415 的人蛋白序列，共 239 个残基。原始 FASTA 和来源哈希保存在 `reports/fixtures`。
- BCL-2 的序列 SHA-256 为 `fff13b41ef65181ecfd036c714d8c453d73781b4e2a5bd73a6cbe02d9829fb19`。
- 阿司匹林预测为 4.8933992386 pKd，咖啡因为 3.9568099976 pKd。独立冷推理的最大绝对差为 0，缓存入口输出也一致。
- 本次 BCL-2 GPU 编码峰值已分配显存约 3.04 GiB，峰值预留显存约 3.25 GiB。首次验收进程峰值 RSS 约 4.85 GiB。RSS 为 50 ms 采样值，显存为 PyTorch 分配器计数。
- 短序列的真实 CPU FP32 路径通过，蛋白编码耗时约 2.92 秒，进程峰值 RSS 约 7.11 GiB，记录在 `cpu-fp32.json`。显存不足切换由模拟故障测试覆盖。
- DeepPK 真实任务 `admet_1793866815.7091815` 通过。两个分子的 CSV 共 197 列，保留 73 个预测字段、49 个概率字段及解释字段，输入 ID 和原始 SMILES 逐行对应。

本次仅有一个 DeepPK 提交任务。第一次解析发现线上字段名与文档示例不同，修正后通过原任务 ID 取回结果，保留两种字段格式。原始结果、状态和验收报告在 `reports/deeppk-acceptance`。

## 重新检查

在 PowerShell 中运行以下命令。亲和力检查会拦截网络，DeepPK 选项会访问其公开服务并优先恢复相同输入的已有任务。

```powershell
Set-Location 'D:\CodexData\Downloads\AgentD-deepseek-local'
$env:TEMP='D:\CodexData\Temp'
$env:TMP=$env:TEMP
$env:PYTHONNOUSERSITE='1'
$env:HF_HOME='D:\CodexData\Temp\huggingface'
$env:HF_MODULES_CACHE='D:\CodexData\Temp\huggingface\modules'
& 'D:\anaconda\envs\agentd-deepseek\python.exe' scripts/verify_affinity.py --output reports/affinity-acceptance
& 'D:\anaconda\envs\agentd-deepseek\python.exe' scripts/verify_affinity.py --deeppk --output reports/deeppk-acceptance
```

两个公开工具入口保持为 `predict_affinity_batch` 和 `get_admet_predictions`。输入 CSV 需要非空 `SMILES` 列，可附加 ID。无效分子在加载模型或提交 DeepPK 前报错。DeepPK 每次请求有连接及读取超时，轮询间隔 30 秒，轮询窗口 600 秒。失败状态保留任务 ID，再次运行相同输入时恢复该任务。提交结果不确定时需要人工核对状态，防止重复提交。OS 文件锁随进程退出释放，残留的锁文件可再次获取。

进程被强制终止时若停在提交与保存任务 ID 之间，状态为 `submitting`。须先确认远端任务情况，再处理该状态文件。

## 后续阶段

下一阶段将候选表、亲和力和 ADMET 表合并为优化输入，并验证修改分子的再次预测和前后对照。随后制作 Linux 增量包，按实际 OSS 大小说明费用并取得上传确认，再进行 A10 验收。Boltz 为独立的后续阶段。

## 官方来源

- [ProtT5 模型卡](https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc)
- [MoLFormer 模型卡](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct)
- [BAPULM 官方实现](https://github.com/radh55sh/BAPULM)
- [DeepPK API](https://biosig.lab.uq.edu.au/deeppk/api_docs)
- [BCL-2 序列](https://rest.uniprot.org/uniprotkb/P10415.fasta)
