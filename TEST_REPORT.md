# 阶段一测试报告：本机 Anaconda / AgentD / DeepSeek

2026-11-05 已完成本机部署、离线验收，以及后续获准的 DeepSeek 最小真实接口检查。DSW 部署仍待后续验收。

接口检查脚本的完整测试为 39 passed、4 subtests passed。用户配置密钥并确认费用后，真实对话、搜索和抓取各调用一次，三项均通过。详见 API_CHECK_REPORT.md。

## 已验证结果

- 环境：D:/anaconda/envs/agentd-deepseek，Python 3.11.18。
- GPU：NVIDIA GeForce RTX 3060 Laptop GPU；torch 2.6.0+cu124，CUDA 可用，实际 CUDA 张量计算通过。
- REINVENT4 提交：3266e54bbe148831796477577d977de8cfbdeb05。源码内部版本标识仍为 4.5.11。
- 实际 GPU 采样：Reinvent 2 个有效 SMILES；Mol2Mol 2 个有效 SMILES。
- 本地 E5：固定 revision 614241f622f53c4eeff9890bdc4f31cfecc418b3；384 维，中英文检索、PDF/Markdown 检索、来源与页码传入问答上下文已验证。
- 完整 pytest：31 passed，2 subtests passed，7 warnings；无失败或跳过。warning 为现有 LangChain/Authlib 弃用提示，详细日志见 reports/pytest-final.log。
- pip check：No broken requirements found。
- MCP 子进程工具清单和配置继承通过；模拟 DeepSeek 请求验证了模型、地址、thinking 关闭、搜索/抓取、空结果、错误格式、鉴权/余额错误、有限重试。
- 源码复核发现的模型继承与 PDF 引用问题已补回归测试：先观察 3 个失败，再修正并运行完整绿灯测试。
- 最终 PowerShell 启动脚本默认离线验证模式运行成功；结果目录：reports/offline-20261105-121750。

## 本机兼容性修正

- pip Windows FAISS 与 torch 的 OpenMP 发生致命冲突；已换为 conda-forge faiss 1.10.0 OpenBLAS 构建，保留 NumPy 1.26.4。未使用 KMP_DUPLICATE_LIB_OK。
- 新 Conda 环境已隔离 Windows user-site，避免混入已有用户包。
- ChEMBL 客户端改为工具实际调用时才加载，避免导入模块时自动联网。
- AgentD 中的 REINVENT 路径和命令来自配置，保留模型/任务路径的跨平台适配；生成失败与无有效 SMILES 会正确失败，runner 不再无条件成功退出。

## 可复现材料

- environment/windows-conda-explicit.lock.txt：Conda 精确包清单，须先安装。
- environment/windows-pip.lock.txt：pip 依赖锁，faiss 由 Conda 提供；源码单独 --no-deps 安装。
- environment/windows-pip-freeze.audit.txt：当前环境原始 freeze，包括本机可编辑源码路径。
- reports/pytest-results.xml、pytest-final.log、launcher-final.log、source-manifest.json。
- reports/source-diff.review-only.txt：含新增文件的只读差异，旧上游凭据已脱敏，不用作部署补丁。
- LOCAL_SETUP.md：本机启动、路径、后续 API 配置及部署边界。

## 验证边界

本阶段使用模拟 HTTP/语言模型验证适配器，所有 pytest 真实 requests/httpx 传输被禁止；实际本地嵌入和 GPU 采样使用真实模型。没有使用真实密钥调用 DeepSeek、OpenAI、Serper 或其他在线预测服务，没有上传 OSS，也没有修改 DSW。

尚未验证：DeepSeek 账号真实接口权限/费用、真实文献问答效果、完整在线药物性质预测、Boltz、Linux/A10 部署。真实 API 和 OSS 操作应在后续分别说明费用并获得确认。
# Affinity and DeepPK subsequent acceptance — 2026-11-05

The current environment passes 58 tests and 4 subtests, with 7 dependency deprecation warnings. Full output is in `reports/pytest-affinity.txt` and JUnit in `reports/pytest-affinity.xml`. `pip check` passes.

Both fixed encoders load with network connections blocked. ProtT5 produces 1024 dimensions, MoLFormer 768, and the CPU FP32 BAPULM head loads strictly. GPU BCL-2 predictions for aspirin/caffeine are 4.8933992386 / 3.9568099976 pKd. An independent cold repeat differs by 0 and the normal AgentD batch-tool cache result agrees. A real CPU FP32 short-sequence run and simulated CUDA OOM recovery also pass.

DeepPK real submission, polling, crash-safe job state and CSV output pass for the same two molecules. One job was submitted. The current service uses descriptive field labels differing from its API documentation sample, so both formats are supported. The output preserves 73 prediction and 49 probability fields, explanations, local IDs and input SMILES. Prior failed parsing state was recovered using the existing job ID.

The two generation modes, CUDA and E5/FAISS retrieval were rerun after dependency changes and pass. Detailed paths, measurements, invocation and the follow-on optimization/deployment stage are documented in `AFFINITY_SETUP.md`.

# Bounded full generation/optimization acceptance — 2026-11-05

The current full suite passes 76 tests and 4 subtests, with 7 dependency deprecation warnings. The approved real batch exits 0 with 4 generated candidates, 4 valid modifications and 4 completed after-predictions. Candidate mapping and finite measured properties were independently checked.

Actual usage is 16 chat requests, 1 Search and 0 Fetch, 45,291 prompt tokens and 4,892 completion tokens. The peak-uncached-price upper bound is RMB 0.134718. Recoveries preserved the original paid-batch counters and deadline. Full startup-failure disclosures, numerical comparison and Linux handoff steps are in `LOOP_TEST_REPORT.md`, with usage in `LOOP_SETUP.md`.

