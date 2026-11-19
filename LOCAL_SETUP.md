# 本机 AgentD / DeepSeek 阶段一

本工作副本固定从 AgentD `2ecae86` 开始修改；配套 REINVENT4 使用提交 `3266e54bbe148831796477577d977de8cfbdeb05`。环境为 D 盘 Anaconda 下的 `agentd-deepseek`。DSW 尚未更新。

## 环境与路径

- Python 环境：`D:\anaconda\envs\agentd-deepseek`
- REINVENT4：`D:\CodexData\Downloads\REINVENT4-3266e54-local`
- 模型：`D:\CodexData\Downloads\agentd-models\multilingual-e5-small\614241f622f53c4eeff9890bdc4f31cfecc418b3`
- 非敏感路径配置：本目录的 `runtime.local.json`，已忽略于 Git。DSW 使用独立路径配置，不应复制本文件中的 Windows 路径。
- 安装缓存和临时文件在 `D:\CodexData\Temp`。Conda 环境已配置 `PYTHONNOUSERSITE=1`，避免读取其他用户安装的 Python 包。
- Windows FAISS 使用 conda-forge 的 `faiss=1.10.0=cpu_openblas*` 构建，搭配 NumPy 1.26.4；pip 的 faiss-cpu Windows wheel 与本机 PyTorch 的 OpenMP 冲突，因此不在本环境使用，也没有设置 KMP_DUPLICATE_LIB_OK。Linux 仍使用 faiss-cpu。

## 离线检查（无需密钥）

在 Anaconda Prompt 中执行：

```bat
conda activate agentd-deepseek
cd /d D:\CodexData\Downloads\AgentD-deepseek-local
python -m pip check
python -m pytest -q
python scripts\verify_local.py --device cuda:0
```

PowerShell 可使用 `scripts/start_local.ps1`；默认仅做本地嵌入/GPU/分子采样检查。生成结果和检查记录写入本目录 `reports/`，不代表真实 API 或完整药物性质预测已验证。

## 本阶段已适配的接口

- `DEEPSEEK_API_KEY`：语言模型和 Search/Fetch 共用；不再要求 OpenAI/Serper 密钥。
- `AGENTD_MODEL`：默认 deepseek-flash；初始关闭 thinking，兼容现有 LangChain 文本动作解析。
- 模型选择顺序：任务 YAML 或 MCP 请求显式指定的 model 优先；否则使用 AGENTD_MODEL、runtime.local.json，最后使用默认模型。同一次任务的嵌套工具沿用该有效模型。
- `AGENTD_CONFIG`：可指定绝对路径的非敏感 JSON 配置；默认读取本工作副本 runtime.local.json。
- `AGENTD_EMBEDDING_PATH`、`AGENTD_EMBEDDING_DEVICE`、`AGENTD_REINVENT_PATH`、`AGENTD_REINVENT_EXECUTABLE`、`AGENTD_SAMPLING_DEVICE` 可覆盖 JSON 配置。
- `AGENTD_ENV_FILE`：显式指定密钥文件；环境变量优先，不自动搜索父目录中的 .env。
- 模型完全本地加载，使用 E5 query/passage 前缀、平均池化及归一化；FAISS 与模型实际维度匹配。
- 网页与文献搜索均走 DeepSeek Search；最多为单次文献发现抓取 3 页，保留来源，失败时可保留标明为摘要的内容。PDF 与 Markdown 都可参与检索。

## 后续真实 API 测试

阶段一只使用模拟凭据和模拟 HTTP，不发送付费请求。后续确认测试费用后，用户可私下将 `.env.deepseek.example` 复制为 `.env.deepseek` 并填写 DEEPSEEK_API_KEY；不要把密钥发到聊天、加入 Git 或上传 OSS。

启动命令（会调用真实服务，当前阶段不要执行）：

```powershell
.\scripts\start_local.ps1 -Mode qna -AllowLive
.\scripts\start_local.ps1 -Mode pipeline -AllowLive
```

首次任务配置为 `pipeline_config.deepseek.yaml`：BCL-2、两个生成模型各采样 2 个分子、迭代 1 次、run_boltz=false。每次真实任务应使用新的 run_id。DeepSeek 对话和搜索/抓取分别计费；这些调用、外部数据库及性质预测服务在阶段一中均不宣称已真实跑通。

## DSW 交付边界

后续上传源码补丁、模型和必要的 Linux 依赖，不上传 Windows Conda 环境。保留 DSW 已配置的 agentd311 与 REINVENT4。先核对目标文件、备份，再更新并重新验证 Linux/A10；本机测试通过不替代 DSW 验收。
