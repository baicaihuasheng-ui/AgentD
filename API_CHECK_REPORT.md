# DeepSeek 真实接口检查报告

2026-11-05 已完成模拟验收和获准的最小真实接口检查，真实检查结果为 passed。用户私下配置密钥，并明确确认了这批测试费用。

- 脚本：scripts/check_deepseek.py，默认配置检查，只有 --live 才发请求。
- 完整测试：39 passed、4 subtests passed、7 个依赖弃用提示、0 失败，18.85 秒。结果见 reports/pytest-live-preflight.xml。
- 新增 8 项测试验证：默认不构造网络客户端；三项各一次；失败立即停止；空搜索跳过抓取；密钥脱敏；非官方地址禁止；Search 429 不重试；SDK 对话 429 不重试及 max_tokens=64 正确传输。
- 修正 pinned LangChain 将 max_tokens 改名的行为，通过适配类恢复 DeepSeek 官方的 max_tokens 参数。
- Search/Fetch 可设置 max_attempts=1，关闭 HTTP 重定向；原有默认有限重试行为保留。
- .env.deepseek 已被 Git 忽略，访问权限限定为当前 Windows 用户和 SYSTEM。用户随后私下填写密钥，检查脚本仅报告其存在状态。
- 实际默认配置检查：credential_present=false、status=needs_key，对话/搜索/抓取尝试次数全部为 0；未发起真实请求。结果见 reports/deepseek-configuration-check.json。
- 真实测试范围：deepseek-flash 非思考模式，固定短提示，最多输出 64 tokens；搜索 BCL-2 PubMed 文献最多 3 条；最多抓取首条可用链接一次；各项失败后停止。
- 预计费用低于 0.02 元，实际按官方账单。用户确认前不运行 --live，不重新尝试或扩大调用。

## 获准的真实测试结果

测试于北京时间 2026-11-05 15:33 开始，脚本退出码为 0，三项接口均通过。记录保存在 reports/deepseek-live-approved-20261105.json。

- 对话使用 deepseek-flash，关闭思考模式，输出上限为 64 tokens。返回内容符合固定短提示，输入 12 tokens，输出 1 token，共 13 tokens。
- Search 返回 3 条 BCL-2 相关 PubMed 结果，每条记录包含标题和来源链接。
- Fetch 读取首条结果 https://pubmed.ncbi.nlm.nih.gov/41152995/#1 ，获得 5,964 字符正文。
- 对话、搜索和抓取各尝试 1 次，自动重试为 0。三类接口使用同一份 DeepSeek 密钥配置。
- 原始凭据和鉴权请求头均由配置及客户端内部处理，结果文件保存状态、用量和公开来源。
- 实际扣费以 DeepSeek 账户账单为准，本次接口返回的数据用于记录调用规模。

本次完成了最小接口连通与返回内容验证。完整 AgentD 小批量端到端流程、在线性质预测及 DSW 部署仍待后续验收。
