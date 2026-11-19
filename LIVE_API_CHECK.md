# DeepSeek 最小真实接口检查

默认只检查配置，不发起网络请求。脚本固定使用官方 DeepSeek 地址、deepseek-flash、关闭思考，限制输出 64 tokens。真实模式最多各尝试一次对话、搜索和抓取；没有自动重试或重定向，也不会启动 AgentD 流水线。

## 私下配置密钥

在本机编辑项目根目录 `.env.deepseek`，将 DEEPSEEK_API_KEY 的值填在等号后；不要把密钥发到聊天，也不要上传该文件。文件已被 Git 忽略。环境变量优先于文件；需要更换已有进程密钥时应重新打开终端或显式更新当前进程。

## 配置检查（不收费）

```bat
conda activate agentd-deepseek
cd /d D:\CodexData\Downloads\AgentD-deepseek-local
python scripts\check_deepseek.py
```

脚本自动选择项目内的 `.env.deepseek`；也可使用 `--env-file` 指定绝对路径。只报告是否存在密钥，不打印密钥内容。

## 真实检查（确认费用后执行一次）

```bat
python scripts\check_deepseek.py --live
```

- 一次短对话，要求回复 OK，输出上限 64 tokens，max_retries=0。
- 一次 BCL-2 PubMed 相关搜索，最多 3 条结果。
- 搜索有可用链接时，对首条链接最多抓取一次正文；否则跳过并报告 partial。
- 任意一步失败即停止；不要直接重复运行。网络超时不能证明服务端未计费，后续重试需要重新确认。
- Search、Fetch 当前各为每次成功请求约 0.005 元；加上短对话，预计本批低于 0.02 元，以官方实际计费为准。

JSON 报告写入 `reports/deepseek-check-*.json`，记录每步尝试次数、状态、模型、token 用量、来源链接和正文字符数；错误只保存异常类别和 HTTP 状态码，不保存原始错误消息、请求头或正文。

退出码：0=配置就绪或三项真实检查通过；1=失败/非官方地址被阻止；2=缺少密钥；3=部分验证（如搜索为空、未能验证抓取）。

官方依据：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/ 、https://api-docs.deepseek.com/zh-cn/guides/search/ 。
