# openagent-compat-lab

[English](README.md) | 简体中文

**在把接口接入 Agent 工作流之前，先探测 Codex、Hermes Agent 和 OpenClaw
常用的 OpenAI 风格协议路径。**

openagent-compat-lab 会从你的电脑发起一组小型、确定性的协议检测，并输出兼容性
矩阵、JSON 或便于审查的 Markdown 报告。它检测的是协议行为，不是模型能力排名；
只要缺少必要行为，命令就会以非零状态退出。

示例输出：

```text
Agent compatibility matrix for provider/model-name
  endpoint: https://provider.example/v1

  profile    passed  failed  duration       result
  ---------- ------- ------- ------------ ----------
  codex       8/8          0   1842.6 ms   passed
  hermes      8/8          0   1604.1 ms   passed
  openclaw    8/8          0   1719.8 ms   passed
```

## 为什么需要它

一次成功的 `curl /chat/completions` 只能证明某个请求返回了响应。Agent 还依赖许多
更容易出问题的协议细节：

- 精确匹配工具调用 ID 和 JSON 参数；
- 正确保持 assistant → tool → assistant 角色顺序；
- 第二轮能够消费工具结果，而不是再次请求同一工具；
- 按索引合并流式返回的工具调用片段；
- 正确处理 Responses API 类型化条目、`call_id` 和 `previous_response_id`；
- 区分 Chat Completions 与 Responses 的结构化输出字段；
- 正确结束 SSE 流并处理非文本输入。

openagent-compat-lab 使用固定提示词和虚构的本地工具结果检测这些路径。它不会执行
Shell 命令、读取你的代码仓库，也不会调用真实的订单或天气服务。

## 相比上游新增了什么

这是一个面向 Agent 集成的衍生项目，不是未经修改的镜像。上游套件覆盖广泛的模型
行为；openagent-compat-lab 在此基础上增加了 Agent 集成验收层：

- 独立的 Codex、Hermes 和 OpenClaw 协议配置；
- 完整的有状态工具结果往返，而不是停在第一次工具调用；
- Responses `call_id` 配对和 `previous_response_id` 续接；
- 严格的 Chat Completions assistant/tool/assistant 角色顺序；
- 按索引和 ID 重组流式并行工具调用；
- 一条命令生成三类 Agent 兼容性矩阵；
- 可只选择实际使用的多个配置，避免无关探测和接口调用；
- 可在配置接口凭据前离线查看准确检测项；
- 可输出已安装版本，便于支持和 CI 环境诊断；
- 每项检测耗时，以及 JSON、Markdown 和 JUnit 证据报告；
- 可选的快速失败模式，用于控制接口调用成本和 CI 等待时间；
- 可按单次运行设置请求超时，适配慢接口和有时限的 CI 任务；
- 凭据脱敏、显式无认证模式和离线回归测试。

## 检测配置

这些命名配置检测的协议路径确实不同，并不是给同一批测试换标签。

| 配置 | API 路径 | 特定协议覆盖 |
|---|---|---|
| `codex` | Responses API | 类型化输出条目、`response.completed`、强制函数调用、按 `call_id` 配对的 `function_call_output`，以及通过 `previous_response_id` 续接 |
| `hermes` | Chat Completions | 严格的 `assistant(tool_calls) → tool(tool_call_id) → assistant(text)` 往返和 `finish_reason: tool_calls` |
| `openclaw` | Chat Completions | 两个并行工具调用的分片流式返回，并按 `index`、ID、名称和 JSON 参数重组 |
| `generic` | Chat Completions | 模型列表、文本、SSE `[DONE]`、工具、可选参数、JSON Schema 和图片 detail 的快速基线 |
| `all` | 三类命名 Agent 路径 | 依次运行 `codex`、`hermes` 和 `openclaw`，然后生成一张兼容性矩阵 |
| `model` | 上游完整套件 | 保留更广泛的 Model Compatibility Suite，用于模型级一致性检测 |

每个命名 Agent 配置还会检测 `GET /models`、基础文本生成、强制工具调用、工具可选
参数省略、严格 JSON Schema 输出、流终止，以及带有 `detail: original` 的 1×1
内联图片。

检测通过只表示这些路径在检测当时正常工作，不代表回答质量、该 Agent 的全部功能、
供应商可用性或生产环境安全已经获得认证。这些配置不会启动对应客户端，也不验证其
端到端配置、认证流程或真实任务表现。

## 快速开始

需要 Python 3.10 或更高版本。

```bash
git clone https://github.com/OkkBtc/openagent-compat-lab.git
cd openagent-compat-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
agent-compat --version
```

请把供应商密钥放在环境变量中，避免它出现在命令行参数里。如果担心 Shell 历史记录
保留密钥，请使用系统密钥管理器或无回显输入方式加载真实密钥：

```bash
export ACL_API_KEY="your-provider-key"

agent-compat \
  --profile all \
  --base-url https://provider.example/v1 \
  --model provider/model-name
```

Base URL 应当是可以继续拼接 `/models`、`/chat/completions` 和 `/responses` 的 API
根路径。大多数供应商的地址以 `/v1` 结尾。

如果你只关心某一种 Agent 集成，可以单独运行对应配置：

```bash
agent-compat --profile codex \
  --base-url "$BASE_URL" \
  --model "$MODEL"

agent-compat --profile hermes \
  --base-url "$BASE_URL" \
  --model "$MODEL"

agent-compat --profile openclaw \
  --base-url "$BASE_URL" \
  --model "$MODEL"
```

如果工作流只使用部分 Agent，可以重复传入 `--profile`，并按指定顺序生成更小的
兼容性矩阵：

```bash
agent-compat \
  --profile codex \
  --profile hermes \
  --base-url "$BASE_URL" \
  --model "$MODEL"
```

选择性矩阵同样支持控制台、JSON、Markdown、JUnit 和快速失败输出。`all`、`model`
不能与其他配置混用，重复的配置值也会被拒绝。

在提供接口、模型或凭据前，先离线查看将执行的准确检测项：

```bash
agent-compat --profile codex --list-checks
agent-compat --profile all --list-checks --json
```

该发现命令不会发送请求，也不需要接口、模型或 API Key；它只适用于 Agent 配置，
不适用于继承的 `model` 套件。

检测明确不使用 Bearer Token 的本地接口：

```bash
agent-compat \
  --profile generic \
  --base-url http://127.0.0.1:11434/v1 \
  --model qwen3 \
  --allow-no-auth
```

除非显式传入 `--allow-no-auth`，空密钥会被拒绝。这样可以在发出请求之前发现常见的
配置错误。

如果首次协议断言失败或发生运行错误后就无需继续，可以启用快速失败：

```bash
agent-compat --profile all \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --fail-fast
```

使用 `--profile all` 或重复配置的选择性矩阵时，某个配置停止后还会跳过后续配置。
在早期失败已经足以判定本次运行不可用时，这可以减少付费请求和 CI 等待。该选项只
用于 Agent 配置，不适用于继承的 `model` 套件。

## 报告与 CI

输出机器可读的 JSON：

```bash
agent-compat --profile all \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --json > compat.json
```

JSON 包含矩阵汇总、每项检测的状态和失败详情，以及检测项和配置级别的
`duration_ms`。

保留控制台矩阵，同时写入 Markdown 报告：

```bash
agent-compat --profile all \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --markdown compat-report.md
```

报告可能包含供应商/模型标识和响应详情；分享到团队以外之前仍需人工检查和脱敏。

为 GitHub Actions、GitLab、Jenkins 或其他 CI 测试报告查看器写入标准 JUnit XML：

```bash
agent-compat --profile all \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --junit compat-results.xml
```

`all` 配置会为每类命名 Agent 写入一个 test suite。协议不匹配记录为 JUnit
failure，传输或运行问题记录为 error，方便 CI 面板区分这两类情况。

只有全部选中检测通过时，退出状态才是 `0`。`FAIL` 表示接口返回了有效响应，但违反
被验证的协议约束；`BROKEN` 表示传输、HTTP、JSON 或其他运行错误导致协议无法完成
验证。两者都会返回退出状态 `1`。

可选的脱敏通信记录可以帮助排查供应商问题：

```bash
agent-compat --profile hermes \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --record-responses ./compat-wire
```

目标目录不能已经存在。文件会按配置和检测项分组，多请求往返会按顺序保存。已知凭据、
Authorization 值、URL 用户信息和常见密钥查询参数都会被脱敏。供应商仍有可能回显
脱敏器不了解的私密数据，因此请始终把通信记录视为敏感文件。

## 配置

命令行参数会写入所有检测共用的同一组环境配置。

| 变量 | 含义 |
|---|---|
| `ACL_API_BASE` | API 根路径，通常以 `/v1` 结尾 |
| `ACL_API_KEY` | Bearer Token；项目有意不提供命令行密钥参数 |
| `ACL_MODEL` | 应当出现在 `GET /models` 中的准确模型 ID |
| `ACL_TIMEOUT` | 单次请求超时秒数，默认 `60` |

无需修改环境变量，也可以只覆盖本次运行的超时：

```bash
agent-compat --profile all \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --timeout 20
```

该值必须大于零，并分别作用于每个 HTTP 请求，而不是整个矩阵的总时长。

项目仍然兼容继承套件使用的 `MCS_*` 变量和 `CSCS_SERVING_API`。旧的 `mcs` 可执行
命令也会继续保留，但新用法应当使用 `agent-compat`。

## 请求、隐私与计费

- 请求会从你的电脑直接发送到你提供的 Base URL。
- 项目没有遥测、托管后端、账号系统或自动报告上传。
- API Key 从环境变量读取，Authorization 请求头不会写入通信记录。
- 项目不会自动重试，因此失败不会被静默隐藏，也不会意外为同一检测重复计费。
- 一次成功的 `all` 检测会发起 26 个 HTTP 请求：23 次小型模型生成和 3 次
  `/models` 读取。某项往返失败后，该项后续请求可能提前停止；`--fail-fast` 还会
  跳过后续检测，并在 `all` 模式下跳过后续配置。
- 提示词、虚构工具 Schema/结果以及一张极小的内联测试图片会发送给供应商。请先
  查看该供应商的数据保留与隐私条款。
- 供应商可能对每次生成收费。在 CI 中重复运行矩阵前，请先检查当前价格和限流规则。

工具 Schema 使用虚构的订单和天气数据。提交 Issue 时不要把它们替换为生产密钥或
真实私密数据。

## 完整模型套件

上游的确定性模型一致性套件仍然可用：

```bash
agent-compat --profile model \
  --base-url "$BASE_URL" \
  --model "$MODEL"

agent-compat --profile model \
  --base-url "$BASE_URL" \
  --model "$MODEL" \
  --capability tools,streaming
```

只有暴露非标准 `/tokenize` 和 `/detokenize` 端点的服务才应使用 `--spec dev`。
继承套件的设计说明见 [SPEC.md](SPEC.md)。

## 排错

| 现象 | 可能原因 |
|---|---|
| 找不到目标模型 | 配置的模型 ID 与 `/models` 不一致，或接口会按密钥过滤可见模型 |
| 流结束时没有 `[DONE]` | Chat Completions SSE 帧不完整 |
| 没有 `response.completed` | Responses 流没有发送必须的类型化完成事件 |
| 工具结果返回后再次请求工具 | 接口没有正确消费 assistant/tool 历史记录 |
| `call_id`/`previous_response_id` 错误 | 接口只实现了部分 Responses 续接协议 |
| 流式工具 JSON 解码失败 | 工具参数片段缺失、顺序错误或格式不合法 |
| `/responses` 返回 HTTP 404 | 供应商可能只支持 Chat Completions；请分别检测 `hermes`、`openclaw` 或 `generic` |

如果一行失败详情不足以定位问题，可以把 `--record-responses` 指向一个新目录。报告
和错误会限制过长的供应商响应，降低意外泄露风险。

## 开发

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

ruff check .
ruff format --check .
pytest tests -q
pytest mcs/suites --collect-only -q
python -m build
```

面向 Agent 的测试使用 Python 标准库提供的本地 HTTP 服务器，不需要网络连接或真实
供应商密钥。CI 会运行静态检查、全部离线 Agent 测试、完整套件收集，以及 wheel 和
源码分发包构建。

## 范围与路线图

项目会有意保持精简：提供确定性的协议验收、清晰证据和安全诊断。它不是 Agent
运行时，也不是模型排行榜。适合继续加入的功能应当对应可复现的供应商协议缺口，并
附带离线回归测试。

欢迎提交 Issue 和范围明确的 Pull Request。请说明检测配置、脱敏失败详情、供应商/
模型标识，以及问题是否能够复现。

## 来源与许可证

openagent-compat-lab 衍生自
[`swiss-ai/model-compatibility-suite`](https://github.com/swiss-ai/model-compatibility-suite)，
基于提交
[`531a52813d9be66d9fdf13c6a9d30875a770df66`](https://github.com/swiss-ai/model-compatibility-suite/commit/531a52813d9be66d9fdf13c6a9d30875a770df66)。

项目保留了上游 Apache-2.0 许可证和通知。Agent 专用配置、有状态工具往返、Responses
API 检测、兼容性矩阵、凭据脱敏、耗时统计、报告和离线协议测试均为本项目修改。
准确的归属信息见 [NOTICE](NOTICE)。

本项目是独立社区项目，与 OpenAI、Hermes Agent、OpenClaw 或上游维护者不存在关联，
也未获得其背书。
