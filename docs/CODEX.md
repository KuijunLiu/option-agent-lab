# 使用已有 Codex 登录运行反馈对照实验

这个桥接脚本调用本机 Codex CLI，使用其已保存的登录方式。若 CLI 使用 ChatGPT 订阅登录，运行消耗相应的 Codex 额度；若 CLI 配置为 API key 登录，仍可能产生 API 费用。脚本不读取、创建或保存 API key。

先按官方文档安装 CLI 并完成登录：

- [Codex 身份验证](https://learn.chatgpt.com/docs/auth)
- [Codex 非交互运行](https://learn.chatgpt.com/docs/non-interactive-mode)

在终端确认 `codex --version` 可运行，并使用 `codex login` 按提示选择 ChatGPT 登录。模型名称与额度由账户和本地配置决定；本项目不保证某个具体模型可用。

## 相同调用次数的两个条件

| 条件 | 可见信息 | 默认预算 |
|---|---|---|
| `feedback` | 共同初始测试结果、之前提议、每轮测得的价格误差及对应参数点 | 初始 128 点 + 7 次 LLM 决策，每次 128 点 |
| `no_feedback` | 相同初始测试结果、之前提议；后续实测结果显示为 `null` | 初始 128 点 + 7 次 LLM 决策，每次 128 点 |

控制组同样有七次思考和提议的机会；区别是后续数值反馈是否可见。两个条件使用同一份提示词模板、相同的 JSON 字段和相同的历史条目结构。第一次 LLM 决策的可见内容完全相同；模型本身可能具有随机性，因此第一轮提议不保证相同。

**唯一的搜索目标是：在给定预算下发现最大的绝对价格误差 `abs(C_network-C_reference)/K`。** 覆盖范围、delta 误差和价格界限违背属于诊断，不作为复合目标。Delta 数值保留在最终诊断中，不输入这两个条件的提示词。

`no_feedback` 的数值测试仍然真实执行、消耗预算并保存结果，只是在下次提示中被过滤。控制器不向 LLM 传递累计 summary，也不传递采样种子、评估时间、后台元数据或其他搜索方法结果。纯函数 `make_agent_view` 通过白名单构建新视图，不修改后台状态；测试验证后续反馈不会通过嵌套字段泄漏。

## 运行一组成对测试

先按 README 训练网络。下面的 `MODEL_ID` 要替换为你账户可用的明确模型名称：

```bash
python scripts/run_codex_agent.py --run runs/demo --name feedback-s123 --condition feedback --budget 1024 --seed 123 --model MODEL_ID
python scripts/run_codex_agent.py --run runs/demo --name control-s123 --condition no_feedback --budget 1024 --seed 123 --model MODEL_ID
```

`runs/demo` 只适合检查安装和理解流程；正式比较应使用研究方案中预留的新网络。单次本地检查允许省略 `--model`，但日志会标记为 `local_default_unpinned`，不应把这样的测试并入固定模型的正式研究。批量正式研究应明确指定模型并保留 CLI 版本。

每组共 8 个数值批次：首批由控制器固定为全域 Sobol 初始点，与相同种子的数值基线一致；后七批由 LLM 提议矩形区域。区域内由 Python 用 scrambled Sobol 采样，因此完整方法是 **LLM 选择区域 + Sobol 生成测试点**。`--max-rounds` 包含初始化，默认是 8；若更改预算，应同时保证轮次足够。非正式探索允许最后一批不足 128 点，非 2 的幂时会使用均匀随机采样，不能把这种配置与默认正式条件直接混合。

控制器强制每次返回的 `n` 等于计划值。返回较少样本不会换来额外决策机会。JSON 无效、区域越界、CLI 失败或超时都会终止该次测试；没有自动重试或虚构 fallback。报告失败率，并为失败后的新尝试使用新的名称，不能挑选成功运行而隐藏失败。每个新实验必须使用新的 `--name`；现有日志和 session 不会被覆盖。

## 保存什么证据

`runs/对应模型/codex-agent-logs/会话名称/` 中保存：

- `metadata.json`：条件、明确请求的模型或未固定状态、CLI 版本、预算、计划及实际调用次数、提示词模板版本和 SHA-256、开始结束时间、完成或失败状态。
- `round-XX-backend-status.json`：后台完整状态，仅供研究者审计，不进入提示词。
- `round-XX-visible-state.json` 与 `round-XX-prompt.txt`：该次 LLM 真正获得的状态与提示词；每轮保存哈希。
- `round-XX-command.json`、stdout/stderr、原始与通过验证的提议：执行证据。
- `round-XX-call.json`：单次调用的时间戳、等待时间、返回码和 CLI 若报告的 token usage。没有 usage 时明确记录 `unknown_not_reported`，不估算也不记作零成本。
- `final-status.json` 或 `failure.json`：完整成功状态或停止原因；已完成的真实数值批次保留在 session 中。

采样种子控制数值实验的随机性，不保证 LLM 文本输出一致。两个条件输入长度不同，所以相同调用次数并不表示相同 token 数或实际延迟；应单独报告成本。CLI JSON 事件若报告模型名称会记录下来，否则实际解析到的服务端模型标为未知，不能仅根据请求名称声称固定了服务端版本。

## 信息边界与当前验证范围

Codex 每次在新临时目录中运行，且不复用对话。目录中只有输出 schema，数值状态通过标准输入传入；提示词禁止使用工具。脚本使用 `--sandbox read-only` 并检查常见工具调用事件，发现时停止并标记该次实验不合规。

但是 **read-only 不是文件系统保密隔离**。本机完整结果仍可能被读取，提示约束和事件检查也不是安全证明。正式研究应在仅挂载允许文件、无其他结果访问权限的容器或独立运行环境中执行，或者审计完整事件日志并披露此限制。不要把这个原型称为经过硬隔离的盲测系统。

本交付环境没有安装且认证的 Codex CLI。控制器单元测试使用明确的测试替身来检验调用次数、反馈过滤和失败处理；它们不是实际 LLM 实验。现有 `assistant_pilot` 是此前对话中的单次助手引导测试，既不是本协议的成对对照实验，也不是自动 Codex 运行；不能并入新研究的正式统计结果。真实 CLI 登录、模型输出和实际 token 成本仍需在你的本机运行后记录。
