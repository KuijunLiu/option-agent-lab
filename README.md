# Option Agent Lab · Erdos v2

**研究问题：实验反馈能否帮助 LLM 更有效地发现神经网络期权定价模型的错误？**

训练小型 PyTorch 网络近似 Black–Scholes 看涨期权价格，冻结网络，再比较测试方法。LLM 提出测试区域，Python 计算参考价格和评分。主数据是合成理论价格，不是真实市场报价；项目不声称能预测股价或获得交易收益。

## 从这里开始

- [实际数值报告](studies/erdos_v2/REPORT.md)：三个新网络、四种数值方法、delta 诊断，以及真实 LLM 实验的完成状态。
- [实验方案](docs/EXPERIMENT_PLAN.md)：假设、六组方法、预算、开发/最终比较分离、Erdos 第一次 check-in 内容。
- [学习路线](docs/LEARNING_PATH.md)：四周约40–60小时，每阶段的概念、练习与可能发现。
- [数据来源](docs/DATA_SOURCES.md)：合成数据、22条 QuantLib 公开校验样例、来源和许可。
- [Codex 说明](docs/CODEX.md)：本机登录、两组条件、逐次日志和验证范围。

旧 [runs/demo](runs/demo/REPORT.md) 保留为开发样例，单次助手 pilot 不混入 v2 比较。交付已运行新的数值部分；**Codex LLM 配对实验仍须在安装且登录 CLI 的机器上执行**。代码不会用人工提议或测试替身冒充真实模型输出。

## 安装与检查

需要 Python 3.10+。macOS/Linux 在解压后的仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,oracle]'
python -m pytest -q
```

CPU 即可，不要求 GPU。oracle 额外安装 QuantLib，交叉检查价格与 delta；其余测试可在没有 QuantLib 时运行。实际环境版本见 requirements-verified.txt；机器与依赖版本不同可能产生小的浮点差异。

## 查看或复现实验

包内包括数据、权重、逐点评估、图表和中文报告。重建已有报告：

```bash
python scripts/run_study.py --phase report --study studies/erdos_v2
```

从头运行时使用新目录，程序拒绝覆盖已有研究：

```bash
python scripts/run_study.py --phase prepare --study studies/my_study
python scripts/run_study.py --phase report --study studies/my_study
```

prepare 保存协议，在旧模型上选择 DE 种群，然后生成三份独立数据、训练新模型、运行数值搜索和 delta 诊断，最后锁定结果。配置在 configs/erdos_v2.json。

| 设置 | 默认值 |
|---|---|
| 训练种子 | 17、29、43 |
| 数据基础种子 | 20261001、20261011、20261021 |
| 每个模型的数据量 | 训练32768、验证4096、测试8192、压力诊断8192 |
| 搜索种子 | 701、702、703 |
| 每次搜索 | 1024点，含共同初始128点 |
| LLM条件 | feedback、no_feedback |
| 计划LLM决策数 | 3模型 × 3搜索种子 × 2条件 × 7次 = 126次 |

三个实例同时改变数据与训练种子，差异不能仅归因于初始化。验证集选择 checkpoint；独立测试/压力诊断不进入 LLM 提示词。正式实验期间不要修改锁定文件；改变方法应另建研究。

## 运行真实 LLM 对照

先按 CODEX.md 安装并登录 Codex CLI，将 MODEL_ID 换成账号实际可用的明确模型标识：

```bash
python scripts/run_study.py --phase agents --study studies/erdos_v2 --model MODEL_ID
python scripts/run_study.py --phase report --study studies/erdos_v2
```

若运行自己新建的研究，统一换成 studies/my_study。控制器锁定模型和超时设置，每个配对的条件顺序预先随机化。重复命令可跳过已完成会话、继续尚未开始的会话；失败或部分完成的会话不会自动重试或重置预算，且始终显示在报告中。未完成全部配对前，不能只据成功的子集宣称优势。

无反馈组只看到共同初始测量和此前提议；反馈组另外看到自己的新结果。两组提示词模板、数值预算和调用次数相同，但反馈组上下文更长，**并非严格等 token 或等算力实验**。逐次用量仅在 CLI 提供时记录，否则标为未知。保留 CLI 版本、模型标识和原始日志；服务模型与本地配置也会影响重现。

只读沙箱不是整个文件系统的保密隔离。桥接检测常见的违规工具调用并停止，但不能宣称实现了强隔离。

## 方法与指标

比较 Random、Sobol、固定金融压力方案、DE、无新反馈 LLM 和有反馈 LLM。DE 种群只在旧开发模型上从16/32/64选择，再用于新模型。

主要目标是找到最大的 abs(C_pred-C_BS)/K；大于0.001算超阈值，即 K=100 时价格误差0.10。覆盖度按固定120个参数网格去重。更大的已发现误差表示更会找问题，不代表网络改善；网格也不是错误机制分类。

Delta 只作独立诊断，不进入搜索评分。网络输出 C/K=f(S/K,T,sigma)，因此 delta=df/dm，还需包含内部归一化的链式法则。价格误差小不保证 delta 误差小；0.05 的 delta 误差标记是研究阈值，不是行业标准或已实现的对冲损益。

## 代码阅读顺序

| 文件（模块位于 option_agent_lab/） | 学习内容 |
|---|---|
| pricing.py、data.py | 理论标签、单位、随机数据和划分 |
| model.py | PyTorch训练、验证选模、冻结模型 |
| greeks.py | 自动微分、链式法则、敏感度验证 |
| evaluation.py、benchmarks.py | 指标、Sobol、压力测试、DE和预算 |
| agent_protocol.py | JSON接口和反馈信息过滤 |
| scripts/run_codex_agent.py | 真实模型调用、记录和错误处理 |
| study.py、study_report.py | 实验矩阵、完整性检查和报告 |

测试目录中的人工数据仅用于单元测试，不计入实验结果。研究报告应同时说明：反馈是否有用、金融规则是否已经很强、数值优化是否更快，以及价格与敏感度验证是否给出不同判断。
