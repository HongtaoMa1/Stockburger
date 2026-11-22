# Stockburger（股堡）

面向个人与中小算力场景的量化研究「算法生成器」：研究者以自然语言提出策略假设或交易约束，闭源前沿大模型（顶层）将其转译为结构化研究方案并调度迭代，本地确定性脚本与消费级显卡（中间层）批量训练、筛选候选特征与模型，最终冻结并交付专注特定细分领域的轻量化量化模型（底层）。这一架构旨在避开机构在私有数据覆盖、大型投研团队与低延迟执行设施上的结构性壁垒，将大模型的研究规划能力与本地严格的多重检验、防泄漏评估协议结合起来。

An "algorithm generator" for individual and resource-constrained quantitative research: a human researcher states hypotheses or constraints in plain language; a frontier LLM (the *top layer*) translates them into structured research plans and directs iterative search; deterministic local scripts and consumer GPU hardware (the *middle layer*) batch-train and screen candidates; and the pipeline freezes and delivers a compact, domain-specific quantitative model (the *bottom layer*). Rather than competing head-on with institutional advantages in proprietary data, large research teams, or low-latency infrastructure, Stockburger pairs LLM-driven research planning with strict local protocols for multiple-testing control and data-leakage prevention.

- **初始计划 / Initial plan**：[Stockburger-初始计划.pdf](Stockburger-初始计划.pdf)（中文 / in Chinese）
- **作者 / Author**：Hongtao Ma (Rick)
- **首次发布 / First published**：2026-10-21
- **当前版本 / Current version**：0.2.0（2026-11-22）
- **项目状态 / Status**：协议工具包原型已发布；生成器训练与演化流水线仍在内部迭代 / Protocol toolkit prototype released; the training and evolution pipeline remains in internal development
- **许可摘要 / License**：文档采用 [CC BY-NC 4.0](LICENSE)，代码采用 [PolyForm Noncommercial 1.0.0](LICENSE-CODE)，附个人自有资金使用补充许可（详见文末[版权与许可](#版权与许可--copyright--license)）/ Documents under [CC BY-NC 4.0](LICENSE), code under [PolyForm Noncommercial 1.0.0](LICENSE-CODE), with an additional permission for personal proprietary trading (see [Copyright & License](#版权与许可--copyright--license))

---

## v0.2.0：协议工具包 / The Protocol Toolkit

在大模型参与的自动化量化研究中，如果缺乏对数据切分、试验记账和反馈通道的硬性约束，多轮搜索极易演变为对历史噪声的反复迎合或隐性前视泄漏。

In LLM-assisted quantitative research, an unconstrained iterative loop can easily overfit historical noise or leak future information unless data partitioning, trial accounting, and feedback channels are strictly governed.

本次发布的 **v0.2.0** 是生成器的**评估与研究协议工具包**（`stockburger/`）：将模型训练、筛选与验收全流程必须遵循的统计与工程规则——时间序列三段切分与标签窗清洗、只追加试验账本、多重检验阈值与自相关校正、测试集单次盲考、顶层汇总表信息隔离、大模型知识截止期审计，以及确定性的需求收单规则——实现为可独立调用、带完整单元测试的 Python 模块。

This release (**v0.2.0**) provides the **research-protocol and evaluation toolkit** (`stockburger/`): standalone, unit-tested Python modules that codify the statistical and engineering rules governing the generator loop — temporal data splitting and label-window purging, append-only trial accounting, multiple-testing thresholds and autocorrelation corrections, one-shot test-set examination, information isolation for the planning LLM, knowledge-cutoff leakage auditing, and deterministic request intake.

### 本版包含与不包含的内容 / What Is and Is Not Included

- **本版包含 / Included in this release**：
  - `stockburger/`：10 个协议与评估核心模块（纯 Python 实现，依赖 `numpy`、`pandas`、`scipy`、`statsmodels`，可选依赖 `arch`）。
  - `demo.py`：基于合成市场的端到端主线演示，串联从数据切分、粗筛、登记、过闸、汇总、冻结考试、影子记分到需求收单的全部步骤，并附 200 个合成世界的统计频率表。
  - `tests/`：覆盖各模块协议规则与边界条件的单元测试集。

- **本版未包含（仍在内部迭代）/ Not included (still in internal development)**：
  - **生成器训练流水线**：真实行情数据接入、特征计算、底层模型训练与回测执行。
  - **跨代演化机制**：候选特征与模型在多轮迭代中的生成、变异与改进策略。
  - **具体研究课题资产**：任何实际研究课题的模型权重、配置文件、行情数据或评估报告。本仓库不含任何真实市场数据，亦不输出任何交易信号。

  *Not included:* the generator training pipeline (real market data ingestion, feature engineering, model training, and backtesting), the cross-generation evolution mechanism, and any models, configurations, datasets, or reports for specific research topics. This repository contains no market data files, trains no predictive models, and generates no trading signals; the bundled demo runs exclusively on synthetic data.

---

## 核心概念与模块结构 / Core Concepts & Modules

### 三个核心设定 / Three Core Concepts

- **S 段、V 段与测试段（时间序列三段切分）**：历史交易日按时间先后划分为 **S 段（搜索段）**、**V 段（验证段）** 与 **测试段**，相邻段之间保留不短于标签期的清洗空档（purging gap）。S 段用于训练、调参与粗筛；V 段仅用于最终那一次多重检验显著性判定（闸门）；测试段为最近一段历史，仅对最终确认的唯一候选模型执行一次盲考。

  **S, V, and test segments (chronological three-way split)**: Historical trading days are partitioned chronologically into **S (search)**, **V (validation)**, and **test** segments, separated by purging gaps at least as long as the label horizon. S is used for training, tuning, and coarse screening; V is reserved for the final multiple-testing significance check; the test segment is the most recent window and examines the single confirmed candidate once.

- **闸门（V 段显著性检验与三值结论）**：在 V 段上执行的统计检验，输出 **「过闸」**、**「未过闸」** 或 **「不可认证」** 三种结论之一。按本项目评估口径（`U-6`），统计检验服务于诊断而非硬性准入：「过闸」与「未过闸」仅作参考标签，不在代码层面阻断后续模拟盘或实盘使用；唯有发生可查证的协议破坏（如访问 V 段后篡改已登记参数、送检数或轮数越界、清洗检查未通过、账本显示提前触碰测试段）时，才判定为 **「不可认证」**，宣告当次认证无效。

  **Gate (V-segment significance test and three verdicts)**: The statistical test on V returns one of three verdicts: **过闸 (*pass*)**, **未过闸 (*fail*)**, or **不可认证 (*uncertifiable*)**. Under the project's evaluation convention (`U-6`), statistical gates serve as diagnostic references rather than hard deployment blockers: neither *pass* nor *fail* prevents paper or live use. **不可认证 (*uncertifiable*)** is triggered only by a verifiable protocol breach (e.g., altering parameters after pre-V registration, exceeding registered trial or round caps, failing the purge check, or premature test-segment access), which voids the result.

- **顶层与信息隔离**：指挥研究全程的闭源前沿大模型称为「顶层」（本版尚未接入大模型 API，由确定性脚本代替）。为防止顶层在多轮迭代中根据验证集或测试集的数值反馈做适应性微调，顶层每轮仅能读取一张经白名单校验的汇总表——表中 V 段只给出三值结论、不暴露具体统计量，测试段成绩与冻结后的影子成绩则严格隔离在仅面向人类研究者的日报中。

  **Top layer and information isolation**: The frontier LLM directing the research process is termed the *top layer* (not connected in this release; deterministic scripts stand in for it). To prevent adaptive overfitting across rounds, the top layer reads only a whitelisted summary table each round: V-segment results appear strictly as three-valued verdicts without numerical statistics, while test-segment scores and post-freeze shadow performance are isolated in human-only daily reports.

### 模块说明 / Modules (`stockburger/`)

- **`segments.py` · 三段切分、样本清洗与日历块账本 / Temporal splitting, purging, and calendar-block ledger**

  将交易日序列切分为 S / V / 测试三段，并剔除标签窗口（计算单个样本标签所需的未来交易日区间）跨入下一段的样本，防止跨段前视泄漏。内置日历块账本按「市场 × 日历块」（默认按月）持久记录各区间的使用历史：追踪哪些日历块曾作为 V 段服务过多少个任务（V 段允许跨任务复用，自第 2 个任务起在报告与汇总表中自动加注），以及测试段已被哪些需求考过（同一需求仅限考一次，重考即判「不可认证」）。

  Splits trading days into S / V / test segments and purges samples whose forward label window extends into the next segment. A calendar-block ledger, keyed by market × calendar block (monthly by default), records which blocks have served as V and for how many tasks (V reuse is permitted with an automatic annotation from the second task onward), and which requests have examined the test segment (one exam per request; any retake is marked *uncertifiable*).

- **`registry.py` · 只追加试验账本与考前登记 / Append-only trial ledger and pre-V registration**

  按候选模型逐行记录试验轨迹，中途终止或放弃的候选一并入账，账本仅支持追加、不可删改，以保证多重检验中的实际试验总数 `N_T` 统计真实完整。在首次读取 V 段数据前，必须登记并锁定切分指纹、送检上限、轮数上限、显著性水平与检验统计量规则；登记后若发生参数变更或计数越界，自动触发「不可认证」。

  Logs one immutable row per candidate trial — including aborted and abandoned runs — in an append-only ledger so the total trial count `N_T` is tracked honestly. Split fingerprints, send caps, round limits, significance levels, and test-statistic rules must be registered and frozen before V is touched; any subsequent modification or cap overrun renders the run *uncertifiable*.

- **`screen.py` · 候选收益口径、紧缩夏普比率（DSR）与粗筛 / Return convention, Deflated Sharpe Ratio, and screening**

  统一候选策略的截面收益核算口径：每日按模型预测分数的截面去均值名次构建零投资多空组合（多头权重合计 +1、空头合计 −1，不计交易成本）。同时按实际试验总数 `N_T` 计算紧缩夏普比率（Deflated Sharpe Ratio, DSR；Bailey & López de Prado, 2014），对因多重尝试而虚高的样本内最优夏普比率作选择偏差折减。

  Standardizes candidate returns as a daily zero-investment cross-sectional long–short portfolio weighted by demeaned score ranks (longs sum to +1, shorts to −1, before transaction costs). Computes the Deflated Sharpe Ratio (DSR; Bailey & López de Prado, 2014) to discount in-sample Sharpe ratios for selection bias across `N_T` trials.

- **`gate.py` · 闸门阈值与自相关稳健检验统计量 / Gate threshold and autocorrelation-robust test statistics**

  根据送检候选数 `N` 与显著性水平 `α` 计算过闸阈值 `κ(N, α)`，同时兼顾 `N` 个纯噪声候选的最大值期望与 Bonferroni 多重检验校正，并反推给定检验功效下的最低可识别夏普比率 `SR_mde`。针对陈旧报价（stale quotes）、非同步成交或涨跌停封板引起的一阶正自相关（会夸大常规 t 值），先在 S 段检验一阶自相关系数 `a₁`：若不显著则采用异方差稳健 t，若显著则自动切换为 HAC（Newey–West）t 或块自助（block-bootstrap）t。

  Computes the gate threshold `κ(N, α)` as a function of the number of submitted candidates `N`, accounting for both the expected maximum of `N` pure-noise trials and the Bonferroni correction, along with the minimum detectable Sharpe ratio `SR_mde`. To guard against spurious significance caused by stale quotes or limit-locked prices, it tests lag-1 autocorrelation `a₁` in S and automatically switches from a heteroskedasticity-robust t to a Newey–West HAC t or block-bootstrap t when `a₁` is significant.

- **`summary.py` · 顶层汇总表与白名单断言 / Top-layer summary table and whitelist assertion**

  构建顶层大模型在每轮迭代中唯一可见的输入表，分为常数区、S 区（展示用 DSR 及人类给定的基线规则）与 V 区（仅含三值结论，不含任何 V 段数值）。每次生成汇总表后执行字段白名单断言，确保测试段成绩、影子成绩与审计标签不会混入顶层上下文。

  Builds the single summary table exposed to the top layer each round, structured into Constants, S-zone (display-only DSR scores and human-specified baselines), and V-zone (three-valued verdicts only, with all V-segment numerical statistics stripped). Passes every assembled table through a schema whitelist assertion.

- **`product.py` · 成品冻结、测试段盲考与影子记分 / Model freezing, one-shot test exam, and shadow scoring**

  送交测试段考试的模型在考前即执行冻结，记录冻结时间戳及三份 SHA-256 哈希（模型权重、特征清单、顶层提案清单）；成品自冻结起即标记为「可用」。冻结后，定时脚本每日以冻结权重在后续新到行情上推理记分（称为「影子」），并在面向研究者的日报中将测试段成绩、影子成绩与已运行天数并排呈现。

  Freezes the selected model prior to test-segment examination with a timestamp and three SHA-256 hashes (weights, feature list, and top-layer proposals), marking the product usable immediately upon freeze. Post-freeze, a daily shadow-scoring routine evaluates the frozen weights on newly realized market data and prints test-segment metrics, shadow metrics, and elapsed trading days side by side in the human daily report.

- **`intake.py` · 规则化需求收单 / Deterministic request intake**

  将自然语言需求解析为带原文出处的结构化「需求卡」，并由确定性规则脚本判定 **「拒 / 问 / 转译」**。仅三类需求会以「拒」结案：要求保本或零风险且目标收益高于无风险基准（`F1`）、明确依赖亚秒级响应或主机托管等高频设施（`F2`）、目标市场在数据登记表中无任何价格源（`F4`）；其余情形均通过反问澄清或转译为可检验的慢频/代理方案处理。

  Parses plain-language requests into structured requirement cards with source quotes and applies deterministic rules to decide **reject / ask / translate**. Only three conditions result in outright rejection: demanding guaranteed returns above the risk-free benchmark (`F1`), explicitly requiring sub-second or co-located execution infrastructure (`F2`), or targeting a market with no registered price source (`F4`); all other ambiguities are resolved by asking clarifying questions or translating to testable proxies.

- **`records.py` · 顶层调用审计账与静态分支检查 / Top-layer audit log and static branch check**

  大模型的预训练语料可能覆盖测试期行情，形成知识截止期层面的前视泄漏。本模块在每次调用顶层时记录模型 ID、厂商申报的知识截止月、与测试段的重叠月数及污染标签 `L_label`（`clean` / `partial` / `contaminated`），仅作日报参考；另提供基于 Python AST 的静态检查，核验业务代码未利用 `L_label` 或重叠月数改变闸门判定或选模流程。

  Because an LLM's pretraining corpus may overlap the test window (look-ahead leakage), every top-layer call is logged with its model ID, vendor-reported knowledge cutoff month, months of overlap with the test window, and cutoff label `L_label` (`clean` / `partial` / `contaminated`) for audit reference. Includes an AST-based static check verifying that no code path branches on `L_label` or overlap duration.

- **`toplayer_prompt.py` · 顶层提示组装与测试期禁令句 / Top-layer prompt builder and test-period ban sentence**

  按上下文白名单组装发往顶层的英文提示词：首轮（`R0`）仅包含需求原文、特征族字段说明与测试窗起止日期，不含任何行情或特征数值；默认植入一条明确禁止调用测试期市场记忆的「禁令句」（可配置开关，并记录句子变体与哈希供审计核对）。

  Assembles whitelisted English prompts for the top layer: initial (`R0`) prompts carry only the request text, feature-family schema definitions, and test-window dates, with zero data values. By default, inserts a standardized "ban sentence" instructing the model not to use remembered information from the test period.

- **`synth.py` · 合成市场生成器 / Synthetic market generator**

  生成供单元测试和端到端演示使用的合成截面行情（默认 40 只资产 × 1500 个交易日），包含三类候选特征：与收益独立同分布的**纯噪声**、具有真实跨期预测力的**植入信号**，以及模拟陈旧报价/涨跌停平滑效应（零均值但具显著一阶正自相关）的**旧价特征**。

  Generates synthetic cross-sectional market data (40 assets × 1,500 trading days by default) for unit tests and the walkthrough demo, featuring three candidate archetypes: **pure noise**, a **planted signal** with genuine predictive power, and a **stale-price** feature with zero expected return but strong positive lag-1 autocorrelation.

---

## 怎么跑 / How to Run

环境要求：Python 3.10 或更高版本（已在 Python 3.12 与 3.13 上验证）。代码注释、docstring 与命令行演示输出均为中文。在仓库根目录下执行：

Requires Python 3.10 or later (verified on Python 3.12 and 3.13). Code comments, docstrings, and CLI output are in Chinese. From the repository root:

```bash
pip install -r requirements.txt
python -m pytest -q             # 运行全部单元测试 / run all unit tests
python demo.py                  # 完整演示：主线流程 + 200 个合成世界的频率附表 / walkthrough + 200-world frequency table
python demo.py --worlds 0       # 仅运行单世界主线流程 / single-world walkthrough only
python demo.py --seed 0         # 换用未挑选的随机种子观察对比 / run with an unpicked synthetic seed
python demo.py --stale-mc 4500  # 附加核对：旧价策略零分布的直接蒙特卡洛检验 / direct Monte Carlo check on the stale-price null
```

在普通笔记本电脑上，完整单元测试约需 15 秒，`python demo.py` 约需 20 秒，仅跑主线（`--worlds 0`）约 1–2 秒。`arch` 为可选依赖，仅在登记使用块自助 t 统计量（`block_bootstrap_t`）时调用；未安装 `arch` 时，与之相关的两条单元测试会自动跳过。

On a standard laptop, the unit test suite takes ~15 seconds, the full demo ~20 seconds, and the walkthrough alone (`--worlds 0`) 1–2 seconds. `arch` is an optional dependency required only when the block-bootstrap t-statistic is selected; without it, the two corresponding unit tests are skipped automatically.

### 演示看点与种子说明 / What the Demo Shows

主线演示在合成市场上完整走通 12 个步骤（第 0 步至第 11 步），重点展示三类候选特征在闸门处的分化表现：
1. **纯噪声候选**：未能通过 V 段闸门；
2. **植入信号候选**：稳定通过 V 段闸门并进入测试段盲考；
3. **旧价候选（伪自相关）**：均值真实为 0，但因价格平滑带有一阶正自相关（`a₁` ≈ 0.5）。若一律使用常规异方差稳健 t，其统计量会越过阈值 `κ`（看似显著）；按协议（`B14`）在 S 段检出显著 `a₁` 并自动切换为 HAC t 后，被成功拦截在闸门外。

The walkthrough steps through all 12 stages (Steps 0–11) on synthetic data and highlights how the V-segment gate separates the three feature archetypes: (1) **pure-noise** candidates fail the gate; (2) the **planted signal** passes; and (3) the **stale-price** candidate (true mean zero, lag-1 autocorrelation ≈ 0.5) exceeds `κ` under an ordinary heteroskedasticity-robust t, but is stopped once the protocol detects significant S-segment autocorrelation (`B14`) and switches to the Newey–West HAC t.

**关于默认种子（`--seed 10`）与有限样本边界的说明**：
默认种子 `10` 是特意挑选的演示种子——从 `0` 往上枚举，它是第一个在单次运行中同时呈现上述三种典型结果的合成世界。为了如实展示这一挑选的选择性，`demo.py` 默认在末尾统计并打印 `0–199` 共 200 个合成世界里各事件的出现频率。若换用未挑选的 `--seed 0`，可以看到另一面：该世界中的旧价候选即便切换为 HAC t 依然越过了阈值 `κ`。这说明在有限样本下，切换自相关稳健统计量能显著降低伪过闸率，但无法将尺度畸变完全归零——这也是本协议坚持将「过闸 / 未过闸」定位于诊断参考、而非绝对准入保证（`U-6`）的方法论原因之一。

**Note on seed selection (`--seed 10` vs. `--seed 0`) and finite-sample limits**:
The default seed `10` was deliberately selected for pedagogical clarity — counting up from `0`, it is the first synthetic world where all three outcomes co-occur in a single run. To make the selectivity of that pick transparent, `demo.py` appends a 200-world Monte Carlo table (`seeds 0–199`) tallying how often each archetype enters V and passes the gate. Running `python demo.py --seed 0` shows an unselected world where the stale-price candidate passes even under the HAC t. In finite samples, switching to a HAC estimator substantially reduces false positives from serial correlation but cannot eliminate size distortion completely — one reason why a gate pass is treated strictly as a diagnostic reference rather than a guarantee (`U-6`).

---

## 条款编号对照 / Clause Index

代码注释、docstring 与程序输出中的 `A1`、`B7`、`C3` 等编号，源自作者为生成器流水线制定的内部技术规格《验收标准 v007》（全文未公开；`v007.1` 为 2026-11-22 的小幅修订，明确同一段 V 可供多个任务复用并加注）。`U-2`、`U-6` 等为项目的常设工程原则，`H8` 为立项前提之一。收单模块中的 `F1–F9″` 为「拒 / 问 / 转译」的命中原因码，完整速查表见 `stockburger/intake.py` 顶部文档字符串。代码中提及的「第 X 节」对应内部规格章节：一 总架构、二 数据与切段、三 S 段训练筛选与试验账本、四 V 段闸门、五 汇总表与人读报告、六 测试段考试与成品、七 顶层与前视泄漏、八 收单与拒单、九 开跑前要登记的默认值、十（本版未涉及）、十一 后续扩展项。下表按本版代码的实际实现列出各编号的要点。

The `A1` / `B7` / `C3`-style identifiers in code comments, docstrings, and CLI output refer to clauses in the author's internal pipeline specification, *Acceptance Criteria v007* (unpublished in full; `v007.1` is a minor revision dated 2026-11-22 allowing a single V segment to serve multiple tasks with an annotation). `U-x` identifiers denote standing project principles, and `H8` is a founding premise. The `F1–F9″` codes in intake results are reason codes for *reject / ask / translate* decisions (a complete reference sits in the module docstring of `stockburger/intake.py`). Section references in comments (e.g., 第八节, "Section 8") correspond to the specification's chapters: (1) Overall Architecture, (2) Data & Segmentation, (3) S-Segment Screening & Trial Ledger, (4) V-Segment Gate, (5) Summary Table & Human Reports, (6) Test-Segment Exam & Product, (7) Top Layer & Look-Ahead Leakage, (8) Request Intake & Rejection, (9) Pre-Run Registered Defaults, (10) unused in this release, and (11) Future Extensions. The table below summarizes each clause as implemented in `v0.2.0`.

| 编号 / ID | 实现要点 | Implementation Summary |
|---|---|---|
| A1 | 历史数据按时间切分为 S 段 → V 段 → 测试段，段间空档不短于标签期；切分日期仅限人工设定或取默认值，顶层无权设定或修改。 | History is split chronologically into S → V → test, with inter-segment gaps at least as long as the label horizon; split dates are set by a human or by default, never by the top layer. |
| A2 | 测试段仅由考试脚本读取，且仅对一个经人工确认的候选执行（无人工在场时按预登记规则代为确认，标注「AI 代确认」）；同一需求仅考一次，重考判「不可认证」；不同需求复用同一测试段时，自第 2 次起加注。 | Only the exam script reads the test segment, for a single human-confirmed candidate (or one chosen by a pre-registered rule, marked `AI 代确认`); each request is examined at most once (a retake is *uncertifiable*); subsequent exams of the same period by other requests carry an annotation. |
| A3 | 样本清洗：标签窗口跨入下一段的样本一律剔除；S 段内部切分交叉验证折时遵循同一清洗规则。 | Purging: drop every sample whose forward label window extends into the next segment; the same rule applies to cross-validation folds inside S. |
| A4 | 顶层上下文由脚本按白名单拼装（需求原文、字段说明、汇总表、新颖性标记），不暴露逐期原始数值（附可选的白名单校验）。 | The top layer's context is assembled from a strict whitelist — request text, field descriptions, summary table, and novelty marks — excluding per-period data values (with an optional whitelist check). |
| A6 | 经验设定值（标注 `[GUESS]`，如闸门显著性水平 `α`）在登记锁定前仅用于反问、标注或展示，不得作为拒单或淘汰候选的依据。 | Heuristic default parameters (marked `[GUESS]`, e.g., gate level `α`) may only be queried, flagged, or displayed prior to registration — never used to reject a request or discard a candidate. |
| A6′ / U-6 | 统计评估是诊断参考而非硬性准入：「过闸 / 未过闸」均不阻断候选进入模拟盘或实盘；唯有协议被破坏（「不可认证」）才视为认证失效。 | Honest evaluation is a reference, not a blocker: neither *pass* nor *fail* prevents paper or live deployment; only a protocol breach (*uncertifiable*) invalidates certification. |
| A7 / C6 | 测试段成绩、影子成绩、审计账与 `[GUESS]` 标注等仅写入面向研究者的日报，一律不进入顶层读取的汇总表。 | Test scores, shadow scores, audit logs, and `[GUESS]` annotations appear only in the human daily report and are excluded from the top layer's summary table. |
| A8 | 需求中明确给出的硬规则（如「跌 x% 买入，持有 20 日卖出」）作为基线候选，在汇总表 S 区单独列示。 | An explicit rule stated in the request (e.g., "buy on an x% drop, hold 20 days") is treated as a baseline candidate and listed separately in the S zone. |
| A9 | 研究方向由人指定：需求可点名具体论文或方法；若点名方法无法映射为可检验方案，则转入反问澄清。 | Research direction comes from the human: requests may name specific papers or methods; if a named method cannot be mapped to a testable plan, intake asks for clarification. |
| B1 | 汇总表分为常数区、S 区与 V 区；S 段成绩采用 DSR 形式并标注「展示用」；V 区仅输出三值结论，不暴露任何 V 段数值统计量。 | The summary table comprises Constants, S, and V zones; S scores are reported as DSR marked "display only"; the V zone reports only three-valued verdicts without V-segment numbers. |
| B3 | 首次访问 V 段前登记并锁定参数（切分日期、`α`、`β`、`T_val`、送检上限、轮数上限、t 统计量规则等），据此计算阈值 `κ` 与最低可识别夏普 `SR_mde`；登记后再修改即视为协议破坏。 | Before touching V, register and freeze parameters (split dates, `α`, `β`, `T_val`, send cap, round cap, choice of t, etc.), from which `κ` and `SR_mde` are derived; any post-registration change is a protocol breach. |
| B4 | 过闸阈值 `κ(N, α) = max{E_max(N) + Φ⁻¹(1−α), Φ⁻¹(1−α/N)}`，其中 `N` 为送检候选数；当且仅当 `t ≥ κ` 时判「过闸」。 | Gate threshold `κ(N, α) = max{E_max(N) + Φ⁻¹(1−α), Φ⁻¹(1−α/N)}`, where `N` is the number of candidates sent to V; a candidate passes iff `t ≥ κ`. |
| B5 | 闸门结论取「过闸 / 未过闸 / 不可认证」三值；仅当协议被破坏（登记被篡改、送检数或轮数越界、清洗检查未通过、账本显示曾触碰测试段）时判「不可认证」。另设仅作提示、不改变结论的「闸门无力认证」标记。 | Three gate verdicts: *pass* / *fail* / *uncertifiable*; *uncertifiable* is returned strictly upon a protocol breach (registration altered, counts exceeded, purge failure, or test-segment access). A separate "gate underpowered" warning never alters the verdict. |
| B6 / B10 / B22 | 闸门报告固定声明认证范围（仅校正循环内选择偏差）；过闸候选额外附印「族错误率保证不适用 / 未核」。 | Standardized disclosures in the gate report stating the certification scope (corrects in-loop selection bias only) and, for passing candidates, "family-wise error guarantee not applicable / unverified". |
| B7 | 试验账本跨任务共用且只追加，每个接触数据的候选记一行（含中止、放弃），据此导出试验总数 `N_T`；碰数据前被格式或前视校验拒掉的提案另计 `n_prop`。候选收益统一按日频截面多空口径核算。 | The trial ledger is append-only and shared across tasks, logging one row per data-touching candidate (including aborted/abandoned runs) to export `N_T`; proposals rejected before touching data are counted separately as `n_prop`. Candidate returns follow the daily cross-sectional long–short convention. |
| B9 / B16 | V 段账本按「市场 × 日历块」记账；同一段 V 允许供多个任务复用，自第 2 个任务起在报告与汇总表中加注，不判「不可认证」；访问 V 前中止的任务不计入复用次数。 | The V ledger is keyed by market × calendar block; a V segment may be reused across multiple tasks with an annotation from the second task onward (not a protocol breach); tasks aborted before touching V are not counted. |
| B11 | S 段粗筛后一次性送检、不设阶梯分级（`L = 0`），默认送检上限 `n_fin ≤ 5`；送检候选越多，过闸阈值 `κ` 越高。 | Coarse-screen in S and submit to V in a single batch without multi-tier staging (`L = 0`), `n_fin ≤ 5`; submitting more candidates raises the threshold `κ`. |
| B14 | 检验统计量选择：先在 S 段检验日收益的一阶自相关系数 `a₁`；不显著时采用异方差稳健 t，显著时自动切换为预登记的 HAC t 或块自助 t。 | Statistic selection: test lag-1 return autocorrelation `a₁` in S; if insignificant, use the heteroskedasticity-robust t; if significant, switch automatically to the pre-registered HAC t or block-bootstrap t. |
| B19 | V 段默认紧邻测试段之前，即便曾供其他任务使用亦不前移；包含已考日历块时照常使用，并在登记回执中列明。 | V is placed immediately before the test segment by default and is not shifted backward even if used by prior tasks; any previously examined blocks within V are retained and itemized in the registration receipt. |
| B25 | 成品一经冻结即可用；测试段考试成绩与闸门认证状态仅作为附加标签展示，任何代码路径均不得据其阻断使用。 | A product is usable as soon as it is frozen; exam scores and gate statuses are informational labels only, and no code path may block deployment based on them. |
| C1 | 顶层审计账：每次调用顶层记录一行——模型 ID、调用日期、厂商申报知识截止月、测试段区间、重叠月数、截止日标签 `L_label`、所用禁令句变体及其哈希。 | Top-layer audit log: one row per LLM call recording model ID, call date, vendor-reported cutoff month, test window, overlap months, cutoff label `L_label`, and the ban-sentence variant and hash. |
| C2 | 业务代码严禁依据 `L_label` 或重叠月数驱动闸门判定、挑选候选或改变执行分支（配套可选的 AST 静态检查）。 | Code must never branch on `L_label` or overlap months to drive gate verdicts, candidate selection, or workflow routing (enforced via an optional AST static check). |
| C3 | 送交测试段考试的模型即为冻结成品；冻结时记录时间戳与三份哈希（权重、特征清单、提案清单）；冻结后若重新训练（含加入测试段数据重训）即视为新产物，仍可使用，但不再挂载原考试成绩。 | The examined model is the exact artifact that gets frozen with a timestamp and three hashes (weights, features, proposals); any retraining after the freeze produces a new product that remains usable but does not inherit the prior exam score. |
| C4 / C5 | 影子记分：成品冻结后，每日以冻结权重在新到行情上推理记分；日报将测试段成绩、影子成绩与已运行天数并排打印；统计量 t 仅作展示，不设过线判定。 | Shadow scoring: after freezing, score daily on new market data using the frozen weights; the daily report displays test scores, shadow scores, and days run side by side, reporting t-values without pass/fail thresholds. |
| C8 | 新颖性标记仅限「已知 / 未匹配」两值，禁用「新发现」等表述，亦不用于过滤候选（本版仅校验标记合法性，不含相似度计算实现）。 | Novelty labels are restricted to `已知` (*known*) or `未匹配` (*unmatched*) — never "new discovery" — and are never used to filter candidates (this release validates label values without computing similarity). |
| C13 | 禁令句：发往顶层的提示默认附加「禁止利用测试期市场记忆」指令（可关闭），置于任务指令之后、作答格式之前；不带汇总表的提示用 `W2`，带汇总表的用 `W1`。 | Ban sentence: top-layer prompts include a "do not use test-period market knowledge" instruction by default (toggleable), placed between the task directive and the output format (`W2` for schema-only prompts, `W1` when a summary table is attached). |
| D1–D3 | 需求收单流水线：需求原文 → 结构化需求卡（固定字段，逐项标注来源与原文引文）→ 确定性规则脚本判定「拒 / 问 / 转译」；当原文表述与字段抽取结果存在冲突、或用户对字段解读提出异议时，原本触发拒单的条款降级为反问。 | Request intake pipeline: raw text → structured requirement card (fixed fields with source attribution and quotes) → deterministic rule script deciding *reject / ask / translate*; if raw text and extracted fields conflict, or if the user disputes a field reading, a would-be rejection downgrades to a clarifying question. |
| D4 | 要求零风险或保本，且目标收益率高于短期国债与政府担保存款利率中的较大者 → 拒单（`F1`）；否则转入反问（`F1′`）。 | Demands zero risk or capital protection with a target return above the higher of the T-bill and insured-deposit rates → reject (`F1`); otherwise ask (`F1′`). |
| D5 | 明确要求亚秒级响应、机房托管或逐笔订单簿报撤单设施才能成立 → 拒单（`F2`）；若存在不依赖高频设施的慢频同类机制则转译（`F2′`）。 | Explicitly requires sub-second latency, co-location, or order-book-level execution infrastructure → reject (`F2`); if a slower-frequency analogue exists, translate (`F2′`). |
| D6 | 用户点名交易衍生品时只问不拒（`F3`）；顶层自行推断出的衍生品敞口自动转译为非衍生品代理候选（`F3′`）。 | User-named derivatives trigger a clarifying question rather than rejection (`F3`); derivative exposure inferred by the top layer is translated to non-derivative proxies (`F3′`). |
| D7 | 目标市场在数据登记表中无任何可用价格源 → 拒单（`F4`）；点名特征缺失时，有代理特征则替换、无代理则剔除并注明（`F4′`）。 | Target market has no registered price source → reject (`F4`); unavailable named features are replaced by registered proxies or dropped with a note (`F4′`). |
| D8 | 风险偏好与收益诉求明显冲突时转入反问（`F6`）；反问轮数达到上限时将任务挂起，不作拒单处理。 | Conflicting risk tolerance and return goals trigger a clarifying question (`F6`); reaching the question-round limit suspends the ticket rather than rejecting it. |
| D9 | 面向研究者展示的评估指标优先从需求阐述的市场观点与尾部风险情景中提取；均未指明时才默认使用夏普比率。 | Human-facing evaluation metrics are derived from the request's stated market view and tail-risk scenario, defaulting to the Sharpe ratio only when neither is specified. |
| D10 | 研究方案抬头固定展示「在当前送检上限与验证期长度下，纯噪声候选预期可达的表观成绩」，仅作直观对照。 | The plan header displays the expected apparent Sharpe ratio reachable by pure-noise candidates given the send cap and validation window length, for reference only. |
| D11 | 考前工程审查（前视特征、标签对齐、幸存者偏差等核查项）结果照实印入日报，不阻断流程；本版负责将该审查结果格式化输出至日报。 | Pre-exam engineering checklist results (look-ahead features, label alignment, survivorship bias, etc.) are printed in the daily report without blocking execution; this release formats and prints the checklist entries. |
| D12 | 策略新颖度不足、顶层怀疑难以盈利、表述宽泛、高风险高收益偏好或日内交易诉求，均不构成拒单理由。 | Lack of novelty, top-layer skepticism about profitability, vague phrasing, high-risk / high-return preferences, or intraday horizons are never grounds for rejection. |
| H8 | 测试段仅由考试脚本读取，依赖研究者与调用方的工程纪律，不设机器级防篡改校验（账本仅能拦截按规申报的读取）。 | Restricting test-segment access solely to the exam script relies on operational discipline rather than cryptographic enforcement (ledgers only track calls that register through the protocol API). |
| U-2 | 衍生品定价类研究暂缓支持；把可交易衍生品作为投资标的默认不作限制（因此收单环节对点名衍生品只问不拒）。 | Derivative-pricing research is currently out of scope, whereas trading listed derivatives is not restricted by default (hence intake asks rather than rejects). |
| U-3 | 顶层大模型知识截止日覆盖测试期所造成的前视泄漏不设硬性阻断闸门，采取尽量规避、如实记录供参考的原则。 | Look-ahead leakage via the top layer's pretraining cutoff has no hard blocking gate; it is mitigated where possible and logged in the audit trail for reference. |
| U-5 | 工程卫生类核查保持为轻量级断言与审计记录，不上升为阻断业务的硬规则。 | Engineering-hygiene checks remain lightweight assertions and audit logs rather than hard blocking rules. |

---

## 版本记录 / Changelog

| 版本 / Version | 日期 / Date | 内容 / Contents |
|---|---|---|
| 0.2.0 | 2026-11-22 | 协议工具包原型：核心模块代码、单元测试与合成市场端到端演示 / Protocol toolkit prototype: core modules, unit tests, and synthetic-market end-to-end demo |
| 0.1.0 | 2026-10-21 | 初始计划 PDF / Initial plan (PDF) |

---

## 版权与许可 / Copyright & License

Copyright © 2026 Hongtao Ma (Rick). All rights reserved except as licensed below.

| 内容 / Content | 许可 / License |
|---|---|
| 文档（含 PDF）/ Documents, including the PDF | [CC BY-NC 4.0](LICENSE) |
| 代码（`stockburger/`、`demo.py`、`tests/`）/ Code (`stockburger/`, `demo.py`, `tests/`) | [PolyForm Noncommercial 1.0.0](LICENSE-CODE) |

每个代码文件开头都有一行 `Required Notice` 版权声明；按 PolyForm 许可的要求，转发代码时请连同这些行和许可证一起保留。

Each source file begins with a `Required Notice` line; as the PolyForm license requires, keep these lines and the license with any copy you pass on.

**补充许可（对以上两份许可都适用）**：个人用自有资金做投资、交易决策，包括为自己盈利，视为非商业用途，允许。公司、基金等机构使用，或用来给他人提供服务，属于商业用途，须事先取得作者书面授权——请通过 GitHub 联系作者。

**Additional permission (applies to both licenses):** An individual may use this work to make investment or trading decisions for their own account with their own money, including for profit; this is treated as non-commercial use. Any use by or on behalf of a company, fund or other organisation, or to provide services to others, is commercial use and requires a separate written license from the author. Please contact the author via GitHub.

本项目及其代码、演示的任何输出都不构成投资建议。This project, including any output of its code or demo, is not investment advice.
