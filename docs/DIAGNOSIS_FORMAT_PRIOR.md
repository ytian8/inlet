# Inlet 的 reasoning 任务为什么崩，以及怎么修

## 一句话

问题不在 eval、不在 checkpoint 选择、也不在 pooling 的容量。**训练集的输出格式先验被写死在 100% 的训练 prompt 里，而它对 hypernetwork 的输入（description）完全不可见** —— 因此它只能被学成一个与输入无关的常数；而这个常数，恰好就是三个生成任务与训练分布之间的**唯一系统性差异**。

## 1. 代码里的那一行

`third_party/text-to-lora/src/hyper_llm_modulator/utils/preprocessing.py:29`

```python
if ds_name.startswith("lol_"):
    def f(example):
        task_def = txt.split("Definition: ")[1].split("\n\nPositive Example")[0]
        task_def += " Please complete the task without any explanation."   # ← 无条件
```

`LOL_TEMPLATE = "{task_def}\n\n{problem}"`，所以这句话就是 user prompt 的一部分。
训练集 479 个任务**全部**是 `lol_*` → **479/479** 条训练 prompt 都带这句话。

## 2. 它和成败的分界线完全重合

| 组 | n | prompt 含 "without any explanation" | assistant_prefill | 目标长度 | Inlet 相对 zero-shot |
|---|---|---|---|---|---|
| 训练任务 | 479 | **479/479** | `''` | 中位 1 词 | — |
| MC benchmark | 7 | **7/7** | `''` | 1 个字母 | **+4 ~ +12** |
| 生成 benchmark | 3 | **0/3** | 非空 | 100~200 token | **−21 ~ −33** |

7/7 对 0/3，符号完全跟着这条线翻转。

**训练目标长度实测**（随机 51 个训练任务 × 30 条 = 1530 条样本，用上游自己的 preprocessing）：

- 中位 **1 词**，均值 3.7，p90 = 11，p99 = 32，最长 83
- **65.9%** 的训练目标是单个词，89.6% 在 10 词以内
- 51 个任务中 31 个的中位目标就是 1 词

训练分布的 **p99（32 词）还够不到一条 gsm8k 的推理链**。

## 3. 为什么 hypernetwork 无法学会区分

关键的不对称：

- 训练任务的 description 里，**0/60** 提到 "without any explanation" —— 格式要求只存在于 LM 的 prompt，**从不出现在 hypernetwork 的输入里**
- gsm8k / humaneval / mbpp 的 description **有**格式词（"step by step"、"code"、"program"）

也就是说：训练期间，格式先验对 hypernetwork 是一个**与输入无关的常数**。它只能被学成常数 —— 而 `P = base + head(...) × emb_rms` 里的 `base` 正是承接常数的完美载体。等 eval 时 gsm8k 的 description 说 "step by step"，模型从未见过「格式随 description 变化」，也就没有任何已学到的通路去改变格式。

这一条同时解释了 `probe_prompt` 的结果：cos(真实 desc, 随机垃圾 desc) = **0.9999**，`varying_fraction` = 5.4%，随机描述对照 **C = +0.41 分**。hypernetwork 的输出几乎不依赖它的输入 —— 因为训练数据里唯一强而一致的信号，正是一个它的输入解释不了的东西。

## 4. 为什么 T2L 用同样的数据却不崩

T2L 官方 checkpoint（`trained_t2l/mistral_7b_t2l`，同一套任务、同一批 description）在生成任务上是 gsm8k **44.45** / mbpp **51.68** / humaneval **39.23**，**全部高于 zero-shot**。

所以：**光是「训练数据简短」并不足以导致崩溃，崩溃需要「input-layer 前缀」这个参数化方式**。一个 32 向量的前缀坐在 context 里，会被模型当作指令读取，格式先验因此被直接表达出来；LoRA 的权重扰动则不占据 prompt 位置，压不过 `` ```python `` 这样的 prefill。

**这正是这篇 paper 可能真正的贡献点**：soft prompt 对训练集格式先验的敏感度远高于 LoRA。

## 5. 已经排除的解释

| 假设 | 证据 | 结论 |
|---|---|---|
| eval 有 bug | 同一条 Inlet eval 路径、**0 个 virtual token** → humaneval **39.63**（已发表 zero-shot 37.80） | 排除 |
| BOS 位置错 | `baseline_ref.prepend_soft_prompt` 拼接方式相同，而 per-task prompt tuning 走同一路径拿到 gsm8k 49.61 | 排除 |
| 32 个向量的容量不够 | per-task prompt tuning 用同样的 32×4096 拿到 8 任务均值 **76.33**（Oracle LoRA 75.8） | 排除 |
| prompt norm 太大 | 4K step 时 `prompt_norm` ≈ 1.05× 真实 token，生成任务仍低于 zero-shot ~16 分 | 基本排除 |
| checkpoint 选错 | 确实是个 bug（按 dict 顺序选到 ~130k），修完 humaneval 4.47 → 21.54；但**仍远低于 39.63** | 部分因素，非根因 |

## 5b. 判决性对照：不是「前缀存在」有害，是训练把它变成有害的

在同一台 A40 上，humaneval，只换前缀，其余一切不变：

| 前缀 | base pass@1 | plus pass@1 | 生成长度中位 |
|---|---|---|---|
| 无前缀 | 38.41 | 31.71 | 139 词 |
| **32 个未训练向量**（`init_base_from_vocab`，逐位等于训练第 0 步） | **40.24** | 34.76 | 133 词 |
| 32 个随机方向向量（逐 token 范数对齐） | 31.10 | 25.61 | 136 词 |
| **训练过的 Inlet prompt**（先前 A100 实测） | **20.33**（130k step 时 4.47） | — | — |

读法：

- 在输入层挂 32 个**未训练**的真实 token 向量，**完全无害**（+1.8，2 道题）
- 随机方向扣 7.3 分 —— 有损，但离崩溃很远
- **训练过的 prompt 比同量级的纯噪声还差 11~27 分**

也就是说，「32 个向量占据了 prompt 位置」这个竞争解释被排除了。**是训练往里面放的东西有害**，而训练数据里唯一强而一致的信号就是 §1–§3 那个格式先验。

（注：本机 zero-prompt = 38.41，先前 A100 记录为 39.63，差 2 道题 / 164，属于已记录的硬件抖动。上表四行同机同 build，比较是干净的。）

## 5c. 判决性对照之二：那句话本身**不足以**复现崩溃 —— 这条否证了我假设的强版本

同一台机器，humaneval，zero-prompt，只在 prompt 前面加上训练集那句话：

| | base pass@1 | plus pass@1 | 生成长度中位 | <10 词 |
|---|---|---|---|---|
| 原始 prompt | 38.41 | 31.71 | **139 词** | 0% |
| + "Please complete the task without any explanation." | 36.59 | 31.10 | **75 词** | 8% |

**方向完全命中**：那句话让生成长度**腰斩**（139 → 75 词），这正是预测的行为。
**量级不成立**：它只扣了 1.8 分（3 道题），离 20.33、更离 4.47 差得很远。

所以必须修正结论。不是「prompt 学会了这一句话」，而是：

> 训练把这个格式先验学成了一个**远比英文指令强的杠杆**。soft prompt 不受「一句话能表达多少」的限制 —— 在 479 个简短任务上做 130k 步梯度下降，产出的前缀对简短的强制程度，**远超过那句一直陪着它的英文指令**。

配合 §5b 的对照，能站得住的因果链是：

1. 前缀存在本身无害（未训练前缀 40.24 ≥ 无前缀 38.41）
2. 训练放进去的东西有害，且比同量级噪声更有害（20.33 vs 31.10）
3. 有害的**性质**是格式/长度压制 —— 与训练分布一致，且那句话能定向复现这个方向
4. 有害的**量级**超出文本指令能达到的范围

这条链子里第 4 步是新的，也是我原来没有的。它把「模型学了一句话」换成了「连续 prompt 是一个无界的格式先验放大器」—— 后者更难，但也更有意思。

## 5d. gsm8k 的那一组否证了这个假设 —— 必须写在前面

同机，zero-prompt，只在 prompt 前加上训练集那句话：

| 任务 | 原 prompt | + 那句话 | 差 | 生成长度中位 |
|---|---|---|---|---|
| humaneval pass@1 | 38.41 | 36.59 | **−1.8** | 139 → 75 词 |
| **gsm8k acc** | 40.56 | **43.67** | **+3.1** | 130 → 114 词 |

humaneval 上它把生成长度腰斩却只扣 1.8 分；gsm8k 上它**反而有帮助** —— prefill
`Let's think step by step.` 仍然强制了 CoT，更简洁反而让最终数字抽取更干净。

**一句在两个任务上一个扣 1.8、一个加 3.1 的指令，解释不了 −20.7 和 −33。**
格式先验作为**因果解释**站不住了。confidence 从 ~80% → ~65% → **~25%**。

混淆本身仍然是真的（479/479 vs 0/3），仍然要修、要披露 —— 但它是一个 artifact，
不是病因。§5c 里我写的「无界放大器」也随之作废：我没有证据说被放大的就是它。

### 修正后立得住的三条，和一个空缺

1. **伤害来自训练，不来自前缀存在。** 两任务一致：未训练前缀 +1.8 / +1.9，
   随机方向 −7.3 / −6.0，训练过的 −18 / −20.7（130k step 时 −33）。
2. **训练过的 prompt 比同量级噪声还差**（humaneval 20.33 vs 31.10）。不是没学到，
   是主动学到了有害的东西。
3. **崩溃是 input-layer 前缀特有的**：同数据同 description，T2L 的 LoRA 不崩。
4. **空缺：那个有害的东西是什么，仍然不知道。** 最合理的剩余候选是训练目标的
   *分布*本身（中位 1 词、65.9% 单词），但**没测过**。不拿未测的解释去替补被
   否证的解释。

### 决定胜负的下一个实验

在 lol_ 训练任务的混合上**直接训练 32 个 soft prompt 向量（不走 hypernetwork）**，
在 humaneval / gsm8k 上评测。

* 崩到 ~20 → **训练分布本身就足够**，hypernetwork 无辜，修复动数据（F2）
* 不崩 → 问题在 hypernetwork 的优化（`base` 吸收、SFT 不奖励区分描述），修复动
  架构与损失（F5）

这一个实验把修复方向劈成两半，而两半现在先验差不多。代价：上游
`train_prompt_tuning.py` 没随仓库发布，`baseline_ref.py` 只重建了辅助函数，
所以要写训练代码 —— 约半天，加一台机器几小时。

## 6. 顺带发现的第二个 bug（影响 baseline 列）

`vllm_eval.py:196-210`，`eval_gsm8k` 里：

```python
problem = template.format(**sample)
if in_context_message:
    problem = in_context_message + "\n\n" + problem   # 算好了
...
MathSample(problem=template.format(**sample), ...)    # 又重算一遍，丢掉了
```

`in_context_message` 对 gsm8k 是**死代码**。所以我们表里的 "3-shot ICL" gsm8k = **39.95** 其实是 zero-shot，不是 3-shot。旁证：ICL 在 7 个 MC 上是 +0.2 ~ +12.5，唯独 gsm8k 是 −0.76。一行可修，但要修 —— 这是 reviewer 会查的 baseline。

## 7. 怎么改（按杠杆排序）

### F2 是承重墙，其余都是配套

**F2. 引入「输出格式随任务变化、且变化可由 description 预测」的训练数据**

这是唯一针对根因的修复。三个条件缺一不可：

1. 目标要长（CoT / 代码 / 多句）
2. 格式要**跨任务变化**
3. 变化要**能从 description 预测出来**

缺了第 3 条，任何架构改动都学不到条件化 —— 包括 cross-attention。

我查过一条本来最省事的路：Natural Instructions 的正例常带 `Explanation:` 字段，若在就能零成本造出长输出变体。**但这批 `lol_*` 数据集在建库时已经把它去掉了**（抽查的任务里 `Explanation:` 出现 0 次），这条路不通。

现实做法：从 Natural Instructions 里挑长输出的任务（生成 / 摘要 / 解释类），按 T2L 原样的方式生成 description，用现有 pipeline 对 10 个 benchmark 去污染。**成本是「天」，不是「小时」**，但没有更便宜的替代。

**F3. 一起改 `equally_weight_sample`**（`sft_trainer.py:368`）

```python
loss = (loss.view(bs, max_seq_len) / seq_len).sum(-1).mean()   # 按样本等权
```

一个 200 token 的目标，每 token 的梯度权重只有 1 token 目标的 1/200。现在全是短目标所以无所谓；**F2 一旦加进长任务，这个 flag 会精确地压制掉新加的信号**。必须同时改成 token 级加权。

**F4. validation 要能看见生成行为** —— 代码已写完，还没上机跑过

`generative_val.py` + `canary.py` + `--model_select_split`。`val/benchmark` 只有 7 个 MC 任务，gsm8k/mbpp/humaneval **不在任何一个 split 里** —— 现在的 validation 结构上就看不见这个失败。上面所有改动，没有这个都无法在训练中被观测。

**F5. 别让 `base` 吸走条件信号** —— 只有在 F2 之后才有意义

`base` 是常数，**结构上不可能降低对比损失**，所以对比 / 任务判别项的梯度必然流向 head。另可加 staged `--freeze_base`（目前只有 `--freeze_head`）。但要说清楚：**数据里没有可由 description 预测的格式变化时，F5 什么也改不了。**

**F1. 去掉 / 随机化训练 prompt 里的格式指令** —— 成本约 0，单独效果有限

去掉那句话只消除了**词面**重合；1 词目标带来的隐式简短先验还在。做，因为便宜、且这是 reviewer 会翻到的 artifact，但别指望它单独解决问题。

**F6. 修 `eval_gsm8k` 的死代码** —— 一行，修正 baseline 列

### 关于 cross-attention 的判断

不是做错了，是**没打在瓶颈上**。它提升的是「读 description 的能力」，而现在的问题是 **description 里没有值得读的东西**。等 F2 落地、格式开始随任务变化，cross-attention 才会真正被用上 —— 那时它是有价值的。

## 8. 我的 confidence，逐条

| 结论 | confidence | 依据 |
|---|---|---|
| 479/479 训练 prompt 带格式指令；7/7 MC 带、0/3 生成任务不带 | **确定** | 读上游代码 + 用上游 preprocessing 实跑 |
| 训练目标极短（65.9% 单词，p99 = 32 词） | **确定** | n = 1530，51 个随机训练任务 |
| 训练 description 不含格式信号 | **高** | 抽查 0/60 |
| 格式先验是生成任务崩溃的**主因** | **~25%** | 作为因果解释已被 §5d 否证（humaneval −1.8、gsm8k +3.1）。原估 ~80% |
| 训练本身产出了有害 prompt（而非前缀存在） | **确定** | 两任务一致，见 §5b |
| 那个有害的东西具体是什么 | **未知** | 剩余候选未测，见 §5d |
| 崩溃是 input-layer 前缀特有的（LoRA 不崩） | **高** | T2L 官方 checkpoint 同数据不崩。注意：是他们发布的 checkpoint，不是我们复现的训练 |
| F1+F2+F3 之后能恢复任务条件化（C 明显 > 0） | **~40%** | 消除了混淆、补上了缺失信号，但「hypernetwork 能学到有用的条件化」本身仍未被证明 |
| 这样改足以中 ICLR | **低到中** | 见 §10 |

## 9. 还没测的两件事（诚实标注）

1. **E2（2×2 的另一半）**：把那句话从 7 个 MC 任务里删掉，看 Inlet 的增益是否缩水。如果缩水，说明我们 headline 的 MC 增益很大程度上是格式先验而非任务知识 —— **这个我们必须自己先测，不能等 reviewer 测**。本次没跑成：旧 pod 停机期间 GPU 被回收、无法启动，checkpoint 只在那块 volume 上，Drive 上也没有备份。**下次训练出 checkpoint 后第一件事就做这个。**
2. F2 落地后能不能真的产生条件化 —— 只能靠重训验证。

## 10. 能不能帮我们中 ICLR

**现状不能投。** 两个硬伤，任何一个称职的 reviewer 都会问出来：

- 随机描述对照 **C = +0.41 分**（164 道题里 0.7 道）。一个 output 几乎不依赖 input 的 text-to-parameter 方法，方法本身不成立。
- 10 任务均值 **54.58 < zero-shot 56.09**。方法比什么都不做还差。

两条路：

**(A) 修好再 claim 成功。** 需要 F2 + 重训。时间风险高，且第 6 行的 ~50% 就是它的成功率。

**(B) 把 paper 重构成「诊断 + 协议 + 修复」。** 论点是：**input-layer soft prompt 对训练集格式先验的敏感度远高于 LoRA** —— 同样的数据、同样的 description，T2L 的 LoRA 在生成任务上全面高于 zero-shot，而 soft prompt 崩了 21~33 分。配套交付：检测协议（随机描述对照 + 生成 canary）、`val/benchmark` 结构性盲区的分析、以及修复方向。这个对照**我们手上已经有了**。

**我的建议：以 (B) 为骨架，争取 (A) 的部分结果。** 纯 (B) 不带任何正面结果，ICLR 偏难，workshop 稳。(B) + 「修复后条件化确实出现了（哪怕生成任务只追平 zero-shot）」，就是一篇完整的 ICLR 投稿：先给出一个反直觉的失败模式，再给出可复现的诊断协议，最后证明修复有效。

要注意的是，(B) 的叙事**必须诚实地包含 §9.1**。如果 MC 增益也大部分来自格式先验，那 (B) 的故事会更强（"整个方法学到的就是一个格式先验"），但我们必须是先说出来的那一方。
