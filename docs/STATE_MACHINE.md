# 快速点击循环的状态机

- 项目：`D:\小说世界架构器` · 目录：`local_decisions/browser/`
- 上游：`docs/DEPLOYMENT.md`（导出与 CDP 接入）· `README.md`（训练与数据格式）
- 本文件讲**循环内部的状态机**：滚动处理、跨步状态记忆、向 Codex 移交表单输入权限。

---

## 0. 跑过的证据

| 项 | 状态 |
|---|---|
| 状态机 + 驱动层自测（**65 项**，含 10 项用假 IO 跑完整循环的集成测试） | ✅ `node browser/selftest.mjs` → **65/65 passed** |
| 其中：状态机纯 reducer 分支 | ✅ 42 项 |
| 其中：驱动层集成（迁移、移交、超时、陈旧恢复、滚动路径、不收敛检测） | ✅ 10 项 |
| 其中：浏览器层执行原语（dry-run 默认、不可逆闸门、候选上限、state 截断） | ✅ 13 项 |
| 全部 `.mjs` / `.js` 语法检查 | ✅ `node --check` 全通过 |
| **连真实 Chrome 跑过** | ❌ **没有**。需要 Chrome 带 `--remote-debugging-port` 启动 |

---

## 1. 分层：为什么状态机是纯 reducer

```
state_machine.mjs   纯 reducer：(ctx, event) → { ctx, effects }     ← 决定"下一步做什么"
loop_driver.mjs     解释器：把 effects 变成 IO，把 IO 结果变成事件   ← 唯一碰外部世界的地方
cdp_loop.mjs        执行原语：CDP 连接 / 点击填字滚动 / 就绪等待 / 安全闸门
memory.mjs          跨步状态记忆（零依赖）
handoff.mjs         范围化授权 / capsule / 核验计划 / 防乒乓台账
snapshot.js         注入页面的原子快照
```

**为什么值得这么分**：状态机的 bug 几乎全是"某条分支永远走不到"或"顺序反了"，
而这类问题在真浏览器上极难复现（页面每次都不一样、时序不稳定）。
纯 reducer 可以用构造事件序列把**每条分支**喂一遍——本文档里所有行为都有一条对应测试。

代价是多了一层间接。收益是：本轮的 52 项状态机/驱动测试**一条都不需要 Chrome**。

---

## 2. 状态图

```
         START
           │
           ▼
      ┌─────────┐  snapshot  ┌──────────┐  decide   ┌──────────┐
      │ OBSERVE │───────────▶│  DECIDE  │──────────▶│ (分支)   │
      └─────────┘            └──────────┘           └──────────┘
           ▲                      │                      │
           │                      │ 硬信号/低置信度        │ CLICK/TYPE/SELECT
           │                      ▼                      ▼
           │              ┌────────────────┐        ┌──────────┐
           │              │ HANDOFF_PREPARE│        │ EXECUTE  │
           │              └────────────────┘        └──────────┘
           │                   │      │                  │
           │        需人工确认  │      │ 可继续            │
           │                   ▼      ▼                  ▼
           │              ┌────────┐ ┌──────────────┐ ┌────────┐
           │              │BLOCKED │ │ HANDOFF_WAIT │ │ SETTLE │
           │              │(等人工)│ └──────────────┘ └────────┘
           │              └────────┘        │
           │                                ▼
           │                        ┌─────────────────┐
           │                        │ HANDOFF_VERIFY  │
           │                        └─────────────────┘
           │                                │
           │   SCROLL_DONE / ACTION_DONE / SETTLE_DONE / VERIFY ok / 恢复
           │                                │
           └────────────────────────────────┘
                     SCROLL ──(kind="scroll")
                     FIND   ──(滚动查找)
                     VERIFY ──(模型自报 DONE)
```

---

## 3. 状态表

| 状态 | 含义 | 接受的进入事件 | 产出的 effect |
|---|---|---|---|
| `INIT` | 起始 | `START` | `snapshot` |
| `OBSERVE` | 取一次原子快照并更新记忆 | `SNAPSHOT_OK` / `SNAPSHOT_STALE` / `SNAPSHOT_FAILED` | `decide` / `snapshot` |
| `DECIDE` | 一次前向问完全部问题 | `DECISION_OK` / `DECISION_FAILED` / `DECISION_DEGRADED` | `execute` / `scroll` / `verify` / `advance` |
| `SCROLL` | 执行滚动并记账 | `SCROLL_DONE` / `SCROLL_FAILED` | `snapshot` |
| `FIND` | 滚动查找某段文字（有界扫描） | `FIND_STEP` / `SCROLL_DONE` / `SNAPSHOT_OK` | `scroll` / `snapshot` / `advance` |
| `EXECUTE` | 派发一次输入 | `ACTION_DONE` / `ACTION_STALE` / `ACTION_FAILED` | `settle` / `snapshot` |
| `SETTLE` | 等页面就绪 | `SETTLE_DONE` | `snapshot` |
| `VERIFY` | 独立核验"目标达成" | `VERIFY_DONE` | `stop` / `snapshot` |
| `HANDOFF_PREPARE` | 构造范围化授权与恢复载荷 | `HANDOFF_PREPARE` | `handoff` / `stop` |
| `HANDOFF_WAIT` | 等 Codex 交回控制权 | `HANDOFF_RETURNED` / `HANDOFF_TIMEOUT` | `snapshot` / `stop` |
| `HANDOFF_VERIFY` | 交回后**必核验** | `SNAPSHOT_OK` / `RESUME_CHECK_DONE` | `verify` / `snapshot` / `stop` |
| `DONE` / `BLOCKED` | 终止，不再接受任何事件 | — | `stop` |

**`DECIDE` 只认三种事件，其余一律忽略**（只记一条 `unexpected_event`）。
早期版本把杂散事件当成"决策失败 → 恢复"，于是一条接线错误会**静默把状态推到别处**，
调用方只看到"后面的决策好像没生效"。现在有专门一条测试锁住这个行为。

### 3.1 三条不变量

| | 不变量 | 实现 | 测试 |
|---|---|---|---|
| **I1** | 模型自报 `DONE` **不算数**：必须进 `VERIFY` 并拿到独立证据 | `DECIDE` 遇 `DONE` 或 `done>0.8` 一律转 `VERIFY` | `I1 模型自报 DONE 不直接接受` / `I1 核验不过不算完成` / `驱动：…核验不过 → 不完成` |
| **I2** | 滚动之后**必须重新观测**（滚动改变几何与 `page_key`） | `SCROLL_DONE` 只通向 `OBSERVE`，并自增 `scroll.epoch` | `滚动完成后必须重新观测，且自增 epoch` / `驱动：滚动路径能走通` |
| **I3** | 移交期间目标字段**加锁**；收回后**必核验**；核验不过不得再交同一字段 | `HandoffLedger.locked` + `counts`；候选集过滤；`maxPerField=2` | `移交期间目标字段加锁并从候选集剔除` / `核验不过…第二次仍不过则终止` |

---

## 4. 滚动处理

滚动不是"发个 wheel 事件"，它有**四种终止条件**，全部有测试：

| 情况 | 判据 | 处理 |
|---|---|---|
| **预算用尽** | `budget.scrolls >= maxScrolls`（默认 8） | `BLOCKED(scroll_budget_exhausted)` |
| **停滞** | 连续 2 次 `scrollY` 不变 **且** 页面高度不增 | 禁用该方向 + 记事实 `scroll_{dir}_stalled`，回到决策 |
| **往复震荡** | 最近 3 次方向交替（down/up/down） | `BLOCKED(scroll_oscillating)` |
| **到底** | `scrollY + viewportH >= pageHeight - 2` 且方向向下 | 禁用向下 + 记事实 `scroll_bottom_reached` |

三个容易做错的点：

1. **"命令发出去了" ≠ "滚动发生了"**。`SCROLL_DONE` 必须带上**实测**的 `newScrollY/newHeight`
   （`createCdpIo().scroll` 是滚完再读一次 `scrollY` 与 `document.documentElement.scrollHeight`）。
2. **页面高度增长算"懒加载出内容"，但不等于"滚动成功"**。两者分开记：
   `moved` 决定是否算停滞，`grew` 另记一条 `lazy_load_observed` 事实。
3. **到底之后不要立刻永久禁用**——懒加载会让高度增长。当前实现里"到底"这一路是直接禁用的，
   而"停滞"这一路才需要两次；如果你要接无限滚动的站点，改这里。

### 4.1 `FIND`：把开放式滚动变成可测量扫描

`jev-browser` 明确把"scroll to load more"列为**已知限制**："Open-ended goals → make them
measurable or check yourself"。`FIND` 就是那个"measurable"：

```
startFind(ctx, { pattern: "Zürich", direction: "down", maxScrolls: 6 })
```

每步先在**当前可见文本 + 元素标签**里查该串；找到就记 `found:<pattern>` 事实并回决策；
扫够 `maxScrolls` 或该方向被禁用就记 `not_found:<pattern>` 并回决策。
**有界、有结论、不会无限滚**。

---

## 5. 跨步状态记忆

`memory.mjs` 的 `LoopMemory`。快速点击循环每轮都重新观测，模型本身没有跨轮记忆；
只把最近几条动作文本塞进 state，会稳定地出现三类失败：

1. **原地打转**——同一处点两次都无效，模型照点第三次；
2. **重复探索**——已展开的面板、已滚过的位置，重新走一遍；
3. **交接失忆**——交回控制权后不知道之前填了什么、原始状态是什么，"恢复原状"做不到。

### 5.1 存什么

| 结构 | 内容 | 用途 |
|---|---|---|
| `visits` | `pageKeyHash → 访问次数` | 打转检测 |
| `attempts` | `动作签名 → {count, lastError, lastStep}` | 失败计数 |
| `suppressed` | 已知无效的动作签名 | **从候选集里剔除** |
| `facts` | `键 → {value, evidence, previous}` | 已核实事实 + 变更溯源 |
| `transitions` | 环形缓冲：`{step, action, target, outcome, detail}` | "我做了什么、页面怎么回应" |
| `original` | 原始 `pageKey/url/scrollY/formValues` | "做完恢复原状" |
| `budget` | `steps / scrolls / handoffs / staleRecoveries` | 预算与上报 |

### 5.2 两个关键设计

**（a）失败判定不需要额外往返。** 每轮观测都会算一份 `snapshotStamp`（对比上一次的
`elementIds` / `textHash` / `url` / `pageKeyHash`）。如果上一步执行过动作而**世界没变**，
就直接给那个动作记一次 `no_effect` 失败。这就是"点了没反应"被抓住的地方——
零额外 CDP 调用、零额外模型调用。

**（b）改候选集，而不是劝模型。** 同一动作失败 2 次后签名进 `suppressed`，
`filterCandidates()` 直接把它从送给模型的候选表里拿掉。提示词说"别再点它"模型仍可能点；
从候选里拿掉才是确定的。移交中的字段同理（`HandoffLedger.filterLocked`）。

**（c）合法回访不误判。** 打转条件是**三者同时成立**：页面反复访问 **且** 连续 4 轮无世界变化
**且** 存在失败签名。只看访问次数会误伤"修好后回到原页"和"来回比较两个视图"。

### 5.3 渲染进 state：有预算、按优先级丢

记忆要进每轮的 `state`，所以必须有预算。优先级固定，超预算时**从低往高丢**，
并且**明确标注被丢的部分**：

```
目标 > 已核实事实 > 最近状态转移 > 失败与剔除 > 与原始状态的差异
[memory truncated: DIFFERENT FROM ORIGINAL STATE omitted]
```

不标注就等于让模型以为记忆是完整的——它会据此下结论，而结论建立在它没看到的东西上。

---

## 6. 向 Codex 移交表单输入权限

依据（来源：`wy-coliney/jev-browser-use`）：

> "Jev handles the controls. **Codex enters text, handles images, and checks the outcome.**
> When a step needs help, Codex takes over and Jev resumes afterward."
> "Jev receives accessibility text, not screenshots. Codex handles typing, visual
> interpretation, and unsupported controls."

关键不是"把整页交出去"，而是**范围化授权**：交出去的是"这一个字段的这一次输入"，
不是"这台浏览器的控制权"。

### 6.1 移交触发（在做决策的**同一次前向**里判定）

10 个问题一次性问完，其中 6 个是移交信号。**硬信号优先，置信度只作兜底**：

| 问题 | 触发 | 授权范围 | 收回后核验 |
|---|---|---|---|
| `needs_login` | 登录墙 | `page` | 登录控件消失 |
| `needs_credentials` | 需要凭据 | `field` | 同登录 |
| `captcha` | 验证码 | `page` | **停下等人工** |
| `needs_vision` | 需要看图判断 | `page` | 页面有与目标一致的变化 |
| `unsupported` | 控件不在元素表内（canvas/iframe/shadow DOM/文件选择/富文本/日期选择） | `field` | 字段值发生变化 |
| `needs_compose` | 值需要**创作**而非从状态里抄 | `field` | 字段值发生变化 |
| `irreversible` | 不可逆动作 | `none` | **停下等人工** |
| （兜底）`target.confidence < 0.35` | 目标元素不确定 | `field` | 字段值发生变化 |

**注意 `confidence` 不是正确率**——它是适配器按 `(p_max − 1/K)/(1 − 1/K)` 算的
"领先答案离均匀分布有多远"（见 `export/spec.py::confidence`）。所以它只当兜底，
阈值也放低；有硬信号时完全不看它（有专门测试锁住"登录墙优先于低置信度"）。

### 6.2 授权对象 `grant`

```jsonc
{
  "scope": "field",                 // field | form | page | none
  "reason": "needs_compose",
  "fieldSignature": "abc123…|102|fill|Name",   // 绑定到**具体字段**
  "field": { "node": 102, "role": "textbox", "label": "Name", "currentValue": "" },
  "allowedOps": ["CLICK", "TYPE_TEXT", "SELECT"],
  "allowIrreversible": false,        // ← 常量，不接受配置覆盖
  "expiresOn": "control_returned",   // ← 控制权一收回即失效，不是"若干步之后"
  "maxSteps": 3
}
```

`allowIrreversible` 是**常量 false**：即便调用方传 `config.allowIrreversible = true` 也仍是 false
（有测试专门试过覆盖）。不可逆动作只能作为"请求人工确认"的理由上报，永不进授权范围。

### 6.3 恢复载荷 `capsule`

Codex 拿到**这一个对象**就能接管，不用再去看别的上下文：

```jsonc
{
  "requestId": "ho-…", "reason": "needs_compose", "goal": "…",
  "pageKeyHash": "…", "url": "…", "scrollY": 1240, "pageHeight": 5000,
  "visibleText": "…(前 1200 字符)",
  "elementTable": ["[e1] button Go", "…(前 120 行)"],
  "target": { "node": 102, "role": "textbox", "label": "Name" },
  "grant": { … },
  "instructions": "GOAL (unchanged, do not reinterpret): …\nHANDOFF REASON: …\nSCOPE: this ONE field only …",
  "mustNot": ["do not navigate away from this URL",
              "do not submit, pay, publish, send, delete, upload or authorize",
              "do not change browser account settings",
              "do not paste credentials into chat"],
  "resumePlan": { "question": "did_field_change", "expect": true, "node": 102 },
  "memoryDigest": "…(800 字符预算内的记忆摘要)",
  "memoryStats": { … }
}
```

`instructions` 是按 reason 分别生成的，写死语气（不给"你自己看着办"的空间）。
`mustNot` 有四项**固定**条目，调用方只能追加。

### 6.4 收回控制权：必核验

`HANDOFF_RETURNED` 之后**先重新快照**，再执行 `resumePlan`。
核验**优先用页面实测**，而不是再问一次模型：

- `did_field_change` → 直接读 `window.__jevFast.nodes.get(node).value`，与 capsule 记录的原值比对；
- `signed_in` → 看元素表里还有没有登录控件；
- `requiresHuman` → 读人工决定。

目标达成的核验（`VERIFY`）同理：优先用**结构化断言** `--expect "Zürich:present:text"`；
没有断言时才退回问模型一次，但证据来源会被标成 `model_reported` 并写进终止证据里
——**这不是独立核验，报告里不能与结构化证据混为一谈**。

### 6.5 防乒乓

`HandoffLedger` 记 `fieldSignature → 次数`，`maxPerField = 2`：

```
第 1 次移交 → 核验不过 → 允许再交一次
第 2 次移交 → 核验不过 → BLOCKED(handoff_not_effective)
核验通过 → 计数清零（允许将来对同一字段做合法的二次输入）
```

不设这个上限就会出现"交出去 → 没收成 → 再交出去"的无限来回，
每一轮都在烧 Codex 的回合。超时同样终止，且**不解除锁定**——带病继续比起停住更贵。

---

## 7. 效果与解释器契约

reducer 不改外部世界，只产出 `effects`。解释器（`loop_driver.mjs`）负责执行并把结果变回事件。
对应关系是**一对一**的：

| effect | 产出的事件 |
|---|---|
| `snapshot` | `SNAPSHOT_OK` / `SNAPSHOT_STALE` / `SNAPSHOT_FAILED` |
| `decide` | `DECISION_OK` / `DECISION_FAILED` |
| `execute` | `ACTION_DONE` / `ACTION_STALE` / `ACTION_FAILED` |
| `settle` | `SETTLE_DONE` |
| `scroll` | `SCROLL_DONE` / `SCROLL_FAILED` |
| `verify` | `VERIFY_DONE` 或 `RESUME_CHECK_DONE`（按当前状态区分） |
| `handoff` | `HANDOFF_RETURNED` / `HANDOFF_TIMEOUT` |
| `advance` | 直接把携带的事件送回机器（用于状态入场） |
| `emit` / `stop` | 不产出事件 |

**一批 effects 里最多只能有一个会产出事件，多于一个直接抛错。**
这不是洁癖：如果不能确定"在等哪个"，就一定会写错。

同一条思路还有一个更重要的检查：**如果一批 effect 里一个都不产出事件，
而机器又没到终止态，就抛"接线缺口"**。这条检查在本轮开发中**真的抓到了一个 bug**——
`DECIDE` 判定要移交时进入 `HANDOFF_PREPARE`，但没有产出任何会推进状态的事件，
于是循环会静默挂住。修法是显式加一个 `advance` effect，而不是让解释器"猜"。
（驱动程序集成测试里的两条移交用例就是因为在测这个才失败的，现在都过了。）

`io` 接口是注入的，所以整条循环可以在没有 Chrome、没有决策服务的情况下跑完：

```js
const io = makeIo({ snapshots: [...], decisions: [...], verifyResults: [...], handoffResults: [...] });
const { ctx, trace } = await drive({ goal: "find Zürich", io });
```

---

## 8. 怎么跑

```bash
NODE="C:/Users/26516/.workbuddy/binaries/node/versions/22.22.2-3/node.exe"

# 全部浏览器层自测（执行原语 13 + 状态机 42 + 驱动 10 = 65）
"$NODE" browser/selftest.mjs

# 只跑状态机与驱动
"$NODE" browser/sm_selftest.mjs

# 真跑（需要 Chrome 带调试端口）
"$NODE" browser/loop_driver.mjs \
  --url "https://www.google.com/travel/flights?hl=en" \
  --goal "Find one-way flights from Zurich to London on September 20, 2026 for one adult in economy." \
  --decide-url "http://127.0.0.1:<本地决策服务端口>/v1/systemone" \
  --expect "Zürich:present:text" --expect "one-way:present:label" \
  --max-steps 12 --max-scrolls 8
```

默认 **dry-run**（只打印决策轨迹，不动页面）；加 `--live` 才真的操作。
`--expect` 强烈建议至少给一条——不给的话"完成"只能靠模型自述，证据强度会标成
`model_reported`。

---

## 9. 已知落差

1. **连真实 Chrome 未跑**。CDP 命令集对齐 `browser-use/jev-ultrafast` 的公开实现，
   但本机没有带调试端口的 Chrome，所以滚动距离、就绪等待时长、命中测试这些**未经实测**。
2. **`FIND` 未接进主循环**。目前是外部 API（`startFind`），没有从 `DECIDE` 自动触发的路径。
   要自动触发，得让模型能表达"我要找某段文字"——那需要再加一个问题，或从 `GOAL` 里解析引号内容。
3. **`form` 范围的授权定义了但没有构造入口**。`OPS_BY_SCOPE.form` 存在，`buildGrant` 目前只会给出
   `field` / `page` / `none` 三种。多字段表单（例如登录的用户名+密码）应该走 `form`，需要补。
4. **嵌套滚动未支持**（滚动容器而非整页）。`jev-ultrafast` 也把它列为 MVP 之外。
5. **懒加载站点的"到底"判定偏保守**：一见到底就禁用向下，而某些站点会先加高再加载内容。
   已在 §4 标注改法。
6. **核验规则的强度参差**。结构化断言最强；`model_reported` 最弱。
   终止证据里带了 `source` 字段，但**目前没有强制要求"至少一条结构化证据才能算 DONE"**——
   要做成硬门，需要在 `VERIFY` 里检查 `evidence` 的来源。
7. **交接是同步等待的**。`io.handoff` 是个 Promise；真正的 Codex 集成需要一个能跨进程/跨回合
   的信号（文件、队列或 MCP），本文件只定义了契约（`--handoff-inbox` 参数占位，未实现轮询）。
