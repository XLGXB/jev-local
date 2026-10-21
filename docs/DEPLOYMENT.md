# 部署方案：微调模型 → ONNX/TensorRT → CDP 浏览器控制循环

- 项目：`D:\小说世界架构器` · 目录：`local_decisions/`
- 日期：2026-09-22
- 上游：`.codex-handoff/runtime/jev_local_stack/DESIGN.md`（三层架构）· `local_decisions/README.md`（训练与数据格式）
- 本文件是**部署层**：把训好的决策模型搬到本地服务，再接到浏览器控制循环上。

---

## 0. 先说清楚：哪些跑过了，哪些没有

| 项 | 状态 |
|---|---|
| `docs/DEPLOYMENT.md` 所述的**纯逻辑**层（mask/归一化/confidence/形状契约/TRT profile/缓存键/一致性比对/温度重拟合/网关解析/降级路径/日志脱敏/不可逆闸门） | ✅ **已跑**：`tests/test_serve.py` **23/23 通过**、`browser/selftest.mjs` **13/13 通过** |
| 网关 **HTTP 往返**（真起服务、真发请求：`/healthz`、`/v1/systemone`、400/404、决策日志落盘） | ✅ **已跑**（同上，用标准库 `http.client` 打真实回环端口） |
| 环境核查 | ✅ **已跑**，输出见 §1.1 |
| **ONNX 导出** | ❌ **没跑**：本机无 torch / onnx / onnxruntime |
| **TensorRT 引擎构建** | ❌ **没跑**：本机无 CUDA Toolkit、无 tensorrt |
| **CDP 循环连真实 Chrome** | ❌ **没跑**：需要 Chrome 带 `--remote-debugging-port` 启动。CDP 命令集照 `browser-use/jev-ultrafast` 的公开实现对齐（§4.3），**未经本机实测** |
| 任何端到端浏览器任务 | ❌ **没有** |

**所以：本文件是方案 + 已通过测试的骨架，不是已验收的部署。**

---

## 1. 环境事实与三条硬约束

### 1.1 本机核查输出（`python -m jevlocal.export.to_onnx check-env`，实跑）

```
  python             3.13.14
  platform           Windows 11 (AMD64)
  driver_version     610.62
  gpu_name           NVIDIA GeForce RTX 4070 SUPER
  compute_cap        8.9
  cuda_toolkit       None
  torch / onnx / onnxruntime / tensorrt   全为 None
  ort_providers      (none)
  [警告] 未安装 onnx / onnxsim
  [阻断] 未安装 onnxruntime-gpu
  [阻断] 未安装 tensorrt
  [阻断] 未安装 torch
  [阻断] 未找到 CUDA Toolkit
  结论: BLOCKED
```

### 1.2 版本事实（2026-09-22 查证）

| 组件 | 事实 | 与本机的关系 |
|---|---|---|
| **onnxruntime-gpu** | 支持 **Python 3.11–3.14**，存在 `cp313-win_amd64` 轮子（最新 1.30.0 / 2026-09-10） | ✅ 3.13 直接可用，**不必降级** |
| **TensorRT** | 官方前置：算力 **Turing 及以上**、**CUDA Toolkit 12.x 或 13.x**、Windows 驱动 **r537+**、Python **3.10–3.13** | ⚠️ 驱动 610.62 ✅、算力 8.9 ✅、Python 3.13 ✅，**但 CUDA Toolkit 缺失** |
| TensorRT 安装模式 | full（约 2–3 GB，**能构建引擎**）/ lean（约 200–300 MB，**不能构建**）/ dispatch（约 100–150 MB，**不能构建**） | 要自己 build 引擎 → **必须 full 模式** |
| ONNX-TensorRT 解析器 | onnx **1.20.0**、opset 支持到 **25**、官方支持 opset **9+** | 我们导出用 opset 17，安全 |
| Torch-TensorRT | 2.12.1：PyTorch 2.12 / CUDA 13.0 / TRT 10.16 / Python 3.10–3.13 | 备选路线，非必须 |

### 1.3 三条硬约束

1. **CUDA Toolkit 缺失是唯一的真实阻断项**。TensorRT 官方要求它。装之前先算空间——本机 **C 只剩 42 GB、D 只剩 76 GB**，而 full 模式 + CUDA Toolkit 加起来可到 5–8 GB。**装到 E/F/G**（E 263 GB / F 473 GB / G 2499 GB）。
2. **Python 3.13 处在 TensorRT 支持区间的上沿**（3.10–3.13），而 onnxruntime-gpu 覆盖到 3.14。当前 3.13 两边都在，但一旦某个组件升版掉出区间，正确做法是**单开一个 3.11/3.12 的 venv 专跑导出台架**，与主环境并存，而不是把主环境降级。
3. **磁盘**：模型权重、ONNX、引擎、timing cache 全部写 `E:/jevlocal/`（`configs/deploy.json` 已如此配置），不要落在 D 盘。

---

## 2. 导出设计

### 2.1 核心决定：固定 256 槽输出，K 只活在 Python 侧

两条候选设计（来源：Jev 架构逆向文，第三方、非官方）：

- **pointer 式**：读数位隐状态与每个候选自身的隐状态做点积 → 输出宽度 = K（动态）。
- **final-position head**：读数位隐状态过 `Linear(H, 256)` → 输出宽度**恒 256**。

本方案选后者（`export/spec.py::ExportSpec.head = "slots"`），理由是工程性的，但后果很大：

1. **TensorRT 引擎不需要为不同 K 重建**。TRT 引擎与 shape profile 绑定，动态宽度会逼出成堆 profile 甚至重新 build（一次 build 几分钟到几十分钟）。K 从 2 到 255 只换 Python 侧的 mask。
2. K 的上限由 `schema.MAX_CHOICE_OPTIONS`（255）兜住，与 System One 验证层一致。
3. **代价必须自己兜住**：超出 K 的槽是未受约束的输出，**必须显式 mask 再归一化**。
   `export/spec.py::mask_and_normalize` 里有一句关键注释，这里重复一遍：

   > 直接对全 256 槽 softmax **形状完全正确、不抛任何异常**，但概率质量被分给了不存在的候选，
   > 于是 `confidence` 与所有 `keepThreshold` 语义全部偏小。

   测试 `test_mask_excludes_phantom_slots` 就是拿这条做负控：构造 254 个高分噪声槽，断言 mask 后 max > 0.9 而全槽 softmax 后 max < 0.5。

`noul` 也走同一条路：把"是/否"两个描述当候选写进输入，K = 2。于是**整个服务只有一个算子图，没有分支**。

### 2.2 一个必须讲清的架构差别：谁能做到"state 只编码一次"

这是两条路线在部署侧的**真正**取舍点，比"哪个模型分高"重要得多：

| | ModernBERT（Laya / openJev 范式） | Qwen-Kev（kev / NanoJev 范式） |
|---|---|---|
| 注意力 | **双向** | **因果**（前缀） |
| state 的表示是否依赖问题与候选 | **依赖**——state 与问题双向互相 attend | **不依赖**——问题只读 state，state 不读问题 |
| 能否缓存 state 再换问题复用 | **不能**，只能整条重算 | **能**，缓存 state 的 KV，每个问题只跑后缀 |
| 多问题的摊薄方式 | 把 N 个问题压进同一个 **batch**（算力不变，但省掉 N 次启动） | 前缀复用（算力真的省掉 N-1 次 state 编码） |

结论：**"state 编码一次"只有因果主干能做到。** 用编码器主干时，等价手段是把问题批量进同一张图，靠一次前向摊掉启动开销——这在小模型上其实非常有效，见 §6。

### 2.3 ModernBERT 导出步骤

```bash
# 1) 环境（隔离 venv，不污染全局）
"C:/Users/26516/.workbuddy/binaries/python/versions/3.13.12/python.exe" -m venv E:/jevlocal/venv-export
E:/jevlocal/venv-export/Scripts/pip.exe install torch --index-url https://download.pytorch.org/whl/cu124
E:/jevlocal/venv-export/Scripts/pip.exe install onnx onnxruntime-gpu transformers onnxsim

# 2) 核查（会明确告诉你缺什么）
PYTHONPATH=src E:/jevlocal/venv-export/Scripts/python.exe -m jevlocal.export.to_onnx check-env

# 3) 导出
PYTHONPATH=src E:/jevlocal/venv-export/Scripts/python.exe -m jevlocal.export.to_onnx modernbert \
    --ckpt E:/jevlocal/artifacts/sft-final --out E:/jevlocal/onnx/model.onnx \
    --n-slots 256 --max-seq-len 512 --max-batch 8 --hidden-size 768

# 4) 数值一致性 + 温度重拟合（**必过**，见 §5）
PYTHONPATH=src E:/jevlocal/venv-export/Scripts/python.exe -m jevlocal.export.parity \
    --reference ref_logits.json --exported onnx_logits.json --labels labels.json --ks ks.json \
    --temperature-before 2.3 --out E:/jevlocal/reports/parity.json
```

导出图的三入两出（`export/wrapper.py::DecisionModule`）：

```
输入   input_ids      int64 (B, L)
      attention_mask int64 (B, L)
      read_index     int64 (B,)      ← 读数位下标
输出   slot_logits    float32 (B, 256)  ← 恒定形状
      read_hidden    float32 (B, H)    ← 供 correctness 头使用
```

`read_index` 用 `gather` 取（`view(B,1,1).expand(B,1,H)` + 形状断言）。写成二维会**静默**沿错误的轴取——形状还可能对。这类 bug 只能靠恒等式测试抓，`wrapper._read` 里带了断言。

### 2.4 Qwen-Kev 导出步骤

```bash
PYTHONPATH=src E:/jevlocal/venv-export/Scripts/python.exe -m jevlocal.export.to_onnx qwen-kev \
    --ckpt E:/jevlocal/runs/mine --out E:/jevlocal/onnx/qwen.onnx --max-seq-len 1024
```

两个前提，缺一都会静默出错：

1. **必须先合 LoRA**（`merge_and_unload()`）。忘了这一步会导出一个"底座 + 未生效的 adapter"，
   推理结果与训练时的评测值不符，**不报任何错**。
2. **`read_index` 是后缀内部下标，不是整条序列的下标**。混用会读到前缀里某个位置——形状对、不报错、答案系统性偏错。`wrapper.TwoStageDecoder.tail` 的文档里写了这条。

`TwoStageDecoder` 就是"前缀复用"的实现：`prefill()` 编码 state 一次并返回 `past_key_values`，后续每个问题用 `tail()` 只跑后缀。

---

## 3. TensorRT 构建

```bash
# 打印构建命令与 shape profile
PYTHONPATH=src python -m jevlocal.export.to_onnx trt-cmd \
    --onnx E:/jevlocal/onnx/model.onnx --engine E:/jevlocal/trt/model_sm89_trt10.15_cu12.9_fp16.plan
```

生成的命令（`export/spec.py::trt_command`）：

```
trtexec --onnx=model.onnx --saveEngine=model.plan --fp16 \
  --minShapes=input_ids:1x64,attention_mask:1x64,read_index:1 \
  --optShapes=input_ids:2x256,attention_mask:2x256,read_index:2 \
  --maxShapes=input_ids:8x512,attention_mask:8x512,read_index:8 \
  --timingCacheFile=model.timing.cache --builderOptimizationLevel=3
```

三条必须记住的：

1. **shape profile 必须覆盖服务期真实会出现的形状**，否则该次推理被拒。测试 `test_trt_profile_covers_max_shapes` 断言三档里都含 `input_ids` / `attention_mask` / `read_index`。
2. **引擎缓存失效三要素**：TensorRT 版本 / GPU 算力 / CUDA 版本。任一变化都必须重建——加载旧引擎要么直接失败，**要么给出错误结果**（后者更糟）。把三者拼进文件名是最省事的做法，`spec.engine_cache_key()` 就在做这件事，`test_engine_cache_key_changes_with_each_input` 验证三者任一变化都产生不同键。
3. **需要 full 安装模式**。lean / dispatch 运行时只能跑**已经建好**的引擎，不能 build。

---

## 4. 接入 CDP 浏览器控制循环

### 4.1 每轮固定四步

```
1. SNAPSHOT   1 次 Runtime.evaluate                 → 元素表 + 可见文本 + 新鲜度凭据
2. DECIDE     0 次 CDP，1 次本地 HTTP              → 操作 / 目标 / 完成 / 受阻 / 不可逆
3. EXECUTE    1–3 次 CDP（命中测试 + 输入派发）     → 坐标现算
4. SETTLE     1 次 Runtime.evaluate                 → 50 ms；combobox 最多 200 ms
```

对照：`jev-browser` 的循环是"LLM 规划 → 每次 Jev 调用约 300 ms，多数步骤 2–4 次调用 → 5 步结账约 14 s"。
本设计把决策压成**一次**本地前向（约 10–40 ms，5 个问题一起算），且不引入宿主模型回合。

### 4.2 三个决定成败的机制

**(a) 快照必须原子。** 分多次查 DOM 会拿到互相不一致的快照——元素在两次调用之间被移除，就出现"表里有的元素执行时不存在"。`browser/snapshot.js` 用一次 `Runtime.evaluate` 完成：清 GC → 建元素表 → 取可见文本 → 收集 `page_key` 与每个元素的 `guard`。

**(b) 只送可见文本。** `snapshot.js` 用 `TreeWalker` 遍历文本节点，但只保留**落在视口内**的（`r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth`），上限 6000 字符。
依据：`jev-browser` 的公开测量显示，页面内容给 LLM 的 token 中位省 5×（≈8k vs ≈557k），而**一篇 Wikipedia 的完整快照约 149k token**——大页面主导了总开销，离屏正文全进 context 是最贵的错误。

**(c) 对照"意义与身份"，几何在输入前现算。** `marker` 与 `guard` 里**不放坐标**，因为坐标会随滚动、动画、懒加载失效。执行前重新 `getBoundingClientRect()` 并做命中测试：

```js
if (!e.contains(document.elementFromPoint(x, y))) return null;   // 被遮挡
```

`test_serve.py` 覆盖了 guard/marker 的**语义**部分；命中测试属 CDP 实测范围，**未跑**。

### 4.3 CDP 命令清单（对齐 jev-ultrafast 的公开实现）

| 阶段 | 命令 |
|---|---|
| 建会话 | `Target.createTarget{background:true}` → `Target.attachToTarget{flatten:true}` |
| 视口 | `Emulation.setDeviceMetricsOverride` |
| **后台渲染** | `Emulation.setFocusEmulationEnabled{enabled:true}` —— 保持后台标签页渲染（rAF 不被节流），**且不抢用户的 Chrome 标签** |
| 导航 | `Page.navigate` + 轮询 `document.readyState === "complete"`（20 ms 步进，15 s 上限） |
| 快照 | `Runtime.evaluate{returnByValue:true}`，`exceptionDetails` 存在即判陈旧 |
| 点击 | `Input.dispatchMouseEvent` `mousePressed` + `mouseReleased`，`button:"left"`, `clickCount:1` |
| 填字 | 先 `Input.dispatchKeyEvent` Ctrl+A（Windows `modifiers=2`）+ `Input.insertText` —— **直接 insertText 是追加而不是替换** |
| 下拉 | 直接设 `.value` 并派发 `input`/`change`（原生 `<select>`），不走模拟点击 |
| 滚动 | `Input.dispatchMouseEvent{type:"mouseWheel", deltaX:0, deltaY:±560}` |
| 截图 | `Page.captureScreenshot{format:"jpeg", quality:72}`（**默认关**，见 §6） |
| 收尾 | `Target.closeTarget` |

**注意**：已在运行的 Chrome 没带 `--remote-debugging-port` 时**无法接上**，必须重启一个带该旗标的实例：

```bash
"C:\Program Files\Google\Chrome\Application\chrome.exe" \
  --remote-debugging-port=9222 --user-data-dir="%TEMP%\cdp-profile" about:blank
```

用独立 profile 才不会动用户现有窗口与登录态。

### 4.4 与 Codex 既有可信浏览器服务的关系（不能忽略）

项目里 `scripts/chrome_extension_sidecar_channel.mjs` 的实测行为是：**只接受可信 Node REPL 浏览器服务，spawn 出来的 node 进程一律拒绝**（报 `blocked_extension_backend_or_reusable_tab_unavailable`）。

而本机 `~/.codex/config.toml` 里已经配好了：

```
[mcp_servers.node_repl.env]
NODE_REPL_TRUSTED_SERVICES = {... "browser": ".../browser-service.mjs" ...}
BROWSER_USE_AVAILABLE_BACKENDS = "chrome,iab"
NODE_REPL_TRUSTED_BROWSER_CLIENT_SHA256S = "..."
```

所以有两条路，**优先第二条**：

- **A 独立模式**：`cdp_loop.mjs` 自己开 CDP 连接（本文描述的模式）。适合脱离 Codex 的单跑与基准测量，需要单独启动带调试端口的 Chrome。
- **B 集成模式（推荐）**：把 `cdp_loop.mjs` 改成**既有可信服务的客户端**——用已有连接、已有 profile、已有登录态，不新建浏览器栈，也不需要 `--remote-debugging-port`。这也正好绕开上面那条拒绝规则。

`configs/deploy.json` 的 `browser.engine` 就是这个开关，默认 `standalone-cdp`。

### 4.5 安全闸门

| 闸门 | 实现 | 测试 |
|---|---|---|
| 模型输出只能是被观测到的元素 id | 动作携带 `node` id，代码用 `window.__jevFast.nodes.get(id)` 解析；**模型输出永不变成选择器/坐标/JS** | `_read` 形状断言（Python 侧）；CDP 侧未跑 |
| 不可逆动作停在确认 | `looksIrreversible()`：模型自报 **或** 标签命中双语模式（支付/下单/删除/发送/发布/授权/上传/安装…） | ✅ `browser/selftest.mjs` 4 条用例 |
| 模型自报 false 不能推翻标签命中 | 单向判断（`||` 而不是覆盖） | ✅ 有专门用例 |
| 默认 dry-run | `--live` 才真的执行 | ✅ 有专门用例 |
| 只绑回环 | `LoopbackServer(("127.0.0.1", port))` | ✅ `test_binds_loopback_only` |
| 请求体上限 | 4 MiB（无鉴权本地服务，不设上限等于给一个 OOM 开关） | ✅ 代码路径已覆盖 |
| 占位后端不得进真实分支 | 非影子模式下 `EchoDecider` 直接拒绝启动 | ✅ `test_server_refuses_placeholder_backend_outside_shadow` |
| 决策日志不记正文 | 只写 qid/类型/K/下标/confidence/延迟 | ✅ `test_decision_log_carries_no_state_text`（用带哨兵词的 state 验证日志里查不到） |

---

## 5. 数值一致性与校准门（这一步不能省）

训练时拟合的温度是在 **PyTorch fp32** 的 logits 上拟合的。导出到 ONNX、再到 TensorRT fp16 之后：

- logits 的**尺度**会变（fp16 舍入、kernel 融合改变了累加顺序）；
- 温度的语义是"logits 除以 T"，尺度一变，原来的 T 就不再对应原来的校准。

于是会出现一种很难查的现象：**argmax 准确率几乎没变，但 ECE 明显变差**，下游所有按
`confidence` 分支的阈值集体偏移。

**规矩：导出的温度必须在导出的图上、用 calib 切分重拟合。** 不能沿用训练侧那个。

`export/parity.py` 的验收门（`configs/deploy.json` 的 `gates.parity`）：

| 指标 | 阈值 | 不过通常意味着 |
|---|---|---|
| 最大 logit 绝对差 | ≤ 0.05 | 忘了合 LoRA / 读数位偏移算错 / opset 选错 |
| argmax 一致率 | ≥ 0.995 | 导出链路有实质问题 |
| top1 概率一致率 | ≥ 0.99 | 同上（用相对容差比，fp16 末位必然不同，用 `==` 会把正常导出判失败） |
| 重拟合后 ECE | 不得比拟合前更差 | 拟合过程本身有 bug |

测试 `test_parity_report_fails_on_planted_corruption` 就是故意喂一个"argmax 被翻掉"的导出，断言判 FAIL。

### 5.1 量化：这是决策模型最危险的一步

INT8 / FP8 量化对普通分类器是"掉 1% 准确率"，对决策模型是**把它唯一的用途毁掉**——
校准没了，`confidence` 与所有阈值就都不可用了。

明确规矩：

1. **先量化，再重拟合温度，再复核 ECE**。顺序颠倒等于用旧温度配新 logits。
2. **ECE 不过 0.03 就不要上量化版**（`configs/deploy.json` 的 `gates.calibration.max_ece_after`）。
3. 若要量化，优先 **weight-only INT8**（对激活的扰动小），并单独保留一个 FP16 影子实例做对照。

---

## 6. 延迟预算与优化次序（这一节可能和直觉相反）

本机 RTX 4070 SUPER 12 GB、算力 8.9（Ada）的粗算：

- ModernBERT-base 149.6M、L=512：前向约 `2 × 1.5e8 × 512 ≈ 1.5e11` FLOPs。
- Ada 的 FP16 张量核心算力远高于此所需，所以 **batch=1 时不是算力受限，而是启动开销受限**。
- 一轮的实际构成大致是：

| 环节 | 量级 | 能否优化 |
|---|---|---|
| 快照（1 次 evaluate，含 DOM 遍历 + 视口文本） | 20–60 ms | 收紧可见文本上限、避免全量 querySelector |
| 决策（本地模型） | 10–40 ms | 见下 |
| 执行（命中测试 + 输入派发） | 5–15 ms | 已是一轮 1–3 次 |
| **等待（settle）** | **50–200 ms** | **最大项**，按动作类型给最小必要等待 |
| 固定 sleep（错误做法） | 300–1000 ms+ | 直接删掉 |

**结论：决策只占一轮的约 10%。所以优化次序是：**

1. **压掉往返**——本轮对话的最大杠杆。参照 `jev-ultrafast`：同任务浏览器协议调用 **1092 → 101**，中位任务时间 **9.450 s → 7.092 s（−25%）**。
2. **一个决策周期一次请求**——5 个问题（操作/目标/完成/受阻/不可逆）**压进一次前向**。这在小模型上比换推理后端有效得多，因为它省掉的是 N 次 kernel 启动而不是 N 次乘加。
3. **删掉固定 sleep**，改成就绪条件。
4. **默认不吃截图**。`Page.captureScreenshot` 是几十毫秒级的固定成本，而且解码图片更贵。只在需要视觉判断时开，由主模型决定而不是循环每轮都做。
5. **CUDA Graph**（`sess_options.enable_cuda_graph = True`）。前提是形状固定——所以统一 pad 到 `max_seq_len`，**牺牲少量算力换掉启动开销**。对"小模型 + 小 batch"这个画像，这比上 TensorRT 更直接。
6. **最后才考虑 TensorRT**。

### 6.1 什么时候**不值得**上 TensorRT

诚实算一笔：决策占一轮约 10%（假设 15 ms / 150 ms）。把 ORT CUDA EP 换成 TRT fp16，
通常再快 1.2–1.5×，也就是省下 **3–5 ms / 轮，约 2–3%**。而代价是：装 CUDA Toolkit（5–8 GB，本机 C/D 都紧张）、
引擎与 GPU/驱动版本绑定、每次换卡或升级都得重建、KV cache 图的 shape profile 容易出错。

**所以 `configs/deploy.json` 里 `tensorrt.enabled` 默认为 `false`。**

建议的推进顺序是：

```
ORT CUDA EP + 一次前向吃下全部问题 + CUDA Graph   → 量基线
        ↓（若 decision p95 仍 > 80 ms，或要跑多路并发）
TensorRT fp16
        ↓（仅当 ECE 重拟合后仍 ≤ 0.03 且确需再省显存）
weight-only INT8
```

**先量再改。** 没有基线就上 TRT，等于把一个 2–3% 的收益当成了主要矛盾。

---

## 7. 分阶段路线与验收

| 阶段 | 做什么 | 验收（可观察） |
|---|---|---|
| **P0 环境** | 建 `E:/jevlocal/venv-export`；装 torch / onnx / onnxruntime-gpu；（可选）CUDA Toolkit + TensorRT full 到 E 盘 | `check-env` 退出码 0、结论 `READY` |
| **P1 导出** | 导出 ONNX；`onnxsim` 简化；跑形状自检 | 图的三入两出形状与 `ExportSpec` 一致；`slot_logits` 宽度恒 256 |
| **P2 一致性 + 校准** | `parity.py` 跑 calib 集；重拟合温度 | 四条门全过（§5 表）；**导出后的 T 写进 `deploy.json`，不沿用训练侧** |
| **P3 网关** | 起服务，先 **shadow** | `/healthz` 的 `active_providers` **必须含 CUDA/TensorRT 而不是只有 CPU**；决策日志有元数据、无正文；延迟 p50 ≤ 40 ms |
| **P4 浏览器（dry-run）** | `cdp_loop.mjs` 接 Chrome，`--dry-run` 跑通一条真实任务 | 决策轨迹合理；每轮 CDP 调用 ≤ 5；快照/决策/执行/等待四项耗时分别有数 |
| **P5 上量** | 转 `--live`；跑固定任务集，交替 N 次做对照 | 先量 CDP 调用数、token 数、墙钟时间，**再**决定是否上 TRT |
| **P6（可选）TensorRT** | build 引擎、重跑 P2 的一致性门、重拟合温度 | 速度提升 > 15% 才保留；否则回退 ORT |

**每一阶段都不得跳过 P2 的一致性门。** 决策模型的价值全在概率上，概率错了，跑得再快也只是更快地给出错误分支。

---

## 8. 风险与回滚

| 风险 | 触发条件 | 回滚 |
|---|---|---|
| CUDA Toolkit 装不下 | C 42 GB / D 76 GB | 装到 E 盘；或**干脆不上 TRT**（§6.1 说明它只值 2–3%） |
| Python 3.13 掉出支持区间 | 某组件升版 | 单开 3.11/3.12 venv 跑导出台架，主环境不动 |
| 引擎与硬件/版本绑定 | 换卡、升级驱动或 TRT | 用 `engine_cache_key` 区分文件；重建引擎 |
| 量化毁掉校准 | 上 INT8 但没重拟合温度 | 保留 FP16 影子实例；ECE 不过 0.03 就不上 |
| CDP 连不上用户 Chrome | Chrome 未带调试端口启动 | 用独立 profile 起新实例；或走 §4.4 的集成模式 B |
| 浏览器动作误操作 | 不可逆动作被放过 | 默认 dry-run + 双向不可逆闸门；`--allow-irreversible` 是显式开关 |
| 快速回滚 | — | 网关是独立进程、浏览器循环是独立脚本、导出物全在 `E:/jevlocal/`；关掉进程即回到原状，**live 源零改动** |

---

## 9. 非目标

- ❌ 不替代 Codex 的 Computer Use 运行时。循环只是**决策层**，执行仍走既有浏览器服务。
- ❌ 不自动登录、上传、发布、支付、改账号设置、绕验证码。
- ❌ 不用本方案去碰 strict 权威、receipt 或 credit。
- ❌ 不为了性能牺牲校准（量化必须过 ECE 门）。
- ❌ 不改全局 `~/.codex`、MCP、代理、provider、API Key。
