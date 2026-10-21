#!/usr/bin/env node
/* CDP 浏览器决策循环：**本地决策模型决定动作，代码负责执行与校验**。
 *
 * 零依赖：只用 Node 22 的全局 `WebSocket` 与 `fetch`（本机 node 22.22.2 具备）。
 * 不引入 puppeteer / playwright——它们的抽象层会把"每次操作几次往返"藏起来，
 * 而往返次数正是这个方案要压的指标。
 *
 * 每轮固定形状（与 jev-ultrafast 的"一个决策周期一次请求"一致）：
 *
 *   1. SNAPSHOT  1 次 Runtime.evaluate                     → 元素表 + 可见文本 + 新鲜度凭据
 *   2. DECIDE    0 次 CDP 调用，1 次本地 HTTP                → 操作 / 目标 / 完成 / 受阻 / 不可逆
 *   3. EXECUTE   1–3 次 CDP 调用（命中测试 + 输入派发）      → 真实坐标现算，不用快照里的坐标
 *   4. SETTLE    1 次 Runtime.evaluate                     → 50ms；combobox 等候选最多 200ms
 *
 * 安全边界（不可通过参数绕过，只可通过显式旗标放开）：
 *   • 模型输出**永远**只能是一个已被观测到的元素 id，不能是选择器/坐标/JS；
 *   • 不可逆动作（支付/下单/删除/发送/发布/授权/上传/验证码/安装/系统设置）默认停在确认；
 *   • `--dry-run` 为默认值，必须显式 `--live` 才会真的点击。
 *
 * 与 Codex 集成时的差别：项目里的 scripts/chrome_extension_sidecar_channel.mjs
 * 只接受"可信 Node REPL 浏览器服务"，**拒绝 spawn 出来的 node 进程**。
 * 所以那时本循环应作为**既有可信服务的客户端**运行，而不是自己开一条 CDP 连接。
 */

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

// 提示词构造的唯一真源在状态机模块（见 browser/state_machine.mjs）。
// 本文件只提供**执行原语**：CDP 连接、元素操作、就绪等待、安全闸门。
import { buildQuestions, buildStateText } from "./state_machine.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const SNAPSHOT_SRC = fs.readFileSync(path.join(HERE, "snapshot.js"), "utf8");

// --------------------------------------------------------------------------- //
// 参数
// --------------------------------------------------------------------------- //
export function parseArgs(argv) {
  const out = {
    url: "",
    goal: "",
    cdpPort: 9222,
    decideUrl: "",
    maxSteps: 12,
    headless: false,
    live: false,
    allowIrreversible: false,
    width: 1120,
    height: 780,
    stateTokenBudget: 6000,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === "--url") out.url = next();
    else if (a === "--goal") out.goal = next();
    else if (a === "--cdp-port") out.cdpPort = Number(next());
    else if (a === "--decide-url") out.decideUrl = next();
    else if (a === "--max-steps") out.maxSteps = Number(next());
    else if (a === "--live") out.live = true;
    else if (a === "--allow-irreversible") out.allowIrreversible = true;
    else if (a === "--width") out.width = Number(next());
    else if (a === "--height") out.height = Number(next());
    else if (a === "--help" || a === "-h") out.help = true;
    else throw new Error(`未知参数: ${a}`);
  }
  return out;
}

const USAGE = `
用法:
  node cdp_loop.mjs --url <起始页> --goal <目标> [选项]

必填
  --url            起始 URL
  --goal           自然语言目标（建议英文；Jev 系模型英文最准）
  --decide-url     本地决策服务，例如 http://127.0.0.1:38123/v1/systemone

选项
  --cdp-port       Chrome 远程调试端口，默认 9222
  --max-steps      最大轮数，默认 12
  --live           真的执行动作。默认 dry-run，只打印决策
  --allow-irreversible  允许执行不可逆动作（仍会在日志里逐条记录）
  --width/--height 视口尺寸，默认 1120x780

启动带调试端口的 Chrome（独立 profile，不动用户现有窗口）:
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" \\
    --remote-debugging-port=9222 --user-data-dir="%TEMP%\\cdp-profile" about:blank
`;

// --------------------------------------------------------------------------- //
// CDP 客户端
// --------------------------------------------------------------------------- //
export class Cdp {
  #ws;
  #nextId = 1;
  #pending = new Map();
  #sessionId = null;
  calls = 0;

  static async discover(port) {
    const res = await fetch(`http://127.0.0.1:${port}/json/version`, { method: "GET" });
    if (!res.ok) throw new Error(`/json/version 返回 ${res.status}`);
    const info = await res.json();
    if (!info.webSocketDebuggerUrl) throw new Error("/json/version 未给出 webSocketDebuggerUrl");
    return info;
  }

  static async connect(port) {
    const info = await Cdp.discover(port);
    const client = new Cdp();
    await client.#open(info.webSocketDebuggerUrl);
    return client;
  }

  async #open(wsUrl) {
    if (typeof WebSocket === "undefined") {
      throw new Error("本 Node 没有全局 WebSocket（需要 Node >= 21）；请升级 node 或改用 ws 包");
    }
    this.#ws = new WebSocket(wsUrl);
    await new Promise((resolve, reject) => {
      const t = setTimeout(() => reject(new Error("CDP WebSocket 连接超时")), 15000);
      this.#ws.addEventListener("open", () => {
        clearTimeout(t);
        resolve();
      });
      this.#ws.addEventListener("error", (e) => {
        clearTimeout(t);
        reject(new Error(`CDP 连接失败: ${e?.message || "unknown"}`));
      });
    });
    this.#ws.addEventListener("message", (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      if (msg.id && this.#pending.has(msg.id)) {
        const { resolve, reject } = this.#pending.get(msg.id);
        this.#pending.delete(msg.id);
        if (msg.error) reject(new Error(`${msg.error.message} (${msg.error.code})`));
        else resolve(msg.result);
      }
    });
  }

  get sessionId() {
    return this.#sessionId;
  }

  async send(method, params = {}, { session = true } = {}) {
    const id = this.#nextId++;
    const payload = { id, method, params };
    if (session && this.#sessionId) payload.sessionId = this.#sessionId;
    this.calls += 1;
    const result = await new Promise((resolve, reject) => {
      const t = setTimeout(() => {
        this.#pending.delete(id);
        reject(new Error(`CDP 超时: ${method}`));
      }, 30000);
      this.#pending.set(id, {
        resolve: (r) => {
          clearTimeout(t);
          resolve(r);
        },
        reject: (e) => {
          clearTimeout(t);
          reject(e);
        },
      });
      this.#ws.send(JSON.stringify(payload));
    });
    return result;
  }

  async evaluate(expression, { awaitPromise = false } = {}) {
    const r = await this.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise });
    if (r.exceptionDetails) throw new StalePage("求值期间文档发生了变化");
    return r.result?.value;
  }

  async createSession(url, width, height) {
    const { targetId } = await this.send("Target.createTarget", { url: "about:blank", background: true }, { session: false });
    const { sessionId } = await this.send("Target.attachToTarget", { targetId, flatten: true }, { session: false });
    this.#sessionId = sessionId;
    this.targetId = targetId;
    await this.send("Emulation.setDeviceMetricsOverride", {
      width,
      height,
      deviceScaleFactor: 1,
      mobile: false,
    });
    // 后台标签页里 rAF 会被节流，等待逻辑就会挂住。开焦点模拟可以在**不抢用户标签页**的前提下保持渲染。
    await this.send("Emulation.setFocusEmulationEnabled", { enabled: true });
    await this.navigate(url);
  }

  async navigate(url) {
    await this.send("Page.navigate", { url });
    const deadline = Date.now() + 15000;
    while (Date.now() < deadline) {
      if ((await this.evaluate("document.readyState")) === "complete") return;
      await sleep(20);
    }
    throw new Error("页面在 15s 内未就绪");
  }

  async close() {
    if (this.targetId) {
      try {
        await this.send("Target.closeTarget", { targetId: this.targetId }, { session: false });
      } catch {
        /* 忽略 */
      }
    }
    try {
      this.#ws.close();
    } catch {
      /* 忽略 */
    }
  }
}

export class StalePage extends Error {}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// --------------------------------------------------------------------------- //
// 决策服务客户端
// --------------------------------------------------------------------------- //
export class Decider {
  constructor(endpoint, tokenBudget) {
    this.endpoint = endpoint;
    this.tokenBudget = tokenBudget;
    this.lastLatencyMs = 0;
    this.lastStateChars = 0;
    this.calls = 0;
  }

  /**
   * state 组装委托给 `state_machine.buildStateText`（**唯一真源**）。
   * 之前这里有一份独立实现，两份很快就会漂开——提示词是模型行为的一部分，
   * 有两份实现就等于有两个不同的模型。
   */
  buildState(goal, snapshot, history = [], memory = null) {
    const state = buildStateText({
      goal,
      snapshot,
      memory,
      config: { memoryCharBudget: 800, stateCharBudget: this.tokenBudget },
      historyLine: history.slice(-4).join(" ; "),
    });
    this.lastStateChars = state.length;
    return state;
  }

  /** 问题集同样委托给唯一真源。默认 10 个问题在**同一次前向**里问完。 */
  static questions(snapshot, { config = {}, scrollDisabled = { down: false, up: false } } = {}) {
    return buildQuestions({ actions: snapshot.actions, config, scrollDisabled });
  }

  async decide(state, questions) {
    const t0 = performance.now();
    const res = await fetch(this.endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state, questions }),
    });
    if (!res.ok) throw new Error(`决策服务返回 ${res.status}`);
    const body = await res.json();
    this.lastLatencyMs = performance.now() - t0;
    this.calls += 1;
    return body;
  }
}

// --------------------------------------------------------------------------- //
// 安全闸门
// --------------------------------------------------------------------------- //
const IRREVERSIBLE_PATTERNS = [
  /\b(pay|payment|purchase|buy|order|checkout|place\s+order|subscribe|confirm\s+order)\b/i,
  /\b(delete|remove|discard|unsubscribe|revoke|cancel\s+account)\b/i,
  /\b(send|submit|post|publish|share|tweet)\b/i,
  /\b(authorize|grant|allow\s+access|install|uninstall)\b/i,
  /支付|付款|下单|购买|订阅|确认订单|结账/,
  /删除|移除|注销|退订|撤销/,
  /发送|提交|发布|分享|投稿|上传/,
  /授权|允许访问|安装|卸载/,
];

export function looksIrreversible(action, decisionFlags) {
  if (decisionFlags?.irreversible) return true;
  const haystack = `${action?.label ?? ""} ${action?.role ?? ""} ${action?.value ?? ""}`;
  return IRREVERSIBLE_PATTERNS.some((re) => re.test(haystack));
}

// --------------------------------------------------------------------------- //
// 执行
// --------------------------------------------------------------------------- //
export async function execute(cdp, action, snapshot, text) {
  if (action.kind === "wait") {
    await sleep(100);
    return { kind: "wait" };
  }
  if (action.kind === "scroll") {
    await cdp.send("Input.dispatchMouseEvent", {
      type: "mouseWheel",
      x: Math.floor(snapshot.w / 2),
      y: Math.floor(snapshot.h / 2),
      deltaX: 0,
      deltaY: action.delta,
    });
    return { kind: "scroll", delta: action.delta };
  }

  // 现算几何 + 命中测试。**不用快照里的坐标**：快照之后页面可能已滚动/动画/懒加载。
  const point = await cdp.evaluate(
    `(a => {
       const e = window.__jevFast?.nodes.get(a.node);
       if (!e) return null;
       if (!e.isConnected || e.matches(':disabled') ||
           e.closest('[aria-disabled="true"],[inert]') ||
           !e.checkVisibility({checkOpacity:true,checkVisibilityCSS:true})) return null;
       if (a.kind === 'fill' && (e.readOnly || e.getAttribute('aria-readonly') === 'true')) return null;
       const r = e.getBoundingClientRect();
       const x = r.x + r.width / 2, y = r.y + r.height / 2;
       if (!r.width || !r.height || x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) return null;
       if (!e.contains(document.elementFromPoint(x, y))) return null;
       return { x, y };
     })(${JSON.stringify({ node: action.node, kind: action.kind })})`,
  );
  if (point === null) throw new StalePage("目标元素已变化或被遮挡，需要重新观测");

  if (action.kind === "select") {
    const ok = await cdp.evaluate(
      `(a => {
         const e = window.__jevFast?.nodes.get(a.node);
         if (!e || e.tagName !== 'SELECT') return false;
         if (![...e.options].some(o => o.value === a.value && !o.disabled)) return false;
         e.value = a.value;
         e.dispatchEvent(new Event('input', { bubbles: true }));
         e.dispatchEvent(new Event('change', { bubbles: true }));
         return true;
       })(${JSON.stringify({ node: action.node, value: action.value })})`,
    );
    if (!ok) throw new StalePage("下拉执行未被确认");
    return { kind: "select", value: action.value };
  }

  const { x, y } = point;
  await cdp.send("Input.dispatchMouseEvent", { type: "mousePressed", x, y, button: "left", clickCount: 1 });
  await cdp.send("Input.dispatchMouseEvent", { type: "mouseReleased", x, y, button: "left", clickCount: 1 });

  if (action.kind === "fill") {
    // Ctrl+A（Windows/Linux modifiers=2）再 insertText：直接 insertText 会追加而不是替换
    const mods = process.platform === "darwin" ? 4 : 2;
    await cdp.send("Input.dispatchKeyEvent", { type: "keyDown", key: "a", code: "KeyA", modifiers: mods, commands: ["selectAll"] });
    await cdp.send("Input.dispatchKeyEvent", { type: "keyUp", key: "a", code: "KeyA", modifiers: mods });
    await cdp.send("Input.insertText", { text });
  }
  return { kind: action.kind, x, y };
}

export async function settle(cdp, action) {
  // 固定 sleep 是延迟的主要来源之一；这里按动作类型给最小必要等待：
  // combobox 填完要等可见候选（上限 200ms），其它最多两帧或 50ms。
  const isCombobox = action.kind === "fill";
  await cdp.evaluate(
    `(a => new Promise(resolve => {
       const field = window.__jevFast?.nodes.get(a.node);
       let frames = 0, stopped = false;
       const finish = () => { stopped = true; resolve(); };
       setTimeout(finish, a.isCombobox ? 200 : 50);
       const ready = () => {
         if (stopped) return;
         const ids = (field?.getAttribute('aria-controls') || field?.getAttribute('aria-owns') || '')
           .split(/\\s+/).filter(Boolean);
         const roots = ids.length ? ids.map(id => document.getElementById(id)).filter(Boolean) : [document];
         const options = roots.flatMap(r => [...r.querySelectorAll('[role="option"]')]);
         const hasVisible = options.some(el => {
           const r = el.getBoundingClientRect();
           return r.width && r.height && r.bottom > 0 && r.top < innerHeight &&
             el.checkVisibility({checkOpacity:true, checkVisibilityCSS:true});
         });
         if (++frames >= 2 && (!a.isCombobox || hasVisible)) finish();
         else requestAnimationFrame(ready);
       };
       requestAnimationFrame(ready);
     }))(${JSON.stringify({ node: action.node, isCombobox })} )`,
    { awaitPromise: true },
  );
}

// --------------------------------------------------------------------------- //
// 主循环
// --------------------------------------------------------------------------- //
export async function runLoop(opts) {
  const cdp = await Cdp.connect(opts.cdpPort);
  const decider = new Decider(opts.decideUrl, opts.stateTokenBudget);
  const metrics = { cycles: 0, cdpCalls: 0, snapshotMs: [], decideMs: [], executeMs: [], settleMs: [], stateChars: [] };
  const history = [];
  const decisions = [];

  try {
    await cdp.createSession(opts.url, opts.width, opts.height);

    for (let step = 1; step <= opts.maxSteps; step += 1) {
      const cycleStart = performance.now();

      // 1) SNAPSHOT —— 一次调用
      const tSnap = performance.now();
      const snapshot = await cdp.evaluate(SNAPSHOT_SRC, {});
      if (!snapshot) throw new StalePage("页面正在导航，快照为空");
      metrics.snapshotMs.push(performance.now() - tSnap);

      // 2) DECIDE —— 零次 CDP 调用
      const tDec = performance.now();
      const state = decider.buildState(opts.goal, snapshot, history);
      const resp = await decider.decide(state, Decider.questions(snapshot));
      metrics.decideMs.push(performance.now() - tDec);
      metrics.stateChars.push(state.length);

      const results = resp.results || {};
      const operation = results.operation;
      const target = results.target;
      if (!operation || !target) {
        console.log(`[${step}] 决策缺失（degraded=${resp.degraded ?? "none"}），停止`);
        break;
      }

      const op = operation.choice;
      const doneScore = results.done?.noul ?? 0;
      const flags = { irreversible: (results.irreversible?.noul ?? 0) > 0.5 };

      if (op === "DONE" || doneScore > 0.8) {
        console.log(`[${step}] 判定完成：DONE(conf=${operation.confidence.toFixed(3)}) done_p=${doneScore.toFixed(3)}`);
        decisions.push({ step, op: "DONE", confidence: operation.confidence });
        break;
      }

      const action = snapshot.actions.find((a) => a.id === target.choice);
      if (!action) {
        console.log(`[${step}] 目标 ${target.choice} 不在观测表内 → 视为陈旧，重观测`);
        continue;
      }

      const irreversible = looksIrreversible(action, flags);
      const record = {
        step,
        op,
        target: action.id,
        label: action.label,
        kind: action.kind,
        opConfidence: operation.confidence,
        targetConfidence: target.confidence,
        irreversible,
        executed: false,
      };

      if (irreversible && !opts.allowIrreversible) {
        console.log(
          `[${step}] 停在确认：${action.label} 疑似不可逆（op_conf=${operation.confidence.toFixed(3)}）` +
            "——确认后加 --allow-irreversible",
        );
        decisions.push(record);
        break;
      }

      if (!opts.live) {
        console.log(
          `[${step}] dry-run: ${op} → [${action.id}] ${action.role} "${action.label}"` +
            ` (op_conf=${operation.confidence.toFixed(3)}, target_conf=${target.confidence.toFixed(3)})`,
        );
        decisions.push(record);
        history.push(`${op} ${action.label}`);
        continue;
      }

      // 3) EXECUTE —— 命中测试 + 输入派发
      const tExec = performance.now();
      try {
        await execute(cdp, action, snapshot, action.value || "");
        record.executed = true;
      } catch (err) {
        if (err instanceof StalePage) {
          console.log(`[${step}] ${err.message}`);
          continue;
        }
        throw err;
      }
      metrics.executeMs.push(performance.now() - tExec);

      // 4) SETTLE
      const tSet = performance.now();
      await settle(cdp, action);
      metrics.settleMs.push(performance.now() - tSet);

      decisions.push(record);
      history.push(`${op} ${action.label}`);
      metrics.cycles += 1;
      console.log(`[${step}] ${op} → ${action.label} (${Math.round(performance.now() - cycleStart)}ms)`);
    }
  } finally {
    metrics.cdpCalls = cdp.calls;
    await cdp.close();
  }
  return { metrics, decisions, history };
}

const avg = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : 0);

export function printMetrics(metrics) {
  const total = ["snapshotMs", "decideMs", "executeMs", "settleMs"].reduce(
    (acc, k) => acc + metrics[k].reduce((a, b) => a + b, 0),
    0,
  );
  console.log("\n=== 每轮成本分解 ===");
  console.log(`  轮数(已执行):       ${metrics.cycles}`);
  console.log(`  CDP 调用总数:       ${metrics.cdpCalls}`);
  console.log(`  快照 平均:          ${avg(metrics.snapshotMs).toFixed(1)} ms`);
  console.log(`  决策 平均:          ${avg(metrics.decideMs).toFixed(1)} ms   ← 本地模型，0 次 CDP`);
  console.log(`  执行 平均:          ${avg(metrics.executeMs).toFixed(1)} ms`);
  console.log(`  等待 平均:          ${avg(metrics.settleMs).toFixed(1)} ms`);
  console.log(`  三项合计:           ${total.toFixed(1)} ms`);
  console.log(`  state 平均字符数:   ${Math.round(avg(metrics.stateChars))}`);
  console.log("\n对照（来源：browser-use/jev-ultrafast 公开测量）:");
  console.log("  优化前后 中位任务时间 9.450s → 7.092s（-25%），浏览器协议调用 1092 → 101。");
  console.log("  若本循环的 CDP 调用数远高于 100，先查是不是每步都重新 attach 或重复截图。");
}

if (import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith("cdp_loop.mjs")) {
  const opts = parseArgs(process.argv.slice(2));
  if (opts.help || !opts.url || !opts.goal || !opts.decideUrl) {
    console.log(USAGE);
    process.exit(opts.help ? 0 : 2);
  }
  try {
    const { metrics, decisions } = await runLoop(opts);
    printMetrics(metrics);
    console.log("\n=== 决策轨迹 ===");
    for (const d of decisions) {
      console.log(
        `  #${d.step} ${d.op} ${d.target ?? ""} ${d.label ?? ""}` +
          `${d.irreversible ? " [不可逆]" : ""}${d.executed ? " [已执行]" : ""}`,
      );
    }
    if (!opts.live) console.log("\n（dry-run：未执行任何动作。加 --live 才真的操作页面。）");
  } catch (err) {
    console.error(`\n失败: ${err.message}`);
    if (/ECONNREFUSED|fetch failed/.test(String(err.message))) {
      console.error(
        "看起来 Chrome 的调试端口没开。用独立 profile 起一个：\n" +
          '  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe" ' +
          '--remote-debugging-port=9222 --user-data-dir="%TEMP%\\cdp-profile" about:blank',
      );
      console.error("注意：已经在跑的 Chrome 没带 --remote-debugging-port 时无法被接上。");
    }
    process.exit(1);
  }
}
