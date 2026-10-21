#!/usr/bin/env node
/* 状态机解释器：把 `state_machine.mjs` 产出的 effects 变成真实的 CDP 调用。
 *
 * 分层：
 *   state_machine.mjs   纯 reducer，决定"下一步做什么"（无副作用、可全分支单测）
 *   loop_driver.mjs     ← 本文件。把 effects 变成 IO，把 IO 结果变成事件
 *   cdp_loop.mjs        执行原语（CDP 连接、点击/填字/滚动、就绪等待、安全闸门）
 *
 * **IO 通过 `io` 接口注入**，所以整条循环可以在没有 Chrome、没有决策服务的情况下
 * 用假 IO 跑完（见 sm_selftest.mjs 的驱动用例）。这是这套分层最实际的收益：
 * 真浏览器上极难复现的状态迁移，在这里可以一条条喂出来。
 *
 * 效果 → 事件的对应是**一对一**的：一批 effects 里最多只能有一个会产出后续事件，
 * 多于一个说明 reducer 接线错了，这里直接报错而不是静默取最后一个。
 */

import process from "node:process";

import { Cdp, Decider, StalePage, execute, looksIrreversible, settle } from "./cdp_loop.mjs";
import { HANDOFF_REASONS } from "./handoff.mjs";
import { TERMINAL, createContext, reduce, S } from "./state_machine.mjs";

/** 一次效果批里允许产出事件的 effect 类型（其余是 fire-and-forget）。 */
const EVENT_PRODUCING = new Set(["snapshot", "decide", "execute", "settle", "scroll", "verify", "handoff", "advance"]);

const LOGIN_PATTERN = /\b(sign\s?in|log\s?in|登录|登入)\b/i;
const CAPTCHA_PATTERN = /\b(captcha|recaptcha|verify you are human|验证码)\b/i;

// --------------------------------------------------------------------------- //
// 结构化核验规则：把"目标达成了没有"变成可机器判定的断言
// --------------------------------------------------------------------------- //
/**
 * 优先用**结构化断言**核验，而不是再问一次模型。
 *
 * 依据：browser-use/jev-ultrafast 的示例是"supplies a goal and independently
 * verifies the outcome"，且明确说"A DONE choice still requires independent outcome
 * verification"。这里的 `expectations` 就是那个独立核验：
 *
 *   { pattern: "Zürich", expect: "present", where: "text" }
 *   { pattern: "one-way", expect: "absent",  where: "label" }
 *
 * 没有给定 expectations 时退回问模型一次，但证据来源会被标成 `model_reported`
 * ——**这不是独立核验**，报告里必须如实标注，不能与结构化证据混为一谈。
 */
export function checkExpectations(snapshot, expectations) {
  if (!expectations?.length) return { decided: false, ok: null, details: [] };
  const text = String(snapshot.text || "");
  const labels = (snapshot.actions || []).map((a) => `${a.label} ${a.role}`).join("\n");
  const details = [];
  let ok = true;
  for (const e of expectations) {
    const haystack = e.where === "label" ? labels : text;
    const present = new RegExp(escapeRe(e.pattern), "i").test(haystack);
    const pass = e.expect === "absent" ? !present : present;
    if (!pass) ok = false;
    details.push({ ...e, observed: present, pass });
  }
  return { decided: true, ok, details };
}

function escapeRe(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// --------------------------------------------------------------------------- //
// 真实 CDP 侧的 IO 实现
// --------------------------------------------------------------------------- //
export function createCdpIo({ cdp, decider, opts, snapshotSrc, log = console.log }) {
  let cachedSnapshot = null;

  return {
    async snapshot() {
      // 一次 Runtime.evaluate 原子取得全表（见 browser/snapshot.js 的注释）
      const snapshot = await cdp.evaluate(snapshotSrc, {});
      if (!snapshot) throw new StalePage("页面正在导航，快照为空");
      cachedSnapshot = snapshot;
      return snapshot;
    },

    async decide(state, questions) {
      return decider.decide(state, questions);
    },

    async scroll(effect) {
      const snap = cachedSnapshot;
      await cdp.send("Input.dispatchMouseEvent", {
        type: "mouseWheel",
        x: Math.floor((snap?.w ?? 1120) / 2),
        y: Math.floor((snap?.h ?? 780) / 2),
        deltaX: 0,
        deltaY: effect.delta,
      });
      // 重新读一次滚动位置；不要把"命令发出去了"当作"滚动发生了"
      const info = await cdp.evaluate(
        "(() => ({ y: Math.round(scrollY), height: document.documentElement.scrollHeight }))()",
      );
      return { newScrollY: info.y, newHeight: info.height };
    },

    async execute(action, text) {
      await execute(cdp, action, cachedSnapshot, text);
    },

    async settle(action) {
      await settle(cdp, action);
    },

    async verify(plan, ctx) {
      // 1) 移交回来的核验：用**页面实测**而不是模型说法
      if (ctx.state === S.HANDOFF_VERIFY) {
        const capsule = ctx.handoff?.capsule;
        const node = capsule?.target?.node;
        if (plan.question === "did_field_change" && node != null) {
          const now = await cdp.evaluate(
            `(() => { const e = window.__jevFast?.nodes.get(${node}); return e ? String(e.value ?? "") : null; })()`,
          );
          const before = capsule?.target?.value ?? null;
          return {
            ok: now !== null && now !== before,
            evidence: `field ${node}: ${JSON.stringify(before)} → ${JSON.stringify(now)}`,
            source: "structural",
          };
        }
        if (plan.question === "signed_in") {
          const blocked = (cachedSnapshot?.actions || []).some((a) => LOGIN_PATTERN.test(a.label || ""));
          return { ok: !blocked, evidence: blocked ? "sign-in control still present" : "no sign-in control", source: "structural" };
        }
        if (plan.requiresHuman) {
          return { ok: opts.humanDecision === true, evidence: `human decision = ${opts.humanDecision}`, source: "human" };
        }
      }

      // 2) 目标达成的核验：结构化断言优先
      const snap = cachedSnapshot || (await this.snapshot());
      const structural = checkExpectations(snap, opts.expectations);
      if (structural.decided) {
        return {
          ok: structural.ok,
          evidence: `structural: ${JSON.stringify(structural.details)}`,
          source: "structural",
        };
      }
      // 3) 没有断言可核 → 只能问模型，并且**如实标注证据来源弱**
      const resp = await decider.decide(
        `GOAL: ${ctx.goal}\nURL: ${snap.url}\n\nPAGE TEXT:\n${String(snap.text || "").slice(0, 2000)}`,
        { goal_satisfied: { type: "noul", instructions: "Is the GOAL satisfied? Answer only from the page content above." } },
      );
      const p = resp?.results?.goal_satisfied?.noul ?? 0;
      return {
        ok: p > 0.5,
        evidence: `model_reported p=${p.toFixed(3)} (NOT an independent check — provide --expect for hard evidence)`,
        source: "model_reported",
      };
    },

    async handoff(capsule, ctx) {
      const handlers = opts.onHandoff;
      if (typeof handlers !== "function") {
        log("—— 移交请求（无 onHandoff 处理器，按超时处理）——");
        log(capsule.instructions);
        const err = new Error("no handoff handler registered");
        err.timeout = true;
        throw err;
      }
      const t0 = Date.now();
      const result = await handlers(capsule, ctx);
      return { ...result, elapsedMs: Date.now() - t0 };
    },

    emit(effect) {
      log(`[${effect.event?.kind ?? "emit"}] ${JSON.stringify(effect.event)}`);
    },
  };
}

// --------------------------------------------------------------------------- //
// 驱动器
// --------------------------------------------------------------------------- //
export async function drive({
  goal,
  url = "",
  config = {},
  io,
  onEvent = null,
  maxIterations = 200,
  log = console.log,
}) {
  const ctx = createContext({ goal, url, config });
  const trace = [];
  let event = { type: "START" };
  let iterations = 0;

  const record = (effect, result) => {
    const row = { step: ctx.memory.step, state: ctx.state, effect: effect.type, result };
    trace.push(row);
    if (onEvent) onEvent(row);
  };

  while (!TERMINAL.has(ctx.state) && iterations < maxIterations) {
    iterations += 1;
    const { effects } = reduce(ctx, event);
    event = null;

    const producing = effects.filter((e) => EVENT_PRODUCING.has(e.type));
    if (producing.length > 1) {
      throw new Error(
        `状态 ${ctx.state} 一次产出了 ${producing.length} 个会产出事件的 effect ` +
          `(${producing.map((e) => e.type).join(", ")})——reduce 接线错误，无法确定等待哪一个`,
      );
    }

    for (const eff of effects) {
      if (eff.type === "stop") {
        record(eff, "terminal");
        return { ctx, trace };
      }
      if (eff.type === "emit") {
        record(eff, "emitted");
        if (io.emit) io.emit(eff);
        continue;
      }

      try {
        if (eff.type === "advance") {
          // "立刻把这条事件送回机器"。用于 reducer 内部的状态推进
          // （例如 DECIDE 判定要移交 → 进入 HANDOFF_PREPARE 后需要一条入场事件）。
          // 不做成隐式约定，是因为隐式的后果正是本文件开头那条"接线缺口"报错。
          event = eff.event;
          record(eff, `→ ${eff.event.type}`);
        } else if (eff.type === "snapshot") {
          const snapshot = await io.snapshot();
          event = { type: "SNAPSHOT_OK", snapshot };
          record(eff, `ok(${(snapshot.actions || []).length} actions)`);
        } else if (eff.type === "decide") {
          const resp = await io.decide(eff.state, eff.questions);
          event = { type: "DECISION_OK", response: resp };
          record(eff, `ok(${Object.keys(eff.questions).length}q, excl=${JSON.stringify(eff.excluded)})`);
        } else if (eff.type === "scroll") {
          const r = await io.scroll(eff);
          event = { type: "SCROLL_DONE", direction: eff.direction, ...r };
          record(eff, `y=${r.newScrollY} h=${r.newHeight}`);
        } else if (eff.type === "execute") {
          await io.execute(eff.action, eff.text);
          event = { type: "ACTION_DONE", action: eff.action };
          record(eff, `${eff.action.kind} ${eff.action.label}`);
        } else if (eff.type === "settle") {
          await io.settle(eff.action);
          event = { type: "SETTLE_DONE" };
          record(eff, "settled");
        } else if (eff.type === "verify") {
          const r = await io.verify(eff.plan, ctx);
          event = ctx.state === S.HANDOFF_VERIFY
            ? { type: "RESUME_CHECK_DONE", ok: r.ok, detail: r.evidence }
            : { type: "VERIFY_DONE", ok: r.ok, evidence: `${r.source}: ${r.evidence}` };
          record(eff, `${r.ok ? "ok" : "reject"} (${r.source})`);
        } else if (eff.type === "handoff") {
          const r = await io.handoff(eff.capsule, ctx);
          event = { type: "HANDOFF_RETURNED", result: r, elapsedMs: r.elapsedMs };
          record(eff, `returned ok=${r.ok}`);
        }
        // emit 之外的效果都已处理；出现未知类型说明 reducer 加了新 effect 但这里没接
        else {
          throw new Error(`未接线的 effect 类型: ${eff.type}`);
        }
      } catch (err) {
        const stale = err instanceof StalePage || /stale|changed since|陈旧/i.test(String(err?.message));
        const timeout = err?.timeout === true || /timeout|超时/i.test(String(err?.message));
        record(eff, `error: ${err.message}`);
        if (eff.type === "snapshot") {
          event = stale ? { type: "SNAPSHOT_STALE" } : { type: "SNAPSHOT_FAILED", error: err.message };
        } else if (eff.type === "decide") {
          event = { type: "DECISION_FAILED", error: err.message };
        } else if (eff.type === "execute") {
          event = stale ? { type: "ACTION_STALE" } : { type: "ACTION_FAILED", error: err.message };
        } else if (eff.type === "scroll") {
          event = { type: "SCROLL_FAILED", error: err.message };
        } else if (eff.type === "handoff") {
          event = timeout ? { type: "HANDOFF_TIMEOUT", detail: err.message } : { type: "HANDOFF_TIMEOUT", detail: err.message };
        } else if (eff.type === "verify") {
          event = ctx.state === S.HANDOFF_VERIFY
            ? { type: "RESUME_CHECK_DONE", ok: false, detail: err.message }
            : { type: "VERIFY_DONE", ok: false, evidence: err.message };
        } else {
          throw err;
        }
      }
    }

    if (!event) {
      throw new Error(`机器停在 ${ctx.state}：本批 effect 没有产出后续事件（接线缺口）`);
    }
  }

  if (!TERMINAL.has(ctx.state)) {
    throw new Error(`超过 maxIterations=${maxIterations} 仍未终止，最后状态 ${ctx.state}`);
  }
  return { ctx, trace };
}

// --------------------------------------------------------------------------- //
// 汇总
// --------------------------------------------------------------------------- //
export function summarize({ ctx, trace }) {
  const byEffect = {};
  for (const t of trace) byEffect[t.effect] = (byEffect[t.effect] || 0) + 1;
  return {
    terminal: ctx.state,
    reason: ctx.reason,
    steps: ctx.memory.step,
    handoffs: ctx.memory.budget.handoffs,
    handoffMs: ctx.memory.budget.handoffMs,
    scrolls: ctx.memory.budget.scrolls,
    scrollEpoch: ctx.scroll.epoch,
    disabled: { ...ctx.scroll.disabled },
    suppressed: ctx.memory.suppressedSignatures().length,
    facts: ctx.memory.verifiedFacts().length,
    memory: ctx.memory.stats(),
    effects: byEffect,
    evidence: ctx.terminal?.evidence ?? null,
    originalDiff: ctx.terminal?.originalDiff ?? null,
  };
}

// --------------------------------------------------------------------------- //
// 端到端 CLI
// --------------------------------------------------------------------------- //
const USAGE = `
用法:
  node loop_driver.mjs --url <起始页> --goal <目标> --decide-url <本地决策服务> [选项]

必填
  --url            起始 URL
  --goal           自然语言目标（建议英文）
  --decide-url     本地决策服务，例如 http://127.0.0.1:38123/v1/systemone

选项
  --cdp-port        Chrome 远程调试端口，默认 9222
  --max-steps       最大步数，默认 12
  --max-scrolls     滚动预算，默认 8
  --live            真的执行动作（默认 dry-run）
  --allow-irreversible  允许执行不可逆动作（默认停在确认）
  --expect "pattern:present[:text|label]"   结构化核验断言，可重复
  --handoff-inbox   移交载荷落盘的目录（集成模式下由宿主轮询该目录）

结构化核验示例（强烈建议至少给一条，否则"完成"只能靠模型自述）:
  --expect "Zürich:present:text" --expect "one-way:present:label"
`;

export function parseDriverArgs(argv) {
  const out = {
    url: "",
    goal: "",
    decideUrl: "",
    cdpPort: 9222,
    maxSteps: 12,
    maxScrolls: 8,
    live: false,
    allowIrreversible: false,
    expectations: [],
    handoffInbox: "",
    help: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    const next = () => argv[++i];
    if (a === "--url") out.url = next();
    else if (a === "--goal") out.goal = next();
    else if (a === "--decide-url") out.decideUrl = next();
    else if (a === "--cdp-port") out.cdpPort = Number(next());
    else if (a === "--max-steps") out.maxSteps = Number(next());
    else if (a === "--max-scrolls") out.maxScrolls = Number(next());
    else if (a === "--live") out.live = true;
    else if (a === "--allow-irreversible") out.allowIrreversible = true;
    else if (a === "--handoff-inbox") out.handoffInbox = next();
    else if (a === "--expect") {
      const [pattern, expect, where] = String(next()).split(":");
      out.expectations.push({ pattern, expect: expect || "present", where: where || "text" });
    } else if (a === "--help" || a === "-h") out.help = true;
    else throw new Error(`未知参数: ${a}`);
  }
  return out;
}

if (process.argv[1]?.endsWith("loop_driver.mjs")) {
  const opts = parseDriverArgs(process.argv.slice(2));
  if (opts.help || !opts.url || !opts.goal || !opts.decideUrl) {
    console.log(USAGE);
    process.exit(opts.help ? 0 : 2);
  }
  const fs = await import("node:fs");
  const path = await import("node:path");
  const { fileURLToPath } = await import("node:url");
  const snapshotSrc = fs.readFileSync(
    path.join(path.dirname(fileURLToPath(import.meta.url)), "snapshot.js"),
    "utf8",
  );

  const cdp = await Cdp.connect(opts.cdpPort);
  await cdp.createSession(opts.url, 1120, 780);
  const decider = new Decider(opts.decideUrl, 6000);

  const io = createCdpIo({ cdp, decider, opts, snapshotSrc });

  try {
    const result = await drive({ goal: opts.goal, url: opts.url, io, config: { maxSteps: opts.maxSteps, maxScrolls: opts.maxScrolls } });
    const s = summarize(result);
    console.log("\n=== 终止 ===");
    console.log(`  state=${s.terminal} reason=${s.reason ?? "(none)"}`);
    console.log(`  steps=${s.steps} scrolls=${s.scrolls}(epoch ${s.scrollEpoch}) handoffs=${s.handoffs}`);
    console.log(`  候选被剔除=${s.suppressed}  已核实事实=${s.facts}`);
    console.log(`  effects=${JSON.stringify(s.effects)}`);
    if (s.evidence) console.log(`  完成证据=${s.evidence}`);
    if (s.originalDiff) console.log(`  与原始状态差异=${JSON.stringify(s.originalDiff)}`);
    console.log("\n=== 轨迹 ===");
    for (const t of result.trace) {
      console.log(`  #${t.step} ${t.state.padEnd(16)} ${t.effect.padEnd(9)} ${t.result}`);
    }
  } finally {
    await cdp.close();
  }
}
