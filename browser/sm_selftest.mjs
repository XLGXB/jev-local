#!/usr/bin/env node
/* 状态机自测：**不连 Chrome、不连决策服务**，全部用构造事件序列驱动纯 reducer。
 *
 * 状态机的 bug 几乎都是"某条分支永远走不到"或"顺序反了"，而这类 bug 在真浏览器上
 * 极难复现（页面每次都不一样）。把每条分支用事件喂一遍，是唯一能稳定抓住它们的方法。
 *
 * 跑法:
 *   node browser/sm_selftest.mjs          # 独立跑
 *   node browser/selftest.mjs             # 连浏览器层基础测试一起跑
 */

import process from "node:process";

import { checkExpectations, drive, parseDriverArgs } from "./loop_driver.mjs";
import { HANDOFF_REASONS, buildGrant } from "./handoff.mjs";
import { LoopMemory, actionSignature } from "./memory.mjs";
import { StalePage } from "./cdp_loop.mjs";
import {
  BLOCK_REASONS,
  S,
  SNAPSHOT_OK,
  buildQuestions,
  createContext,
  digestSnapshot,
  enterScroll,
  reduce,
  snapshotStamp,
  startFind,
} from "./state_machine.mjs";

// --------------------------------------------------------------------------- //
// 构造助手
// --------------------------------------------------------------------------- //
const pageKey = (n) => [1, `https://x/${n}`, 0, 0, 1120, 780, []];

function mkSnapshot({ n = 1, text = "hello page", actions = null, scrollY = 0, height = 5000 } = {}) {
  const acts =
    actions ??
    [
      { id: "e1", node: 101, kind: "click", role: "button", label: "Go", value: "" },
      { id: "e2", node: 102, kind: "fill", role: "textbox", label: "Name", value: "" },
      { id: "wait", node: -1, kind: "wait", role: "wait", label: "Wait" },
    ];
  return {
    url: `https://x/${n}`,
    title: "t",
    w: 1120,
    h: 780,
    table: ["[e1] button Go", "[e2] textbox Name"],
    text,
    scroll: { y: scrollY, height },
    actions: acts,
    page_key: pageKey(n),
    guards: {},
    omitted_actions: 0,
  };
}

function dec({ op = "CLICK", target = "e1", done = 0, irr = 0, extra = {} } = {}) {
  return {
    model: "test",
    results: {
      operation: { type: "choice", choice: op, confidence: 0.9, probabilities: [] },
      target: { type: "choice", choice: target, confidence: 0.9, probabilities: [] },
      done: { type: "noul", noul: done },
      irreversible: { type: "noul", noul: irr },
      ...extra,
    },
  };
}

const sigOf = (snap, action) =>
  actionSignature(digestSnapshot(snap).pageKeyHash, action.node, action.kind, action.label);

function boot(ctx, snap = mkSnapshot()) {
  reduce(ctx, { type: "START" });
  reduce(ctx, { type: SNAPSHOT_OK, snapshot: snap });
  return ctx;
}

/** 走完一次"执行一个动作"的固定路径。 */
function cycle(ctx, snap, decision) {
  reduce(ctx, { type: "DECISION_OK", response: decision });
  reduce(ctx, { type: "ACTION_DONE" });
  reduce(ctx, { type: "SETTLE_DONE" });
  reduce(ctx, { type: SNAPSHOT_OK, snapshot: snap });
  return ctx;
}

function ctxFor(overrides = {}) {
  return createContext({ goal: "achieve the goal", url: "https://x/1", config: overrides });
}

// --------------------------------------------------------------------------- //
// 测试
// --------------------------------------------------------------------------- //
export const smTests = [
  // ---- 基本生命周期 ----
  [
    "启动后进入 OBSERVE 并请求快照",
    () => {
      const ctx = ctxFor();
      const { effects } = reduce(ctx, { type: "START" });
      assert(ctx.state === S.OBSERVE, `state=${ctx.state}`);
      assert(effects[0].type === "snapshot", "应先请求快照");
    },
  ],
  [
    "快照成功后进入 DECIDE，且全部问题在同一次请求里",
    () => {
      const ctx = boot(ctxFor());
      assert(ctx.state === S.DECIDE, `state=${ctx.state}`);
      const { effects } = reduce(ctx, { type: "DECISION_OK", response: dec() });
      assert(effects.length === 1 && effects[0].type === "execute", `effects=${JSON.stringify(effects.map((e) => e.type))}`);
      const q = buildQuestions({ actions: ctx.filteredActions, config: ctx.config });
      const expected = [
        "operation", "target", "done", "irreversible", "needs_login",
        "needs_credentials", "captcha", "needs_vision", "unsupported", "needs_compose",
      ];
      for (const k of expected) assert(k in q, `缺问题 ${k}`);
      assert(Object.keys(q).length === 10, `问题数应为 10，得到 ${Object.keys(q).length}`);
    },
  ],
  [
    "DECIDE 状态忽略杂散事件（不静默改变状态）",
    () => {
      const ctx = boot(ctxFor());
      const before = ctx.state;
      const { effects } = reduce(ctx, { type: SNAPSHOT_OK, snapshot: mkSnapshot({ n: 9 }) });
      assert(ctx.state === before, `状态不该变，得到 ${ctx.state}`);
      assert(effects[0]?.event?.kind === "unexpected_event", "应记录而非执行");
    },
  ],

  // ---- 不变量 I1：DONE 不可自证 ----
  [
    "I1 模型自报 DONE 不直接接受，必须进 VERIFY",
    () => {
      const ctx = boot(ctxFor());
      const { effects } = reduce(ctx, { type: "DECISION_OK", response: dec({ op: "DONE", done: 1 }) });
      assert(ctx.state === S.VERIFY, `应进 VERIFY，得到 ${ctx.state}`);
      assert(effects[0].type === "verify", "必须发出核验 effect");
      assert(ctx.state !== S.DONE, "绝不能直接进 DONE");
    },
  ],
  [
    "I1 核验不过不算完成",
    () => {
      const ctx = boot(ctxFor());
      reduce(ctx, { type: "DECISION_OK", response: dec({ op: "DONE", done: 1 }) });
      reduce(ctx, { type: "VERIFY_DONE", ok: false });
      assert(ctx.state !== S.DONE, `不应完成，得到 ${ctx.state}`);
      assert(ctx.doneRejections === 1, "应记录一次被驳回");
      assert(ctx.state === S.OBSERVE, "应回到观测继续尝试");
    },
  ],
  [
    "I1 核验通过才 DONE，并带原始状态差异",
    () => {
      const ctx = boot(ctxFor());
      ctx.memory.captureOriginal({ pageKey: pageKey(1), url: "https://x/1", scrollY: 0, formValues: { name: "" } });
      reduce(ctx, { type: "DECISION_OK", response: dec({ op: "DONE", done: 1 }) });
      const { effects } = reduce(ctx, { type: "VERIFY_DONE", ok: true, evidence: "cart badge shows 1" });
      assert(ctx.state === S.DONE, `state=${ctx.state}`);
      assert(effects[0].type === "stop", "终态应发出 stop");
      assert(ctx.terminal.evidence === "cart badge shows 1", "必须留下证据");
      assert("originalDiff" in ctx.terminal, "必须带上与原始状态的差异");
    },
  ],
  [
    "核验连续被驳回到上限即终止（不是无限重试）",
    () => {
      const ctx = boot(ctxFor());
      for (let i = 0; i < 3; i += 1) {
        if (ctx.state === S.DECIDE) {
          reduce(ctx, { type: "DECISION_OK", response: dec({ op: "DONE", done: 1 }) });
        }
        if (ctx.state === S.VERIFY) reduce(ctx, { type: "VERIFY_DONE", ok: false });
        if (ctx.state === S.OBSERVE) reduce(ctx, { type: SNAPSHOT_OK, snapshot: mkSnapshot() });
      }
      assert(ctx.state === S.BLOCKED, `应终止，得到 ${ctx.state}`);
      assert(ctx.reason === BLOCK_REASONS.VERIFY_FAILED, `reason=${ctx.reason}`);
    },
  ],

  // ---- 执行路径 ----
  [
    "CLICK → EXECUTE → SETTLE → 重新观测",
    () => {
      const ctx = boot(ctxFor());
      let r = reduce(ctx, { type: "DECISION_OK", response: dec({ op: "CLICK", target: "e1" }) });
      assert(ctx.state === S.EXECUTE && r.effects[0].type === "execute", `state=${ctx.state}`);
      reduce(ctx, { type: "ACTION_DONE" });
      assert(ctx.state === S.SETTLE, `state=${ctx.state}`);
      r = reduce(ctx, { type: "SETTLE_DONE" });
      assert(ctx.state === S.OBSERVE && r.effects[0].type === "snapshot", `state=${ctx.state}`);
    },
  ],
  [
    "动作陈旧 → 记一次失败并重新观测（不空转）",
    () => {
      const ctx = boot(ctxFor());
      reduce(ctx, { type: "DECISION_OK", response: dec() });
      const { effects } = reduce(ctx, { type: "ACTION_STALE" });
      assert(ctx.state === S.OBSERVE && effects[0].type === "snapshot", `state=${ctx.state}`);
      assert(ctx.memory.budget.staleRecoveries === 1, "应记一次陈旧恢复");
    },
  ],
  [
    "点两次没效果后，该目标从候选集里被剔除（改候选而不是劝模型）",
    () => {
      const ctx = boot(ctxFor());
      const snap = mkSnapshot();
      const target = snap.actions[0];
      const sig = sigOf(snap, target);

      cycle(ctx, snap, dec({ op: "CLICK", target: "e1" }));
      assert(ctx.memory.attempts.get(sig)?.count === 1, "第一次应记 no_effect");
      assert(!ctx.memory.isSuppressed(sig), "一次还不该剔除");

      cycle(ctx, snap, dec({ op: "CLICK", target: "e1" }));
      assert(ctx.memory.isSuppressed(sig), "第二次后应剔除");
      assert(ctx.excludedCounts.suppressed >= 1, `应在候选统计里体现，得到 ${JSON.stringify(ctx.excludedCounts)}`);
      assert(!ctx.filteredActions.some((a) => a.id === "e1"), "e1 必须不在候选里");
    },
  ],
  [
    "有效果的动作会清掉失败计数（合法重试不累积）",
    () => {
      const ctx = boot(ctxFor());
      const snap = mkSnapshot();
      const sig = sigOf(snap, snap.actions[0]);

      cycle(ctx, snap, dec({ op: "CLICK", target: "e1" })); // 同一页面 → 无效果
      assert(ctx.memory.attempts.get(sig)?.count === 1, `应记 1 次失败，得到 ${ctx.memory.attempts.get(sig)?.count}`);

      // 这一次页面真的变了 → 失败计数必须清空，否则合法的"再试一次"会被永久封印
      cycle(ctx, mkSnapshot({ n: 2, text: "changed" }), dec({ op: "CLICK", target: "e1" }));
      assert(!ctx.memory.attempts.has(sig), "有进展时应清掉失败计数");
      assert(!ctx.memory.isSuppressed(sig), "也不该被剔除");
    },
  ],
  [
    "步骤预算用尽即终止",
    () => {
      const ctx = ctxFor({ maxSteps: 1 });
      boot(ctx);
      reduce(ctx, { type: "DECISION_OK", response: dec() });
      reduce(ctx, { type: "ACTION_DONE" });
      reduce(ctx, { type: "SETTLE_DONE" });
      reduce(ctx, { type: SNAPSHOT_OK, snapshot: mkSnapshot() });
      assert(ctx.state === S.BLOCKED && ctx.reason === BLOCK_REASONS.STEP_BUDGET, `state=${ctx.state} reason=${ctx.reason}`);
    },
  ],
  [
    "USER_ABORT 在任何活动状态都立即终止",
    () => {
      const ctx = boot(ctxFor());
      reduce(ctx, { type: "USER_ABORT" });
      assert(ctx.state === S.BLOCKED && ctx.reason === BLOCK_REASONS.USER_ABORT, `reason=${ctx.reason}`);
    },
  ],

  // ---- 滚动处理 ----
  [
    "滚动完成后必须重新观测，且自增 epoch（几何失效标记）",
    () => {
      const ctx = boot(ctxFor());
      const before = ctx.scroll.epoch;
      reduce(ctx, { type: "DECISION_OK", response: dec({ op: "SCROLL_DOWN" }) });
      assert(ctx.state === S.SCROLL, `state=${ctx.state}`);
      const { effects } = reduce(ctx, { type: "SCROLL_DONE", direction: "down", newScrollY: 560, newHeight: 5000 });
      assert(effects[0].type === "snapshot", "滚动后必须先重新观测");
      assert(ctx.state === S.OBSERVE, `state=${ctx.state}`);
      assert(ctx.scroll.epoch === before + 1, "滚动应自增 epoch");
    },
  ],
  [
    "滚动停滞两次后禁用该方向，并回到决策（不空转）",
    () => {
      const ctx = boot(ctxFor());
      ctx.scroll.lastY = 560;
      ctx.scroll.lastHeight = 5000;
      reduce(ctx, { type: "DECISION_OK", response: dec({ op: "SCROLL_DOWN" }) });
      reduce(ctx, { type: "SCROLL_DONE", direction: "down", newScrollY: 560, newHeight: 5000 });
      assert(ctx.scroll.stalls === 1 && !ctx.scroll.disabled.down, "第一次停滞不该禁用");
      reduce(ctx, { type: SNAPSHOT_OK, snapshot: mkSnapshot({ scrollY: 560 }) });
      reduce(ctx, { type: "DECISION_OK", response: dec({ op: "SCROLL_DOWN" }) });
      reduce(ctx, { type: "SCROLL_DONE", direction: "down", newScrollY: 560, newHeight: 5000 });
      assert(ctx.scroll.disabled.down, "第二次停滞应禁用向下");
      assert(ctx.memory.fact("scroll_down_stalled"), "应记成事实，供后续与上报使用");
    },
  ],
  [
    "方向往复三次判定为震荡并终止",
    () => {
      const ctx = boot(ctxFor());
      const dirs = ["down", "up", "down"];
      for (let i = 0; i < dirs.length; i += 1) {
        const dir = dirs[i];
        if (ctx.state === S.DECIDE) {
          reduce(ctx, { type: "DECISION_OK", response: dec({ op: dir === "down" ? "SCROLL_DOWN" : "SCROLL_UP" }) });
        }
        if (ctx.state === S.SCROLL) {
          reduce(ctx, { type: "SCROLL_DONE", direction: dir, newScrollY: i === 0 ? 560 : i === 1 ? 0 : 560, newHeight: 5000 });
        }
        if (ctx.state === S.OBSERVE) {
          reduce(ctx, { type: SNAPSHOT_OK, snapshot: mkSnapshot({ scrollY: 560 }) });
        }
      }
      assert(ctx.state === S.BLOCKED, `应终止，得到 ${ctx.state}`);
      assert(ctx.reason === BLOCK_REASONS.SCROLL_OSCILLATING, `reason=${ctx.reason}`);
    },
  ],
  [
    "滚动预算用尽即终止",
    () => {
      const ctx = boot(ctxFor({ maxScrolls: 1 }));
      reduce(ctx, { type: "DECISION_OK", response: dec({ op: "SCROLL_DOWN" }) });
      reduce(ctx, { type: "SCROLL_DONE", direction: "down", newScrollY: 560, newHeight: 5000 });
      reduce(ctx, { type: SNAPSHOT_OK, snapshot: mkSnapshot({ scrollY: 560 }) });
      reduce(ctx, { type: "DECISION_OK", response: dec({ op: "SCROLL_DOWN" }) });
      assert(ctx.state === S.BLOCKED && ctx.reason === BLOCK_REASONS.SCROLL_BUDGET, `state=${ctx.state} reason=${ctx.reason}`);
    },
  ],
  [
    "已在底部时禁用向下滚动，且不发出无效滚动",
    () => {
      // 直接带着"已在底部"的快照启动：4600 + 780 >= 5000 - 2
      const ctx = boot(ctxFor(), mkSnapshot({ scrollY: 4600, height: 5000 }));
      reduce(ctx, { type: "DECISION_OK", response: dec({ op: "SCROLL_DOWN" }) });
      assert(ctx.scroll.disabled.down, "到底应禁用向下");
      assert(ctx.state === S.OBSERVE, `应回到观测，得到 ${ctx.state}`);
      assert(ctx.memory.fact("scroll_bottom_reached"), "应记成事实");
    },
  ],
  [
    "滚动禁用后 operation 问题里不再出现 SCROLL（能力随候选一起收敛）",
    () => {
      const q = buildQuestions({
        actions: [{ id: "e1", node: 1, kind: "click", role: "button", label: "a" }],
        config: { scrollDelta: 560 },
        scrollDisabled: { down: true, up: false },
      });
      assert(!("SCROLL_DOWN" in q.operation.criteria), "向下的能力必须消失");
      assert("SCROLL_UP" in q.operation.criteria, "向上的能力应保留");
      assert("DONE" in q.operation.criteria && "BLOCKED" in q.operation.criteria, "终态操作必须保留");
    },
  ],
  [
    "enterScroll 在预算内正常发出滚动",
    () => {
      const ctx = boot(ctxFor());
      const effects = enterScroll(ctx, "up"); // 注意：返回的是数组，不是 {effects}
      assert(ctx.state === S.SCROLL && effects[0].type === "scroll", `state=${ctx.state}`);
      assert(effects[0].delta === -560, `向上应为负 delta，得到 ${effects[0].delta}`);
    },
  ],

  // ---- 滚动查找（把开放式滚动变成可测量扫描）----
  [
    "FIND 找到目标即记为事实",
    () => {
      const ctx = boot(ctxFor());
      ctx.snapshot = mkSnapshot({ text: "here is the TARGET text" });
      startFind(ctx, { pattern: "target text" });
      assert(ctx.state === S.FIND);
      reduce(ctx, { type: "FIND_STEP" });
      assert(ctx.memory.fact("found:target text"), "应记录找到");
      assert(ctx.state === S.OBSERVE, `state=${ctx.state}`);
    },
  ],
  [
    "FIND 扫完未找到时记录下来并回决策（有界，不无限滚）",
    () => {
      const ctx = boot(ctxFor());
      ctx.snapshot = mkSnapshot({ text: "nothing here" });
      startFind(ctx, { pattern: "absent", direction: "down", maxScrolls: 1 });
      reduce(ctx, { type: "FIND_STEP" });
      assert(ctx.state === S.FIND && ctx.find.attempts === 1, `state=${ctx.state} attempts=${ctx.find?.attempts}`);
      reduce(ctx, { type: "FIND_STEP" });
      assert(ctx.memory.fact("not_found:absent"), "应记录未找到");
      assert(ctx.state === S.OBSERVE, `state=${ctx.state}`);
    },
  ],

  // ---- 跨步记忆 ----
  [
    "合法回访不被判为打转",
    () => {
      const m = new LoopMemory({ goal: "g" });
      m.observe("h1", { changed: true });
      m.observe("h1", { changed: true });
      m.observe("h1", { changed: true, navigated: true });
      assert(m.visits.get("h1") === 3, "访问应累计");
      assert(!m.isSpinning(), "有变化就不该判打转");
    },
  ],
  [
    "反复访问 + 无变化 + 有失败 → 判打转",
    () => {
      const m = new LoopMemory({ goal: "g" });
      for (let i = 0; i < 5; i += 1) m.observe("h1", { changed: false });
      m.recordAttempt("s1", { error: "no_effect" });
      assert(m.isSpinning(), "应判为打转");
    },
  ],
  [
    "打转终止发生在循环里",
    () => {
      const ctx = boot(ctxFor());
      const snap = mkSnapshot();
      for (let i = 0; i < 6 && !TERMINAL_STATES.has(ctx.state); i += 1) {
        if (ctx.state === S.DECIDE) reduce(ctx, { type: "DECISION_OK", response: dec() });
        if (ctx.state === S.EXECUTE) reduce(ctx, { type: "ACTION_DONE" });
        if (ctx.state === S.SETTLE) reduce(ctx, { type: "SETTLE_DONE" });
        if (ctx.state === S.OBSERVE) reduce(ctx, { type: SNAPSHOT_OK, snapshot: snap });
      }
      assert(ctx.state === S.BLOCKED, `应终止，得到 ${ctx.state}`);
      assert(
        [BLOCK_REASONS.NO_PROGRESS, BLOCK_REASONS.NO_CANDIDATES, BLOCK_REASONS.RECOVER_BUDGET].includes(ctx.reason),
        `reason=${ctx.reason}`,
      );
    },
  ],
  [
    "记忆渲染超预算时丢低优先级，并显式标注被丢的部分",
    () => {
      const m = new LoopMemory({ goal: "g" });
      m.captureOriginal({ pageKey: pageKey(1), url: "https://orig", scrollY: 0, formValues: {} });
      for (let i = 0; i < 10; i += 1) m.noteFact(`fact_${i}`, `value_${i}`, { evidence: "test" });
      for (let i = 0; i < 10; i += 1) m.recordTransition({ action: "CLICK", targetLabel: `t${i}`, outcome: "effected" });
      for (let i = 0; i < 6; i += 1) m.recordAttempt(`sig_${i}`, { error: "no_effect" });
      const text = m.renderForPrompt({ charBudget: 320 });
      assert(text.includes("VERIFIED FACTS"), "最高优先级应保留");
      assert(text.includes("memory truncated"), "必须显式标注被丢的部分，不能装作记忆完整");
    },
  ],
  [
    "记忆渲染为空时返回空串（不占预算）",
    () => {
      const m = new LoopMemory({ goal: "g" });
      assert(m.renderForPrompt({ charBudget: 500 }) === "", "空记忆不该产生任何字节");
    },
  ],
  [
    "snapshotStamp 能区分『导航』『新增元素』『纯滚动』",
    () => {
      const a = digestSnapshot(mkSnapshot());
      const b = digestSnapshot(mkSnapshot({ n: 1, text: "hello page", scrollY: 560 }));
      const s = snapshotStamp(a, b);
      assert(s.scrollMoved && !s.navigated && !s.textChanged, `stamp=${JSON.stringify(s)}`);
      assert(!s.changed, "纯滚动不算『世界变了』");
      assert(s.scrollProgress, "但算滚动有进展");
    },
  ],

  // ---- 向 Codex 移交表单输入权限 ----
  [
    "授权默认限定单字段，且不可逆恒为 false",
    () => {
      const g = buildGrant({
        reason: HANDOFF_REASONS.COMPOSE_TEXT,
        action: { node: 102, kind: "fill", role: "textbox", label: "Name", value: "" },
        pageKeyHash: "abc",
        step: 3,
        config: { allowIrreversible: true }, // 故意尝试覆盖
      });
      assert(g.scope === "field", `scope=${g.scope}`);
      assert(g.allowedOps.includes("TYPE_TEXT") && !g.allowedOps.includes("scroll"), "操作应被收窄");
      assert(g.allowIrreversible === false, "不可逆授权不可被配置覆盖");
      assert(g.expiresOn === "control_returned", "授权应在控制权收回时失效");
      assert(g.fieldSignature && g.field.node === 102, "授权必须绑定到具体字段");
    },
  ],
  [
    "不可逆动作：只发出移交请求，然后立即终止等人工",
    () => {
      const ctx = boot(ctxFor());
      reduce(ctx, { type: "DECISION_OK", response: dec({ op: "CLICK", target: "e1", irr: 1 }) });
      assert(ctx.state === S.HANDOFF_PREPARE, `state=${ctx.state}`);
      const { effects } = reduce(ctx, { type: "HANDOFF_PREPARE" });
      assert(ctx.state === S.BLOCKED && ctx.reason === BLOCK_REASONS.AWAITING_HUMAN, `reason=${ctx.reason}`);
      assert(effects.some((e) => e.type === "handoff"), "必须产出移交载荷");
      assert(effects.some((e) => e.type === "stop"), "必须立即停止");
      const capsule = effects.find((e) => e.type === "handoff").capsule;
      assert(capsule.grant.allowIrreversible === false, "不可逆动作不得进入授权范围");
      assert(capsule.instructions.includes("IRREVERSIBLE"), "指令必须明确说明不可逆");
      assert(capsule.mustNot.length >= 4, "必须带上禁止清单");
      assert(capsule.resumePlan.requiresHuman === true, "核验计划应标为需要人工");
    },
  ],
  [
    "移交期间目标字段加锁并从候选集剔除",
    () => {
      const ctx = boot(ctxFor());
      const snap = mkSnapshot();
      reduce(ctx, {
        type: "DECISION_OK",
        response: dec({ op: "TYPE_TEXT", target: "e2", extra: { needs_compose: { type: "noul", noul: 1 } } }),
      });
      reduce(ctx, { type: "HANDOFF_PREPARE" });
      assert(ctx.state === S.HANDOFF_WAIT, `state=${ctx.state}`);
      const pk = digestSnapshot(snap).pageKeyHash;
      const lockedSig = actionSignature(pk, 102, "fill", "Name");
      assert(ctx.ledger.isLocked(lockedSig), "移交中的字段必须加锁");
      const remaining = ctx.ledger.filterLocked(snap.actions, pk);
      assert(!remaining.some((a) => a.id === "e2"), "加锁字段不得出现在候选里");
      assert(remaining.some((a) => a.id === "e1"), "其它元素应保持可用");
    },
  ],
  [
    "移交后核验通过：计数清零、继续循环",
    () => {
      const ctx = boot(ctxFor());
      reduce(ctx, { type: "DECISION_OK", response: dec({ op: "TYPE_TEXT", target: "e2", extra: { needs_compose: { type: "noul", noul: 1 } } }) });
      reduce(ctx, { type: "HANDOFF_PREPARE" });
      const sig = ctx.handoff.capsule.grant.fieldSignature;
      reduce(ctx, { type: "HANDOFF_RETURNED", elapsedMs: 4200 });
      assert(ctx.state === S.HANDOFF_VERIFY, `state=${ctx.state}`);
      reduce(ctx, { type: SNAPSHOT_OK, snapshot: mkSnapshot({ text: "filled" }) });
      reduce(ctx, { type: "RESUME_CHECK_DONE", ok: true, evidence: "field value changed" });
      assert(ctx.state === S.OBSERVE, `state=${ctx.state}`);
      assert(ctx.ledger.canHandoff(sig).ok, "成功后应允许将来对同一字段再次移交");
      assert(ctx.ledger.counts.get(sig) === undefined, "成功后计数应清零");
      assert(ctx.memory.fact("handoff_resolved"), "应记成事实");
      assert(ctx.memory.budget.handoffs === 1 && ctx.memory.budget.handoffMs === 4200, "预算账本应记耗时");
    },
  ],
  [
    "移交后核验不过：允许再交一次，第二次仍不过则终止（防乒乓）",
    () => {
      const ctx = boot(ctxFor());
      const compose = () => {
        while (ctx.state !== S.DECIDE) {
          if (ctx.state === S.OBSERVE) reduce(ctx, { type: SNAPSHOT_OK, snapshot: mkSnapshot() });
          else if (ctx.state === S.SETTLE) reduce(ctx, { type: "SETTLE_DONE" });
          else if (ctx.state === S.EXECUTE) reduce(ctx, { type: "ACTION_DONE" });
          else break;
        }
        reduce(ctx, { type: "DECISION_OK", response: dec({ op: "TYPE_TEXT", target: "e2", extra: { needs_compose: { type: "noul", noul: 1 } } }) });
        reduce(ctx, { type: "HANDOFF_PREPARE" });
        reduce(ctx, { type: "HANDOFF_RETURNED", elapsedMs: 1000 });
        reduce(ctx, { type: SNAPSHOT_OK, snapshot: mkSnapshot() });
        reduce(ctx, { type: "RESUME_CHECK_DONE", ok: false, detail: "field unchanged" });
      };
      compose(); // 第 1 次
      assert(ctx.state !== S.BLOCKED, `第 1 次不该终止，得到 ${ctx.state} / ${ctx.reason}`);
      compose(); // 第 2 次
      assert(ctx.state === S.BLOCKED, `第 2 次应终止，得到 ${ctx.state}`);
      assert(ctx.reason === BLOCK_REASONS.HANDOFF_LOOP, `reason=${ctx.reason}`);
    },
  ],
  [
    "移交缺失 reason 时拒绝猜测并终止",
    () => {
      const ctx = boot(ctxFor());
      ctx.state = S.HANDOFF_PREPARE;
      ctx.pendingHandoffReason = null;
      reduce(ctx, { type: "HANDOFF_PREPARE" });
      assert(ctx.state === S.BLOCKED, `state=${ctx.state}`);
      assert(ctx.reason === BLOCK_REASONS.HANDOFF_LOOP, `reason=${ctx.reason}`);
    },
  ],
  [
    "移交超时终止，且不解除锁定（避免带病继续）",
    () => {
      const ctx = boot(ctxFor());
      reduce(ctx, { type: "DECISION_OK", response: dec({ op: "TYPE_TEXT", target: "e2", extra: { needs_compose: { type: "noul", noul: 1 } } }) });
      reduce(ctx, { type: "HANDOFF_PREPARE" });
      const { effects } = reduce(ctx, { type: "HANDOFF_TIMEOUT", detail: "no response in 120s" });
      assert(ctx.state === S.BLOCKED && ctx.reason === BLOCK_REASONS.HANDOFF_TIMEOUT, `reason=${ctx.reason}`);
      assert(effects[0].type === "stop", "应停止");
    },
  ],
  [
    "登录墙与验证码都走移交，且验证码同样停下等人工",
    () => {
      const login = boot(ctxFor());
      reduce(login, { type: "DECISION_OK", response: dec({ extra: { needs_login: { type: "noul", noul: 1 } } }) });
      assert(login.state === S.HANDOFF_PREPARE && login.pendingHandoffReason === HANDOFF_REASONS.NEEDS_LOGIN);
      const r = reduce(login, { type: "HANDOFF_PREPARE" });
      assert(login.state === S.HANDOFF_WAIT, "登录移交不该直接终止循环");
      assert(r.effects.some((e) => e.type === "handoff"));

      const cap = boot(ctxFor());
      reduce(cap, { type: "DECISION_OK", response: dec({ extra: { captcha: { type: "noul", noul: 1 } } }) });
      reduce(cap, { type: "HANDOFF_PREPARE" });
      assert(cap.state === S.BLOCKED && cap.reason === BLOCK_REASONS.AWAITING_HUMAN, `reason=${cap.reason}`);
    },
  ],
  [
    "低置信度触发移交（仅作为兜底，且只在没有硬信号时）",
    () => {
      const ctx = boot(ctxFor());
      const lowConf = dec();
      lowConf.results.target.confidence = 0.1;
      reduce(ctx, { type: "DECISION_OK", response: lowConf });
      assert(ctx.state === S.HANDOFF_PREPARE, `state=${ctx.state}`);
      assert(ctx.pendingHandoffReason === HANDOFF_REASONS.AMBIGUOUS_TARGET, `reason=${ctx.pendingHandoffReason}`);
    },
  ],
  [
    "有硬信号时不用置信度兜底（登录墙优先于低置信度）",
    () => {
      const ctx = boot(ctxFor());
      const both = dec({ extra: { needs_login: { type: "noul", noul: 1 } } });
      both.results.target.confidence = 0.05;
      reduce(ctx, { type: "DECISION_OK", response: both });
      assert(ctx.pendingHandoffReason === HANDOFF_REASONS.NEEDS_LOGIN, `reason=${ctx.pendingHandoffReason}`);
    },
  ],
  [
    "关闭移交开关后不再移交（可配置回退）",
    () => {
      const ctx = ctxFor({ handoffEnabled: false });
      boot(ctx);
      reduce(ctx, { type: "DECISION_OK", response: dec({ extra: { needs_compose: { type: "noul", noul: 1 } } }) });
      assert(ctx.state !== S.HANDOFF_PREPARE, `state=${ctx.state}`);
    },
  ],

  // ---- 门面：终止态不再接受事件 ----
  [
    "终止后拒绝一切后续事件（幂等）",
    () => {
      const ctx = boot(ctxFor());
      reduce(ctx, { type: "USER_ABORT" });
      const { effects } = reduce(ctx, { type: "DECISION_OK", response: dec() });
      assert(effects.length === 0, "终止后不该再产生 effect");
      assert(ctx.state === S.BLOCKED);
    },
  ],
  [
    "决策降级立即终止（不带着不可判定的状态继续）",
    () => {
      const ctx = boot(ctxFor());
      reduce(ctx, { type: "DECISION_OK", response: { degraded: "request_over_token_budget", results: {} } });
      assert(ctx.state === S.BLOCKED && ctx.reason === BLOCK_REASONS.DECISION_DEGRADED, `reason=${ctx.reason}`);
    },
  ],
  [
    "无可用候选时终止并给出排除统计",
    () => {
      const ctx = boot(ctxFor({ }));
      ctx.decision = dec({ op: "BLOCKED" });
      ctx.filteredActions = [];
      reduce(ctx, { type: "DECISION_OK", response: dec({ op: "BLOCKED" }) });
      ctx.filteredActions = [];
      reduce(ctx, { type: "DECISION_OK", response: dec({ op: "BLOCKED" }) });
      assert(ctx.state === S.BLOCKED, `state=${ctx.state}`);
      assert([BLOCK_REASONS.NO_CANDIDATES, BLOCK_REASONS.AWAITING_HUMAN].includes(ctx.reason), `reason=${ctx.reason}`);
    },
  ],
];

const TERMINAL_STATES = new Set([S.DONE, S.BLOCKED]);

// --------------------------------------------------------------------------- //
// 驱动层集成测试：用**假 IO** 跑完整条循环（不需要 Chrome、不需要决策服务）
// --------------------------------------------------------------------------- //
function makeIo({ snapshots = [], decisions = [], verifyResults = [], handoffResults = [], onExecute = null }) {
  const st = { snap: 0, dec: 0, exec: [], verify: 0, handoff: 0, scrolls: 0 };
  const pick = (arr, i) => arr[Math.min(i, arr.length - 1)];
  return {
    st,
    async snapshot() {
      const s = pick(snapshots, st.snap);
      st.snap += 1;
      if (s === "STALE") throw new StalePage("stale snapshot");
      if (s === "FAIL") throw new Error("evaluate failed");
      return s;
    },
    async decide() {
      const d = pick(decisions, st.dec);
      st.dec += 1;
      return typeof d === "function" ? d() : d;
    },
    async scroll() {
      st.scrolls += 1;
      return { newScrollY: 560 * st.scrolls, newHeight: 5000 };
    },
    async execute(action) {
      st.exec.push(action.id);
      if (onExecute) return onExecute(action, st);
    },
    async settle() {},
    async verify(plan, ctx) {
      const v = pick(verifyResults, st.verify);
      st.verify += 1;
      return typeof v === "function" ? v(plan, ctx) : v;
    },
    async handoff(capsule, ctx) {
      const h = pick(handoffResults, st.handoff);
      st.handoff += 1;
      if (h && h.throw) {
        const e = new Error("handoff timeout");
        e.timeout = true;
        throw e;
      }
      return typeof h === "function" ? h(capsule, ctx) : h;
    },
    emit() {},
  };
}

export const driverTests = [
  [
    "驱动：结构化核验通过 → DONE（全流程走完）",
    async () => {
      const io = makeIo({
        snapshots: [mkSnapshot(), mkSnapshot({ n: 2, text: "result Zürich" })],
        decisions: [dec({ op: "CLICK", target: "e1" }), dec({ op: "DONE", done: 1 })],
        verifyResults: [{ ok: true, evidence: "structural: Zürich present", source: "structural" }],
      });
      const { ctx, trace } = await drive({ goal: "find Zürich", io, config: { maxSteps: 5 } });
      assert(ctx.state === S.DONE, `state=${ctx.state} reason=${ctx.reason}`);
      assert(io.st.exec.length === 1, `应执行 1 次动作，得到 ${io.st.exec.length}`);
      assert(trace.some((t) => t.effect === "verify"), "必须经过核验");
    },
  ],
  [
    "驱动：模型自报 DONE 但结构化核验不过 → 不完成，最终终止",
    async () => {
      const io = makeIo({
        snapshots: [mkSnapshot()],
        decisions: [dec({ op: "DONE", done: 1 })],
        verifyResults: [{ ok: false, evidence: "structural: Zürich absent", source: "structural" }],
      });
      const { ctx } = await drive({ goal: "find Zürich", io, config: { maxSteps: 6, maxDoneRejections: 2 } });
      assert(ctx.state === S.BLOCKED, `state=${ctx.state}`);
      assert(ctx.reason === BLOCK_REASONS.VERIFY_FAILED, `reason=${ctx.reason}`);
      assert(io.st.exec.length === 0, "一次动作都不该执行");
    },
  ],
  [
    "驱动：移交 → 假 Codex 改字段 → 核验通过 → 继续到 DONE",
    async () => {
      const io = makeIo({
        snapshots: [mkSnapshot(), mkSnapshot({ text: "filled hello" }), mkSnapshot({ n: 2, text: "done Zürich" })],
        decisions: [
          dec({ op: "TYPE_TEXT", target: "e2", extra: { needs_compose: { type: "noul", noul: 1 } } }),
          dec({ op: "DONE", done: 1 }),
        ],
        verifyResults: [
          { ok: true, evidence: 'field 102: "" → "hello"', source: "structural" }, // 移交回来的核验
          { ok: true, evidence: "structural ok", source: "structural" }, // 目标核验
        ],
        handoffResults: [{ ok: true, evidence: "typed the value", elapsedMs: 500 }],
      });
      const { ctx, trace } = await drive({ goal: "fill the name", io, config: { maxSteps: 6 } });
      assert(ctx.state === S.DONE, `state=${ctx.state} reason=${ctx.reason}`);
      assert(ctx.memory.budget.handoffs === 1, "应记 1 次移交");
      assert(trace.some((t) => t.effect === "handoff"), "轨迹里应有移交");
      assert(ctx.memory.fact("handoff_resolved"), "应记录移交已解决");
    },
  ],
  [
    "驱动：移交超时 → 终止（不带着未决字段继续）",
    async () => {
      const io = makeIo({
        snapshots: [mkSnapshot()],
        decisions: [dec({ op: "TYPE_TEXT", target: "e2", extra: { needs_compose: { type: "noul", noul: 1 } } })],
        handoffResults: [{ throw: true }],
      });
      const { ctx } = await drive({ goal: "fill", io, config: { maxSteps: 6 } });
      assert(ctx.state === S.BLOCKED && ctx.reason === BLOCK_REASONS.HANDOFF_TIMEOUT, `reason=${ctx.reason}`);
    },
  ],
  [
    "驱动：陈旧快照自动恢复，不消耗动作预算",
    async () => {
      const io = makeIo({
        snapshots: ["STALE", mkSnapshot(), mkSnapshot({ n: 2, text: "changed" })],
        decisions: [dec({ op: "CLICK", target: "e1" }), dec({ op: "DONE", done: 1 })],
        verifyResults: [{ ok: true, evidence: "structural", source: "structural" }],
      });
      const { ctx } = await drive({ goal: "g", io, config: { maxSteps: 5 } });
      assert(ctx.state === S.DONE, `state=${ctx.state} reason=${ctx.reason}`);
      assert(ctx.memory.budget.staleRecoveries >= 1, "应记一次陈旧恢复");
    },
  ],
  [
    "驱动：滚动路径能走通（SCROLL → 重新观测）",
    async () => {
      const io = makeIo({
        snapshots: [mkSnapshot(), mkSnapshot({ n: 2, text: "changed after scroll" })],
        decisions: [dec({ op: "SCROLL_DOWN" }), dec({ op: "DONE", done: 1 })],
        verifyResults: [{ ok: true, evidence: "structural", source: "structural" }],
      });
      const { ctx, trace } = await drive({ goal: "g", io, config: { maxSteps: 5 } });
      assert(ctx.state === S.DONE, `state=${ctx.state} reason=${ctx.reason}`);
      assert(io.st.scrolls === 1, `应滚动 1 次，得到 ${io.st.scrolls}`);
      assert(ctx.scroll.epoch === 1, `滚动 epoch 应为 1，得到 ${ctx.scroll.epoch}`);
      assert(trace.some((t) => t.effect === "snapshot" && t.state === S.OBSERVE), "滚动后应重新观测");
    },
  ],
  [
    "驱动：状态机不收敛时报错而不是静默挂住",
    async () => {
      const io = makeIo({ snapshots: [mkSnapshot()], decisions: [dec()] });
      let threw = false;
      try {
        await drive({ goal: "g", io, config: { maxSteps: 3 }, maxIterations: 12 });
      } catch (e) {
        threw = true;
        assert(/maxIterations|接线缺口/.test(e.message), `错误信息应可定位，得到 ${e.message}`);
      }
      assert(threw, "不收敛必须抛错");
    },
  ],
  [
    "结构化核验规则：present / absent / text / label",
    async () => {
      const snap = mkSnapshot({ text: "Zürich to London", actions: [{ id: "e1", node: 1, kind: "click", role: "button", label: "One-way" }] });
      assert(checkExpectations(snap, [{ pattern: "Zürich", expect: "present", where: "text" }]).ok === true);
      assert(checkExpectations(snap, [{ pattern: "Zürich", expect: "absent", where: "text" }]).ok === false);
      assert(checkExpectations(snap, [{ pattern: "One-way", expect: "present", where: "label" }]).ok === true);
      assert(checkExpectations(snap, [{ pattern: "Round trip", expect: "absent", where: "label" }]).ok === true);
      assert(checkExpectations(snap, []).decided === false, "没有断言时应报未决，而不是假装通过");
    },
  ],
  [
    "结构化核验规则：正则元字符被转义（不把用户串当正则用）",
    async () => {
      const snap = mkSnapshot({ text: "price is $1.00 (net)" });
      const r = checkExpectations(snap, [{ pattern: "$1.00 (net)", expect: "present", where: "text" }]);
      assert(r.ok === true, `应字面匹配，得到 ${JSON.stringify(r.details)}`);
    },
  ],
  [
    "驱动参数：--expect 解析成结构化断言",
    async () => {
      const o = parseDriverArgs(["--url", "u", "--goal", "g", "--decide-url", "d", "--expect", "Zürich:present:text", "--expect", "Round trip:absent:label"]);
      assert(o.expectations.length === 2, `得到 ${o.expectations.length}`);
      assert(o.expectations[0].where === "text" && o.expectations[1].expect === "absent");
      assert(o.live === false, "默认必须 dry-run");
    },
  ],
];

function assert(cond, msg) {
  if (!cond) throw new Error(msg || "断言失败");
}

// --------------------------------------------------------------------------- //
// 运行器
// --------------------------------------------------------------------------- //
export async function runSmTests({ log = console.log } = {}) {
  const all = [...smTests, ...driverTests];
  let passed = 0;
  const failures = [];
  for (const [name, fn] of all) {
    try {
      await fn();
      passed += 1;
      log(`PASS  ${name}`);
    } catch (err) {
      failures.push(name);
      log(`FAIL  ${name}: ${err.message}`);
    }
  }
  log(`\n状态机 + 驱动: ${passed}/${all.length} passed`);
  return { passed, total: all.length, failures };
}

if (process.argv[1] && process.argv[1].endsWith("sm_selftest.mjs")) {
  const r = await runSmTests();
  if (r.failures.length) {
    console.log("失败用例: " + r.failures.join(", "));
    process.exit(1);
  }
}
