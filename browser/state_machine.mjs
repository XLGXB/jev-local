/* 快速点击循环的状态机 —— **纯 reducer**：(ctx, event) → { ctx, effects }。
 *
 * 为什么做成纯函数：状态机的 bug 几乎全是"某个分支没被走到"或"顺序反了"，
 * 而这类 bug 在真浏览器上极难复现（页面每次都不一样）。纯 reducer 可以用构造事件
 * 序列的方式把每条分支都跑一遍——`selftest.mjs` 里就是这么测的。
 * 解释器（loop_driver.mjs）只负责把 effects 变成真实的 CDP 调用。
 *
 * 三个被要求的能力各自落在哪里：
 *
 *   • **滚动处理** → `SCROLL` 状态 + `scrollState`（预算 / 停滞 / 往复 / 到底）
 *                    + `FIND` 状态（"滚动去找某段文字"这种开放式目标变成可测量扫描）
 *   • **跨步状态记忆** → `memory.mjs` 的 LoopMemory，在 OBSERVE 时更新，
 *                        在 DECIDE 时按优先级渲染进 state，并**改候选集**
 *   • **向 Codex 移交表单输入权限** → `HANDOFF_PREPARE / HANDOFF_WAIT / HANDOFF_VERIFY`
 *                        三态 + `handoff.mjs` 的范围化授权、加锁与核验
 *
 * 三条不可让步的不变量（都有对应测试）：
 *   I1. 模型报 DONE **不算数**：必须进 VERIFY 并拿到独立证据才能进 DONE。
 *   I2. 滚动之后**必须重新观测**（滚动会改变几何与 page_key）。
 *   I3. 移交期间目标动作**加锁**，收回后**必须核验**，核验不过不得继续交同一个字段。
 */

import { LoopMemory, actionSignature, sha256 } from "./memory.mjs";
import {
  DEFAULT_TARGET_CONFIDENCE_FLOOR,
  HANDOFF_REASONS,
  HandoffLedger,
  buildCapsule,
  buildGrant,
  buildResumePlan,
  detectHandoff,
} from "./handoff.mjs";

export const S = {
  INIT: "INIT",
  OBSERVE: "OBSERVE",
  DECIDE: "DECIDE",
  SCROLL: "SCROLL",
  FIND: "FIND",
  EXECUTE: "EXECUTE",
  SETTLE: "SETTLE",
  VERIFY: "VERIFY",
  HANDOFF_PREPARE: "HANDOFF_PREPARE",
  HANDOFF_WAIT: "HANDOFF_WAIT",
  HANDOFF_VERIFY: "HANDOFF_VERIFY",
  RECOVER: "RECOVER",
  DONE: "DONE",
  BLOCKED: "BLOCKED",
};

export const TERMINAL = new Set([S.DONE, S.BLOCKED]);

/** 快照成功事件名。导出成常量，避免解释器与 reducer 各写一份字符串。 */
export const SNAPSHOT_OK = "SNAPSHOT_OK";

export const BLOCK_REASONS = {
  STEP_BUDGET: "step_budget_exhausted",
  SCROLL_BUDGET: "scroll_budget_exhausted",
  SCROLL_OSCILLATING: "scroll_oscillating",
  SCROLL_STALLED: "scroll_stalled",
  NO_PROGRESS: "no_progress_spinning",
  OBSERVE_FAILED: "observe_failed",
  DECISION_DEGRADED: "decision_degraded",
  RECOVER_BUDGET: "recover_budget_exhausted",
  HANDOFF_LOOP: "handoff_not_effective",
  HANDOFF_TIMEOUT: "handoff_timeout",
  AWAITING_HUMAN: "awaiting_human_confirmation",
  VERIFY_FAILED: "verification_failed",
  NO_CANDIDATES: "no_usable_candidates",
  USER_ABORT: "user_abort",
};

export const DEFAULT_CONFIG = {
  maxSteps: 12,
  maxRecoveries: 6,
  maxScrolls: 8,
  maxScrollStalls: 2,
  maxDirectionReversals: 3,
  maxFindScrolls: 6,
  maxDoneRejections: 2,
  targetConfidenceFloor: DEFAULT_TARGET_CONFIDENCE_FLOOR,
  stateCharBudget: 6000,
  memoryCharBudget: 1400,
  scrollDelta: 560,
  handoffEnabled: true,
};

// --------------------------------------------------------------------------- //
// 上下文
// --------------------------------------------------------------------------- //
export function createContext({ goal, url = "", config = {}, memory = null } = {}) {
  return {
    state: S.INIT,
    goal,
    url,
    config: { ...DEFAULT_CONFIG, ...config },

    snapshot: null,
    lastDigest: null,
    /** 上一次执行的动作；用于"只观测、按效果判定成败" */
    pendingAction: null,
    pendingActionSignature: null,
    /** 模型决策结果 */
    decision: null,
    filteredActions: [],
    excludedCounts: { suppressed: 0, locked: 0, capability: 0 },

    scroll: {
      epoch: 0,
      count: 0,
      lastY: 0,
      lastHeight: 0,
      stalls: 0,
      directions: [],
      disabled: { down: false, up: false },
    },
    find: null,

    recoveries: 0,
    doneRejections: 0,
    spins: 0,

    memory: memory || new LoopMemory({ goal }),
    ledger: new HandoffLedger(),
    handoff: null,
    awaitingHuman: null,

    terminal: null,
    reason: null,
    facts: [],
    log: [],
  };
}

// --------------------------------------------------------------------------- //
// 快照摘要与"世界变了没有"
// --------------------------------------------------------------------------- //
export function digestSnapshot(snapshot) {
  return {
    url: snapshot.url,
    title: snapshot.title,
    scrollY: snapshot.scroll?.y ?? 0,
    pageHeight: snapshot.scroll?.height ?? 0,
    textHash: sha256(String(snapshot.text || "").slice(0, 6000)),
    textChars: (snapshot.text || "").length,
    actionCount: (snapshot.actions || []).length,
    elementIds: (snapshot.actions || []).map((a) => a.node),
    pageKeyHash: sha256(JSON.stringify(snapshot.page_key || [])),
    omittedActions: snapshot.omitted_actions || 0,
  };
}

/**
 * 用两份摘要算"世界变了没有"。
 * 这是**零额外往返的效果检测**：动作有没有起作用，看下一次观测的 diff 就知道，
 * 不需要为每个动作单独问一次模型。
 */
export function snapshotStamp(prev, cur) {
  if (!prev) return { first: true, changed: true, navigated: true, newElements: 0, textChanged: true, scrollMoved: false };
  const prevIds = new Set(prev.elementIds);
  const newElements = cur.elementIds.filter((id) => !prevIds.has(id)).length;
  const navigated = prev.pageKeyHash !== cur.pageKeyHash || prev.url !== cur.url;
  const textChanged = prev.textHash !== cur.textHash;
  const scrollMoved = prev.scrollY !== cur.scrollY;
  const grew = cur.pageHeight > prev.pageHeight;
  return {
    first: false,
    navigated,
    newElements,
    textChanged,
    scrollMoved,
    grew,
    changed: navigated || newElements > 0 || textChanged,
    scrollProgress: scrollMoved || grew,
  };
}

// --------------------------------------------------------------------------- //
// 问题构造（全部问题在**同一次前向**里问完）
// --------------------------------------------------------------------------- //
export function buildQuestions({ actions, config, scrollDisabled = { down: false, up: false } }) {
  const ops = {
    CLICK: "click one observed element",
    TYPE_TEXT: "type a value into one observed field",
    SELECT: "choose one option of a native dropdown",
  };
  if (!scrollDisabled.down) ops.SCROLL_DOWN = "scroll down to reveal more of the page";
  if (!scrollDisabled.up) ops.SCROLL_UP = "scroll up";
  ops.WAIT = "wait for the page to update";
  ops.DONE = "the goal is already satisfied on this page";
  ops.BLOCKED = "nothing available can make progress";

  const ids = actions.filter((a) => a.kind !== "wait").map((a) => a.id);
  if (ids.length > 255) throw new Error(`候选数 ${ids.length} 超过 System One 上限 255`);

  return {
    operation: { type: "choice", instructions: "Which single operation should be executed next to make progress toward the GOAL?", criteria: ops },
    target: { type: "choice", instructions: "Which observed element should that operation act on?", criteria: Object.fromEntries(ids.map((id) => [id, null])) },
    done: { type: "noul", instructions: "Does the page already show that the GOAL is satisfied?" },
    irreversible: { type: "noul", instructions: "Would that operation be irreversible (payment, order, delete, send, publish, authorize, upload)?" },
    needs_login: { type: "noul", instructions: "Is a sign-in wall blocking progress?" },
    needs_credentials: { type: "noul", instructions: "Is a credential (password, one-time code, API key) required that must not be handled here?" },
    captcha: { type: "noul", instructions: "Is a captcha or access-denied wall present?" },
    needs_vision: { type: "noul", instructions: "Does this step require looking at an image or visual layout rather than the element table?" },
    unsupported: { type: "noul", instructions: "Is the needed control outside the observed element table (canvas, iframe, shadow DOM, file picker, rich editor, date picker)?" },
    needs_compose: { type: "noul", instructions: "Does the value to enter have to be composed rather than copied from the state?" },
  };
}

export function buildStateText({ goal, snapshot, memory, config, historyLine = "" }) {
  const lines = [`GOAL: ${goal}`, `URL: ${snapshot.url}`];
  if (historyLine) lines.push(`DONE SO FAR: ${historyLine}`);
  const mem = memory?.renderForPrompt ? memory.renderForPrompt({ charBudget: config.memoryCharBudget }) : "";
  if (mem) lines.push("", mem);
  lines.push("", "PAGE TEXT (visible only):", String(snapshot.text || "").slice(0, config.stateCharBudget));
  lines.push("", "INTERACTIVE ELEMENTS:", (snapshot.table || []).join("\n"));
  if (snapshot.omitted_actions) {
    // 截断必须可见，否则模型会反复找它看不见的控件
    lines.push(`(only the first 250 elements are listed; ${snapshot.omitted_actions} more were omitted)`);
  }
  return lines.join("\n");
}

// --------------------------------------------------------------------------- //
// 内部辅助
// --------------------------------------------------------------------------- //
function sig(ctx, action) {
  if (!action || !ctx.snapshot) return null;
  const pageKeyHash = sha256(JSON.stringify(ctx.snapshot.page_key || []));
  return actionSignature(pageKeyHash, action.node, action.kind, action.label);
}

function block(ctx, reason, detail = "") {
  ctx.state = S.BLOCKED;
  ctx.reason = reason;
  ctx.terminal = { state: S.BLOCKED, reason, detail, step: ctx.memory.step };
  return [{ type: "stop", terminal: ctx.terminal }];
}

function goto(ctx, state, effects = []) {
  ctx.state = state;
  return effects;
}

function pickAction(ctx) {
  const choice = ctx.decision?.results?.target?.choice;
  if (!choice) return null;
  return (ctx.snapshot.actions || []).find((a) => a.id === choice) || null;
}

/** 候选集过滤：剔除已失效的（被记忆剔除 / 移交加锁 / 能力被禁用）。 */
function filterCandidates(ctx) {
  const pageKeyHash = sha256(JSON.stringify(ctx.snapshot.page_key || []));
  const all = ctx.snapshot.actions || [];
  const { kept: afterSuppressed, removed } = ctx.memory.filterCandidates(all, pageKeyHash);
  const afterLocked = ctx.ledger.filterLocked(afterSuppressed, pageKeyHash);
  const finalKept = afterLocked.filter((a) => {
    if (a.id === "scroll_down" && ctx.scroll.disabled.down) return false;
    if (a.id === "scroll_up" && ctx.scroll.disabled.up) return false;
    return true;
  });
  ctx.excludedCounts = {
    suppressed: removed.length,
    locked: afterSuppressed.length - afterLocked.length,
    capability: afterLocked.length - finalKept.length,
  };
  ctx.filteredActions = finalKept;
  return finalKept;
}

function decideEffect(ctx) {
  const questions = buildQuestions({
    actions: ctx.filteredActions,
    config: ctx.config,
    scrollDisabled: ctx.scroll.disabled,
  });
  const state = buildStateText({ goal: ctx.goal, snapshot: ctx.snapshot, memory: ctx.memory, config: ctx.config });
  return [{ type: "decide", state, questions, excluded: { ...ctx.excludedCounts } }];
}

function enterObserve(ctx) {
  if (ctx.memory.budget.steps >= ctx.config.maxSteps) {
    return block(ctx, BLOCK_REASONS.STEP_BUDGET, `used ${ctx.memory.budget.steps}/${ctx.config.maxSteps}`);
  }
  return goto(ctx, S.OBSERVE, [{ type: "snapshot" }]);
}

// --------------------------------------------------------------------------- //
// reducer
// --------------------------------------------------------------------------- //
export function reduce(ctx, event) {
  const cfg = ctx.config;

  if (event.type === "USER_ABORT" && !TERMINAL.has(ctx.state)) {
    return { ctx, effects: block(ctx, BLOCK_REASONS.USER_ABORT) };
  }
  if (TERMINAL.has(ctx.state)) return { ctx, effects: [] };

  switch (ctx.state) {
    // ------------------------------------------------------------------ //
    case S.INIT: {
      if (event.type !== "START") return { ctx, effects: [] };
      return { ctx, effects: enterObserve(ctx) };
    }

    // ------------------------------------------------------------------ //
    case S.OBSERVE: {
      if (event.type === "SNAPSHOT_FAILED") {
        ctx.recoveries += 1;
        if (ctx.recoveries > cfg.maxRecoveries) return { ctx, effects: block(ctx, BLOCK_REASONS.OBSERVE_FAILED, event.error) };
        return { ctx, effects: enterObserve(ctx) };
      }
      if (event.type === "SNAPSHOT_STALE") {
        ctx.recoveries += 1;
        ctx.memory.budget.staleRecoveries += 1;
        if (ctx.recoveries > cfg.maxRecoveries) return { ctx, effects: block(ctx, BLOCK_REASONS.OBSERVE_FAILED, "too many stale observations") };
        return { ctx, effects: enterObserve(ctx) };
      }
      if (event.type !== "SNAPSHOT_OK") return { ctx, effects: [] };

      const snapshot = event.snapshot;
      if (!snapshot) {
        ctx.recoveries += 1;
        return { ctx, effects: enterObserve(ctx) };
      }
      ctx.snapshot = snapshot;
      const cur = digestSnapshot(snapshot);
      const stamp = snapshotStamp(ctx.lastDigest, cur);
      ctx.lastDigest = cur;

      // I2 的配套：滚动是"观测上下文变更"，必须重新采样几何（本循环每轮都重新快照，天然满足）
      const observed = ctx.memory.observe(cur.pageKeyHash, stamp);

      // --- 动作效果检测（零额外往返）---
      if (ctx.pendingAction && !stamp.first) {
        const changed = stamp.changed;
        ctx.memory.recordTransition({
          action: ctx.pendingAction.kind,
          targetLabel: ctx.pendingAction.label,
          outcome: changed ? "effected" : "no_effect",
          detail: stamp.navigated ? "navigated" : `+${stamp.newElements} elements`,
        });
        if (!changed && !stamp.scrollProgress) {
          // "点了没反应" 就在这里被记成失败 —— 而不是等模型第三次点同一个地方
          ctx.memory.recordAttempt(ctx.pendingActionSignature, { error: "no_effect" });
        } else {
          ctx.memory.clearAttempts(ctx.pendingActionSignature);
        }
        ctx.pendingAction = null;
        ctx.pendingActionSignature = null;
      }

      // --- 打转检测 ---
      if (ctx.memory.isSpinning()) {
        ctx.spins += 1;
        return { ctx, effects: block(ctx, BLOCK_REASONS.NO_PROGRESS, `barren rounds=${observed.barrenRounds}`) };
      }

      filterCandidates(ctx);
      return { ctx, effects: goto(ctx, S.DECIDE, decideEffect(ctx)) };
    }

    // ------------------------------------------------------------------ //
    case S.DECIDE: {
      // 只认三种事件。**其余事件一律忽略**（只记录），不做状态迁移——
      // 否则一个来接线错误的杂散事件会静默把状态推到别处，而调用方看到的只是
      // "后面的决策好像没生效"。这类"静默错位"是状态机里最难查的一类 bug。
      if (event.type !== "DECISION_OK" && event.type !== "DECISION_FAILED" && event.type !== "DECISION_DEGRADED") {
        return { ctx, effects: [{ type: "emit", event: { kind: "unexpected_event", inState: S.DECIDE, got: event.type } }] };
      }
      if (event.type !== "DECISION_OK") {
        ctx.recoveries += 1;
        if (ctx.recoveries > cfg.maxRecoveries) return { ctx, effects: block(ctx, BLOCK_REASONS.DECISION_DEGRADED, event.error) };
        return { ctx, effects: enterObserve(ctx) };
      }
      const resp = event.response || {};
      if (resp.degraded || !resp.results?.operation || !resp.results?.target) {
        return { ctx, effects: block(ctx, BLOCK_REASONS.DECISION_DEGRADED, resp.degraded || "missing operation/target") };
      }
      ctx.decision = resp;

      // --- 1) 移交判定（硬信号优先）---
      const detection = detectHandoff({
        results: resp.results,
        action: pickAction(ctx),
        config: { targetConfidenceFloor: cfg.targetConfidenceFloor },
      });
      if (detection && cfg.handoffEnabled) {
        ctx.pendingHandoffReason = detection.reason;
        return {
          ctx,
          effects: goto(ctx, S.HANDOFF_PREPARE, [
            { type: "emit", event: { kind: "handoff_needed", ...detection } },
            // 进入 HANDOFF_PREPARE 后需要一条入场事件，否则机器会停在这里发不出 effect。
            // 显式写出而不是让解释器猜（猜的后果就是静默挂住）。
            { type: "advance", event: { type: S.HANDOFF_PREPARE } },
          ]),
        };
      }

      // --- 2) 模型自报 DONE：进 VERIFY，**不直接接受** ---
      const op = resp.results.operation.choice;
      const doneP = resp.results.done?.noul ?? 0;
      if (op === "DONE" || doneP > 0.8) {
        return {
          ctx,
          effects: goto(ctx, S.VERIFY, [
            {
              type: "verify",
              plan: { question: "goal_satisfied", expect: true, describe: "GOAL 必须由页面证据独立确认，不能采信模型自报" },
            },
          ]),
        };
      }

      // --- 3) 无候选可用 ---
      if (op === "BLOCKED") {
        if (ctx.filteredActions.length === 0) {
          return { ctx, effects: block(ctx, BLOCK_REASONS.NO_CANDIDATES, `excluded=${JSON.stringify(ctx.excludedCounts)}`) };
        }
        ctx.pendingHandoffReason = HANDOFF_REASONS.UNKNOWN;
        return {
          ctx,
          effects: goto(ctx, S.HANDOFF_PREPARE, [
            { type: "emit", event: { kind: "handoff_needed", reason: HANDOFF_REASONS.UNKNOWN, from: "blocked" } },
            { type: "advance", event: { type: S.HANDOFF_PREPARE } },
          ]),
        };
      }

      // --- 4) 滚动 ---
      if (op === "SCROLL_DOWN" || op === "SCROLL_UP") {
        const dir = op === "SCROLL_DOWN" ? "down" : "up";
        return { ctx, effects: enterScroll(ctx, dir) };
      }

      // --- 5) 普通执行 ---
      const action = pickAction(ctx);
      if (!action) {
        ctx.recoveries += 1;
        if (ctx.recoveries > cfg.maxRecoveries) {
          return { ctx, effects: block(ctx, BLOCK_REASONS.RECOVER_BUDGET, `target ${resp.results.target.choice} not in element table`) };
        }
        return { ctx, effects: enterObserve(ctx) };
      }
      const s = sig(ctx, action);
      ctx.pendingAction = action;
      ctx.pendingActionSignature = s;
      return {
        ctx,
        effects: goto(ctx, S.EXECUTE, [
          { type: "execute", action, text: action.kind === "fill" ? action.value || "" : null, signature: s },
        ]),
      };
    }

    // ------------------------------------------------------------------ //
    case S.EXECUTE: {
      if (event.type === "ACTION_DONE") {
        return { ctx, effects: goto(ctx, S.SETTLE, [{ type: "settle", action: ctx.pendingAction }]) };
      }
      if (event.type === "ACTION_STALE") {
        ctx.memory.recordAttempt(ctx.pendingActionSignature, { error: "stale" });
        ctx.memory.budget.staleRecoveries += 1;
        ctx.pendingAction = null;
        ctx.pendingActionSignature = null;
        ctx.recoveries += 1;
        if (ctx.recoveries > cfg.maxRecoveries) return { ctx, effects: block(ctx, BLOCK_REASONS.RECOVER_BUDGET, "too many stale actions") };
        return { ctx, effects: enterObserve(ctx) };
      }
      if (event.type === "ACTION_FAILED") {
        ctx.memory.recordAttempt(ctx.pendingActionSignature, { error: String(event.error || "failed") });
        ctx.memory.recordTransition({
          action: ctx.pendingAction?.kind || "?",
          targetLabel: ctx.pendingAction?.label || "",
          outcome: "failed",
          detail: String(event.error || ""),
        });
        ctx.pendingAction = null;
        ctx.pendingActionSignature = null;
        ctx.recoveries += 1;
        if (ctx.recoveries > cfg.maxRecoveries) return { ctx, effects: block(ctx, BLOCK_REASONS.RECOVER_BUDGET, String(event.error || "")) };
        return { ctx, effects: enterObserve(ctx) };
      }
      return { ctx, effects: [] };
    }

    // ------------------------------------------------------------------ //
    case S.SETTLE: {
      if (event.type !== "SETTLE_DONE") return { ctx, effects: [] };
      // 动作执行完总要重新观测：既确认效果，也让"没反应"能被记成失败
      return { ctx, effects: enterObserve(ctx) };
    }

    // ------------------------------------------------------------------ //
    case S.VERIFY: {
      if (event.type !== "VERIFY_DONE") return { ctx, effects: [] };
      if (event.ok) {
        ctx.state = S.DONE;
        ctx.terminal = {
          state: S.DONE,
          step: ctx.memory.step,
          evidence: event.evidence || null,
          originalDiff: ctx.memory.diffFromOriginal({
            url: ctx.snapshot?.url,
            scrollY: ctx.snapshot?.scroll?.y,
          }),
        };
        return { ctx, effects: [{ type: "stop", terminal: ctx.terminal }] };
      }
      // I1 的另一半：核不过就不能当完成
      ctx.doneRejections += 1;
      if (ctx.doneRejections >= cfg.maxDoneRejections) {
        return { ctx, effects: block(ctx, BLOCK_REASONS.VERIFY_FAILED, `rejected ${ctx.doneRejections} DONE claims`) };
      }
      ctx.recoveries += 1;
      return { ctx, effects: enterObserve(ctx) };
    }

    // ------------------------------------------------------------------ //
    case S.HANDOFF_PREPARE: {
      // reason 来自 DECIDE 时存下的判定结果（ctx.pendingHandoffReason）或本次事件。
      // 两者都没有 → 说明接线断了；这时**不能猜**一个理由交出去：
      // 猜错的范围化授权比不授权更危险（会把"只能填这一个字段"错判成"整页控制权"）。
      const reason = event.reason || ctx.pendingHandoffReason;
      if (!reason) {
        return { ctx, effects: block(ctx, BLOCK_REASONS.HANDOFF_LOOP, "handoff entered without a reason") };
      }
      ctx.pendingHandoffReason = null;

      const action = pickAction(ctx);
      const pageKeyHash = sha256(JSON.stringify(ctx.snapshot?.page_key || []));
      const grant = buildGrant({ reason, action, pageKeyHash, step: ctx.memory.step, config: cfg });

      const gate = ctx.ledger.canHandoff(grant.fieldSignature);
      if (!gate.ok) {
        return { ctx, effects: block(ctx, BLOCK_REASONS.HANDOFF_LOOP, `field handed off ${gate.attempts}× without effect`) };
      }

      const resumePlan = buildResumePlan({ reason, grant, action });
      const capsule = buildCapsule({ reason, goal: ctx.goal, grant, resumePlan, snapshot: ctx.snapshot, memory: ctx.memory, action });
      const pending = ctx.ledger.open(capsule);
      ctx.handoff = { capsule, resumePlan, attempt: pending.attempt };

      // 不可逆 / 验证码：交出去之后循环**必须停**，等人工决定
      if (reason === HANDOFF_REASONS.IRREVERSIBLE_ACTION || reason === HANDOFF_REASONS.CAPTCHA || resumePlan.requiresHuman) {
        ctx.awaitingHuman = { reason, capsule };
        const effects = [{ type: "handoff", capsule }, { type: "emit", event: { kind: "awaiting_human", reason } }];
        ctx.state = S.BLOCKED;
        ctx.reason = BLOCK_REASONS.AWAITING_HUMAN;
        ctx.terminal = { state: S.BLOCKED, reason: BLOCK_REASONS.AWAITING_HUMAN, capsule };
        effects.push({ type: "stop", terminal: ctx.terminal });
        return { ctx, effects };
      }

      ctx.memory.recordTransition({
        action: "HANDOFF",
        targetLabel: grant.field?.label || "(page)",
        outcome: "handoff",
        detail: reason,
      });
      return { ctx, effects: goto(ctx, S.HANDOFF_WAIT, [{ type: "handoff", capsule }]) };
    }

    // ------------------------------------------------------------------ //
    case S.HANDOFF_WAIT: {
      if (event.type === "HANDOFF_TIMEOUT") {
        ctx.ledger.close({ verified: false });
        ctx.handoff = null;
        return { ctx, effects: block(ctx, BLOCK_REASONS.HANDOFF_TIMEOUT, String(event.detail || "")) };
      }
      if (event.type !== "HANDOFF_RETURNED") return { ctx, effects: [] };
      ctx.memory.budget.handoffs += 1;
      if (typeof event.elapsedMs === "number") ctx.memory.budget.handoffMs += event.elapsedMs;
      // I3 的前半：收回后必须先重新观测，再核验
      return { ctx, effects: goto(ctx, S.HANDOFF_VERIFY, [{ type: "snapshot" }]) };
    }

    // ------------------------------------------------------------------ //
    case S.HANDOFF_VERIFY: {
      if (event.type === "SNAPSHOT_STALE" || event.type === "SNAPSHOT_FAILED") {
        ctx.recoveries += 1;
        if (ctx.recoveries > cfg.maxRecoveries) return { ctx, effects: block(ctx, BLOCK_REASONS.OBSERVE_FAILED) };
        return { ctx, effects: [{ type: "snapshot" }] };
      }
      if (event.type === SNAPSHOT_OK) {
        ctx.snapshot = event.snapshot;
        const cur = digestSnapshot(event.snapshot);
        const stamp = snapshotStamp(ctx.lastDigest, cur);
        ctx.lastDigest = cur;
        ctx.memory.observe(cur.pageKeyHash, stamp);
        filterCandidates(ctx);
        return { ctx, effects: goto(ctx, S.HANDOFF_VERIFY, [{ type: "verify", plan: ctx.handoff.resumePlan }]) };
      }
      if (event.type !== "RESUME_CHECK_DONE") return { ctx, effects: [] };

      const attempt = ctx.handoff?.attempt ?? 1;
      const describe = ctx.handoff?.resumePlan?.describe || "control returned";
      const closed = ctx.ledger.close({ verified: Boolean(event.ok) });
      ctx.handoff = null;

      if (event.ok) {
        ctx.memory.noteFact("handoff_resolved", describe, { evidence: `attempt ${attempt}` });
        return { ctx, effects: enterObserve(ctx) };
      }

      ctx.memory.noteFact("handoff_failed", `attempt ${attempt}: ${event.detail || "no change"}`, { evidence: "verify" });
      if (attempt >= ctx.ledger.maxPerField || !closed.ok) {
        return { ctx, effects: block(ctx, BLOCK_REASONS.HANDOFF_LOOP, `field still unchanged after ${attempt} handoffs`) };
      }
      ctx.recoveries += 1;
      if (ctx.recoveries > cfg.maxRecoveries) return { ctx, effects: block(ctx, BLOCK_REASONS.RECOVER_BUDGET) };
      return { ctx, effects: enterObserve(ctx) };
    }

    // ------------------------------------------------------------------ //
    case S.SCROLL: {
      if (event.type !== "SCROLL_DONE") {
        if (event.type === "SCROLL_FAILED") {
          ctx.recoveries += 1;
          if (ctx.recoveries > cfg.maxRecoveries) return { ctx, effects: block(ctx, BLOCK_REASONS.RECOVER_BUDGET, String(event.error || "")) };
          return { ctx, effects: enterObserve(ctx) };
        }
        return { ctx, effects: [] };
      }

      const { newScrollY, newHeight } = event;
      const sc = ctx.scroll;
      const moved = newScrollY !== sc.lastY;
      const grew = newHeight > sc.lastHeight;
      sc.lastY = newScrollY;
      sc.lastHeight = newHeight;
      sc.epoch += 1;
      sc.count += 1;
      ctx.memory.budget.scrolls += 1;

      ctx.memory.recordTransition({
        action: "SCROLL",
        targetLabel: event.direction === "up" ? "scroll up" : "scroll down",
        outcome: "scroll",
        detail: moved ? `y=${newScrollY}` : grew ? `height=${newHeight} (lazy load)` : "no movement",
      });

      if (!moved && !grew) {
        sc.stalls += 1;
        if (sc.stalls >= cfg.maxScrollStalls) {
          sc.disabled[event.direction] = true;
          ctx.memory.noteFact(`scroll_${event.direction}_stalled`, `disabled after ${sc.stalls} stalls`, { evidence: "scroll" });
          ctx.scroll.stalls = 0;
        }
      } else {
        sc.stalls = 0;
        if (!moved && grew) {
          // 高度增长说明懒加载出了新内容，但这**不算滚动成功**：要把它记进事实里
          ctx.memory.noteFact("lazy_load_observed", `page height ${sc.lastHeight}`, { evidence: "scroll" });
        }
      }

      // 往复检测：方向交替太频繁说明在震荡
      sc.directions.push(event.direction);
      if (sc.directions.length > cfg.maxDirectionReversals * 2) sc.directions.shift();
      const tail = sc.directions.slice(-cfg.maxDirectionReversals);
      const alternating = tail.length === cfg.maxDirectionReversals && tail.every((d, i) => i === 0 || d !== tail[i - 1]);
      if (alternating) {
        return { ctx, effects: block(ctx, BLOCK_REASONS.SCROLL_OSCILLATING, tail.join(">")) };
      }
      if (ctx.scroll.disabled.down && ctx.scroll.disabled.up) {
        return { ctx, effects: block(ctx, BLOCK_REASONS.SCROLL_STALLED, "both directions disabled") };
      }

      // I2：滚动后必须重新观测
      return { ctx, effects: enterObserve(ctx) };
    }

    // ------------------------------------------------------------------ //
    case S.FIND: {
      if (event.type === "FIND_STEP") {
        const plan = ctx.find;
        const haystack = `${ctx.snapshot?.text || ""}\n${(ctx.snapshot?.actions || []).map((a) => a.label).join("\n")}`.toLowerCase();
        if (haystack.includes(plan.pattern.toLowerCase())) {
          ctx.memory.noteFact(`found:${plan.pattern}`, `at scrollY=${ctx.snapshot?.scroll?.y}`, { evidence: "find" });
          ctx.find = null;
          return { ctx, effects: enterObserve(ctx) };
        }
        if (plan.attempts >= plan.maxScrolls || ctx.scroll.disabled[plan.direction]) {
          ctx.memory.noteFact(`not_found:${plan.pattern}`, `scanned ${plan.attempts} screens`, { evidence: "find" });
          ctx.find = null;
          return { ctx, effects: enterObserve(ctx) };
        }
        plan.attempts += 1;
        return { ctx, effects: goto(ctx, S.FIND, [{ type: "scroll", direction: plan.direction, delta: cfg.scrollDelta }]) };
      }
      if (event.type === "SCROLL_DONE") {
        // FIND 复用滚动机制，但自己管计数
        ctx.scroll.epoch += 1;
        ctx.scroll.count += 1;
        ctx.scroll.lastY = event.newScrollY;
        ctx.scroll.lastHeight = event.newHeight;
        ctx.memory.budget.scrolls += 1;
        return { ctx, effects: [{ type: "snapshot" }] };
      }
      if (event.type === "SNAPSHOT_OK") {
        ctx.snapshot = event.snapshot;
        ctx.lastDigest = digestSnapshot(event.snapshot);
        return { ctx, effects: goto(ctx, S.FIND, [{ type: "emit", event: { kind: "find_step" } }]) };
      }
      return { ctx, effects: [] };
    }

    default:
      return { ctx, effects: [] };
  }
}

// --------------------------------------------------------------------------- //
// 进入 SCROLL 的守卫（预算 / 到底 / 已禁用）
// --------------------------------------------------------------------------- //
export function enterScroll(ctx, direction) {
  const cfg = ctx.config;
  const sc = ctx.scroll;

  if (ctx.memory.budget.scrolls >= cfg.maxScrolls) {
    return block(ctx, BLOCK_REASONS.SCROLL_BUDGET, `used ${ctx.memory.budget.scrolls}/${cfg.maxScrolls}`);
  }
  if (sc.disabled[direction]) {
    // 该方向已被判定无效：不要让循环空转，回到决策（候选集里该方向已被剔除）
    ctx.recoveries += 1;
    if (ctx.recoveries > cfg.maxRecoveries) return block(ctx, BLOCK_REASONS.SCROLL_STALLED, `${direction} disabled`);
    return enterObserve(ctx);
  }
  // 到底检测：向下滚但已经到底。
  // 这里**不**因为"看起来到底了"就永久禁用该方向——懒加载会让 height 增长，
  // 所以只在真正滚不动（SCROLL_DONE 里 moved=false && grew=false）时才禁用。
  const atBottom = (ctx.snapshot?.scroll?.y ?? 0) + (ctx.snapshot?.h ?? 0) >= (ctx.snapshot?.scroll?.height ?? 0) - 2;
  if (direction === "down" && atBottom) {
    const lastY = sc.lastY;
    const lastHeight = sc.lastHeight;
    sc.lastY = ctx.snapshot?.scroll?.y ?? lastY;
    sc.lastHeight = ctx.snapshot?.scroll?.height ?? lastHeight;
    sc.disabled.down = true;
    ctx.memory.noteFact("scroll_bottom_reached", `height=${sc.lastHeight}`, { evidence: "scroll" });
    return enterObserve(ctx);
  }
  const delta = direction === "up" ? -cfg.scrollDelta : cfg.scrollDelta;
  return goto(ctx, S.SCROLL, [{ type: "scroll", direction, delta }]);
}

/** 主动发起一次"滚动查找"（把开放式滚动变成可测量扫描）。 */
export function startFind(ctx, { pattern, direction = "down", maxScrolls = null }) {
  if (TERMINAL.has(ctx.state)) return { ctx, effects: [] };
  ctx.find = { pattern, direction, attempts: 0, maxScrolls: maxScrolls ?? ctx.config.maxFindScrolls };
  return {
    ctx,
    effects: goto(ctx, S.FIND, [
      { type: "emit", event: { kind: "find_start", pattern } },
      { type: "advance", event: { type: "FIND_STEP" } },
    ]),
  };
}
