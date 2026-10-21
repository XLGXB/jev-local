/* 表单输入权限的移交（Jev 点击 / Codex 输入与校验）。
 *
 * 依据（来源：wy-coliney/jev-browser-use 的公开设计）：
 *   "Jev handles the controls. Codex enters text, handles images, and checks the outcome.
 *    When a step needs help, Codex takes over and Jev resumes afterward."
 *   "Jev receives accessibility text, not screenshots. Codex handles typing, visual
 *    interpretation, and unsupported controls."
 *
 * 关键点不是"把整页交出去"，而是**范围化授权**：交出去的是
 * "这一个字段的这一次输入"，不是"这台浏览器的控制权"。所以每次移交都带三样东西：
 *
 *   1. **grant**    —— 允许做什么、允许在哪、什么时候失效；
 *   2. **capsule**  —— 恢复所需的一切（页面身份、滚动位置、待填字段、记忆摘要）；
 *   3. **resumePlan** —— 收回控制权后**用什么问题去核验**，而不是相信"我做完了"。
 *
 * 三条不可让步的规矩：
 *   • 移交期间该动作签名被**加锁**，循环不得再碰（否则和 Codex 抢同一个字段）；
 *   • 收回后必须核验，核验不过计入 handoff 失败；同字段连续失败 2 次即判 BLOCKED，
 *     防止"交出去—没收成—再交出去"的乒乓；
 *   • 不可逆动作**永远不在授权范围内**，只能作为请求人工确认的理由上报。
 */

import { actionSignature, sha256 } from "./memory.mjs";

export const HANDOFF_REASONS = {
  NEEDS_LOGIN: "needs_login",
  UNSUPPORTED_CONTROL: "unsupported_control",
  VISION_REQUIRED: "vision_required",
  COMPOSE_TEXT: "compose_text",
  AMBIGUOUS_TARGET: "ambiguous_target",
  IRREVERSIBLE_ACTION: "irreversible_action",
  CAPTCHA: "captcha",
  CREDENTIALS_REQUIRED: "credentials_required",
  UNKNOWN: "unknown",
};

/** 低置信度阈值。注意：**confidence 不是正确率**，见 spec.confidence 的说明。 */
export const DEFAULT_TARGET_CONFIDENCE_FLOOR = 0.35;
export const MAX_HANDOFFS_PER_FIELD = 2;

const SCOPE_BY_REASON = {
  [HANDOFF_REASONS.NEEDS_LOGIN]: "page",
  [HANDOFF_REASONS.CREDENTIALS_REQUIRED]: "field",
  [HANDOFF_REASONS.UNSUPPORTED_CONTROL]: "field",
  [HANDOFF_REASONS.VISION_REQUIRED]: "page",
  [HANDOFF_REASONS.COMPOSE_TEXT]: "field",
  [HANDOFF_REASONS.AMBIGUOUS_TARGET]: "field",
  [HANDOFF_REASONS.IRREVERSIBLE_ACTION]: "none",
  [HANDOFF_REASONS.CAPTCHA]: "page",
  [HANDOFF_REASONS.UNKNOWN]: "field",
};

const OPS_BY_SCOPE = {
  field: ["CLICK", "TYPE_TEXT", "SELECT"],
  form: ["CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "SCROLL_UP"],
  page: ["CLICK", "TYPE_TEXT", "SELECT", "SCROLL_DOWN", "SCROLL_UP", "WAIT"],
  none: [],
};

/**
 * 从一次决策里判定是否需要移交。**先看硬信号，再看置信度**。
 *
 * 这些信号来自与决策同一次前向里的 noul 问题（每个问题只占一个槽，几乎不增加延迟），
 * 所以移交判定不是启发式猜的，而是模型自己报的。
 */
export function detectHandoff({ results, action, config = {} } = {}) {
  const floor = config.targetConfidenceFloor ?? DEFAULT_TARGET_CONFIDENCE_FLOOR;
  const y = (qid) => (results?.[qid]?.noul ?? 0) > 0.5;

  if (y("needs_login")) return { reason: HANDOFF_REASONS.NEEDS_LOGIN, from: "signal" };
  if (y("needs_credentials")) return { reason: HANDOFF_REASONS.CREDENTIALS_REQUIRED, from: "signal" };
  if (y("captcha")) return { reason: HANDOFF_REASONS.CAPTCHA, from: "signal" };
  if (y("needs_vision")) return { reason: HANDOFF_REASONS.VISION_REQUIRED, from: "signal" };
  if (y("unsupported")) return { reason: HANDOFF_REASONS.UNSUPPORTED_CONTROL, from: "signal" };
  if (y("needs_compose")) return { reason: HANDOFF_REASONS.COMPOSE_TEXT, from: "signal" };
  if ((results?.irreversible?.noul ?? 0) > 0.5) {
    return { reason: HANDOFF_REASONS.IRREVERSIBLE_ACTION, from: "signal" };
  }
  // 置信度只作为**兜底**信号：它衡量"领先答案离均匀分布多远"，不是正确率，
  // 所以只在没有硬信号时才用它，且阈值放低。
  if (action && typeof results?.target?.confidence === "number" && results.target.confidence < floor) {
    return {
      reason: HANDOFF_REASONS.AMBIGUOUS_TARGET,
      from: "confidence",
      detail: `target confidence ${results.target.confidence.toFixed(3)} < ${floor}`,
    };
  }
  return null;
}

/** 构造范围化授权。 */
export function buildGrant({ reason, action, pageKeyHash, step, config = {} }) {
  const scope = SCOPE_BY_REASON[reason] ?? "field";
  const sig = action ? actionSignature(pageKeyHash, action.node, action.kind, action.label) : null;
  return {
    scope,
    reason,
    step,
    fieldSignature: sig,
    field: action
      ? { node: action.node, role: action.role, label: action.label, currentValue: action.value ?? null }
      : null,
    allowedOps: OPS_BY_SCOPE[scope],
    /** 不可逆动作永不在授权内 —— 这是常量，不接受覆盖 */
    allowIrreversible: false,
    /** 授权失效条件：控制权一收回即失效（不是"若干步之后"） */
    expiresOn: "control_returned",
    maxSteps: config.grantMaxSteps ?? 3,
  };
}

/** Codex 侧要照着做的指令块。写死语气，避免"你自己看着办"。 */
export function buildInstructions({ reason, grant, goal, action }) {
  const base = [
    `GOAL (unchanged, do not reinterpret): ${goal}`,
    `HANDOFF REASON: ${reason}`,
  ];
  if (grant.scope === "field" && grant.field) {
    base.push(
      `SCOPE: this ONE field only — node id ${grant.field.node}, role ${grant.field.role}, label "${grant.field.label}".`,
      `Current value: ${JSON.stringify(grant.field.currentValue)}`,
    );
  } else if (grant.scope === "page") {
    base.push("SCOPE: the current page only. Do not navigate away.");
  } else {
    base.push("SCOPE: none — this handoff is a request for human confirmation only.");
  }
  base.push(
    `ALLOWED: ${grant.allowedOps.join(", ") || "(none)"}`,
    `TOTAL STEPS ALLOWED BEFORE RETURNING: ${grant.maxSteps}`,
  );
  switch (reason) {
    case HANDOFF_REASONS.NEEDS_LOGIN:
    case HANDOFF_REASONS.CREDENTIALS_REQUIRED:
      base.push(
        "A sign-in wall blocks progress. Do NOT ask for credentials in chat.",
        "If you have an authorized session, use it; otherwise stop and report that credentials are needed.",
      );
      break;
    case HANDOFF_REASONS.COMPOSE_TEXT:
      base.push(
        "The value must be COMPOSED, not copied from the page. Write it yourself.",
        "Type it into the field above, then return control. Do not submit.",
      );
      break;
    case HANDOFF_REASONS.UNSUPPORTED_CONTROL:
      base.push(
        "The control is outside the observed element table (canvas / shadow DOM / iframe / file picker / rich editor / date picker).",
        "Operate it directly, then return control.",
      );
      break;
    case HANDOFF_REASONS.VISION_REQUIRED:
      base.push(
        "This step needs visual judgment, which the decision model cannot do (it never receives screenshots).",
        "Take a screenshot, decide, act, then return control.",
      );
      break;
    case HANDOFF_REASONS.AMBIGUOUS_TARGET:
      base.push(
        `The decision model was unsure which element to act on (confidence below the floor).`,
        "Pick the correct element yourself, act once, then return control.",
      );
      break;
    case HANDOFF_REASONS.IRREVERSIBLE_ACTION:
      base.push(
        "The next action looks IRREVERSIBLE. Nothing has been executed.",
        "Do NOT execute it. Ask the human for explicit confirmation and stop.",
      );
      break;
    case HANDOFF_REASONS.CAPTCHA:
      base.push("A captcha or access wall is present. Do not attempt to bypass it. Report and stop.");
      break;
    default:
      base.push("Explain what you did before returning control.");
  }
  return base.join("\n");
}

/** 收回控制权后的核验计划：**用问题去核，而不是相信回执**。 */
export function buildResumePlan({ reason, grant, action }) {
  switch (reason) {
    case HANDOFF_REASONS.COMPOSE_TEXT:
    case HANDOFF_REASONS.UNSUPPORTED_CONTROL:
    case HANDOFF_REASONS.AMBIGUOUS_TARGET:
      return {
        question: "did_field_change",
        expect: true,
        node: grant.field?.node ?? null,
        describe: "目标字段的值必须与移交前不同",
      };
    case HANDOFF_REASONS.NEEDS_LOGIN:
    case HANDOFF_REASONS.CREDENTIALS_REQUIRED:
      return {
        question: "signed_in",
        expect: true,
        describe: "登录墙必须消失",
      };
    case HANDOFF_REASONS.VISION_REQUIRED:
      return {
        question: "page_advanced",
        expect: true,
        describe: "页面必须发生与目标一致的变化（由 Codex 汇报 + 快照对照）",
      };
    case HANDOFF_REASONS.IRREVERSIBLE_ACTION:
    case HANDOFF_REASONS.CAPTCHA:
      return {
        question: "human_decision_recorded",
        expect: true,
        describe: "必须有人工决定被记录，否则循环终止",
        requiresHuman: true,
      };
    default:
      return { question: "page_advanced", expect: true, describe: "默认：页面必须有进展" };
  }
}

/** 完整移交载荷。Codex 拿到这一个对象就能接管，不需要再去看别的上下文。 */
export function buildCapsule({
  requestId = null,
  reason,
  goal,
  grant,
  resumePlan,
  snapshot,
  memory,
  action,
  mustNot = [],
}) {
  const pageKeyHash = snapshot?.page_key ? sha256(JSON.stringify(snapshot.page_key)) : "unknown";
  return {
    schemaVersion: 1,
    requestId: requestId || `ho-${Date.now().toString(36)}-${pageKeyHash.slice(0, 8)}`,
    reason,
    goal,
    pageKeyHash,
    url: snapshot?.url ?? null,
    title: snapshot?.title ?? null,
    scrollY: snapshot?.scroll?.y ?? null,
    pageHeight: snapshot?.scroll?.height ?? null,
    visibleText: (snapshot?.text ?? "").slice(0, 1200),
    elementTable: (snapshot?.table ?? []).slice(0, 120),
    target: action
      ? { node: action.node, role: action.role, label: action.label, kind: action.kind, value: action.value ?? null }
      : null,
    grant,
    instructions: buildInstructions({ reason, grant, goal, action }),
    mustNot: [
      "do not navigate away from this URL",
      "do not submit, pay, publish, send, delete, upload or authorize",
      "do not change browser account settings",
      "do not paste credentials into chat",
      ...mustNot,
    ],
    resumePlan,
    memoryDigest: memory?.renderForPrompt ? memory.renderForPrompt({ charBudget: 800 }) : "",
    memoryStats: memory?.stats ? memory.stats() : null,
  };
}

/**
 * 移交台账：加锁、计数、防乒乓。
 *
 * "同字段连续移交 2 次"是真实会发生的失败：Codex 说做完了、核验没过、循环又交出去。
 * 到第 2 次就停下并报 BLOCKED，比无限乒乓好。
 */
export class HandoffLedger {
  constructor({ maxPerField = MAX_HANDOFFS_PER_FIELD } = {}) {
    this.maxPerField = maxPerField;
    this.counts = new Map(); // fieldSignature → 次数
    this.locked = new Set(); // 移交期间被锁住的动作签名
    this.history = [];
    this.pending = null;
  }

  canHandoff(fieldSignature) {
    const n = this.counts.get(fieldSignature || "(none)") || 0;
    return { ok: n < this.maxPerField, attempts: n, limit: this.maxPerField };
  }

  open(capsule) {
    const sig = capsule.grant?.fieldSignature || "(none)";
    const n = (this.counts.get(sig) || 0) + 1;
    this.counts.set(sig, n);
    if (sig !== "(none)") this.locked.add(sig);
    this.pending = { capsule, fieldSignature: sig, attempt: n };
    this.history.push({ requestId: capsule.requestId, reason: capsule.reason, at: Date.now(), attempt: n });
    return this.pending;
  }

  /** 收回控制权：解锁，并根据核验结果决定是否还允许再交。 */
  close({ verified }) {
    const p = this.pending;
    this.pending = null;
    if (!p) return { ok: false, reason: "no_pending_handoff" };
    if (p.fieldSignature !== "(none)") this.locked.delete(p.fieldSignature);
    if (verified) this.counts.delete(p.fieldSignature); // 成功则清零，允许将来对同一字段再交（合法的二次输入）
    return { ok: true, verified, attempt: p.attempt };
  }

  isLocked(sig) {
    return this.locked.has(sig);
  }

  /** 把加锁签名从候选里拿掉——**同样是改候选集，不是靠提示词劝住模型**。 */
  filterLocked(actions, pageKeyHash) {
    return actions.filter(
      (a) => !this.isLocked(actionSignature(pageKeyHash, a.node, a.kind, a.label)),
    );
  }

  stats() {
    return { fields: this.counts.size, locked: this.locked.size, pending: Boolean(this.pending), history: this.history.length };
  }
}
