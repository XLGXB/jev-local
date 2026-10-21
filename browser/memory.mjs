/* 跨步状态记忆：把"上一步做了什么、页面怎么回应、哪些坑已经踩过"变成可渲染、可预算的确定性数据。
 *
 * 为什么需要它：快速点击循环每轮都会重新观测页面，模型本身**没有**跨轮记忆。
 * 如果只把最近几条动作文本塞进 state，会出现三类典型失败：
 *
 *   1. **原地打转**：同一处点两次都无效，模型照点第三次；
 *   2. **重复探索**：已经展开过的面板、已经滚动过的位置，重新走一遍；
 *   3. **交接失忆**：把输入权交给 Codex 再收回来之后，不知道之前填了什么、
 *      原始状态是什么，于是"恢复原状"这类要求做不到。
 *
 * 本模块只做数据与渲染，不做决策；模块级零依赖（除 node:crypto），可在无浏览器环境单测。
 */

import { createHash } from "node:crypto";

export function sha256(text) {
  return createHash("sha256").update(String(text)).digest("hex");
}

/** 动作签名：**必须包含页面身份**，否则"同一个按钮在不同页面"会被当成同一个动作。 */
export function actionSignature(pageKeyHash, nodeId, kind, label) {
  const norm = String(label || "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 120);
  return `${pageKeyHash.slice(0, 12)}|${nodeId}|${kind}|${norm}`;
}

export const DEFAULT_MEMORY_LIMITS = {
  maxTransitions: 40,
  maxAttempts: 120,
  maxVisits: 60,
  maxFacts: 60,
  maxSuppressed: 40,
  failureThreshold: 2, // 同一动作失败多少次后从候选里剔除
};

export class LoopMemory {
  constructor({ goal = "", limits = {}, original = null } = {}) {
    this.goal = goal;
    this.limits = { ...DEFAULT_MEMORY_LIMITS, ...limits };

    /** pageKeyHash → 访问次数（用于打转检测） */
    this.visits = new Map();
    /** pageKeyHash → 上一次访问时"世界变了没有"的标记 */
    this.visitStamps = new Map();
    /** 动作签名 → {count, lastError, lastStep} */
    this.attempts = new Map();
    /** 动作签名 → 被剔除的步骤号（避免无界增长） */
    this.suppressed = new Map();
    /** 键 → {value, step, evidence} 已核实的既有事实 */
    this.facts = new Map();
    /** 环形缓冲：最近的状态转移 */
    this.transitions = [];
    /** 原始状态快照，用于"做完恢复原状" */
    this.original = original;
    /** 预算账本 */
    this.budget = { steps: 0, scrolls: 0, handoffs: 0, handoffMs: 0, staleRecoveries: 0 };
    this.step = 0;
    /** 自上次"世界变化"以来的轮数：循环检测的核心量 */
    this.barrenRounds = 0;
  }

  // ------------------------------------------------------------------ //
  // 页面身份与访问计数
  // ------------------------------------------------------------------ //
  pageKeyHash(pageKey) {
    return sha256(JSON.stringify(pageKey));
  }

  /**
   * 记录一次观测。
   * @param stamp 本轮"世界是否变化"的证据：新元素数、文本是否变、是否导航。
   *              用它区分"合法回访"（例如修好后再回来）与"原地打转"。
   */
  observe(pageKeyHash, stamp = {}) {
    this.step += 1;
    this.budget.steps += 1;
    const n = (this.visits.get(pageKeyHash) || 0) + 1;
    this.visits.set(pageKeyHash, n);
    this.visitStamps.set(pageKeyHash, { ...stamp, step: this.step });

    const changed = Boolean(stamp.navigated || stamp.newElements > 0 || stamp.textChanged);
    this.barrenRounds = changed ? 0 : this.barrenRounds + 1;

    this.#trim(this.visits, this.limits.maxVisits, (k) => (this.visits.get(k) || 0) > 1);
    this.#trim(this.visits, this.limits.maxVisits);
    return { visitCount: n, changed, barrenRounds: this.barrenRounds };
  }

  /**
   * 打转判定：同一页面被反复访问 **且** 一直没有世界变化。
   *
   * 只看"访问次数"会误伤合法回访（修好之后回到原页、来回比较两个视图）。
   * 所以条件是二者同时成立，并且要求已经有人试过动作却都失败了。
   */
  isSpinning() {
    const maxVisit = Math.max(0, ...this.visits.values());
    return this.barrenRounds >= 4 && maxVisit >= 2 && this.failingSignatures().length > 0;
  }

  // ------------------------------------------------------------------ //
  // 失败计数与候选剔除
  // ------------------------------------------------------------------ //
  recordAttempt(sig, { step = this.step, error = null } = {}) {
    const rec = this.attempts.get(sig) || { count: 0, lastError: null, lastStep: 0 };
    rec.count += 1;
    rec.lastError = error;
    rec.lastStep = step;
    this.attempts.set(sig, rec);
    this.#trim(this.attempts, this.limits.maxAttempts, (k) => k === sig);
    if (rec.count >= this.limits.failureThreshold) this.suppress(sig, step);
    return rec;
  }

  clearAttempts(sig) {
    this.attempts.delete(sig);
    this.suppressed.delete(sig);
  }

  suppress(sig, step = this.step) {
    this.suppressed.set(sig, step);
    this.#trim(this.suppressed, this.limits.maxSuppressed, (k) => k === sig);
  }

  isSuppressed(sig) {
    return this.suppressed.has(sig);
  }

  suppressedSignatures() {
    return [...this.suppressed.keys()];
  }

  failingSignatures() {
    return [...this.attempts.entries()].filter(([, r]) => r.count > 0).map(([k]) => k);
  }

  /**
   * 按签名过滤候选。**改的是候选集而不是提示词**——提示词说"别再点它"模型仍然可能点，
   * 直接从候选里拿掉才是确定的。
   */
  filterCandidates(actions, pageKeyHash) {
    const kept = [];
    const removed = [];
    for (const a of actions) {
      const sig = actionSignature(pageKeyHash, a.node, a.kind, a.label);
      if (this.isSuppressed(sig)) removed.push({ action: a, sig });
      else kept.push(a);
    }
    return { kept, removed };
  }

  // ------------------------------------------------------------------ //
  // 已核实事实
  // ------------------------------------------------------------------ //
  noteFact(key, value, { evidence = "", step = this.step } = {}) {
    // 同一键的新值与旧值不同 → 记为变更（用于最终校验与"是否恢复原状"）
    const prev = this.facts.get(key);
    this.facts.set(key, {
      value,
      evidence,
      step,
      previous: prev && prev.value !== value ? prev.value : prev?.previous ?? null,
    });
    this.#trim(this.facts, this.limits.maxFacts);
    return this.facts.get(key);
  }

  fact(key) {
    return this.facts.get(key)?.value ?? null;
  }

  verifiedFacts() {
    return [...this.facts.entries()].filter(([, v]) => v.evidence).map(([k, v]) => ({ key: k, ...v }));
  }

  // ------------------------------------------------------------------ //
  // 状态转移（模型看不到"页面怎么变"，只看到"我做了什么、结果如何"）
  // ------------------------------------------------------------------ //
  recordTransition({ action, targetLabel, outcome, detail = "" }) {
    this.transitions.push({
      step: this.step,
      action,
      target: String(targetLabel || "").slice(0, 80),
      outcome, // executed | stale | failed | handoff | scroll
      detail: String(detail).slice(0, 120),
    });
    this.#trimTransitions();
  }

  recentTransitions(n = 6) {
    return this.transitions.slice(-n);
  }

  // ------------------------------------------------------------------ //
  // 原始状态
  // ------------------------------------------------------------------ //
  captureOriginal({ pageKey, url, scrollY, formValues = {}, guards = null }) {
    this.original = {
      pageKey,
      pageKeyHash: this.pageKeyHash(pageKey),
      url,
      scrollY,
      formValues,
      guards,
      step: this.step,
    };
    return this.original;
  }

  /** 与原始状态的差异清单，用于"做完恢复原状"这类要求。 */
  diffFromOriginal(current = {}) {
    if (!this.original) return { hasOriginal: false, items: [] };
    const items = [];
    if (current.url && current.url !== this.original.url) {
      items.push({ kind: "url", from: this.original.url, to: current.url });
    }
    if (typeof current.scrollY === "number" && current.scrollY !== this.original.scrollY) {
      items.push({ kind: "scroll", from: this.original.scrollY, to: current.scrollY });
    }
    for (const [k, v] of Object.entries(current.formValues || {})) {
      const was = this.original.formValues?.[k];
      if (was !== undefined && was !== v) items.push({ kind: "field", key: k, from: was, to: v });
    }
    return { hasOriginal: true, items };
  }

  // ------------------------------------------------------------------ //
  // 渲染进 state（**有预算、按优先级裁剪**）
  // ------------------------------------------------------------------ //
  /**
   * 记忆渲染。优先级固定，超预算时**从低优先级开始丢**：
   *   目标 > 已核实事实 > 最近状态转移 > 失败与剔除 > 原始状态提示
   * 低优先级的先丢，且丢的内容要显式标注，不能让模型以为记忆是完整的。
   */
  renderForPrompt({ charBudget = 1400 } = {}) {
    const blocks = [];

    if (this.verifiedFacts().length) {
      blocks.push({
        priority: 2,
        title: "VERIFIED FACTS",
        lines: this.verifiedFacts().map((f) => `- ${f.key} = ${fmt(f.value)}${f.evidence ? ` (${f.evidence})` : ""}`),
      });
    }
    if (this.transitions.length) {
      blocks.push({
        priority: 3,
        title: "RECENT STEPS",
        lines: this.recentTransitions(6).map(
          (t) => `- ${t.action} ${t.target} → ${t.outcome}${t.detail ? ` (${t.detail})` : ""}`,
        ),
      });
    }
    const failing = this.failingSignatures();
    if (failing.length) {
      blocks.push({
        priority: 4,
        title: "ALREADY TRIED (do not repeat blindly)",
        lines: failing.slice(-8).map((sig) => {
          const r = this.attempts.get(sig);
          const mark = this.isSuppressed(sig) ? " [excluded from candidates]" : "";
          return `- ${sig.split("|").slice(2).join(" ")} ×${r.count}${mark}`;
        }),
      });
    }
    const diff = this.diffFromOriginal();
    if (diff.hasOriginal && diff.items.length) {
      blocks.push({
        priority: 5,
        title: "DIFFERENT FROM ORIGINAL STATE",
        lines: diff.items.map((i) => `- ${i.kind}${i.key ? `:${i.key}` : ""} ${fmt(i.from)} → ${fmt(i.to)}`),
      });
    }

    blocks.sort((a, b) => a.priority - b.priority);
    const out = [];
    let used = 0;
    const dropped = [];
    for (const b of blocks) {
      const text = `${b.title}\n${b.lines.join("\n")}`;
      if (used + text.length <= charBudget) {
        out.push(text);
        used += text.length + 1;
      } else {
        dropped.push(b.title);
      }
    }
    if (dropped.length) out.push(`[memory truncated: ${dropped.join(", ")} omitted]`);
    return out.join("\n");
  }

  stats() {
    return {
      step: this.step,
      visits: this.visits.size,
      attempts: this.attempts.size,
      suppressed: this.suppressed.size,
      facts: this.facts.size,
      transitions: this.transitions.length,
      barrenRounds: this.barrenRounds,
      budget: { ...this.budget },
    };
  }

  // ------------------------------------------------------------------ //
  #trimTransitions() {
    const over = this.transitions.length - this.limits.maxTransitions;
    if (over > 0) this.transitions.splice(0, over);
  }

  #trim(map, max, keepKey = null) {
    if (map.size <= max) return;
    // Map 保持插入顺序；从最早的开始删，但保留 keepKey（通常是刚写入的那个）
    const keys = [...map.keys()];
    for (const k of keys) {
      if (map.size <= max) break;
      if (keepKey && k === keepKey) continue;
      map.delete(k);
    }
  }
}

function fmt(v) {
  if (v === null || v === undefined) return "(none)";
  const s = typeof v === "string" ? v : JSON.stringify(v);
  return s.length > 60 ? s.slice(0, 60) + "…" : s;
}
