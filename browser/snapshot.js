/* 原子 DOM 快照：一次 Runtime.evaluate 拿到"可交互元素表 + 可见正文 + 新鲜度凭据"。
 *
 * 机制来源（设计参考，非逐行照搬）：browser-use/jev-ultrafast 的 snapshot.js
 * （该仓库许可证未核，故此处是独立实现，只借用其公开思路）。三个关键点：
 *
 *   1. **一次调用原子完成**。分多次查 DOM 会拿到互相不一致的快照——
 *      元素在两次调用之间被移除，就会出现"表里有的元素执行时不存在"。
 *   2. **只送可见文本**。离屏的正文与页脚不进模型 context。
 *      实测参考：一篇 Wikipedia 的完整快照约 149k token，只取视口内文本可压到几千。
 *   3. **对照"意义与身份"，几何在输入前现算**。marker/guard 里**不放坐标**，
 *      因为坐标会随滚动、动画、懒加载失效；执行前重新解析并做命中测试。
 *
 * 返回结构：
 *   { url, title, w, h, table[], text, scroll, actions[], marker, page_key, guards, omitted_actions }
 *
 * 安全边界：`password` / `file` / `hidden` 三类输入一律不进表（`safe()`）。
 */

(() => {
  if (!document.body) return null;

  const cache = (window.__jevFast ||= { ids: new WeakMap(), nodes: new Map(), next: 1 });

  const identity = (e) => {
    if (!cache.ids.has(e)) cache.ids.set(e, cache.next++);
    const id = cache.ids.get(e);
    cache.nodes.set(id, e);
    return id;
  };

  // 每次快照回收已断开的节点，否则长会话里 Map 会无限增长并持有已卸载的 DOM
  for (const [id, e] of cache.nodes) if (!e.isConnected) cache.nodes.delete(id);

  const safe = (e) => !["password", "file", "hidden"].includes(e.type);

  const visible = (e) =>
    !e.closest('[aria-hidden="true"],[inert]') &&
    e.checkVisibility({ checkOpacity: true, checkVisibilityCSS: true });

  // 近似可访问名：顺序对齐常见实现（labelledby → label → labels → value → alt → 子文本 → title → placeholder）
  const name = (e, seen = new Set()) => {
    if (!e || seen.has(e)) return "";
    seen.add(e);
    const referenced = (e.getAttribute("aria-labelledby") || "")
      .split(/\s+/)
      .map((id) => name(document.getElementById(id), seen))
      .filter(Boolean)
      .join(" ");
    return (
      referenced ||
      e.getAttribute("aria-label") ||
      [...(e.labels || [])].map((l) => name(l, seen)).filter(Boolean).join(" ") ||
      (["button", "submit", "reset"].includes(e.type) ? e.value : "") ||
      e.getAttribute("alt") ||
      (e.tagName === "INPUT"
        ? ""
        : [...e.childNodes]
            .map((n) =>
              n.nodeType === 3
                ? n.textContent
                : n.nodeType === 1 && n.getAttribute("aria-hidden") !== "true"
                  ? name(n, seen)
                  : ""
            )
            .join(" ")
            .trim()) ||
      e.getAttribute("title") ||
      e.getAttribute("placeholder") ||
      ""
    );
  };

  const ROLES = [
    "button", "link", "checkbox", "radio", "switch", "tab", "menuitem",
    "menuitemradio", "option", "gridcell", "combobox", "textbox", "searchbox", "spinbutton",
  ];
  const SELECTOR =
    "a[href],button,input,textarea,select,summary,[contenteditable=\"true\"]," +
    ROLES.map((r) => `[role="${r}"]`).join(",");

  const role = (e) => {
    const explicit = e.getAttribute("role");
    if (ROLES.includes(explicit)) return explicit;
    if (e.tagName === "BUTTON" || e.tagName === "SUMMARY") return "button";
    if (e.tagName === "A") return "link";
    if (e.tagName === "SELECT") return "combobox";
    if (e.tagName === "TEXTAREA" || e.isContentEditable) return "textbox";
    if (e.tagName === "INPUT") {
      if (["checkbox", "radio"].includes(e.type)) return e.type;
      if (["button", "submit", "reset", "image"].includes(e.type)) return "button";
      if (e.type === "search") return "searchbox";
      if (e.type === "number") return "spinbutton";
      if (["text", "email", "url", "tel"].includes(e.type)) return "textbox";
    }
    return null;
  };

  // 页面身份：timeOrigin + URL + 滚动 + 视口 + 表单状态
  cache.pageKey = () => [
    performance.timeOrigin,
    location.href,
    scrollX,
    scrollY,
    innerWidth,
    innerHeight,
    [...document.querySelectorAll("input,textarea,select")]
      .filter(safe)
      .map((e) => [identity(e), e.value, e.checked, e.selectedIndex, e.disabled, e.readOnly]),
  ];

  // 单元素新鲜度：身份 + 语义（**不含几何**）
  cache.guard = (e) => {
    if (!e?.isConnected || !visible(e)) return null;
    const scope = e.closest('form,dialog,[role="dialog"],article,li,tr,[role="row"]') || e.parentElement;
    return [
      identity(e), role(e), name(e), e.value ?? null, e.checked ?? null, e.selectedIndex ?? null,
      e.readOnly ?? null, e.matches(":disabled"), e.getAttribute("aria-disabled"),
      e.getAttribute("aria-expanded"), e.getAttribute("aria-checked"), e.getAttribute("aria-selected"),
      e.getAttribute("href"), (scope?.innerText || "").slice(0, 6000),
    ];
  };

  // ---- 元素表 ----
  const actions = [];
  for (const e of document.querySelectorAll(SELECTOR)) {
    if (!safe(e) || !visible(e) || e.matches(":disabled") || e.closest('[aria-disabled="true"]')) continue;
    const r = e.getBoundingClientRect();
    const x = r.x + r.width / 2;
    const y = r.y + r.height / 2;
    const rname = role(e);
    if (!rname || r.width <= 0 || r.height <= 0 || x < 0 || y < 0 || x >= innerWidth || y >= innerHeight) continue;
    // gridcell 里还有 button 时只保留内层，否则会把同一处报两次
    if (rname === "gridcell" && e.querySelector('button,[role="button"]')) continue;

    const base = { node: identity(e), role: rname, label: name(e) || rname };
    for (const key of ["checked", "selected", "expanded"]) {
      const v = e.getAttribute("aria-" + key);
      if (v !== null) base[key] = v;
    }
    if (["checkbox", "radio"].includes(e.type)) base.checked = String(e.checked);

    if (e.tagName === "SELECT") {
      for (const o of e.options) {
        if (!o.selected && !o.disabled && !o.closest("optgroup[disabled]")) {
          actions.push({
            ...base, kind: "select", value: o.value,
            current_value: [...e.selectedOptions].map((x) => x.label).join(", "),
            label: base.label + " → " + o.label,
          });
        }
      }
    } else {
      const editable =
        !e.readOnly && e.getAttribute("aria-readonly") !== "true" &&
        (["textbox", "searchbox", "spinbutton"].includes(rname) ||
          (rname === "combobox" && ["INPUT", "TEXTAREA"].includes(e.tagName)));
      const value = "value" in e ? String(e.value) : e.isContentEditable || rname === "combobox" ? e.innerText.trim() : "";
      actions.push({ ...base, kind: editable ? "fill" : "click", value });
      if (editable) actions.push({ ...base, kind: "click", value, label: "Open " + base.label });
    }
  }

  // ---- 只取视口内可见文本 ----
  const words = [];
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const range = document.createRange();
  let node;
  let length = 0;
  while ((node = walker.nextNode()) && length < 6000) {
    const value = node.textContent.trim();
    const parent = node.parentElement;
    if (!value || !parent || parent.closest("script,style,noscript,template") || !visible(parent)) continue;
    range.selectNodeContents(node);
    const r = range.getBoundingClientRect();
    // 视口过滤：离屏正文不进 context。这是"只送可见文本"的落点。
    if (r.width > 0 && r.height > 0 && r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth) {
      words.push(value);
      length += value.length;
    }
  }
  const text = words.join("\n").slice(0, 6000);
  const height = document.documentElement.scrollHeight;

  const page_key = cache.pageKey();
  const guards = {};
  for (const a of actions) if (!(a.node in guards)) guards[a.node] = cache.guard(cache.nodes.get(a.node));

  // marker 对照"意义与身份"，**不含几何**
  const semantics = actions.map(({ ...a }) => a);
  const marker = [
    performance.timeOrigin, location.href, scrollX, scrollY, innerWidth, innerHeight,
    document.title, text, semantics, page_key[6],
  ];

  // 截断必须**可见**：静默砍掉元素表会让模型"看不见"某些控件而反复失败
  const omitted_actions = Math.max(0, actions.length - 250);
  actions.splice(250);
  actions.forEach((a, i) => (a.id = "e" + (i + 1)));

  if (scrollY + innerHeight < height - 2) actions.push({ id: "scroll_down", kind: "scroll", label: "Scroll down", delta: 560 });
  if (scrollY > 0) actions.push({ id: "scroll_up", kind: "scroll", label: "Scroll up", delta: -560 });
  actions.push({ id: "wait", kind: "wait", label: "Wait for the page to update" });

  // 给模型看的元素表：一行一个候选，编号即 e1..eN
  const table = actions
    .filter((a) => a.kind !== "wait")
    .map((a) => {
      const v = a.current_value ?? a.value;
      const shown = v === undefined || v === null || v === "" ? "" : " · " + String(v).slice(0, 60);
      return `[${a.id}] ${a.role} ${a.label}${shown}`;
    });

  return {
    url: location.href,
    title: document.title,
    w: innerWidth,
    h: innerHeight,
    table,
    text,
    scroll: { y: scrollY, height },
    actions,
    marker,
    page_key,
    guards,
    omitted_actions,
  };
})()
