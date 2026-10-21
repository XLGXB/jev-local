#!/usr/bin/env node
/* 浏览器层的可执行自测：只覆盖**纯逻辑**，不连 Chrome、不连决策服务。
 *
 * 跑法:
 *   node browser/selftest.mjs
 *
 * 为什么这些逻辑值得单测：它们都是"形状对、不报错、行为错"的高发区——
 * 默认值写反（dry-run 变成真执行）、不可逆闸门被绕过、候选数超出 255 被静默截断。
 */

import process from "node:process";

import { Decider, looksIrreversible, parseArgs } from "./cdp_loop.mjs";
import { runSmTests } from "./sm_selftest.mjs";

let passed = 0;
const failures = [];

function check(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`PASS  ${name}`);
  } catch (err) {
    failures.push(name);
    console.log(`FAIL  ${name}: ${err.message}`);
  }
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || "断言失败");
}

// --------------------------------------------------------------------------- //
// 1. 默认必须是 dry-run
// --------------------------------------------------------------------------- //
check("默认 dry-run（不显式 --live 绝不执行动作）", () => {
  const opts = parseArgs(["--url", "https://a", "--goal", "g", "--decide-url", "http://127.0.0.1:1/x"]);
  assert(opts.live === false, `live 应为 false，得到 ${opts.live}`);
  assert(opts.allowIrreversible === false, "不可逆默认必须为 false");
});

check("--live 与 --allow-irreversible 只有显式传才生效", () => {
  const opts = parseArgs([
    "--url", "https://a", "--goal", "g", "--decide-url", "u", "--live", "--allow-irreversible",
  ]);
  assert(opts.live === true && opts.allowIrreversible === true);
});

check("未知参数必须报错而不是被忽略", () => {
  let threw = false;
  try {
    parseArgs(["--nope"]);
  } catch {
    threw = true;
  }
  assert(threw, "未知参数应抛错");
});

// --------------------------------------------------------------------------- //
// 2. 不可逆闸门
// --------------------------------------------------------------------------- //
check("不可逆动作被命中（英文）", () => {
  for (const label of [
    "Place order", "Pay now", "Delete account", "Send message", "Publish post",
    "Checkout", "Install extension", "Authorize access", "Unsubscribe",
  ]) {
    assert(looksIrreversible({ label, role: "button" }, {}), `应命中: ${label}`);
  }
});

check("不可逆动作被命中（中文）", () => {
  for (const label of ["立即支付", "确认订单", "删除账号", "发送消息", "发布文章", "授权访问", "卸载插件"]) {
    assert(looksIrreversible({ label, role: "button" }, {}), `应命中: ${label}`);
  }
});

check("普通导航动作不被误判", () => {
  for (const label of ["Log in", "Open filters", "Next page", "Search flights", "切换视图", "展开备注"]) {
    assert(!looksIrreversible({ label, role: "button" }, {}), `不应命中: ${label}`);
  }
});

check("模型自报不可逆即为不可逆（单向不可撤销）", () => {
  assert(looksIrreversible({ label: "Continue", role: "button" }, { irreversible: true }));
});

check("模型自报 false 不能推翻标签命中", () => {
  assert(
    looksIrreversible({ label: "Delete account", role: "button" }, { irreversible: false }),
    "标签命中时模型说 false 也必须拦",
  );
});

// --------------------------------------------------------------------------- //
// 3. 问题构造
// --------------------------------------------------------------------------- //
function fakeSnapshot(n) {
  const actions = [];
  for (let i = 0; i < n; i += 1) actions.push({ id: `e${i + 1}`, kind: "click", role: "button", label: `b${i}` });
  actions.push({ id: "wait", kind: "wait", label: "Wait" });
  return { actions, url: "https://x", text: "t", table: [] };
}

check("候选数不超过 255（System One 验证层上限）", () => {
  const q = Decider.questions(fakeSnapshot(250));
  assert(Object.keys(q.target.criteria).length === 250, "应为 250");
  const q2 = Decider.questions(fakeSnapshot(254));
  assert(Object.keys(q2.target.criteria).length <= 255, "不得超过 255");
});

check("wait 不进入候选表（它不是可点元素）", () => {
  const q = Decider.questions(fakeSnapshot(3));
  assert(!("wait" in q.target.criteria), "wait 不该出现在候选中");
  // 但 operation 里仍要有 WAIT 这个操作，否则模型无法表达"等页面更新"
  assert("WAIT" in q.operation.criteria);
});

check("四个问题齐备且类型正确", () => {
  const q = Decider.questions(fakeSnapshot(3));
  assert(q.operation.type === "choice" && q.target.type === "choice");
  assert(q.irreversible.type === "noul" && q.done.type === "noul");
});

// --------------------------------------------------------------------------- //
// 4. state 组装
// --------------------------------------------------------------------------- //
check("state 里同时含目标、可见文本与元素表", () => {
  const d = new Decider("http://127.0.0.1:1/x", 500);
  const snap = { url: "https://x", text: "HELLOTEXT", table: ["[e1] button Go"] };
  const state = d.buildState("MYGOAL", snap, ["clicked A"]);
  for (const token of ["MYGOAL", "HELLOTEXT", "[e1] button Go", "clicked A", "https://x"]) {
    assert(state.includes(token), `state 缺: ${token}`);
  }
});

check("超预算的可见文本被截断到 token 预算内", () => {
  const d = new Decider("http://127.0.0.1:1/x", 50);
  const snap = { url: "u", text: "A".repeat(10_000), table: [] };
  const state = d.buildState("g", snap, []);
  assert(state.length < 1000, `应被截断，实际 ${state.length}`);
});

// --------------------------------------------------------------------------- //
// --------------------------------------------------------------------------- //
// 状态机 + 驱动层（同一命令一起跑，避免"改完忘了跑另一半"）
// --------------------------------------------------------------------------- //
const sm = await runSmTests();
passed += sm.passed;
failures.push(...sm.failures);

console.log(`\n合计: ${passed}/${passed + failures.length} passed`);
if (failures.length) {
  console.log("失败用例: " + failures.join(", "));
  process.exit(1);
}
