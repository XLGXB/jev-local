import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import {
  PiBridge, activeToolMetadata, chooseActiveTool, minimalWorkerEnvironment,
  validateEnvelope, validatePacket,
} from './bridge.mjs';

const env = {
  JEVLOCAL_PI_PYTHON: 'C:\\python\\python.exe',
  JEVLOCAL_PI_CONFIG: 'C:\\jev\\config.json',
  SystemRoot: 'C:\\Windows', PATH: 'C:\\Windows\\System32',
  SECRET_TOKEN: 'must-not-reach-child',
};
const question = { type: 'choice', instructions: 'Choose one', criteria: { a: 'Alpha', b: 'Beta' } };

function envelope(observation_id = 'obs-1') {
  return { request_id: 'request-1', observation_id, state: { task: 'brief' }, questions: { q: question } };
}

function packetFor(request, selected = 'a') {
  const answers = {};
  for (const [qid, q] of Object.entries(request.questions)) {
    const options = q.type === 'noul' ? ['false', 'true']
      : q.type === 'score' ? q.criteria.map((_, i) => String(i)) : Object.keys(q.criteria);
    const pick = options.includes(selected) ? selected : options[0];
    const probabilities = Object.fromEntries(options.map(key => [key, key === pick ? 1 : 0]));
    answers[qid] = { selected_key: pick, probabilities, p_selected: 1, calibrated: false };
  }
  return {
    schema_version: 1, request_id: request.request_id, observation_id: request.observation_id,
    ok: true,
    reply: { request_id: request.request_id, observation_id: request.observation_id,
      runtime_fingerprint: 'native-sha', answers },
    metadata: { runtime_fingerprint: 'native-sha', advisory_only: true,
      execution_authorized: false, calibrated: false, semantic_quality: 'unqualified' },
  };
}

function fakeSpawn(onWrite = () => {}, { closeOnKill = true, closeOnEnd = true, killResult = true } = {}) {
  const calls = [];
  const children = [];
  const spawn = (command, args, options) => {
    calls.push({ command, args, options });
    const child = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.writes = [];
    child.killed = false;
    child.ended = false;
    child.kill = () => {
      child.killed = true;
      if (closeOnKill) queueMicrotask(() => child.emit('close'));
      return killResult;
    };
    child.stdin = Object.assign(new EventEmitter(), { end(callback) {
      child.ended = true;
      if (closeOnEnd) queueMicrotask(() => child.emit('close', 0, null));
      callback?.();
    }, write(line, encoding, callback) {
      assert.equal(encoding, 'utf8');
      child.writes.push(line);
      callback?.();
      onWrite(child, JSON.parse(line));
      return true;
    } });
    children.push(child);
    return child;
  };
  return { spawn, calls, children };
}

const existingFile = () => ({ isFile: () => true });
const makeBridge = fake => new PiBridge({ env, spawn: fake.spawn, stat: existingFile,
  workerPath: 'C:\\jev\\src\\jevlocal\\integration\\pi_worker.py', timeoutMs: 100, closeTimeoutMs: 20 });
const ask = (bridge, signal) => bridge.ask({ observation_id: 'obs-1', state: {}, questions: { q: question } }, signal);
const emit = (child, packet) => child.stdout.emit('data', Buffer.from(`${JSON.stringify(packet)}\n`));

test('request contract rejects labels, non-finite JSON, oversized and invalid candidates before spawn', () => {
  const base = envelope();
  assert.deepEqual([...validateEnvelope(base).keysets.get('q')], ['a', 'b']);
  assert.throws(() => validateEnvelope({ ...base, questions: { q: { ...question, label: 'a' } } }));
  assert.throws(() => validateEnvelope({ ...base, state: { x: NaN } }));
  assert.throws(() => validateEnvelope({ ...base, state: { x: 'x'.repeat(131073) } }));
  assert.throws(() => validateEnvelope({ ...base, questions: { q: { ...question, criteria: { a: 'one' } } } }));
  assert.deepEqual(validateEnvelope({ ...base, questions: { q: { type: 'noul', instructions: 'Judge' } } }).keysets.get('q'), ['false', 'true']);
  assert.deepEqual(validateEnvelope({ ...base, questions: { q: { type: 'noul', instructions: 'Judge', criteria: null } } }).keysets.get('q'), ['false', 'true']);
  assert.deepEqual(validateEnvelope({ ...base, questions: { q: { type: 'score', instructions: 'Rate', criteria: ['low', 'high'] } } }).keysets.get('q'), ['0', '1']);
});

test('packet binding rejects stale IDs, changed options, invalid probability and false authority', () => {
  const request = envelope();
  const keysets = validateEnvelope(request).keysets;
  assert.equal(validatePacket(request, keysets, packetFor(request)).ok, true);
  const stale = packetFor(request); stale.request_id = 'other';
  assert.throws(() => validatePacket(request, keysets, stale));
  const changed = packetFor(request); changed.reply.answers.q.probabilities = { a: 1, c: 0 };
  assert.throws(() => validatePacket(request, keysets, changed));
  const bad = packetFor(request); bad.reply.answers.q.probabilities.a = Infinity;
  assert.throws(() => validatePacket(request, keysets, bad));
  const authority = packetFor(request); authority.metadata.execution_authorized = true;
  assert.throws(() => validatePacket(request, keysets, authority));
  const calibration = packetFor(request); calibration.reply.answers.q.calibrated = true;
  assert.throws(() => validatePacket(request, keysets, calibration));
  const identity = packetFor(request); identity.metadata.runtime_fingerprint = 'other';
  assert.throws(() => validatePacket(request, keysets, identity));
});

test('missing absolute opt-in configuration never spawns; child gets no ambient secret', async () => {
  const fake = fakeSpawn((child, request) => queueMicrotask(() => emit(child, packetFor(request))));
  const unconfigured = new PiBridge({ env: {}, spawn: fake.spawn, stat: existingFile,
    workerPath: 'C:\\jev\\pi_worker.py' });
  await unconfigured.startSession();
  await assert.rejects(ask(unconfigured), { code: 'config_required' });
  assert.equal(fake.calls.length, 0);
  for (const missing of [env.JEVLOCAL_PI_PYTHON, env.JEVLOCAL_PI_CONFIG, 'C:\\jev\\pi_worker.py']) {
    const nonfile = new PiBridge({ env, spawn: fake.spawn,
      stat: path => ({ isFile: () => path !== missing }), workerPath: 'C:\\jev\\pi_worker.py' });
    await nonfile.startSession();
    await assert.rejects(ask(nonfile), { code: 'config_required' });
  }
  const nonexistent = new PiBridge({ env, spawn: fake.spawn,
    stat: () => { throw new Error('ENOENT'); }, workerPath: 'C:\\jev\\pi_worker.py' });
  await nonexistent.startSession();
  await assert.rejects(ask(nonexistent), { code: 'config_required' });
  assert.equal(fake.calls.length, 0);
  const bridge = makeBridge(fake); await bridge.startSession();
  await ask(bridge);
  assert.equal(fake.calls.length, 1);
  assert.deepEqual(fake.calls[0].args, ['-I', '-B', 'C:\\jev\\src\\jevlocal\\integration\\pi_worker.py', '--config', env.JEVLOCAL_PI_CONFIG]);
  assert.equal(fake.calls[0].options.shell, false);
  assert.equal(fake.calls[0].options.windowsHide, true);
  assert.equal(fake.calls[0].options.env.SECRET_TOKEN, undefined);
  assert.deepEqual(Object.keys(minimalWorkerEnvironment(env)).sort(), ['PATH', 'SystemRoot'].sort());
});

test('owned worker is lazy, reused serially, and refuses concurrent requests', async () => {
  const fake = fakeSpawn();
  const bridge = makeBridge(fake); await bridge.startSession();
  assert.equal(fake.calls.length, 0);
  const first = ask(bridge);
  await assert.rejects(ask(bridge), { code: 'busy' });
  emit(fake.children[0], packetFor(JSON.parse(fake.children[0].writes[0])));
  await first;
  const second = ask(bridge);
  assert.equal(fake.calls.length, 1);
  emit(fake.children[0], packetFor(JSON.parse(fake.children[0].writes[1])));
  await second;
  await bridge.shutdown();
  assert.equal(fake.children[0].ended, true);
  assert.equal(fake.children[0].killed, false);
  await assert.rejects(ask(bridge), { code: 'closed' });
});

test('optional resource policy is host-only, absolute, regular and passed without expanding child env', async () => {
  const policy = 'C:\\jev\\resource-policy.json';
  const configured = { ...env, JEVLOCAL_PI_RESOURCE_POLICY: policy };
  const fake = fakeSpawn((child, request) => queueMicrotask(() => emit(child, packetFor(request))));
  for (const invalid of ['', null, 'relative.json', 42]) {
    const bridge = new PiBridge({ env: { ...configured, JEVLOCAL_PI_RESOURCE_POLICY: invalid },
      spawn: fake.spawn, stat: existingFile, workerPath: 'C:\\jev\\pi_worker.py' });
    await bridge.startSession();
    await assert.rejects(ask(bridge), { code: 'config_required' });
  }
  const missing = new PiBridge({ env: configured, spawn: fake.spawn,
    stat: path => ({ isFile: () => path !== policy }), workerPath: 'C:\\jev\\pi_worker.py' });
  await missing.startSession();
  await assert.rejects(ask(missing), { code: 'config_required' });
  assert.equal(fake.calls.length, 0);
  const bridge = new PiBridge({ env: configured, spawn: fake.spawn, stat: existingFile,
    workerPath: 'C:\\jev\\pi_worker.py', timeoutMs: 100, closeTimeoutMs: 20 });
  await bridge.startSession();
  await ask(bridge);
  assert.deepEqual(fake.calls[0].args, ['-I', '-B', 'C:\\jev\\pi_worker.py', '--config',
    env.JEVLOCAL_PI_CONFIG, '--resource-policy', policy]);
  assert.equal(fake.calls[0].options.env.JEVLOCAL_PI_RESOURCE_POLICY, undefined);
  assert.deepEqual(Object.keys(fake.calls[0].options.env).sort(), ['PATH', 'SystemRoot'].sort());
  await bridge.shutdown();
  assert.equal(fake.children[0].ended, true);
  assert.equal(fake.children[0].killed, false);
});

test('idle session rotation delivers EOF and waits for close before replacement', async () => {
  const fake = fakeSpawn((child, request) => queueMicrotask(() => emit(child, packetFor(request))),
    { closeOnEnd: false });
  const bridge = makeBridge(fake); await bridge.startSession();
  await ask(bridge);
  const rotating = bridge.startSession();
  assert.equal(fake.children[0].ended, true);
  assert.equal(fake.children[0].killed, false);
  assert.equal(fake.calls.length, 1);
  fake.children[0].emit('close', 0, null);
  await rotating;
  await ask(bridge);
  assert.equal(fake.calls.length, 2);
  const shutdown = bridge.shutdown();
  fake.children[1].emit('close', 0, null);
  await shutdown;
});

test('normal EOF timeout remains terminal and cannot replace child', async () => {
  const fake = fakeSpawn((child, request) => queueMicrotask(() => emit(child, packetFor(request))),
    { closeOnEnd: false });
  const bridge = makeBridge(fake); await bridge.startSession();
  await ask(bridge);
  await assert.rejects(bridge.startSession(), { code: 'close_unconfirmed' });
  assert.equal(fake.children[0].ended, true);
  assert.equal(fake.children[0].killed, true);
  assert.equal(fake.calls.length, 1);
  await assert.rejects(bridge.startSession(), { code: 'closed' });
});

test('normal EOF close requires exit zero without signal before replacement', async () => {
  for (const [exitCode, signal] of [[1, null], [null, 'SIGTERM']]) {
    const fake = fakeSpawn((child, request) => queueMicrotask(() => emit(child, packetFor(request))),
      { closeOnEnd: false });
    const bridge = makeBridge(fake); await bridge.startSession();
    await ask(bridge);
    const rotating = bridge.startSession();
    assert.equal(fake.children[0].ended, true);
    fake.children[0].emit('close', exitCode, signal);
    await assert.rejects(rotating, { code: 'close_unconfirmed' });
    assert.equal(bridge.terminal, true);
    assert.equal(fake.calls.length, 1);
    await assert.rejects(bridge.startSession(), { code: 'closed' });
  }
});

test('resource policy cannot change silently while reusing a worker; poisoned bridge fails closed', async () => {
  const configured = { ...env };
  const fake = fakeSpawn((child, request) => queueMicrotask(() => emit(child, packetFor(request))));
  const bridge = new PiBridge({ env: configured, spawn: fake.spawn, stat: existingFile,
    workerPath: 'C:\\jev\\pi_worker.py', timeoutMs: 100, closeTimeoutMs: 20 });
  await bridge.startSession();
  await ask(bridge);
  configured.JEVLOCAL_PI_RESOURCE_POLICY = 'C:\\jev\\resource-policy.json';
  await assert.rejects(ask(bridge), { code: 'resource_policy_changed' });
  assert.equal(fake.children[0].killed, true);
  assert.equal(fake.calls.length, 1);
  await assert.rejects(ask(bridge), { code: 'poisoned' });
  await assert.rejects(bridge.startSession(), { code: 'closed' });
  assert.equal(fake.calls.length, 1);
  await assert.rejects(bridge.shutdown(), { code: 'close_unconfirmed' });
});

test('abort and timeout kill owned worker and fail closed', async () => {
  const fake = fakeSpawn();
  const bridge = makeBridge(fake); await bridge.startSession();
  const controller = new AbortController();
  const pending = ask(bridge, controller.signal);
  controller.abort();
  await assert.rejects(pending, { code: 'aborted' });
  assert.equal(fake.children[0].killed, true);
  await assert.rejects(ask(bridge), { code: 'poisoned' });
  await assert.rejects(bridge.startSession(), { code: 'closed' });
  const timeoutFake = fakeSpawn();
  const timeoutBridge = makeBridge(timeoutFake); await timeoutBridge.startSession();
  await assert.rejects(ask(timeoutBridge), { code: 'timeout' });
  assert.equal(timeoutFake.children[0].killed, true);
  await assert.rejects(timeoutBridge.startSession(), { code: 'closed' });
});

test('pre-aborted request without spawned worker does not close future sessions', async () => {
  const fake = fakeSpawn((child, request) => queueMicrotask(() => emit(child, packetFor(request))));
  const bridge = makeBridge(fake); await bridge.startSession();
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(ask(bridge, controller.signal), { code: 'aborted' });
  assert.equal(fake.calls.length, 0);
  await bridge.startSession();
  await ask(bridge);
  assert.equal(fake.calls.length, 1);
  await bridge.shutdown();
});

test('pending rotation cancels promptly, drains EOF and ignores old response until acknowledged close', async () => {
  const fake = fakeSpawn(() => {}, { closeOnEnd: false });
  const bridge = makeBridge(fake); await bridge.startSession();
  const pending = ask(bridge);
  const old = fake.children[0];
  const rotating = bridge.startSession();
  await assert.rejects(pending, { code: 'session_changed' });
  assert.equal(old.ended, true);
  assert.equal(old.killed, false);
  assert.equal(fake.calls.length, 1);
  assert.equal(bridge.started, false);
  emit(old, packetFor(JSON.parse(old.writes[0])));
  assert.equal(bridge.started, false);
  old.emit('close', 0, null);
  await rotating;
  const next = ask(bridge);
  old.stdout.emit('data', Buffer.from('{}\n'));
  old.emit('exit', 1);
  emit(fake.children[1], packetFor(JSON.parse(fake.children[1].writes[0])));
  await next;
  assert.equal(fake.calls.length, 2);
});

test('pending shutdown cancels promptly then awaits normal EOF close', async () => {
  const fake = fakeSpawn(() => {}, { closeOnEnd: false });
  const bridge = makeBridge(fake); await bridge.startSession();
  const pending = ask(bridge);
  const closing = bridge.shutdown();
  await assert.rejects(pending, { code: 'closed' });
  assert.equal(fake.children[0].ended, true);
  assert.equal(fake.children[0].killed, false);
  fake.children[0].emit('close', 0, null);
  await closing;
  await assert.rejects(bridge.startSession(), { code: 'closed' });
});

test('pending EOF abnormal close and timeout forbid replacement', async () => {
  for (const [exitCode, signal] of [[1, null], [null, 'SIGTERM'], ['timeout', null]]) {
    const fake = fakeSpawn(() => {}, { closeOnEnd: false });
    const bridge = makeBridge(fake); await bridge.startSession();
    const pending = ask(bridge);
    const rotating = bridge.startSession();
    await assert.rejects(pending, { code: 'session_changed' });
    assert.equal(fake.children[0].ended, true);
    assert.equal(fake.children[0].killed, false);
    if (exitCode !== 'timeout') fake.children[0].emit('close', exitCode, signal);
    await assert.rejects(rotating, { code: 'close_unconfirmed' });
    assert.equal(bridge.terminal, true);
    assert.equal(fake.calls.length, 1);
    await assert.rejects(bridge.startSession(), { code: 'closed' });
  }
});

test('new session does not spawn while old close is unconfirmed', async () => {
  const fake = fakeSpawn(() => {}, { closeOnEnd: false });
  const bridge = makeBridge(fake); await bridge.startSession();
  const pending = ask(bridge);
  const rotating = bridge.startSession();
  await assert.rejects(pending, { code: 'session_changed' });
  await assert.rejects(rotating, { code: 'close_unconfirmed' });
  assert.equal(fake.calls.length, 1);
  await assert.rejects(bridge.startSession(), { code: 'closed' });
});

test('kill failure blocks replacement even if old child later closes', async () => {
  const fake = fakeSpawn(() => {}, { killResult: false });
  const bridge = makeBridge(fake); await bridge.startSession();
  const pending = ask(bridge);
  const rejectedPending = assert.rejects(pending, { code: 'aborted' });
  bridge._poison('aborted');
  await rejectedPending;
  await Promise.resolve();
  assert.equal(fake.calls.length, 1);
  await assert.rejects(bridge.startSession(), { code: 'closed' });
});

test('forced launcher close and spontaneous close cannot certify worker stopped', async () => {
  for (const kind of ['forced', 'spontaneous', 'error', 'exit']) {
    const fake = fakeSpawn(() => {}, { closeOnKill: false });
    const bridge = makeBridge(fake); await bridge.startSession();
    const pending = ask(bridge);
    if (kind === 'forced') {
      bridge._poison('aborted');
      fake.children[0].emit('close', 0, null);
      await assert.rejects(pending, { code: 'aborted' });
    } else {
      if (kind === 'error') fake.children[0].emit('error', new Error('fake spawn error'));
      else if (kind === 'exit') fake.children[0].emit('exit', 1);
      else fake.children[0].emit('close', 0, null);
      await assert.rejects(pending, { code: {
        spontaneous: 'worker_close', error: 'worker_error', exit: 'worker_exit',
      }[kind] });
      fake.children[0].emit('close', 0, null);
    }
    await assert.rejects(bridge.startSession(), { code: 'closed' });
    assert.equal(fake.calls.length, 1);
  }
});

test('shutdown retains unconfirmed force or abnormal close after closing reference clears', async () => {
  for (const kind of ['forced', 'abnormal']) {
    const fake = fakeSpawn();
    const bridge = makeBridge(fake); await bridge.startSession();
    const pending = ask(bridge);
    if (kind === 'forced') {
      bridge._poison('aborted');
      await assert.rejects(pending, { code: 'aborted' });
    } else {
      fake.children[0].emit('close', 0, null);
      await assert.rejects(pending, { code: 'worker_close' });
    }
    await new Promise(resolve => setImmediate(resolve));
    assert.equal(bridge.closing, null);
    await assert.rejects(bridge.shutdown(), { code: 'close_unconfirmed' });
    assert.equal(fake.calls.length, 1);
    await assert.rejects(bridge.startSession(), { code: 'closed' });
  }
});

test('stdout frame and stderr budgets poison without leaking diagnostics', async () => {
  const fake = fakeSpawn();
  const bridge = makeBridge(fake); await bridge.startSession();
  const pending = ask(bridge);
  fake.children[0].stdout.emit('data', Buffer.alloc(262146, 65));
  await assert.rejects(pending, { code: 'frame_limit' });
  await assert.rejects(bridge.startSession(), { code: 'closed' });
  const stderrFake = fakeSpawn();
  const stderrBridge = makeBridge(stderrFake); await stderrBridge.startSession();
  const next = ask(stderrBridge);
  stderrFake.children[0].stderr.emit('data', Buffer.alloc(65537, 66));
  await assert.rejects(next, { code: 'stderr_limit' });
  await assert.rejects(stderrBridge.startSession(), { code: 'closed' });
});

test('extra or unsolicited stdout and stale reply poison transport', async () => {
  const fake = fakeSpawn();
  const bridge = makeBridge(fake); await bridge.startSession();
  const pending = ask(bridge);
  const request = JSON.parse(fake.children[0].writes[0]);
  emit(fake.children[0], { ...packetFor(request), observation_id: 'stale' });
  await assert.rejects(pending, { code: 'invalid_reply' });
  assert.equal(fake.children[0].killed, true);
  await assert.rejects(bridge.startSession(), { code: 'closed' });
  const extraFake = fakeSpawn();
  const extraBridge = makeBridge(extraFake); await extraBridge.startSession();
  const next = ask(extraBridge);
  const packet = packetFor(JSON.parse(extraFake.children[0].writes[0]));
  extraFake.children[0].stdout.emit('data', Buffer.from(`${JSON.stringify(packet)}\n{}\n`));
  await assert.rejects(next, { code: 'extra_stdout' });
  await assert.rejects(extraBridge.startSession(), { code: 'closed' });
  const unsolicitedFake = fakeSpawn();
  const unsolicitedBridge = makeBridge(unsolicitedFake); await unsolicitedBridge.startSession();
  const last = ask(unsolicitedBridge);
  emit(unsolicitedFake.children[0], packetFor(JSON.parse(unsolicitedFake.children[0].writes[0])));
  await last;
  unsolicitedFake.children[0].stdout.emit('data', Buffer.from('{}\n'));
  await assert.rejects(ask(unsolicitedBridge), { code: 'poisoned' });
  await assert.rejects(unsolicitedBridge.startSession(), { code: 'closed' });
});

test('tool closed set uses active ∩ all metadata and rechecks after inference', async () => {
  let active = ['read', 'write', 'jev_ask', 'jev_choose_tool'];
  const all = [{ name: 'read', description: 'Read a file' }, { name: 'write', description: 'Write a file' },
    { name: 'inactive', description: 'No' }];
  const getActive = () => active;
  const getAll = () => all;
  assert.deepEqual(Object.keys(activeToolMetadata(getActive, getAll)), ['read', 'write']);
  for (const names of [[], ['read'], ['read', 'read'], ['read', 'inactive'], ['read', 'unknown'], Array(17).fill('read')]) {
    assert.throws(() => activeToolMetadata(getActive, getAll, names));
  }
  const fake = fakeSpawn();
  const bridge = makeBridge(fake); await bridge.startSession();
  const pending = chooseActiveTool(bridge, getActive, getAll,
    { observation_id: 'obs-1', state: {}, instructions: 'Pick', candidate_names: ['read', 'write'] });
  assert.deepEqual(Object.keys(JSON.parse(fake.children[0].writes[0]).questions.tool_choice.criteria), ['read', 'write']);
  active = ['read'];
  emit(fake.children[0], packetFor(JSON.parse(fake.children[0].writes[0]), 'write'));
  await assert.rejects(pending, { code: 'invalid_tools' });
  assert.equal(fake.children[0].killed, false);
});
