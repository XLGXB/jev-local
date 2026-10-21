import { spawn as nodeSpawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { Buffer } from 'node:buffer';
import { statSync } from 'node:fs';
import { dirname, resolve, isAbsolute, win32, posix } from 'node:path';
import { fileURLToPath } from 'node:url';

const MAX_REQUEST_BYTES = 128 * 1024;
const MAX_FRAME_BYTES = 262144;
const MAX_STDERR_BYTES = 65536;
const MAX_QUESTIONS = 8;
const MAX_CANDIDATES = 16;
const DEFAULT_TIMEOUT_MS = 120000;
const CLOSE_TIMEOUT_MS = 5000;
const TOOL_NAMES = new Set(['jev_ask', 'jev_choose_tool']);
const SYSTEM_ENV = ['SystemRoot', 'WINDIR', 'PATH', 'TEMP', 'TMP', 'HOME', 'USERPROFILE'];

export class BridgeError extends Error {
  constructor(code, message) {
    super(message);
    this.name = 'BridgeError';
    this.code = code;
  }
}

const fail = (code, message) => { throw new BridgeError(code, message); };
const record = value => value !== null && typeof value === 'object' && !Array.isArray(value)
  && (Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null);
const keysExactly = (value, keys) => record(value) && Object.keys(value).length === keys.length
  && keys.every(key => Object.hasOwn(value, key));
const validId = value => typeof value === 'string' && value.trim() !== '' && value.length <= 128;
const absolute = value => typeof value === 'string' && value.trim() !== ''
  && (isAbsolute(value) || win32.isAbsolute(value) || posix.isAbsolute(value));

function finiteJson(value, seen = new Set()) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return;
  if (typeof value === 'number' && Number.isFinite(value)) return;
  if (typeof value !== 'object' || seen.has(value)) fail('invalid_request', 'Finite JSON is required');
  seen.add(value);
  if (Array.isArray(value)) {
    if (Object.keys(value).length !== value.length) fail('invalid_request', 'Dense JSON arrays are required');
    for (const item of value) finiteJson(item, seen);
  } else if (record(value)) {
    for (const item of Object.values(value)) finiteJson(item, seen);
  } else {
    fail('invalid_request', 'Plain JSON objects are required');
  }
  seen.delete(value);
}

export function validateEnvelope(envelope) {
  if (!keysExactly(envelope, ['request_id', 'observation_id', 'state', 'questions'])
      || !validId(envelope.request_id) || !validId(envelope.observation_id)) {
    fail('invalid_request', 'A four-field envelope with valid IDs is required');
  }
  if (!record(envelope.questions) || Object.keys(envelope.questions).length < 1
      || Object.keys(envelope.questions).length > MAX_QUESTIONS) {
    fail('invalid_request', '1..8 questions are required');
  }
  const keysets = new Map();
  for (const [qid, question] of Object.entries(envelope.questions)) {
    if (!validId(qid) || !record(question)
        || Object.keys(question).some(key => !['type', 'instructions', 'criteria'].includes(key))
        || typeof question.instructions !== 'string' || !question.instructions.trim()) {
      fail('invalid_request', 'Question ID, instructions or public fields are invalid');
    }
    let options;
    if (question.type === 'choice') {
      if (!record(question.criteria)) fail('invalid_request', 'Choice criteria must be an object');
      options = Object.keys(question.criteria);
      if (options.length < 2 || options.length > MAX_CANDIDATES
          || options.some(key => !validId(key))
          || Object.values(question.criteria).some(value => typeof value !== 'string' || !value.trim())) {
        fail('invalid_request', 'Choice requires 2..16 described candidates');
      }
    } else if (question.type === 'score') {
      if (!Array.isArray(question.criteria) || question.criteria.length < 2
          || question.criteria.length > MAX_CANDIDATES
          || question.criteria.some(value => typeof value !== 'string' || !value.trim())) {
        fail('invalid_request', 'Score requires 2..16 described levels');
      }
      options = question.criteria.map((_, i) => String(i));
    } else if (question.type === 'noul') {
      if (question.criteria != null && (!record(question.criteria)
          || Object.keys(question.criteria).length !== 2
          || !Object.hasOwn(question.criteria, 'false') || !Object.hasOwn(question.criteria, 'true')
          || Object.values(question.criteria).some(value => typeof value !== 'string' || !value.trim()))) {
        fail('invalid_request', 'Noul criteria must describe false and true');
      }
      options = ['false', 'true'];
    } else {
      fail('invalid_request', 'Question type must be choice, score or noul');
    }
    keysets.set(qid, options);
  }
  finiteJson(envelope);
  const line = JSON.stringify(envelope);
  if (Buffer.byteLength(line, 'utf8') > MAX_REQUEST_BYTES) {
    fail('invalid_request', 'Request exceeds 128 KiB');
  }
  return { line, keysets };
}

function validateReply(envelope, keysets, reply) {
  if (!record(reply) || Object.keys(reply).some(key => ![
    'request_id', 'observation_id', 'runtime_fingerprint', 'answers', 'inference_total_ms',
  ].includes(key)) || reply.request_id !== envelope.request_id
      || reply.observation_id !== envelope.observation_id
      || typeof reply.runtime_fingerprint !== 'string' || !reply.runtime_fingerprint
      || !record(reply.answers)
      || Object.keys(reply.answers).length !== keysets.size) {
    fail('invalid_reply', 'Reply identity or question set is invalid');
  }
  if (Object.hasOwn(reply, 'inference_total_ms')
      && (typeof reply.inference_total_ms !== 'number'
        || !Number.isFinite(reply.inference_total_ms) || reply.inference_total_ms < 0)) {
    fail('invalid_reply', 'Inference timing is invalid');
  }
  for (const [qid, options] of keysets) {
    if (!Object.hasOwn(reply.answers, qid)) fail('invalid_reply', 'Reply question set changed');
    const answer = reply.answers[qid];
    if (!record(answer) || Object.keys(answer).some(key => ![
      'selected_key', 'probabilities', 'p_selected', 'calibrated', 'temperature',
    ].includes(key)) || !record(answer.probabilities)
        || Object.keys(answer.probabilities).length !== options.length
        || answer.calibrated !== false) {
      fail('invalid_reply', 'Typed answer is invalid');
    }
    if (Object.hasOwn(answer, 'temperature')
        && (typeof answer.temperature !== 'number'
          || !Number.isFinite(answer.temperature) || answer.temperature <= 0)) {
      fail('invalid_reply', 'Calibration temperature is invalid');
    }
    let total = 0;
    let maximum = -1;
    for (const option of options) {
      const probability = answer.probabilities[option];
      if (typeof probability !== 'number' || !Number.isFinite(probability)
          || probability < 0 || probability > 1) {
        fail('invalid_reply', 'Candidate probability is invalid');
      }
      total += probability;
      maximum = Math.max(maximum, probability);
    }
    if (Math.abs(total - 1) > 1e-6 || !options.includes(answer.selected_key)
        || answer.probabilities[answer.selected_key] !== maximum
        || typeof answer.p_selected !== 'number' || !Number.isFinite(answer.p_selected)
        || Math.abs(answer.p_selected - maximum) > 1e-6) {
      fail('invalid_reply', 'Candidate selection or probabilities are inconsistent');
    }
  }
}

export function validatePacket(envelope, keysets, packet) {
  if (!record(packet) || packet.schema_version !== 1
      || packet.request_id !== envelope.request_id
      || packet.observation_id !== envelope.observation_id) {
    fail('invalid_reply', 'Worker packet identity is invalid');
  }
  if (packet.ok === false) {
    if (!keysExactly(packet, ['schema_version', 'request_id', 'observation_id', 'ok', 'error'])
        || !keysExactly(packet.error, ['code', 'message'])
        || !['invalid_request', 'initialization_failed', 'prediction_failed'].includes(packet.error.code)
        || typeof packet.error.message !== 'string' || packet.error.message.length > 512) {
      fail('invalid_reply', 'Worker error packet is invalid');
    }
    fail(packet.error.code, 'Local worker rejected the request');
  }
  if (!keysExactly(packet, ['schema_version', 'request_id', 'observation_id', 'ok', 'reply', 'metadata'])
      || packet.ok !== true) fail('invalid_reply', 'Worker success packet is invalid');
  validateReply(envelope, keysets, packet.reply);
  const metadata = packet.metadata;
  if (!record(metadata) || metadata.runtime_fingerprint !== packet.reply.runtime_fingerprint
      || metadata.advisory_only !== true || metadata.execution_authorized !== false
      || metadata.calibrated !== false || metadata.semantic_quality !== 'unqualified'
      || Object.entries(metadata).some(([key, value]) =>
        /(?:authoriz|approv|grant|permission)/i.test(key) && value !== false)) {
    fail('invalid_reply', 'Worker metadata is not bound to non-authoritative runtime identity');
  }
  finiteJson(packet);
  if (Buffer.byteLength(JSON.stringify(packet), 'utf8') > MAX_FRAME_BYTES) {
    fail('invalid_reply', 'Worker packet exceeds frame budget');
  }
  return packet;
}

export function activeToolMetadata(getActiveTools, getAllTools, candidateNames) {
  const active = getActiveTools();
  const all = getAllTools();
  if (!Array.isArray(active) || !Array.isArray(all)) fail('invalid_tools', 'Pi tool inventory is unavailable');
  const activeNames = new Set(active);
  const available = new Map();
  for (const item of all) {
    if (record(item) && typeof item.name === 'string' && activeNames.has(item.name)
        && !TOOL_NAMES.has(item.name)) {
      if (available.has(item.name)) fail('invalid_tools', 'Pi tool metadata has duplicate names');
      available.set(item.name, item);
    }
  }
  const names = candidateNames === undefined ? [...available.keys()] : candidateNames;
  if (!Array.isArray(names) || names.length < 2 || names.length > MAX_CANDIDATES
      || new Set(names).size !== names.length
      || names.some(name => !validId(name) || !available.has(name))) {
    fail('invalid_tools', 'Supply 2..16 unique currently active known tool names');
  }
  const criteria = Object.create(null);
  for (const name of names) {
    const description = available.get(name).description;
    if (typeof description !== 'string' || !description.trim()) {
      fail('invalid_tools', 'Every candidate needs current Pi description metadata');
    }
    criteria[name] = description;
  }
  return criteria;
}

export function assertToolsStillActive(getActiveTools, getAllTools, names) {
  activeToolMetadata(getActiveTools, getAllTools, names);
}

export async function chooseActiveTool(bridge, getActiveTools, getAllTools,
  { observation_id, state, instructions, candidate_names }, signal) {
  if (typeof instructions !== 'string' || !instructions.trim()) {
    fail('invalid_request', 'Tool-choice instructions are required');
  }
  const criteria = activeToolMetadata(getActiveTools, getAllTools, candidate_names);
  const names = Object.keys(criteria);
  const packet = await bridge.ask({ observation_id, state, questions: {
    tool_choice: { type: 'choice', instructions, criteria },
  } }, signal);
  assertToolsStillActive(getActiveTools, getAllTools, names);
  const answer = packet.reply.answers.tool_choice;
  if (!names.includes(answer.selected_key)) fail('invalid_reply', 'Selected tool left the candidate set');
  return {
    suggested_tool: answer.selected_key,
    probabilities: answer.probabilities,
    p_selected: answer.p_selected,
    calibrated: answer.calibrated,
    runtime_fingerprint: packet.reply.runtime_fingerprint,
    advisory_only: true,
    execution_authorized: false,
    semantic_quality: 'unqualified',
  };
}

export function minimalWorkerEnvironment(env) {
  const childEnv = Object.create(null);
  for (const key of SYSTEM_ENV) if (typeof env[key] === 'string' && env[key]) childEnv[key] = env[key];
  return childEnv;
}

export class PiBridge {
  constructor({ env = process.env, spawn = nodeSpawn, timeoutMs = DEFAULT_TIMEOUT_MS,
    closeTimeoutMs = CLOSE_TIMEOUT_MS,
    stat = statSync,
    workerPath = resolve(dirname(fileURLToPath(import.meta.url)), '../src/jevlocal/integration/pi_worker.py') } = {}) {
    this.env = env;
    this.spawn = spawn;
    this.stat = stat;
    this.timeoutMs = timeoutMs;
    this.closeTimeoutMs = closeTimeoutMs;
    this.workerPath = workerPath;
    this.started = false;
    this.closed = false;
    this.poisoned = false;
    this.terminal = false;
    this.child = null;
    this.resourcePolicyPath = undefined;
    this.closing = null;
    this.pending = null;
    this.buffer = Buffer.alloc(0);
    this.stderrBytes = 0;
    this.generation = 0;
  }

  async startSession() {
    if (this.closed || this.terminal) fail('closed', 'Extension runtime cannot start a new session');
    this.started = false;
    const generation = ++this.generation;
    const closed = await this._stop('session_changed', { graceful: true });
    if (!closed || this.terminal || this.closed) fail('close_unconfirmed', 'Old worker close was not confirmed');
    if (generation !== this.generation) fail('session_changed', 'A newer session superseded this start');
    this.started = true;
    this.poisoned = false;
  }

  async shutdown() {
    if (this.closed) return;
    this.closed = true;
    this.started = false;
    this.generation++;
    if (!await this._stop('closed', { graceful: true })) {
      fail('close_unconfirmed', 'Owned worker close was not confirmed');
    }
  }

  _stop(code, { alreadyClosed = false, kill = true, graceful = false } = {}) {
    const pending = this.pending;
    this.pending = null;
    if (pending) {
      clearTimeout(pending.timer);
      pending.signal?.removeEventListener('abort', pending.onAbort);
      pending.reject(new BridgeError(code, 'Owned worker session ended'));
    }
    if (this.closing) {
      if (!graceful) this.terminal = true;
      return this.closing.then(confirmed => confirmed && !this.terminal);
    }
    const child = this.child;
    this.child = null;
    this.resourcePolicyPath = undefined;
    this.buffer = Buffer.alloc(0);
    this.stderrBytes = 0;
    if (!child) return Promise.resolve(!this.terminal);
    if (!graceful || alreadyClosed) this.terminal = true;
    if (alreadyClosed) return Promise.resolve(false);
    let finish;
    const closing = new Promise(resolveClose => { finish = resolveClose; });
    this.closing = closing;
    let settled = false;
    const onStdinError = () => failClose();
    const settle = confirmed => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      child.stdin?.removeListener?.('error', onStdinError);
      if (!confirmed) this.terminal = true;
      finish(confirmed);
    };
    const failClose = () => {
      if (settled) return;
      // A launcher close after SIGTERM cannot confirm its redirected interpreter stopped.
      settle(false);
      try { child.kill(); } catch { /* terminal state already recorded */ }
    };
    const timer = setTimeout(failClose, this.closeTimeoutMs);
    child.once('close', (exitCode, signal) =>
      settle(graceful && exitCode === 0 && signal === null));
    if (graceful) {
      try {
        child.stdin.once('error', onStdinError);
        child.stdin.end(error => { if (error) failClose(); });
      } catch { failClose(); }
    } else if (kill) {
      try {
        if (child.kill() !== true) {
          this.terminal = true;
          settle(false);
        }
      } catch {
        settle(false);
      }
    }
    void closing.then(() => { if (this.closing === closing) this.closing = null; });
    return closing;
  }

  _poison(code, options) {
    this.poisoned = true;
    void this._stop(code, options);
  }

  _onStdout(chunk) {
    if (!this.pending) { this._poison('unsolicited_stdout'); return; }
    this.buffer = Buffer.concat([this.buffer, Buffer.from(chunk)]);
    if (this.buffer.length > MAX_FRAME_BYTES + 1) { this._poison('frame_limit'); return; }
    const newline = this.buffer.indexOf(10);
    if (newline < 0) return;
    if (newline > MAX_FRAME_BYTES || newline !== this.buffer.length - 1) {
      this._poison('extra_stdout'); return;
    }
    let packet;
    try {
      packet = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(this.buffer.subarray(0, newline)));
      validatePacket(this.pending.envelope, this.pending.keysets, packet);
    } catch (error) {
      this._poison(error instanceof BridgeError ? error.code : 'invalid_reply');
      return;
    }
    this.buffer = Buffer.alloc(0);
    const pending = this.pending;
    this.pending = null;
    clearTimeout(pending.timer);
    pending.signal?.removeEventListener('abort', pending.onAbort);
    pending.resolve(packet);
  }

  ask({ observation_id, state, questions }, signal) {
    if (!this.started || this.closed) return Promise.reject(new BridgeError('closed', 'No active Pi session'));
    if (this.poisoned) return Promise.reject(new BridgeError('poisoned', 'Worker transport is poisoned'));
    if (this.pending) return Promise.reject(new BridgeError('busy', 'Worker already has one request'));
    const python = this.env.JEVLOCAL_PI_PYTHON;
    const config = this.env.JEVLOCAL_PI_CONFIG;
    const resourcePolicy = this.env.JEVLOCAL_PI_RESOURCE_POLICY;
    if (this.child && resourcePolicy !== this.resourcePolicyPath) {
      this._poison('resource_policy_changed');
      return Promise.reject(new BridgeError('resource_policy_changed', 'Resource policy changed within the owned worker session'));
    }
    if (!absolute(python) || !absolute(config) || !absolute(this.workerPath)) {
      return Promise.reject(new BridgeError('config_required', 'Explicit absolute local Python and config paths are required'));
    }
    if (resourcePolicy !== undefined && !absolute(resourcePolicy)) {
      return Promise.reject(new BridgeError('config_required', 'Resource policy must be an explicit absolute local path'));
    }
    try {
      const files = [python, config, this.workerPath];
      if (resourcePolicy !== undefined) files.push(resourcePolicy);
      if (!files.every(path => this.stat(path).isFile())) {
        return Promise.reject(new BridgeError('config_required', 'Python, config, shipped worker and any resource policy must be regular files'));
      }
    } catch {
      return Promise.reject(new BridgeError('config_required', 'Python, config, shipped worker and any resource policy must exist as regular files'));
    }
    const envelope = { request_id: randomUUID(), observation_id, state, questions };
    let validated;
    try { validated = validateEnvelope(envelope); }
    catch (error) { return Promise.reject(error); }
    if (signal?.aborted) {
      this._poison('aborted');
      return Promise.reject(new BridgeError('aborted', 'Tool request was aborted'));
    }
    if (!Number.isSafeInteger(this.timeoutMs) || this.timeoutMs < 1 || this.timeoutMs > DEFAULT_TIMEOUT_MS) {
      return Promise.reject(new BridgeError('invalid_timeout', 'Timeout must be 1..120000 ms'));
    }
    return new Promise((resolvePromise, rejectPromise) => {
      const onAbort = () => this._poison('aborted');
      const pending = { envelope, keysets: validated.keysets, resolve: resolvePromise,
        reject: rejectPromise, signal, onAbort, timer: null };
      this.pending = pending;
      pending.timer = setTimeout(() => this._poison('timeout'), this.timeoutMs);
      signal?.addEventListener('abort', onAbort, { once: true });
      try {
        if (!this.child) {
          const args = ['-I', '-B', this.workerPath, '--config', config];
          if (resourcePolicy !== undefined) args.push('--resource-policy', resourcePolicy);
          const child = this.spawn(python, args, {
            stdio: ['pipe', 'pipe', 'pipe'], shell: false, windowsHide: true,
            env: minimalWorkerEnvironment(this.env),
          });
          this.child = child;
          this.resourcePolicyPath = resourcePolicy;
          child.stdout.on('data', chunk => {
            if (this.child === child) this._onStdout(chunk);
          });
          child.stderr.on('data', chunk => {
            if (this.child !== child) return;
            this.stderrBytes += Buffer.byteLength(chunk);
            if (this.stderrBytes > MAX_STDERR_BYTES) this._poison('stderr_limit');
          });
          child.on('error', () => { if (this.child === child) this._poison('worker_error'); });
          child.on('exit', () => { if (this.child === child) this._poison('worker_exit', { kill: false }); });
          child.on('close', () => { if (this.child === child) this._poison('worker_close', { alreadyClosed: true }); });
        }
        const child = this.child;
        child.stdin.write(`${validated.line}\n`, 'utf8', error => {
          if (error && this.child === child) this._poison('worker_write_failed');
        });
      } catch {
        this._poison('worker_start_failed');
      }
    });
  }
}
