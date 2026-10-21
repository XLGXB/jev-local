"""Explicit local tools; no automatic installation, model download, or approval."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from .context_tools import ModelRouteError


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))


def _read_state_json_config(path):
    """Bound the operator configuration before parsing or importing a model."""
    if Path(path).stat().st_size > 64 * 1024:
        raise ValueError('state-json configuration byte budget exceeded')
    return _read(path)


def doctor():
    packages = {}
    for name in ("pytest", "numpy", "torch", "safetensors", "transformers", "peft", "fastapi", "httpx",
                 "uvicorn", "filelock", "pydantic", "cryptography", "playwright", "mcp", "onnx", "onnxruntime"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {"schema_version": 1, "project": "JevLocal", "python": sys.version.split()[0],
            "executable": sys.executable, "packages": packages,
            "active_browser_core": "fastloop", "default_mode": "dry_run",
            "codex_host_browser_adapter": "not_connected",
            "native_model_validated": False, "execution_authorized": False,
            "note": "Inventory only. Package availability is not native-model, browser, or quality acceptance."}


def _write_new(path, value):
    content = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(content)


def _write_new_atomic(path, value):
    """Publish a complete sidecar without overwriting an existing destination."""
    content = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf-8')
    target = Path(path).absolute()
    parent = target.parent.resolve(strict=True)
    fd, temporary = tempfile.mkstemp(prefix='.jevlocal-shadow-', suffix='.tmp', dir=parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        # Same-directory hard-link publication is atomic and exclusive; no
        # os.replace fallback may overwrite another caller's concurrent file.
        os.link(temporary, target)
    finally:
        if temporary.resolve().parent == parent:
            temporary.unlink(missing_ok=True)


def _read_scope_artifact(path):
    """Read exactly one caller-supplied bounded JSON artifact, never its paths."""
    import hashlib
    with Path(path).open('rb') as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError('scope artifact exceeds limit')
    def finite_float(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError('nonfinite JSON')
        return parsed
    try:
        value = json.loads(raw.decode('utf-8-sig'), object_pairs_hook=_pairs,
                           parse_float=finite_float,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except RecursionError:
        raise ValueError('scope artifact nesting exceeds limit') from None
    if not isinstance(value, dict):
        raise ValueError('scope artifact must be an object')
    return value, hashlib.sha256(raw).hexdigest()


def _scope_output_paths(outputs, inputs):
    protected = {Path(p).resolve() for p in inputs if p is not None}
    seen = set()
    for name in outputs:
        if name is None:
            continue
        path = Path(name)
        resolved = path.resolve()
        if (resolved in protected or resolved in seen or path.exists() or path.is_symlink()
                or not path.parent.is_dir()):
            raise ValueError('scope outputs must be new distinct files in existing directories')
        seen.add(resolved)


def _scope_command(args):
    from .evidence_scope import scope_evidence, verify_scope_evidence
    original, original_hash = _read_scope_artifact(args.input)
    if args.command == 'scope-evidence':
        _scope_output_paths([args.output, args.rank_input_output], [args.input])
        result = scope_evidence(**original)
        if result['status'] != 'requires_larger_budget':
            # Each artifact is exclusive/atomic individually, not a multi-file
            # transaction. A second write failure is reported by the caller;
            # do not delete the valid first artifact or claim full publication.
            _write_new_atomic(args.output, result)
            if args.rank_input_output and result['rank_input'] is not None:
                _write_new_atomic(args.rank_input_output, result['rank_input'])
        return result
    if bool(args.primary) != bool(args.shadow):
        raise ValueError('combined replay requires both primary and shadow')
    _scope_output_paths([args.output], [args.input, args.prepared, args.rank_input,
                                        args.primary, args.shadow])
    prepared, prepared_hash = _read_scope_artifact(args.prepared)
    derived, derived_hash = _read_scope_artifact(args.rank_input) if args.rank_input else (None, None)
    scope_report = verify_scope_evidence(original, prepared, derived)
    replay_report = None
    hashes = {'input': original_hash, 'prepared': prepared_hash, 'rank_input': derived_hash}
    if args.primary:
        if derived is None:
            raise ValueError('no derived candidates to replay')
        # Use the same in-memory derived object checked above, not a second read
        # that might silently replay a replacement file. No implicit store read.
        from .context_replay import load_replay
        primary, hashes['primary'] = _read_scope_artifact(args.primary)
        shadow, hashes['shadow'] = _read_scope_artifact(args.shadow)
        replay_report = load_replay(derived, primary, shadow).report()
    result = {'version': 1, 'status': 'consistent', 'scope': scope_report, 'replay': replay_report,
              'artifact_sha256': hashes, 'model_calls': 0, 'actions_replayed': 0,
              'execution_authorized': False, 'training_allowed': False, 'origin_authenticated': False}
    if args.output:
        _write_new_atomic(args.output, result)
    return result


def _passive_rank(args, arguments):
    """Publish the unchanged host result before waiting for an optional observer."""
    from . import context_tools
    from .native_context import validate_native_spec
    from .context_shadow import start_shadow, unavailable_shadow

    timeout = 60.0 if args.shadow_timeout is None else args.shadow_timeout
    if not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError('shadow timeout must be in (0, 60] seconds')
    spec = validate_native_spec(_read(args.shadow_runtime))
    sidecar = Path(args.shadow_output).resolve()
    protected = [Path(args.input).resolve(), Path(args.shadow_runtime).resolve()]
    if args.output:
        protected.append(Path(args.output).resolve())
        if Path(args.output).exists():
            raise ValueError('primary output already exists')
    if sidecar in protected or sidecar.exists() or not sidecar.parent.is_dir():
        raise ValueError('shadow output must be a new distinct file in an existing directory')
    query, safe = context_tools._prepare_rank_evidence(**arguments)
    store_dir = arguments.get('store_dir')
    try:
        job = start_shadow(query, safe, spec, timeout_s=timeout)
    except Exception:
        # Caller input/config errors were already rejected above. Failure in the
        # optional observer is not a failure of the independent lexical host.
        job = unavailable_shadow(query, safe, spec, reason='startup_failed')
    exit_code = None
    try:
        result = context_tools._rank_prepared_evidence(query, safe,
            max_chars=arguments['max_chars'], store_dir=store_dir)
        if args.output:
            _write_new(args.output, result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2), flush=True)
        exit_code = 2 if result.get('status') == 'requires_larger_budget' else 0
        try:
            receipt = job.finish(result, store_dir=store_dir)
        except Exception as exc:
            # An optional observer must not replace or reprint the host result.
            receipt = {'schema_version': 1, 'status': 'unavailable', 'shadow': True,
                'reason': 'observer_internal_error', 'error_type': type(exc).__name__,
                'training_allowed': False, 'execution_authorized': False, 'cloud_data_used': False}
        try:
            _write_new_atomic(args.shadow_output, receipt)
        except (ValueError, TypeError, OSError) as exc:
            print(json.dumps({'status': 'shadow_receipt_not_written',
                'error_type': type(exc).__name__, 'execution_authorized': False}), file=sys.stderr)
            exit_code = 2
    finally:
        try:
            shutdown = job.close()
            if isinstance(shutdown, dict) and (shutdown.get('closed') is not True or
                    shutdown.get('temporary_cleaned') is not True):
                print(json.dumps({'status': 'shadow_cleanup_failed',
                    'execution_authorized': False}), file=sys.stderr)
                if exit_code is not None:
                    exit_code = 2
        except Exception as exc:
            print(json.dumps({'status': 'shadow_cleanup_failed',
                'error_type': type(exc).__name__, 'execution_authorized': False}), file=sys.stderr)
            # Do not mask a primary exception; the outer handler preserves it.
            if exit_code is not None:
                exit_code = 2
    return exit_code


def _passive_skills(args, arguments):
    """Publish lexical suggestions before observing declared-description relevance."""
    from . import context_tools
    from .context_shadow import start_shadow, unavailable_shadow
    from .native_context import validate_native_spec
    from .skill_shadow import prepare_skill_projection, finish_skill_shadow

    timeout = 60.0 if args.shadow_timeout is None else args.shadow_timeout
    if not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError('shadow timeout must be in (0, 60] seconds')
    spec = validate_native_spec(_read(args.shadow_runtime))
    sidecar = Path(args.shadow_output).resolve()
    protected = [Path(args.input).resolve(), Path(args.shadow_runtime).resolve()]
    if args.output:
        protected.append(Path(args.output).resolve())
        if Path(args.output).exists() or Path(args.output).is_symlink():
            raise ValueError('primary output already exists')
    if (sidecar in protected or Path(args.shadow_output).exists() or
            Path(args.shadow_output).is_symlink() or not sidecar.parent.is_dir()):
        raise ValueError('shadow output must be a new distinct file in an existing directory')
    task, safe = context_tools._prepare_suggest_skills(**arguments)
    try:
        projection = prepare_skill_projection(task, safe)
    except (ValueError, TypeError, OverflowError, UnicodeError, RecursionError):
        # This is an optional observer capacity/preparation failure after the
        # unchanged skill input boundary has accepted the closed set.
        projection = None
    job = None
    if projection is not None:
        record, query, candidates = projection
        try:
            job = start_shadow(query, candidates, spec, timeout_s=timeout,
                binding=record['binding'], manifest=record['manifest'],
                rank_input=record['rank_input'])
        except Exception:
            job = unavailable_shadow(query, candidates, spec,
                reason='startup_failed', binding=record['binding'])
    exit_code = None
    try:
        primary = context_tools._suggest_prepared_skills(task, safe,
            max_chars=arguments['max_chars'])
        if args.output:
            _write_new(args.output, primary)
        print(json.dumps(primary, ensure_ascii=False, allow_nan=False, indent=2), flush=True)
        exit_code = 2 if primary.get('status') == 'requires_larger_budget' else 0
        try:
            receipt = finish_skill_shadow(job, task, safe, primary,
                max_chars=arguments['max_chars'], projection=projection)
        except Exception as exc:
            receipt = finish_skill_shadow(None, task, safe, primary,
                max_chars=arguments['max_chars'])
            receipt['availability']['reason'] = 'observer_internal_error'
            receipt['error_type'] = type(exc).__name__
        try:
            _write_new_atomic(args.shadow_output, receipt)
        except (ValueError, TypeError, OSError) as exc:
            print(json.dumps({'status': 'shadow_receipt_not_written',
                'error_type': type(exc).__name__, 'execution_authorized': False}), file=sys.stderr)
            exit_code = 2
    finally:
        if job is not None:
            try:
                shutdown = job.close()
                if not shutdown.get('closed') or not shutdown.get('temporary_cleaned'):
                    print(json.dumps({'status': 'shadow_cleanup_failed',
                        'execution_authorized': False}), file=sys.stderr)
                    if exit_code is not None:
                        exit_code = 2
            except Exception as exc:
                print(json.dumps({'status': 'shadow_cleanup_failed',
                    'error_type': type(exc).__name__, 'execution_authorized': False}), file=sys.stderr)
                if exit_code is not None:
                    exit_code = 2
    return exit_code


def _scoped_command(args):
    from .scoped_invocation import prepare_invocation, replay_invocation, validate_snapshot
    if args.command == 'replay-scoped-rank-evidence':
        _scope_output_paths([args.output], [args.input, args.manifest, args.primary, args.shadow])
        original, _ = _read_scope_artifact(args.input)
        record, _ = _read_scope_artifact(args.manifest)
        primary, _ = _read_scope_artifact(args.primary)
        sidecar, _ = _read_scope_artifact(args.shadow) if args.shadow else (None, None)
        result = replay_invocation(original, record, primary, sidecar)
        if args.output:
            _write_new_atomic(args.output, result)
        return result

    from . import context_tools
    from .context_shadow import start_shadow, unavailable_shadow
    from .native_context import validate_native_spec
    _scope_output_paths([args.manifest_output, args.primary_output, args.shadow_output],
                        [args.input, args.shadow_runtime])
    timeout = 60.0 if args.shadow_timeout is None else args.shadow_timeout
    if not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError('shadow timeout must be in (0, 60] seconds')
    original, _ = _read_scope_artifact(args.input)
    spec = validate_native_spec(_read(args.shadow_runtime))
    record = prepare_invocation(original)
    if record.get('status') == 'requires_larger_budget':
        return record
    derived = record['rank_input']
    binding = record['binding']
    if derived is not None:
        query, safe = context_tools._prepare_rank_evidence(**derived)
        # A contract mismatch is invalid input, never an optional observer
        # outage. Reject before publication and outside the startup catch.
        validate_snapshot(binding, record['manifest'], derived, query, safe)
    # Exclusive, complete before-start publication. A later failure preserves it.
    _write_new_atomic(args.manifest_output, record)
    if derived is None:
        primary = {'status': 'no_matches', 'execution_authorized': False,
                   'training_allowed': False}
        _write_new_atomic(args.primary_output, primary)
        print(json.dumps(primary, ensure_ascii=False, allow_nan=False, indent=2), flush=True)
        return None

    try:
        job = start_shadow(query, safe, spec, timeout_s=timeout, binding=binding,
                           manifest=record['manifest'], rank_input=derived)
    except Exception:
        job = unavailable_shadow(query, safe, spec, reason='startup_failed', binding=binding)
    exit_code = None
    try:
        primary = context_tools._rank_prepared_evidence(query, safe,
            max_chars=derived['max_chars'])
        _write_new_atomic(args.primary_output, primary)
        print(json.dumps(primary, ensure_ascii=False, allow_nan=False, indent=2), flush=True)
        exit_code = 2 if primary.get('status') == 'requires_larger_budget' else 0
        try:
            sidecar = job.finish(primary)
        except Exception as exc:
            print(json.dumps({'status': 'scoped_observer_internal_error',
                'error_type': type(exc).__name__, 'execution_authorized': False}), file=sys.stderr)
            return 2
        try:
            _write_new_atomic(args.shadow_output, sidecar)
        except (ValueError, TypeError, OSError) as exc:
            print(json.dumps({'status': 'scoped_receipt_not_written',
                'error_type': type(exc).__name__, 'execution_authorized': False}), file=sys.stderr)
            exit_code = 2
    finally:
        try:
            shutdown = job.close()
            if not shutdown.get('closed') or not shutdown.get('temporary_cleaned'):
                print(json.dumps({'status': 'shadow_cleanup_failed',
                    'execution_authorized': False}), file=sys.stderr)
                if exit_code is not None:
                    exit_code = 2
        except Exception as exc:
            print(json.dumps({'status': 'shadow_cleanup_failed',
                'error_type': type(exc).__name__, 'execution_authorized': False}), file=sys.stderr)
            if exit_code is not None:
                exit_code = 2
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="read dependency inventory without installing or loading a model")
    bridge = sub.add_parser("bridge", help="convert canonical JSONL with an explicit per-sample authorization map")
    bridge.add_argument("--input", required=True)
    bridge.add_argument("--authorization", required=True)
    bridge.add_argument("--output", required=True)
    dataset = sub.add_parser("prepare-dataset", help="preserve and verify all four existing partitions")
    dataset.add_argument("--input", required=True)
    dataset.add_argument("--output-dir", required=True)
    validate = sub.add_parser("validate-envelope")
    validate.add_argument("--input", required=True)
    replay = sub.add_parser("replay-context", help="verify saved passive context artifacts without model/action replay")
    for option in ("input", "primary", "shadow"):
        replay.add_argument("--" + option, required=True)
    replay.add_argument("--store-dir", help="explicit existing store to recheck; input paths are not followed")
    replay.add_argument("--output", help="new report file; existing files are never overwritten")
    scope = sub.add_parser('scope-evidence', help='prepare explicit object/revision applicability; no model or storage')
    scope.add_argument('--input', required=True)
    scope.add_argument('--output', required=True, help='new full scope snapshot/report')
    scope.add_argument('--rank-input-output', help='new matching-only input for existing rank_evidence; omitted when no matches')
    scoped_replay = sub.add_parser('verify-scoped-evidence', help='recompute scope and optionally verify derived passive artifacts')
    scoped_replay.add_argument('--input', required=True)
    scoped_replay.add_argument('--prepared', required=True)
    for option in ('rank-input', 'primary', 'shadow', 'output'):
        scoped_replay.add_argument('--' + option)
    scoped_rank = sub.add_parser('scoped-rank-evidence',
        help='explicit scope + lexical primary + bounded passive observer; no host registration')
    for option in ('input', 'shadow-runtime', 'manifest-output', 'primary-output', 'shadow-output'):
        scoped_rank.add_argument('--' + option, required=True)
    scoped_rank.add_argument('--shadow-timeout', type=float)
    scoped_rank_replay = sub.add_parser('replay-scoped-rank-evidence',
        help='pure consistency check of one explicit v2 scoped invocation')
    for option in ('input', 'manifest', 'primary'):
        scoped_rank_replay.add_argument('--' + option, required=True)
    scoped_rank_replay.add_argument('--shadow')
    scoped_rank_replay.add_argument('--output')
    for command in ("build-qualification", "verify-qualification"):
        qualify = sub.add_parser(command, help="verify quality evidence without authorizing any action")
        qualify.add_argument("--runtime-fingerprint", required=True)
        for option in ("calibration-path", "evaluation-path", "policy-path", "dataset-dir"):
            qualify.add_argument("--" + option, required=True)
        for option in ("parity-path", "golden-path", "bundle-manifest-path"):
            qualify.add_argument("--" + option)
        qualify.add_argument("--receipt", required=True,
                             help="new output for build; existing receipt for verify")
    review = sub.add_parser('review-diff', help='triage actual file diffs against explicit contracts; never approve merge')
    review.add_argument('--input', required=True)
    review.add_argument('--output', help='new risk report; existing files are never overwritten')
    review_route = review.add_mutually_exclusive_group()
    review_route.add_argument('--native-runtime', help='explicit generic native route; represented checkpoints rejected')
    review_route.add_argument('--state-json-runtime', help='explicit pinned raw StateJson route; no calibration or fallback')
    completion = sub.add_parser('verify-completion', help='verify explicit local completion evidence; no command execution or authorization')
    completion.add_argument('--input', required=True)
    completion.add_argument('--output', help='new evidence report; existing files are never overwritten')
    completion_route = completion.add_mutually_exclusive_group()
    completion_route.add_argument('--native-runtime', help='explicit generic native route; represented checkpoints rejected')
    completion_route.add_argument('--state-json-runtime', help='explicit pinned raw StateJson route; no calibration or fallback')
    model_routing = sub.add_parser('advise-model-route', help='closed-catalog model/effort advice; never switch models or authorize execution')
    model_routing.add_argument('--input', required=True)
    model_routing.add_argument('--output', help='new advisory report; existing files are never overwritten')
    model_routing_route = model_routing.add_mutually_exclusive_group()
    model_routing_route.add_argument('--native-runtime', help='explicit generic native route; represented checkpoints rejected')
    model_routing_route.add_argument('--state-json-runtime', help='explicit pinned raw StateJson route; no calibration or fallback')
    navigation = sub.add_parser('navigate-repository', help='observe explicit-root directory candidates and bounded advisory hops; never read file content')
    navigation.add_argument('--input', required=True)
    navigation.add_argument('--output', help='new path-evidence report; existing files are never overwritten')
    navigation_route = navigation.add_mutually_exclusive_group()
    navigation_route.add_argument('--native-runtime', help='explicit generic native route; represented checkpoints rejected')
    navigation_route.add_argument('--state-json-runtime', help='explicit pinned raw StateJson route; no calibration or fallback')
    context = sub.add_parser("context", help="lexical tools with optional active native scoring or passive sidecar")
    context.add_argument("tool", choices=["rank_evidence", "filter_tool_output", "suggest_skills", "restore_lines"])
    context.add_argument("--input", required=True, help="JSON object containing the documented tool arguments")
    context.add_argument("--output")
    route = context.add_mutually_exclusive_group()
    route.add_argument("--native-runtime", help="active native scoring changes ranking/skills; operator runtime JSON")
    route.add_argument("--shadow-runtime", help="passive native observer; rank_evidence/suggest_skills only; primary stays lexical")
    context.add_argument("--shadow-output", help="new separate passive observation/feedback JSON file")
    context.add_argument("--shadow-timeout", type=float, help="observer startup plus inference deadline, (0, 60] seconds")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            result = doctor()
        elif args.command == "bridge":
            from jevlocal.schema import Sample
            from .data_bridge import to_training_record
            authorization = _read(args.authorization)
            if not isinstance(authorization, dict):
                raise ValueError("authorization must map each actual sample ID to explicit metadata")
            rows = []
            ids = set()
            for line in Path(args.input).read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                sample = Sample.from_json(json.loads(line, object_pairs_hook=_pairs))
                if sample.sample_id in ids:
                    raise ValueError("duplicate sample ID")
                ids.add(sample.sample_id)
                metadata = authorization.get(sample.sample_id)
                if not isinstance(metadata, dict) or set(metadata) != {"group_id", "label_source", "task", "training_allowed"}:
                    raise ValueError("missing or invalid explicit per-sample authorization metadata")
                rows.append(to_training_record(sample, **metadata))
            if not rows or set(authorization) != ids:
                raise ValueError("authorization IDs must exactly match nonempty input samples")
            # Serialize every row before creating an output; never publish a partly converted file.
            content = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
            with Path(args.output).open("x", encoding="utf-8") as stream:
                stream.write(content)
            result = {"status": "converted", "records": len(rows), "output": str(Path(args.output).resolve()),
                      "training_executed": False, "partitioning_changed": False}
        elif args.command == "prepare-dataset":
            from decision_ft.data import read_jsonl
            from .dataset import prepare_presplit_dataset
            result = prepare_presplit_dataset(read_jsonl(args.input), args.output_dir)
        elif args.command == "validate-envelope":
            from decision_deploy.protocol import validate_envelope
            result = {"status": "valid", "candidate_keys": validate_envelope(_read(args.input)), "execution_authorized": False}
        elif args.command == "replay-context":
            from .context_replay import read_replay_files
            result = read_replay_files(args.input, args.primary, args.shadow,
                                      store_dir=args.store_dir).report()
            if args.output:
                _write_new_atomic(args.output, result)
        elif args.command in ('scope-evidence', 'verify-scoped-evidence'):
            result = _scope_command(args)
        elif args.command in ('scoped-rank-evidence', 'replay-scoped-rank-evidence'):
            result = _scoped_command(args)
            if args.command == 'scoped-rank-evidence' and (result is None or type(result) is int):
                return 0 if result is None else result
        elif args.command == 'review-diff':
            from .review_diff import review_diff
            if Path(args.input).stat().st_size > 128 * 1024:
                raise ValueError('review input file byte budget exceeded')
            if args.output and Path(args.output).exists():
                raise ValueError('review output must be new')
            arguments = _read(args.input)
            def review_factory():
                if args.state_json_runtime:
                    from .native_review import build_state_json_review_judge
                    return build_state_json_review_judge(_read_state_json_config(args.state_json_runtime))
                from .native_review import build_native_review_judge
                return build_native_review_judge(_read(args.native_runtime))
            result = review_diff(arguments,
                judge_factory=review_factory if (args.native_runtime or args.state_json_runtime) else None)
            if args.output:
                _write_new_atomic(args.output, result)
        elif args.command == 'verify-completion':
            from .completion_verification import verify_completion
            if Path(args.input).stat().st_size > 128 * 1024:
                raise ValueError('completion input file byte budget exceeded')
            if args.output and Path(args.output).exists():
                raise ValueError('completion output must be new')
            arguments = _read(args.input)
            def native_factory():
                if args.state_json_runtime:
                    from .native_completion import build_state_json_completion_judge
                    return build_state_json_completion_judge(_read_state_json_config(args.state_json_runtime))
                from .native_completion import build_native_completion_judge
                return build_native_completion_judge(_read(args.native_runtime))
            result = verify_completion(arguments,
                judge_factory=native_factory if (args.native_runtime or args.state_json_runtime) else None)
            if args.output:
                _write_new_atomic(args.output, result)
        elif args.command == 'advise-model-route':
            from .model_routing import advise_model_route
            if Path(args.input).stat().st_size > 128 * 1024:
                raise ValueError('model routing input file byte budget exceeded')
            if args.output and Path(args.output).exists():
                raise ValueError('model routing output must be new')
            arguments = _read(args.input)
            def model_routing_factory():
                if args.state_json_runtime:
                    from .native_model_routing import build_state_json_model_routing_judge
                    return build_state_json_model_routing_judge(_read_state_json_config(args.state_json_runtime))
                if Path(args.native_runtime).stat().st_size > 64 * 1024:
                    raise ValueError('model routing runtime config byte budget exceeded')
                from .native_model_routing import build_native_model_routing_judge
                return build_native_model_routing_judge(_read(args.native_runtime))
            result = advise_model_route(arguments,
                judge_factory=model_routing_factory if (args.native_runtime or args.state_json_runtime) else None)
            if args.output:
                _write_new_atomic(args.output, result)
        elif args.command == 'navigate-repository':
            from .repository_navigation import navigate_repository
            if Path(args.input).stat().st_size > 64 * 1024:
                raise ValueError('repository navigation input file byte budget exceeded')
            if args.output and Path(args.output).exists():
                raise ValueError('repository navigation output must be new')
            arguments = _read(args.input)
            def repository_navigation_factory():
                if args.state_json_runtime:
                    from .native_repository_navigation import build_state_json_repository_navigation_judge
                    return build_state_json_repository_navigation_judge(_read_state_json_config(args.state_json_runtime))
                if Path(args.native_runtime).stat().st_size > 64 * 1024:
                    raise ValueError('repository navigation runtime config byte budget exceeded')
                from .native_repository_navigation import build_native_repository_navigation_judge
                return build_native_repository_navigation_judge(_read(args.native_runtime))
            result = navigate_repository(arguments,
                judge_factory=repository_navigation_factory if (args.native_runtime or args.state_json_runtime) else None)
            if args.output:
                _write_new_atomic(args.output, result)
        elif args.command in ("build-qualification", "verify-qualification"):
            from .qualification import build_receipt, verify_receipt
            evidence = {key: getattr(args, key) for key in ("runtime_fingerprint", "calibration_path",
                "evaluation_path", "policy_path", "dataset_dir", "parity_path", "golden_path", "bundle_manifest_path")}
            if args.command == "build-qualification":
                result = build_receipt(**evidence)
                _write_new(args.receipt, result)
            else:
                result = {"status": "PASS" if verify_receipt(_read(args.receipt), **evidence) else "FAIL",
                          "action_authorized": False}
        else:
            from . import context_tools
            arguments = _read(args.input)
            if not isinstance(arguments, dict) or any(key.endswith("callback") for key in arguments):
                raise ValueError("CLI context input must be data only, without executable callbacks")
            shadow_requested = any(value is not None for value in
                                   (args.shadow_runtime, args.shadow_output, args.shadow_timeout))
            if shadow_requested:
                if (not args.shadow_runtime or not args.shadow_output or args.native_runtime or
                        args.tool not in ('rank_evidence', 'suggest_skills')):
                    raise ValueError('passive shadow requires ranking/skills, runtime and separate output')
                return (_passive_rank(args, arguments) if args.tool == 'rank_evidence'
                        else _passive_skills(args, arguments))
            if args.native_runtime:
                if args.tool not in ("rank_evidence", "suggest_skills"):
                    raise ValueError("native runtime is only supported for ranking/skills")
                from .native_context import build_native_scorer, invoke_context
                scorer = build_native_scorer(_read(args.native_runtime))
                result = invoke_context(args.tool, arguments, scorer)
            else:
                result = getattr(context_tools, args.tool)(**arguments)
            if args.output:
                _write_new(args.output, result)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
        return 2 if isinstance(result, dict) and result.get("status") in ("FAIL", "requires_larger_budget") else 0
    except (ValueError, TypeError, OSError, KeyError, ModelRouteError) as exc:
        # Do not echo data, field values, credentials, or source text on errors.
        print(json.dumps({"status": "rejected", "error_type": type(exc).__name__, "execution_authorized": False}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
