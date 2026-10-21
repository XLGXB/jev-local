"""Deterministic local evidence gate, not an execution grant.

The caller must choose files in a trusted local directory. Hashes detect changes
between receipt creation and verification; they do not authenticate a writer who
can replace the evidence, policy and receipt together. Model execution and
permission approval remain outside this module.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from decision_ft.data import VERIFIED, atomize, verify_dataset


KINDS = {"choice": "0", "score": "1", "noul": "2"}
POLICY_FIELDS = {
    "schema_version", "required_types", "min_calibration_per_type",
    "min_evaluation_per_type", "min_evaluation_total", "min_action_decisions",
    "min_accepted_actions", "min_accuracy", "max_nll_target",
    "max_brier_target", "max_ece_verified", "max_rps_target",
    "max_score_mean_mae_target", "action_confidence_threshold",
    "max_action_error_rate", "min_action_coverage", "max_parity_logit_abs",
    "max_parity_probability_abs", "min_parity_top1", "min_parity_decisions",
}


class _Evidence(dict):
    """A dict read and byte-hashed by this module, not a caller's assertion."""

    def __init__(self, value: dict, path: Path, sha256: str):
        super().__init__(value)
        self.path = path
        self.sha256 = sha256
        self.dataset_manifest_sha256: str | None = None
        self.actual_counts: Counter[str] | None = None
        self.actual_action_decisions: int | None = None
        self.same_checkpoint = False


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest(value: dict) -> str:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value)


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate evidence key")
        value[key] = item
    return value


def _read(path: str | Path) -> _Evidence:
    source = Path(path).resolve()
    raw = source.read_bytes()
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                       parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {x}")))
    if not isinstance(value, dict):
        raise ValueError("evidence must be a JSON object")
    if not _finite_tree(value):
        raise ValueError("nonfinite evidence number")
    return _Evidence(value, source, hashlib.sha256(raw).hexdigest())


def _finite_tree(value: Any) -> bool:
    if type(value) is float:
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return True


def _finite_number(value: Any) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _number(value: Any, name: str, reasons: list[str], *, low: float | None = None,
            high: float | None = None) -> float | None:
    if not _finite_number(value):
        reasons.append(f"{name}: finite number required")
        return None
    try:
        number = float(value)
    except OverflowError:
        reasons.append(f"{name}: finite number required")
        return None
    if (low is not None and number < low) or (high is not None and number > high):
        reasons.append(f"{name}: outside permitted range")
        return None
    return number


def _count(value: Any, name: str, reasons: list[str], *, positive: bool = False) -> int | None:
    if type(value) is not int or value < (1 if positive else 0):
        reasons.append(f"{name}: nonnegative integer required")
        return None
    return value


def inspect_qualification(*, runtime_fingerprint: str, calibration: dict,
                          evaluation: dict, policy: dict,
                          parity: dict | None = None) -> dict:
    """Inspect already file-verified evidence; plain dicts cannot yield PASS.

    ``build_receipt`` supplies file-backed inputs and independently checks the
    frozen dataset. A standalone caller may use this for diagnostics only.
    """
    reasons: list[str] = []
    if not _is_sha256(runtime_fingerprint):
        reasons.append("runtime_fingerprint: lowercase SHA-256 required")
    if not all(isinstance(x, _Evidence) for x in (calibration, evaluation, policy)):
        reasons.append("untrusted evidence source: load real files through build_receipt")
    if not all(isinstance(x, dict) for x in (calibration, evaluation, policy)):
        return {"status": "FAIL", "reasons": reasons + ["invalid evidence object"],
                "runtime_fingerprint": runtime_fingerprint, "action_authorized": False}
    if not all(_finite_tree(x) for x in (calibration, evaluation, policy, parity)):
        reasons.append("nonfinite evidence number")
    if set(policy) != POLICY_FIELDS or policy.get("schema_version") != 1:
        reasons.append("policy: explicit versioned thresholds required")
    required = policy.get("required_types")
    if not isinstance(required, list) or not required or not all(isinstance(kind, str) for kind in required) or (
            len(required) != len(set(required))) or any(kind not in KINDS for kind in required):
        reasons.append("policy.required_types: nonempty distinct known types required")
        required = []
    # A policy may require a subset, but every present type must also have a fit.
    for name in ("min_calibration_per_type", "min_evaluation_per_type", "min_evaluation_total",
                 "min_action_decisions", "min_accepted_actions", "min_parity_decisions"):
        _count(policy.get(name), f"policy.{name}", reasons, positive=True)
    for name in ("min_accuracy", "max_ece_verified", "max_action_error_rate",
                 "min_action_coverage", "min_parity_top1"):
        _number(policy.get(name), f"policy.{name}", reasons, low=0, high=1)
    for name in ("max_nll_target", "max_brier_target", "max_rps_target",
                 "max_score_mean_mae_target", "max_parity_logit_abs",
                 "max_parity_probability_abs"):
        _number(policy.get(name), f"policy.{name}", reasons, low=0)
    threshold = _number(policy.get("action_confidence_threshold"),
                        "policy.action_confidence_threshold", reasons, low=0, high=1)
    if threshold is not None and threshold == 0:
        reasons.append("action confidence gate must be positive")
    if calibration.get("stage") != "calibrate" or evaluation.get("stage") != "evaluate":
        reasons.append("calibration/evaluation stage mismatch")
    if evaluation.get("partition") != "test":
        reasons.append("held-out test partition required")
    if calibration.get("runtime_fingerprint") != runtime_fingerprint or evaluation.get(
            "runtime_fingerprint") != runtime_fingerprint:
        reasons.append("runtime fingerprint mismatch")
    identity = evaluation.get("runtime_identity")
    if not isinstance(identity, dict) or calibration.get("runtime_identity") != identity:
        reasons.append("runtime identity mismatch")
        identity = {}
    try:
        if _digest(identity) != runtime_fingerprint:
            reasons.append("runtime fingerprint does not hash runtime identity")
    except (TypeError, ValueError, OverflowError):
        reasons.append("runtime identity cannot be hashed")
    mode = identity.get("spec", {}).get("mode") if isinstance(identity.get("spec"), dict) else None
    if mode not in ("native", "ort"):
        reasons.append("unknown runtime mode")
    manifest_hash = calibration.get("dataset_manifest_sha256")
    if not isinstance(manifest_hash, str) or manifest_hash != evaluation.get("dataset_manifest_sha256"):
        reasons.append("dataset manifest mismatch")
    if not isinstance(calibration, _Evidence) or not isinstance(evaluation, _Evidence) or (
            calibration.dataset_manifest_sha256 != manifest_hash or
            evaluation.dataset_manifest_sha256 != manifest_hash):
        reasons.append("dataset manifest lacks file verification")
    if not isinstance(calibration, _Evidence) or evaluation.get("calibration_sha256") != calibration.sha256:
        reasons.append("evaluation is not bound to calibration file bytes")
    if not isinstance(calibration, _Evidence) or not isinstance(evaluation, _Evidence) or not isinstance(
            policy, _Evidence) or any(_sha(x.path) != x.sha256 for x in (calibration, evaluation, policy)
                               if isinstance(x, _Evidence)):
        reasons.append("evidence bytes changed")

    fits = calibration.get("fits")
    temps = calibration.get("temperatures")
    if not isinstance(fits, dict) or not isinstance(temps, dict):
        reasons.append("calibration fits/temperatures missing")
        fits, temps = {}, {}
    metrics = evaluation.get("calibrated")
    if not isinstance(metrics, dict):
        reasons.append("calibrated held-out metrics missing")
        metrics = {}
    if not isinstance(evaluation.get("raw"), dict):
        reasons.append("raw metrics missing")
    actual_cal = calibration.actual_counts if isinstance(calibration, _Evidence) else None
    actual_eval = evaluation.actual_counts if isinstance(evaluation, _Evidence) else None
    if actual_cal is None or actual_eval is None:
        reasons.append("verified dataset counts missing")
        actual_cal, actual_eval = Counter(), Counter()
    all_entry = metrics.get("all")
    if not isinstance(all_entry, dict):
        reasons.append("calibrated all-group metrics missing")
        all_entry = {}
    total = sum(actual_eval.values())
    if _count(all_entry.get("n"), "calibrated.all.n", reasons) != total or _count(
            all_entry.get("n_verified"), "calibrated.all.n_verified", reasons) != total:
        reasons.append("calibrated all-group count differs from verified test data")
    if type(policy.get("min_evaluation_total")) is int and total < policy["min_evaluation_total"]:
        reasons.append("too few held-out decisions")
    for kind in required:
        code = KINDS[kind]
        fit = fits.get(code)
        if not isinstance(fit, dict) or fit.get("status") != "fitted":
            reasons.append(f"{kind}: calibration fit missing")
            fit = {}
        fit_n = _count(fit.get("n"), f"{kind}.fit.n", reasons)
        expected_cal = actual_cal[kind]
        if fit_n != expected_cal or (type(policy.get("min_calibration_per_type")) is int and
                                     expected_cal < policy["min_calibration_per_type"]):
            reasons.append(f"{kind}: calibration sample count insufficient/mismatched")
        temperature = _number(temps.get(code), f"{kind}.temperature", reasons, low=0)
        if temperature == 0:
            reasons.append(f"{kind}: positive temperature required")
        entry = metrics.get(f"type:{kind}")
        if not isinstance(entry, dict):
            reasons.append(f"{kind}: calibrated type metrics missing")
            entry = {}
        expected_eval = actual_eval[kind]
        if _count(entry.get("n"), f"{kind}.n", reasons) != expected_eval or _count(
                entry.get("n_verified"), f"{kind}.n_verified", reasons) != expected_eval or (
                type(policy.get("min_evaluation_per_type")) is int and
                expected_eval < policy["min_evaluation_per_type"]):
            reasons.append(f"{kind}: verified test sample count insufficient/mismatched")
        for metric, bound, lower in (
            ("accuracy_against_supplied_label", "min_accuracy", True),
            ("nll_target", "max_nll_target", False),
            ("brier_target", "max_brier_target", False),
            ("ece_verified_15_bins", "max_ece_verified", False),
        ):
            value = _number(entry.get(metric), f"{kind}.{metric}", reasons, low=0,
                            high=1 if metric in ("accuracy_against_supplied_label", "ece_verified_15_bins") else None)
            limit = policy.get(bound)
            if value is not None and _finite_number(limit) and (
                    value < limit if lower else value > limit):
                reasons.append(f"{kind}.{metric} failed policy")
        if kind == "score":
            for metric, bound in (("rps_target", "max_rps_target"),
                                  ("score_mean_mae_target", "max_score_mean_mae_target")):
                value = _number(entry.get(metric), f"score.{metric}", reasons, low=0)
                limit = policy.get(bound)
                if value is not None and _finite_number(limit) and value > limit:
                    reasons.append(f"score.{metric} failed policy")
    # No present type may be uncalibrated, even if policy omitted it.
    for kind, count in actual_eval.items():
        if count and kind not in required:
            reasons.append(f"{kind}: present type omitted from policy")

    coverage = evaluation.get("action_coverage_excluding_needs_host")
    gate = coverage.get(str(threshold)) if isinstance(coverage, dict) else None
    if not isinstance(gate, dict):
        reasons.append("action confidence threshold absent from evaluated coverage")
        gate = {}
    decisions = _count(gate.get("decisions"), "action.decisions", reasons)
    accepted = _count(gate.get("accepted_actions"), "action.accepted_actions", reasons)
    expected_actions = evaluation.actual_action_decisions if isinstance(evaluation, _Evidence) else None
    if decisions != expected_actions or (type(policy.get("min_action_decisions")) is int and
                                              (decisions is None or decisions < policy["min_action_decisions"])):
        reasons.append("action decision pool insufficient/mismatched")
    if accepted is None or decisions is None or accepted > decisions or (
            type(policy.get("min_accepted_actions")) is int and
            accepted < policy["min_accepted_actions"]):
        reasons.append("accepted action count insufficient/invalid")
    action_coverage = _number(gate.get("coverage"), "action.coverage", reasons, low=0, high=1)
    action_error = _number(gate.get("action_error_rate"), "action.error_rate", reasons, low=0, high=1)
    if accepted is not None and decisions and action_coverage is not None and not math.isclose(
            action_coverage, accepted / decisions, abs_tol=1e-6):
        reasons.append("action coverage inconsistent with counts")
    if action_coverage is not None and type(policy.get("min_action_coverage")) in (int, float) and (
            action_coverage < policy["min_action_coverage"]):
        reasons.append("action coverage below policy")
    if action_error is not None and type(policy.get("max_action_error_rate")) in (int, float) and (
            action_error > policy["max_action_error_rate"]):
        reasons.append("action error rate above policy")

    if mode == "ort":
        if not isinstance(parity, _Evidence) or not parity.same_checkpoint:
            reasons.append("ORT same-checkpoint golden/bundle evidence missing")
        if not isinstance(parity, dict) or parity.get("stage") != "compare" or parity.get("status") != "PASS" or (
                parity.get("runtime_fingerprint") != runtime_fingerprint or
                parity.get("runtime_identity") != identity):
            reasons.append("ORT compare receipt mismatches runtime")
        decisions_list = parity.get("decisions") if isinstance(parity, dict) else None
        if not isinstance(decisions_list, list) or not decisions_list or (
                type(policy.get("min_parity_decisions")) is int and
                len(decisions_list) < policy["min_parity_decisions"]):
            reasons.append("ORT numeric comparison has too few decisions")
            decisions_list = []
        top_count = 0
        for index, item in enumerate(decisions_list):
            if not isinstance(item, dict) or type(item.get("same_top1")) is not bool:
                reasons.append(f"parity decision {index}: invalid item")
                continue
            top_count += item["same_top1"]
            for metric, bound in (("max_logit_abs", "max_parity_logit_abs"),
                                  ("max_probability_abs", "max_parity_probability_abs")):
                value = _number(item.get(metric), f"parity.{index}.{metric}", reasons, low=0)
                limit = policy.get(bound)
                if value is not None and _finite_number(limit) and value > limit:
                    reasons.append(f"parity.{index}.{metric} failed policy")
        if decisions_list:
            top1 = _number(parity.get("top1_agreement"), "parity.top1_agreement", reasons, low=0, high=1)
            observed = top_count / len(decisions_list)
            if top1 is None or not math.isclose(top1, observed, abs_tol=1e-9) or (
                    type(policy.get("min_parity_top1")) in (int, float) and
                    observed < policy["min_parity_top1"]):
                reasons.append("ORT top1 agreement failed policy")
    elif parity is not None:
        reasons.append("native runtime must not use export parity as qualification evidence")

    return {"status": "PASS" if not reasons else "FAIL", "reasons": reasons,
            "runtime_fingerprint": runtime_fingerprint, "action_authorized": False}


def build_receipt(*, runtime_fingerprint: str, calibration_path: str | Path,
                  evaluation_path: str | Path, policy_path: str | Path,
                  dataset_dir: str | Path, parity_path: str | Path | None = None,
                  golden_path: str | Path | None = None,
                  bundle_manifest_path: str | Path | None = None) -> dict:
    """Read local evidence, verify dataset files, and produce a quality receipt."""
    artifacts: dict[str, str] = {}
    policy_hash: str | None = None
    try:
        calibration = _read(calibration_path)
        evaluation = _read(evaluation_path)
        policy = _read(policy_path)
        manifest = _read(Path(dataset_dir) / "manifest.json")
        artifacts.update(calibration=calibration.sha256, evaluation=evaluation.sha256,
                         dataset_manifest=manifest.sha256)
        policy_hash = policy.sha256
        dataset = verify_dataset(dataset_dir, ("calibration", "test"))
        cal_atoms = atomize(dataset["calibration"])
        eval_atoms = atomize(dataset["test"])
        if any(atom.label_source not in VERIFIED for atom in cal_atoms + eval_atoms):
            raise ValueError("unverified calibration/test labels")
        calibration.actual_counts = Counter(atom.question["type"] for atom in cal_atoms)
        evaluation.actual_counts = Counter(atom.question["type"] for atom in eval_atoms)
        evaluation.actual_action_decisions = sum(atom.kind == 0 and "needs_host" in atom.keys
                                                 for atom in eval_atoms)
        calibration.dataset_manifest_sha256 = evaluation.dataset_manifest_sha256 = manifest.sha256
        if calibration.get("calibration_file_sha256") != _sha(Path(dataset_dir) / "calibration.jsonl"):
            raise ValueError("calibration partition bytes changed")
        parity = None
        eval_identity = evaluation.get("runtime_identity")
        eval_spec = eval_identity.get("spec") if isinstance(eval_identity, dict) else None
        mode = eval_spec.get("mode") if isinstance(eval_spec, dict) else None
        if mode == "ort":
            if not all((parity_path, golden_path, bundle_manifest_path)):
                raise ValueError("ORT requires parity, golden and bundle manifest files")
            parity = _read(parity_path)
            golden = _read(golden_path)
            bundle = _read(bundle_manifest_path)
            artifacts.update(parity=parity.sha256, golden=golden.sha256,
                             bundle_manifest=bundle.sha256)
            golden_identity = golden.get("runtime_identity")
            golden_artifact = golden_identity.get("artifact") if isinstance(golden_identity, dict) else None
            native_checkpoint = golden_artifact.get("checkpoint_sha256") if isinstance(golden_artifact, dict) else None
            bundle_checkpoint = bundle.get("checkpoint_sha256")
            parity_identity = parity.get("runtime_identity")
            parity_artifact = parity_identity.get("artifact") if isinstance(parity_identity, dict) else None
            golden_spec = golden_identity.get("spec") if isinstance(golden_identity, dict) else None
            samples = golden.get("samples")
            expected_decisions = None
            if isinstance(samples, list) and samples:
                try:
                    expected_decisions = sum(len(item["request"]["questions"]) for item in samples)
                except (TypeError, KeyError, AttributeError):
                    pass
            compared = parity.get("decisions")
            try:
                golden_fingerprint_matches = golden.get("runtime_fingerprint") == _digest(golden_identity)
            except (TypeError, ValueError, OverflowError):
                golden_fingerprint_matches = False
            parity.same_checkpoint = (golden.get("stage") == "golden" and
                isinstance(golden_spec, dict) and golden_spec.get("mode") == "native" and
                _is_sha256(native_checkpoint) and golden_fingerprint_matches and
                native_checkpoint == bundle_checkpoint and
                parity.get("golden_sha256") == golden.sha256 and
                isinstance(compared, list) and expected_decisions is not None and
                expected_decisions > 0 and len(compared) == expected_decisions and
                isinstance(parity_artifact, dict) and
                parity_artifact.get("bundle_manifest_sha256") == bundle.sha256)
        elif any((parity_path, golden_path, bundle_manifest_path)):
            raise ValueError("native receipt must not include export parity files")
        checks = inspect_qualification(runtime_fingerprint=runtime_fingerprint,
            calibration=calibration, evaluation=evaluation, policy=policy, parity=parity)
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError,
            OverflowError, UnicodeError) as exc:
        checks = {"status": "FAIL", "reasons": [f"evidence load/verification failed: {exc}"],
                  "runtime_fingerprint": runtime_fingerprint, "action_authorized": False}
    return {"schema": "jevlocal.qualification.receipt.v1", "schema_version": 1,
            "status": checks["status"],
            "runtime_fingerprint": runtime_fingerprint, "policy_sha256": policy_hash,
            "artifact_sha256": artifacts, "checks": checks, "action_authorized": False}


def verify_receipt(receipt: dict, **evidence_paths) -> bool:
    """Re-read every evidence file and require an exactly matching PASS receipt."""
    if not isinstance(receipt, dict) or receipt.get("status") != "PASS" or (
            receipt.get("action_authorized") is not False):
        return False
    expected = build_receipt(**evidence_paths)
    return expected["status"] == "PASS" and receipt == expected
