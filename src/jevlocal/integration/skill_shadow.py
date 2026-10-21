"""Passive relevance observation of explicit, caller-declared skill descriptions.

This is not skill discovery, capability validation, availability, or permission.
Only the separately published lexical suggestion is the host primary.
"""
from __future__ import annotations

import copy

from . import context_tools
from .evidence_scope import _digest, _json, candidate_digest
from .scoped_invocation import prepare_invocation, validate_snapshot


_FLAGS = {"execution_authorized": False, "training_allowed": False,
          "origin_authenticated": False}


def verify_skill_primary(task, safe_skills, primary, *, max_chars):
    """Mechanically check the skill primary, independently of projection feedback."""
    # The bounded Hy4 text-only helper supplied the recomputation/canonical
    # comparison pattern; local review uses the project's strict JSON codec.
    expected = context_tools._suggest_prepared_skills(
        task, safe_skills, max_chars=max_chars)
    try:
        same = _json(primary) == _json(expected)
    except (ValueError, TypeError, OverflowError, RecursionError):
        same = False
    if not same:
        return {"status": "failed", "reasons": ["primary_mismatch"],
                "semantic_relevance": "unknown"}
    if expected["status"] == "requires_larger_budget":
        return {"status": "not_available", "reasons": ["primary_budget"],
                "semantic_relevance": "unknown"}
    return {"status": "verified", "reasons": [],
            "semantic_relevance": "unknown"}


def prepare_skill_projection(task, safe_skills):
    """Bind every sanitized declared description to one formal scoped invocation."""
    if not safe_skills:
        return None
    snapshot = {"task": task, "enabled_skills": safe_skills}
    revision = _digest(snapshot)
    target = {"object_id": "caller_declared_skill_descriptions",
              "revision": revision}
    candidates = [{"id": item["id"], "source": "caller_declared_skill_description",
                   "start_line": 1, "text": item["description"]}
                  for item in safe_skills]
    bindings = {candidate["id"]: {**target,
                "candidate_sha256": candidate_digest(candidate)}
                for candidate in candidates}
    record = prepare_invocation({"query": task, "candidates": candidates,
        "target": target, "bindings": bindings, "max_chars": 4 * 1024 * 1024,
        "rank_max_chars": 1024 * 1024})
    if record.get("status") == "requires_larger_budget":
        raise ValueError("skill projection exceeds scope budget")
    derived = record["rank_input"]
    if derived is None:
        raise ValueError("nonempty skill projection has no candidates")
    query, safe = context_tools._prepare_rank_evidence(**derived)
    validate_snapshot(record["binding"], record["manifest"], derived, query, safe)
    return record, query, safe


def finish_skill_shadow(job, task, safe_skills, primary, *, max_chars,
                        projection=None):
    """Finish the owned observer; label nested evidence feedback projection-only."""
    input_sha256 = _digest({"task": task, "enabled_skills": safe_skills})
    primary_feedback = verify_skill_primary(task, safe_skills, primary,
                                            max_chars=max_chars)
    result = {"version": 1, "adapter": "jevlocal-skill-description-shadow-v1",
              "purpose": "caller_declared_skill_description_relevance_only",
              "input_sha256": input_sha256,
              "closed_set_sha256": _digest(safe_skills),
              "primary_result_sha256": _digest(primary),
              "primary_feedback": primary_feedback,
              "host_ranking": copy.deepcopy(primary.get("suggestions", [])),
              "host_top": (primary["suggestions"][0]["id"]
                           if primary.get("suggestions") else None),
              "projection_feedback_scope": "projection_only_not_skill_primary",
              "availability": {"status": "unavailable", "observed_count": 0,
                               "candidate_count": len(safe_skills),
                               "reason": "capacity" if safe_skills else "no_candidates"},
              "projection": None, "shadow": True, "cloud_data_used": False,
              **_FLAGS}
    if projection is None:
        return result
    record, query, safe = projection
    projected_primary = context_tools._rank_prepared_evidence(
        query, safe, max_chars=record["rank_input"]["max_chars"])
    nested = job.finish(projected_primary)
    result["availability"] = copy.deepcopy(nested["availability"])
    result["projection"] = {
        "binding": copy.deepcopy(record["binding"]),
        "manifest_sha256": record["binding"]["manifest_sha256"],
        "rank_input_sha256": record["binding"]["rank_input_sha256"],
        "projected_primary_sha256": _digest(projected_primary),
        "feedback_scope": "projection_only_not_skill_primary",
        "context_shadow": nested,
    }
    return result
