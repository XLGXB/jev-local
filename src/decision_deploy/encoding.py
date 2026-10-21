"""Tokenization uses the pinned native training formatter; no second prompt template."""
from __future__ import annotations
import inspect
from pathlib import Path
import numpy as np
from .common import read, sha
from .identity import decision_source_hashes

def padding(kind, encoded, sequence, candidates, pad_id):
    ids, marks = encoded if kind == "laya" else (encoded["ids"], encoded["opt_idx"][0])
    if len(ids) > sequence or len(marks) > candidates: raise ValueError("profile exceeded; no truncation")
    x = np.full((1, sequence), pad_id, dtype=np.int64)
    att = np.zeros((1, sequence), dtype=np.int64)
    x[0, :len(ids)] = ids; att[0, :len(ids)] = 1
    pos = np.zeros((1, candidates), dtype=np.int64)
    mask = np.zeros((1, candidates), dtype=np.bool_)
    pos[0, :len(marks)] = marks; mask[0, :len(marks)] = True
    if min(marks) < 0 or max(marks) >= len(ids): raise ValueError("invalid candidate position")
    if kind == "laya": return {"input_ids": x, "attention_mask": att, "marker_pos": pos, "marker_mask": mask}
    if len(encoded["decide_idx"]) != 1: raise ValueError("one causal question per row required")
    pids = np.zeros((1, sequence), dtype=np.int64); pids[0, :len(ids)] = encoded["pos"]
    return {"input_ids": x, "attention_mask": att, "position_ids": pids,
            "decide_idx": np.array([encoded["decide_idx"]], dtype=np.int64),
            "option_idx": pos, "option_mask": mask}

class Frontend:
    """Loads tokenizer/config ONLY, not the base model or any decision weights."""
    def __init__(self, root, kind, option_isolation=False):
        from decision_ft.backends import enforce_offline
        enforce_offline()
        from transformers import AutoTokenizer
        self.root, self.kind = Path(root), kind
        self.tok = AutoTokenizer.from_pretrained(str(self.root / "tokenizer"), local_files_only=True, trust_remote_code=False)
        if self.tok.pad_token_id is None: raise ValueError("pad_token_id required")
        self.sources = {}
        if kind == "laya":
            import laya.common as common
            import decision_ft.backends as b
            # v1's strict encoder is reused without calling its weight-loading constructor.
            self.proxy = b.LayaBackend.__new__(b.LayaBackend)
            b.Backend.__init__(self.proxy)
            self.proxy.cfg = read(self.root / "rl_agent_config.json")
            self.proxy.tok, self.proxy.common = self.tok, common
            self.sources = {"laya_common": sha(inspect.getfile(common)), "training_backends": sha(inspect.getfile(b))}
        else:
            if option_isolation: raise ValueError("option isolation is native-only in v1 deployment")
            import kev.model as model
            import kev.data as data
            self.kmodel, self.kdata = model, data
            self.sources = {"kev_model": sha(inspect.getfile(model)), "kev_data": sha(inspect.getfile(data))}
        self.sources.update(decision_source_hashes())
    def encode(self, atom):
        if self.kind == "laya":
            try:
                return self.proxy.encode(atom)
            finally:
                self.proxy.cache.clear()  # also discard observations after failed encoding
        q = dict(atom.question)
        q["label"] = atom.keys[0] if q["type"] == "choice" else False if q["type"] == "noul" else 0
        q["src"] = "local"
        rec = self.kdata.materialize({"state": atom.state, "questions": {"q": q}})
        if rec["questions"][0]["keys"] != atom.keys: raise ValueError("candidate mapping changed")
        enc = self.kmodel.encode(self.tok, rec, strict=True, option_isolation=False)
        enc.pop("labels", None)
        return enc
    def tensors(self, atom, buckets, candidates):
        enc = self.encode(atom)
        n = len(enc[0] if self.kind == "laya" else enc["ids"])
        bucket = next((b for b in sorted(buckets) if b >= n), None)
        if bucket is None: raise ValueError("context budget exceeded; ask host for a shorter authorized projection")
        data = padding(self.kind, enc, bucket, candidates, self.tok.pad_token_id)
        if self.kind == "laya": data["qtype"] = np.array([atom.kind], dtype=np.int64)
        return bucket, data
