"""Thin adapters around native Laya/Kev encoders and heads.
Native integration is version-sensitive: inspect recorded source hashes and run
one-batch/reload parity tests on the locally pinned upstream revisions.
"""
from __future__ import annotations
import copy
import hashlib
import inspect
import json
import os
from dataclasses import replace
from pathlib import Path
import torch
from torch import nn
from .data import Atom, file_sha, write_json


def enforce_offline() -> None:
    # Models must already exist locally/in the HF cache. No local data upload.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["WANDB_DISABLED"] = "true"


class Backend:
    model: nn.Module
    name: str
    source_files: list[str]

    def __init__(self):
        self.cache = {}

    def train_groups(self, backbone_lr: float, head_lr: float) -> list[dict]:
        head, backbone = [], []
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                (backbone if name.startswith(("encoder.", "lm.")) else head).append(p)
        return [g for g in ({"params": backbone, "lr": backbone_lr},
                             {"params": head, "lr": head_lr}) if g["params"]]

    def source_manifest(self):
        return {str(p): file_sha(p) for p in self.source_files}


class LayaBackend(Backend):
    name = "laya"

    def __init__(self, checkpoint: str, device: str, weights_dtype: str = "fp32"):
        super().__init__()
        enforce_offline()
        from transformers import AutoTokenizer
        from safetensors.torch import load_file
        import laya.common as common
        root = Path(checkpoint).resolve()
        for rel in ("encoder/config.json", "rl_agent_config.json", "model.safetensors"):
            if not (root / rel).is_file():
                raise FileNotFoundError(f"Laya checkpoint missing {root / rel}")
        self.cfg = json.loads((root / "rl_agent_config.json").read_text(encoding="utf-8"))
        self.tok = AutoTokenizer.from_pretrained(str(root / "tokenizer"), local_files_only=True,
                                                 trust_remote_code=False)
        self.common = common
        self.model = common.build_model(self.cfg, encoder_dir=str(root / "encoder"))
        self.model.load_state_dict(load_file(str(root / "model.safetensors")), strict=True)
        # Match pinned Laya's small-batch path; avoid implicit torch.compile at
        # first inference. This changes no weights or candidate semantics.
        if hasattr(self.model.encoder.config, "reference_compile"):
            self.model.encoder.config.reference_compile = False
        # The auxiliary action head is NOT trained and MUST NOT become a safety gate.
        for parameter in self.model.act_head.parameters():
            parameter.requires_grad_(False)
        self.model.temperature.fill_(1.0)
        self.model.to(device)
        self.device = device
        self.source_files = [inspect.getfile(common)]
        self.provenance = {"init": str(root), "model_sha256": file_sha(root / "model.safetensors"),
                           "config_sha256": file_sha(root / "rl_agent_config.json"),
                           "encoder": self.cfg.get("encoder"), "raw_temperature": 1.0,
                           "reference_compile": getattr(self.model.encoder.config, "reference_compile", None)}
        if weights_dtype != "fp32":
            raise ValueError("Laya path keeps FP32 master weights; choose --amp bf16 for CUDA autocast")

    def encode(self, atom: Atom):
        cfg = self.cfg
        full_options = "option_max_len" in cfg
        if full_options:
            for key in ("option_max_len", "head_max_len", "max_len"):
                value = cfg.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"Laya {key} must be a positive integer")
        if atom.fingerprint in self.cache:
            return self.cache[atom.fingerprint]
        q = {"t": atom.question["type"], "ins": atom.question["instructions"],
             "crit": atom.question.get("criteria")}
        # Native build_sequence can truncate silently. Reject every such case here.
        tok = self.tok
        max_len, head_limit = (cfg["max_len"], cfg["head_max_len"]) if full_options else (int(cfg["max_len"]), int(cfg["head_max_len"]))
        mask_token = tok.mask_token
        if mask_token is None:
            raise ValueError("Laya tokenizer has no mask token")
        clean = lambda text: text.replace(mask_token, " ")
        options = self.common.render_options(q)
        option_ids = [tok(" " + clean(o), add_special_tokens=False)["input_ids"] for o in options]
        option_lens = [len(ids) for ids in option_ids]
        if any(length > (cfg["option_max_len"] if full_options else 48) for length in option_lens):
            if full_options:
                raise ValueError(f"{atom.uid}: option exceeds option_max_len {cfg['option_max_len']}")
            raise ValueError(f"{atom.uid}: native Laya would truncate an option to 48 tokens")
        options_total = sum(length + 1 for length in option_lens)
        instruction_ids = tok(f"{q['t']} question: {clean(q['ins'])}", add_special_tokens=False)["input_ids"]
        instruction_len = len(instruction_ids)
        budget = head_limit - options_total
        if budget < 16 or instruction_len > max(8, budget):
            if full_options:
                raise ValueError(f"{atom.uid}: head/options exceed head_max_len {head_limit}")
            raise ValueError(f"{atom.uid}: native Laya would truncate head/options; reduce the candidate set")
        state_text = self.common.serialize_state(atom.state)
        state_ids = tok(clean(state_text), add_special_tokens=False)["input_ids"]
        state_len = len(state_ids)
        expected_len = 4 + instruction_len + options_total + state_len
        if expected_len > max_len:
            if full_options:
                raise ValueError(f"{atom.uid}: {expected_len} tokens exceed max_len {max_len}")
            raise ValueError(f"{atom.uid}: {expected_len} tokens exceed checkpoint budget {max_len}; create an evidence-preserving projection")
        if full_options:
            ids = [tok.cls_token_id] + instruction_ids + [tok.sep_token_id]
            markers = []
            for option in option_ids:
                markers.append(len(ids))
                ids.extend([tok.mask_token_id] + option)
            ids.extend([tok.sep_token_id] + state_ids + [tok.sep_token_id])
        else:
            ids, markers = self.common.build_sequence(tok, atom.state, q, max_len, head_limit)
        if len(markers) != len(atom.keys) or len(ids) != expected_len:
            raise ValueError("native encoding contract changed; inspect pinned Laya version")
        self.cache[atom.fingerprint] = (ids, markers)
        return ids, markers

    def forward(self, atoms: list[Atom]) -> list[torch.Tensor]:
        rows = [self.encode(a) for a in atoms]
        n, length, kmax = len(rows), max(len(x[0]) for x in rows), max(len(x[1]) for x in rows)
        pad = self.tok.pad_token_id
        if pad is None:
            raise ValueError("missing tokenizer pad_token_id")
        ids = torch.full((n, length), pad, dtype=torch.long, device=self.device)
        att = torch.zeros_like(ids)
        pos = torch.zeros((n, kmax), dtype=torch.long, device=self.device)
        mask = torch.zeros((n, kmax), dtype=torch.bool, device=self.device)
        for i, (tokens, markers) in enumerate(rows):
            ids[i, :len(tokens)] = torch.tensor(tokens, device=self.device)
            att[i, :len(tokens)] = 1
            pos[i, :len(markers)] = torch.tensor(markers, device=self.device)
            mask[i, :len(markers)] = True
        kind = torch.tensor([a.kind for a in atoms], dtype=torch.long, device=self.device)
        logits, _ = self.model(ids, att, pos, mask, kind)
        return [logits[i, :len(a.keys)].float() for i, a in enumerate(atoms)]

    def export(self, directory: Path):
        from safetensors.torch import save_file
        directory.mkdir(parents=True, exist_ok=False)
        state = {k: v.detach().cpu().contiguous().clone() for k, v in self.model.state_dict().items()}
        save_file(state, str(directory / "model.safetensors"))
        self.model.encoder.config.save_pretrained(str(directory / "encoder"))
        self.tok.save_pretrained(str(directory / "tokenizer"))
        cfg = copy.deepcopy(self.cfg)
        cfg["temperature"] = [1.0, 1.0, 1.0]
        cfg["local_training_note"] = "Raw logits; calibration is a separately hashed sidecar. Auxiliary act_head was frozen."
        write_json(directory / "rl_agent_config.json", cfg)


class KevBackend(Backend):
    name = "kev"

    def __init__(self, checkpoint: str, device: str, weights_dtype: str = "fp32"):
        super().__init__()
        enforce_offline()
        import kev.checkpoint as checkpoints
        import kev.model as models
        import kev.data as data
        root = Path(checkpoint).resolve()
        if not root.is_dir():
            raise FileNotFoundError("--init must be a local Kev checkpoint directory")
        ck = checkpoints.Checkpoint(str(root))
        self.meta = replace(ck.meta, head=None, temperature=1.0, weights_dtype=weights_dtype,
                            extra=copy.deepcopy(ck.meta.extra))
        if not self.meta.lora:
            raise ValueError("expected a native trained Kev LoRA checkpoint, not a plain Qwen base")
        self.tok = models.load_tokenizer(self.meta.base, revision=self.meta.base_revision)
        targets = self.meta.extra.get("args", {}).get("lora_targets", "all")
        self.model = models.DecisionModel(
            self.meta.base, self.tok, device, lora=self.meta.lora,
            revision=self.meta.base_revision, head_dim=self.meta.head_dim,
            option_isolation=self.meta.option_isolation,
            special_embeddings=self.meta.special_embeddings, lora_targets=targets,
            dtype=torch.bfloat16 if weights_dtype == "bf16" else torch.float32)
        self.provenance = ck.warm_start(self.model, self.meta)
        self.provenance["raw_temperature"] = 1.0
        self.model.head.temperature = 1.0
        self.model.lm.config.use_cache = False
        self.device, self.data, self.checkpoints = device, data, checkpoints
        self.source_files = [inspect.getfile(module) for module in (checkpoints, models, data)]
        self.provenance["local_files_sha256"] = {p.name: file_sha(p) for p in root.iterdir()
                                               if p.is_file() and p.suffix in (".pt", ".safetensors", ".json")}

    def encode(self, atom: Atom):
        if atom.fingerprint in self.cache:
            return self.cache[atom.fingerprint]
        q = copy.deepcopy(atom.question)
        # A dummy label is required by the native data parser, but is never tokenized.
        # Real supervision is held outside this path entirely.
        q["label"] = atom.keys[0] if q["type"] == "choice" else (False if q["type"] == "noul" else 0)
        q["src"] = "local"
        record = self.data.materialize({"state": atom.state, "questions": {"q": q}})
        if record["questions"][0]["keys"] != atom.keys:
            raise ValueError("Kev option order contract mismatch")
        encoded = self.model.encode(self.tok, record, strict=True)
        encoded.pop("labels", None)
        if encoded.get("state_truncated"):
            raise ValueError("truncated evidence is not permitted")
        self.cache[atom.fingerprint] = encoded
        return encoded

    def forward(self, atoms: list[Atom]) -> list[torch.Tensor]:
        nested = self.model.forward_batch([self.encode(a) for a in atoms])
        return [row[0].float() for row in nested]

    def export(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=False)
        self.model.lm.save_pretrained(str(directory))
        meta = replace(self.meta,
                       head={k: v.detach().cpu() for k, v in self.model.head.state_dict().items()},
                       temperature=1.0, extra={**self.meta.extra, "local_init_source": self.provenance})
        self.checkpoints.write_meta(str(directory), meta)
        self.tok.save_pretrained(str(directory))


class ToyBackend(Backend):
    """Synthetic contract smoke ONLY. This is neither ModernBERT nor Qwen."""
    name = "toy"

    def __init__(self, checkpoint: str, device: str, weights_dtype: str = "fp32"):
        super().__init__()
        self.device = device
        self.model = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 1)).to(device)
        if checkpoint != "toy-init":
            self.model.load_state_dict(torch.load(Path(checkpoint) / "toy.pt", map_location=device, weights_only=True))
        self.provenance = {"init": checkpoint, "synthetic_only": True}
        self.source_files = [__file__]

    def encode(self, atom: Atom):
        if atom.fingerprint not in self.cache:
            rows = []
            for key in atom.keys:
                desc = atom.question.get("criteria", {})
                text = json.dumps([atom.state, atom.question["instructions"], key, desc], ensure_ascii=False)
                values = [0.] * 64
                for start in range(max(1, len(text) - 2)):
                    index = int.from_bytes(hashlib.blake2b(text[start:start+3].encode(), digest_size=2).digest(), "big") % 64
                    values[index] += 1. / max(1, len(text))
                rows.append(values)
            self.cache[atom.fingerprint] = torch.tensor(rows, device=self.device)
        return self.cache[atom.fingerprint]

    def forward(self, atoms: list[Atom]) -> list[torch.Tensor]:
        return [self.model(self.encode(a)).squeeze(-1) for a in atoms]

    def export(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=False)
        torch.save(self.model.state_dict(), directory / "toy.pt")


def load_backend(name: str, checkpoint: str, device: str, weights_dtype: str = "fp32") -> Backend:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if weights_dtype == "bf16" and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 weights require supported CUDA hardware in this scaffold")
    classes = {"laya": LayaBackend, "kev": KevBackend, "toy": ToyBackend}
    return classes[name](checkpoint, device, weights_dtype)
