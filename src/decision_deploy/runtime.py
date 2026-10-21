from __future__ import annotations
import contextlib, inspect, time
from pathlib import Path
import numpy as np
from .common import atoms_for, digest, read, sha, versions, probabilities
from .identity import decision_source_hashes

class Runtime:
    def __init__(self, spec):
        self.spec = dict(spec); self.mode = spec["mode"]; self.backend = spec["backend"]
        if self.mode == "native":
            from decision_ft.backends import load_backend
            from decision_ft.train import model_digest, validate_amp
            self.device = spec.get("device", "cpu")
            validate_amp(self.device,spec.get("amp","none"))
            self.native = load_backend(self.backend, spec["checkpoint"], self.device, spec.get("weights_dtype", "fp32"))
            self.native.model.eval()
            identity = {"checkpoint_sha256": model_digest(spec["checkpoint"]), "sources": self.native.source_manifest()}
        elif self.mode == "ort":
            import onnxruntime as ort
            from .encoding import Frontend
            self.root = Path(spec["bundle"]).resolve(); self.manifest = read(self.root / "manifest.json")
            if self.manifest["backend"] != self.backend: raise ValueError("bundle backend mismatch")
            for rel, expected in self.manifest["files_sha256"].items():
                f = (self.root / rel).resolve()
                if not f.is_relative_to(self.root) or sha(f) != expected: raise ValueError("bundle integrity check failed")
            self.frontend = Frontend(self.root, self.backend, self.manifest["option_isolation"])
            if self.frontend.sources != self.manifest["frontend_source_sha256"]:
                raise ValueError("native tokenization source changed; re-qualify the export")
            self.buckets = spec.get("active_buckets", self.manifest["buckets"])
            if not self.buckets or any(x not in self.manifest["buckets"] for x in self.buckets): raise ValueError("invalid active_buckets")
            provider = spec.get("provider", "cpu")
            required = {"cpu":"CPUExecutionProvider", "cuda":"CUDAExecutionProvider", "trt":"TensorrtExecutionProvider"}[provider]
            if required not in ort.get_available_providers(): raise RuntimeError(f"required provider is absent: {required}")
            providers = []
            if provider == "trt":
                # This profile targets the ORT TensorRT-10.x EP option API. Pin the compatible stack.
                try:
                    import tensorrt
                    if int(tensorrt.__version__.split(".")[0]) != 10:
                        raise RuntimeError("This ORT EP profile is TensorRT 10.x only; do not mix with TensorRT 11.x")
                except ImportError:
                    pass  # ORT may load TensorRT libraries without its Python binding; verify shared-library versions externally.
                opt = {"device_id": 0, "trt_fp16_enable": spec.get("fp16", False),
                       "trt_engine_cache_enable": True, "trt_timing_cache_enable": True,
                       "trt_max_workspace_size": int(spec.get("workspace_bytes", 2 << 30))}
                cache = Path(spec.get("cache_dir", "./engine-cache")) / digest({"bundle": sha(self.root / "manifest.json"), "fp16":spec.get("fp16",False), "env":versions()})
                cache.mkdir(parents=True, exist_ok=True)
                opt.update(trt_engine_cache_path=str(cache), trt_timing_cache_path=str(cache))
                providers.append((required, opt))
            if provider in ("cuda", "trt"):
                if "CUDAExecutionProvider" not in ort.get_available_providers(): raise RuntimeError("CUDA EP absent")
                providers.append(("CUDAExecutionProvider", {"device_id":0, "use_tf32":0}))
            if provider == "cpu" or spec.get("allow_cpu_fallback", False): providers.append("CPUExecutionProvider")
            self.sessions = {}
            for seq in self.buckets:
                so = ort.SessionOptions(); so.intra_op_num_threads = int(spec.get("cpu_threads", 4))
                so.enable_profiling = bool(spec.get("profile", False))
                if so.enable_profiling:
                    Path(spec.get("profile_dir", "./profiles")).mkdir(parents=True, exist_ok=True)
                    so.profile_file_prefix = str(Path(spec.get("profile_dir", "./profiles")) / f"s{seq}")
                if provider != "cpu" and not spec.get("allow_cpu_fallback", False):
                    so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
                session = ort.InferenceSession(str(self.root / f"s{seq}" / "model.onnx"), sess_options=so, providers=providers)
                session.disable_fallback()  # runtime provider failure must surface, not silently recreate a CPU session
                if required not in session.get_providers(): raise RuntimeError("requested provider failed to register")
                self.sessions[seq] = session
            identity = {"bundle_manifest_sha256": sha(self.root / "manifest.json"), "sources":self.frontend.sources,
                        "registered_providers":{str(k):v.get_providers() for k,v in self.sessions.items()}}
        else: raise ValueError("mode must be native or ort")
        semantic_spec = {k:v for k,v in spec.items() if k not in ("bundle", "checkpoint", "cache_dir", "profile", "profile_dir")}
        code = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
        self.identity = {"spec":semantic_spec, "artifact":identity, "environment":versions(), "deployment_code_sha256":code,
                         "decision_source_sha256":decision_source_hashes(native_backend=self.mode == "native")}
        self.fingerprint = digest(self.identity)
    def logits(self, request):
        pairs = atoms_for(request); result = []
        if self.mode == "native":
            import torch
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.spec.get("amp", "none") == "bf16" else contextlib.nullcontext()
            try:
                with torch.inference_mode(), ctx:
                    zs = self.native.forward([atom for _,atom in pairs])
                if len(zs) != len(pairs): raise RuntimeError("native result count does not match the request")
                for (qid, atom), z in zip(pairs, zs): result.append((qid, atom, z.detach().float().cpu().numpy()))
            finally:
                self.native.cache.clear()
        else:
            for qid, atom in pairs:
                bucket, data = self.frontend.tensors(atom, self.buckets, self.manifest["candidates"])
                session = self.sessions[bucket]
                # Exporters may remove a redundant input in a specific fixed graph.
                names = {x.name for x in session.get_inputs()}
                if not names <= data.keys(): raise ValueError("ONNX input contract changed")
                raw = np.asarray(session.run(["logits"], {k:v for k,v in data.items() if k in names})[0])
                if raw.ndim != 2 or raw.shape != (1, self.manifest["candidates"]):
                    raise RuntimeError("ONNX output shape does not match the export profile")
                z = raw[0, :len(atom.keys)]
                result.append((qid, atom, np.asarray(z, dtype=np.float32)))
        for _, atom, z in result:
            if z.ndim != 1 or len(z) != len(atom.keys):
                raise RuntimeError("logit shape does not match the candidate set")
            if not np.isfinite(z).all(): raise RuntimeError("nonfinite model output")
        return result
    def close_profiles(self):
        return [s.end_profiling() for s in getattr(self,"sessions",{}).values() if self.spec.get("profile",False)]

class Predictor:
    def __init__(self, runtime, calibration=None):
        self.runtime, self.calibration = runtime, calibration
        if calibration is not None and calibration.get("runtime_fingerprint") != runtime.fingerprint:
            raise ValueError("calibration is bound to another graph/provider/precision/environment")
    def predict(self, request):
        started = time.perf_counter(); answers = {}
        for qid, atom, z in self.runtime.logits(request):
            if np.asarray(z).ndim != 1 or len(z) != len(atom.keys):
                raise RuntimeError("predictor logit shape does not match candidates")
            key = str(atom.kind); t = 1.0
            calibrated = bool(self.calibration and self.calibration.get("fits",{}).get(key,{}).get("status") == "fitted")
            if calibrated: t = float(self.calibration["temperatures"][key])
            p = probabilities(z,t); ix = int(p.argmax())
            answers[qid] = {"selected_key":atom.keys[ix], "probabilities":dict(zip(atom.keys,p.tolist())),
                            "p_selected":float(p[ix]), "calibrated":calibrated, "temperature":t}
        return {"runtime_fingerprint":self.runtime.fingerprint,"answers":answers,
                "inference_total_ms":1000*(time.perf_counter()-started)}
