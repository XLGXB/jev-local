"""Export native checkpoints into static B=1 raw-logit ONNX candidates.
Successful serialization is NOT deployment approval. Run verify + runtime calibration.
"""
from __future__ import annotations
import argparse, inspect, shutil
from pathlib import Path
import numpy as np
import torch
from .common import atoms_for, read, save, sha, versions
from .encoding import padding
from .wrappers import LayaGraph, KevGraph

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["laya", "kev"], required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--sample", required=True)
    p.add_argument("--out", required=True); p.add_argument("--buckets", nargs="+", type=int, default=[512])
    p.add_argument("--candidates", type=int, default=16); p.add_argument("--opset", type=int, default=18)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--experimental-hybrid", action="store_true")
    a = p.parse_args()
    from decision_ft.backends import load_backend
    from decision_ft.train import model_digest
    root, out = Path(a.checkpoint).resolve(), Path(a.out).resolve()
    if out.exists(): p.error("output must be a NEW directory")
    if a.candidates < 2 or a.candidates > 16 or any(n < 8 for n in a.buckets): p.error("invalid profile")
    pairs = atoms_for(read(a.sample))
    if len(pairs) != 1: p.error("export probe contains exactly one question")
    atom = pairs[0][1]
    b = load_backend(a.backend, str(root), a.device, "fp32"); b.model.eval()
    if a.backend == "kev" and b.model.hybrid and not a.experimental_hybrid:
        p.error("Qwen3.5 hybrid full ONNX export is experimental; use native deployment or explicitly probe --experimental-hybrid")
    with torch.no_grad(): original = b.forward([atom])[0].cpu().numpy()
    # Merge in memory only; never overwrite original SFT/RLCD checkpoints.
    if a.backend == "kev" and hasattr(b.model.lm, "merge_and_unload"):
        b.model.lm = b.model.lm.merge_and_unload(safe_merge=True)
        with torch.no_grad(): merged = b.forward([atom])[0].cpu().numpy()
        np.testing.assert_allclose(merged, original, rtol=1e-4, atol=1e-4)
    encmod = b.model.encoder if a.backend == "laya" else b.model.lm
    if hasattr(encmod.config, "reference_compile"): encmod.config.reference_compile = False
    if hasattr(encmod, "set_attn_implementation"): encmod.set_attn_implementation("eager")
    else: encmod.config._attn_implementation = "eager"
    if a.backend == "kev": b.model.lm.config.use_cache = False
    b.model.eval()
    wrapper = (LayaGraph if a.backend == "laya" else KevGraph)(b.model).eval()
    encoded = b.encode(atom)
    if a.backend == "laya" and max(a.buckets) > int(b.cfg["max_len"]):
        p.error("export bucket exceeds trained checkpoint max_len")
    out.mkdir(); (out / "tokenizer").mkdir()
    b.tok.save_pretrained(str(out / "tokenizer"))
    if a.backend == "laya": shutil.copy2(root / "rl_agent_config.json", out / "rl_agent_config.json")
    import onnx
    flags = torch.backends.mha.get_fastpath_enabled()
    probes = []
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        for seq in sorted(set(a.buckets)):
            data = padding(a.backend, encoded, seq, a.candidates, b.tok.pad_token_id or 0)
            if a.backend == "laya": data["qtype"] = np.array([atom.kind], dtype=np.int64)
            inputs = tuple(torch.from_numpy(data[n]).to(a.device) for n in wrapper.names)
            with torch.no_grad(): z = wrapper(*inputs).cpu().numpy()[0, :len(atom.keys)]
            np.testing.assert_allclose(z, original, rtol=1e-4, atol=1e-4)
            graphdir = out / f"s{seq}"; graphdir.mkdir()
            with torch.no_grad():
                torch.onnx.export(wrapper, inputs, str(graphdir / "model.onnx"),
                    input_names=wrapper.names, output_names=["logits"],
                    opset_version=a.opset, dynamo=True, external_data=True,
                    report=True, artifacts_dir=str(graphdir))
            onnx.checker.check_model(str(graphdir / "model.onnx"))
            # Frozen synthetic/native export probe; a broader independent corpus is still required.
            np.savez(graphdir / "probe.npz", **data, reference_logits=original)
            probes.append({"sequence": seq, "native_to_wrapper_max_abs": float(np.max(np.abs(z-original)))})
    finally: torch.backends.mha.set_fastpath_enabled(flags)
    # Create the serving manifest only after ALL requested exports succeed.
    files = {str(f.relative_to(out)): sha(f) for f in sorted(out.rglob("*")) if f.is_file()}
    from .encoding import Frontend
    frontend = Frontend(out, a.backend)
    save(out / "manifest.json", {
        "schema_version": 1, "status": "EXPORTED_CANDIDATE_NOT_QUALIFIED", "backend": a.backend,
        "checkpoint_sha256": model_digest(root), "buckets": sorted(set(a.buckets)), "candidates": a.candidates,
        "batch": 1, "raw_logits": True, "option_isolation": False,
        "hybrid_experimental": bool(a.backend == "kev" and b.model.hybrid), "opset": a.opset,
        "frontend_source_sha256": frontend.sources, "environment": versions(),
        "files_sha256": files, "probe_results": probes})
    print(out / "manifest.json")
if __name__ == "__main__": main()
