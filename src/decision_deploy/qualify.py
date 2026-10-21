"""Golden vectors, runtime parity, warm timing, and runtime-specific calibration."""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
from .common import read, save, sha, probabilities, unique_pairs
from .runtime import Runtime

def requests(path):
    import json
    rows=[]
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r=json.loads(line,object_pairs_hook=unique_pairs)
            rows.append({"state":r["state"],"questions":{
                k:{f:v for f,v in q.items() if f in ("type","instructions","criteria")}
                for k,q in r["questions"].items()}})
    if not rows: raise ValueError("empty corpus")
    return rows

def profile_summary(files):
    totals={}
    for filename in files:
        for event in read(filename):
            provider=event.get("args",{}).get("provider")
            if provider:
                item=totals.setdefault(provider,{"events":0,"duration_us":0})
                item["events"]+=1;item["duration_us"]+=event.get("dur",0)
    return totals

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=["golden","compare","warm","calibrate","evaluate"])
    p.add_argument("--runtime",required=True);p.add_argument("--requests");p.add_argument("--golden")
    p.add_argument("--data");p.add_argument("--out",required=True);p.add_argument("--repeats",type=int,default=50)
    p.add_argument("--max-logit-abs",type=float,default=1e-3);p.add_argument("--max-prob-abs",type=float,default=1e-4)
    p.add_argument("--min-top1",type=float,default=1.0);p.add_argument("--min-per-type",type=int,default=30)
    p.add_argument("--partition",choices=["dev","test"],default="dev");p.add_argument("--allow-test",action="store_true");p.add_argument("--calibration")
    a=p.parse_args()
    if a.stage=="evaluate" and a.partition=="test" and not a.allow_test:p.error("locked test requires --allow-test after checkpoint/threshold freeze")
    if Path(a.out).exists():p.error("will not overwrite a qualification receipt")
    runtime=Runtime(read(a.runtime)); start=time.perf_counter()
    out={"runtime_fingerprint":runtime.fingerprint,"runtime_identity":runtime.identity,"stage":a.stage}
    if a.stage=="golden":
        if runtime.mode!="native" or not a.requests:p.error("golden requires native runtime and --requests")
        samples=[]
        for r in requests(a.requests):
            samples.append({"request":r,"logits":{qid:z.tolist() for qid,atom,z in runtime.logits(r)}})
        out.update(samples=samples,request_file_sha256=sha(a.requests))
    elif a.stage=="compare":
        if not a.golden:p.error("--golden required")
        golden=read(a.golden); results=[]
        if golden.get("stage")!="golden":raise ValueError("not a golden vector receipt")
        native_sha=golden["runtime_identity"]["artifact"]["checkpoint_sha256"]
        candidate_sha=runtime.manifest["checkpoint_sha256"] if runtime.mode=="ort" else runtime.identity["artifact"]["checkpoint_sha256"]
        if native_sha!=candidate_sha:raise ValueError("export and native golden use different checkpoints")
        for item in golden["samples"]:
            for qid,atom,z in runtime.logits(item["request"]):
                ref=np.array(item["logits"][qid]); z=np.array(z)
                results.append({"qid":qid,"max_logit_abs":float(np.max(np.abs(ref-z))),
                    "max_probability_abs":float(np.max(np.abs(probabilities(ref)-probabilities(z)))),
                    "same_top1":bool(ref.argmax()==z.argmax())})
        if not results:raise ValueError("no decisions")
        top=float(np.mean([r["same_top1"] for r in results]))
        passed=top>=a.min_top1 and max(r["max_logit_abs"] for r in results)<=a.max_logit_abs and max(r["max_probability_abs"] for r in results)<=a.max_prob_abs
        out.update(status="PASS" if passed else "FAIL",top1_agreement=top,decisions=results,
                   golden_sha256=sha(a.golden),
                   thresholds={"logit":a.max_logit_abs,"probability":a.max_prob_abs,"top1":a.min_top1},
                   warning="Numerical parity only; not task quality, calibration or permission approval.")
    elif a.stage=="warm":
        if not a.requests or a.repeats<1:p.error("positive repeats and --requests required")
        corpus=requests(a.requests); cold=[]; warm=[]
        for r in corpus:
            t=time.perf_counter();runtime.logits(r);cold.append(1000*(time.perf_counter()-t))
        for i in range(a.repeats):
            t=time.perf_counter();runtime.logits(corpus[i%len(corpus)]);warm.append(1000*(time.perf_counter()-t))
        profiles=runtime.close_profiles()
        out.update(first_inference_ms=cold,warm_p50_ms=float(np.percentile(warm,50)),warm_p95_ms=float(np.percentile(warm,95)),
                   profiles=profiles,provider_node_events=profile_summary(profiles),
                   warning="Includes tokenization and host copies, excludes HTTP/browser. Profile durations are not speedup attribution.")
    elif a.stage=="calibrate":
        if not a.data:p.error("--data must be the v1 frozen dataset directory")
        import torch
        from decision_ft.data import atomize,verify_dataset,VERIFIED
        from decision_ft.calibrate import fit_temperature
        atoms=atomize(verify_dataset(a.data,("calibration",))["calibration"])
        if any(x.label_source not in VERIFIED for x in atoms):raise ValueError("unverified calibration labels")
        logits=[]
        for atom in atoms:
            z=runtime.logits({"state":atom.state,"questions":{"q":atom.question}})[0][2]
            logits.append(torch.from_numpy(np.array(z,copy=True)))
        temps,details=fit_temperature(atoms,logits,a.min_per_type)
        out.update(temperatures=temps,fits=details,calibration_file_sha256=sha(Path(a.data)/"calibration.jsonl"),
                   dataset_manifest_sha256=sha(Path(a.data)/"manifest.json"),
                   warning="Fit-set performance is not held-out task quality; this receipt does not approve browser writes.")
    else:
        if not a.data:p.error("--data required")
        import torch
        from decision_ft.data import atomize,verify_dataset,VERIFIED
        from decision_ft.metrics import report
        from .runtime import Predictor
        atoms=atomize(verify_dataset(a.data,(a.partition,))[a.partition])
        logits=[torch.from_numpy(np.array(runtime.logits({"state":x.state,"questions":{"q":x.question}})[0][2],copy=True)) for x in atoms]
        out.update(partition=a.partition,dataset_manifest_sha256=sha(Path(a.data)/"manifest.json"),raw=report(atoms,logits))
        if a.calibration:
            calibration=read(a.calibration);Predictor(runtime,calibration)
            if calibration["dataset_manifest_sha256"]!=out["dataset_manifest_sha256"]:raise ValueError("calibration dataset mismatch")
            if any(calibration["fits"].get(str(x.kind),{}).get("status")!="fitted" for x in atoms):raise ValueError("each evaluated type needs a successful fit")
            out.update(calibrated=report(atoms,logits,calibration["temperatures"]),calibration_sha256=sha(a.calibration))
            # Refusals are NOT automatically completed browser actions.
            pool=[(x,z) for x,z in zip(atoms,logits) if x.label_source in VERIFIED and x.kind==0 and "needs_host" in x.keys]
            coverage={}
            for threshold in (0.8,0.9,0.95,0.99):
                flags=[];correct=[]
                for x,z in pool:
                    probs=probabilities(z.numpy(),calibration["temperatures"]["0"]);i=int(probs.argmax())
                    accepted=x.keys[i]!="needs_host" and probs[i]>=threshold
                    flags.append(accepted)
                    if accepted:correct.append(i==x.label)
                coverage[str(threshold)]={"decisions":len(pool),"accepted_actions":sum(flags),"coverage":sum(flags)/len(pool) if pool else None,"action_error_rate":1-float(np.mean(correct)) if correct else None}
            out["action_coverage_excluding_needs_host"]=coverage
        out["warning"]="Atomic-decision metrics only; not browser end-to-end success or empirical proof of a 99% safety guarantee."
    out["elapsed_s"]=time.perf_counter()-start;save(a.out,out)
    if out.get("status")=="FAIL":raise SystemExit(2)
if __name__=="__main__":main()
