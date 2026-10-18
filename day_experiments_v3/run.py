"""Source-bound modern baseline/candidate experiment runner. Linux only.

All mutations are confined to --root day experiment space. Historical runners
and source pins are never rewritten. Every result directory is write-once.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone

BASELINE = "25c8c1642029a640b153c160c22a89795d29ad91"
ROOT = Path("/var/tmp/codex-tdigest-pr3-20261014/day20261015")
EVIDENCE = ROOT/"evidence"/"v3"
HERE = Path(__file__).resolve().parent
TARGET = Path("/var/tmp/codex-tdigest-pr3-20261014/work/gr-tdigest/target")
PREFIX = "compat/accuracy/matched-budgets-ff73cfa/protocol/held_out_accuracy.rs"
SEEDS = {"screen": 104729, "confirm": 86028157}
ROUND_CAP = 1_000_000_000
ORDERS = ["sorted","reverse","random","nan","ties","nearlysorted"]
PROBE_COUNTS = [1,8,64,256,4096,32767,32768,65536]
OPERATIONS = ["build","merge","merge2","kway","p50","cdf","cdf_atomic","cdf_weighted","cdf_batch","encode","decode"]
OPERATIONS += [f"frontend_{scope}_{n}" for scope in ["owned","clone"] for n in [1,2,40]]
OPERATIONS += [f"batch_{order}_{n}" for order in ORDERS for n in PROBE_COUNTS]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")

def read(path):
    return json.loads(Path(path).read_text())

def command(argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()

def now():
    return datetime.now(timezone.utc).isoformat()

def manifest(root):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob("*")) if p.is_file()}

def env():
    return {**os.environ,"CARGO_TARGET_DIR":str(TARGET),"RAYON_NUM_THREADS":"1","OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","MKL_NUM_THREADS":"1"}

def verify_protocol():
    protocol=read(HERE/"protocol.json")
    for name,expected in protocol["files_sha256"].items():
        assert sha(HERE/name)==expected,f"Protocol file changed: {name}"
    return protocol

def run_native(binary, args, output):
    with output.with_suffix(".stderr").open("w") as err:
        result=subprocess.run(["taskset","-c","0",str(binary),*map(str,args)],text=True,stdout=subprocess.PIPE,stderr=err,env=env(),timeout=180)
    output.write_text(result.stdout)
    if result.returncode:
        raise RuntimeError(f"Native exit {result.returncode}: {output}")
    return [json.loads(s) for s in result.stdout.splitlines() if s]

def build(args):
    repo=args.repo.resolve()
    assert repo.is_relative_to(ROOT) and repo.name != "harness"
    label=args.label
    assert label.replace("-","").replace("_","").isalnum()
    out=EVIDENCE/"builds"/label
    assert not out.exists(),f"Refuse to overwrite {out}"
    out.mkdir(parents=True)
    head=command(["git","rev-parse","HEAD"],repo)
    status=command(["git","status","--short"],repo)
    if label.startswith("baseline"):
        assert head==BASELINE and not status
    source=out/"source"
    source.mkdir()
    names=command(["git","ls-files","--cached","--others","--exclude-standard","-z"],repo).split("\0")
    selected=[name for name in names if name.startswith(("src/","crates/","benches/")) or name in ("Cargo.toml","Cargo.lock","build.rs","README.md","LICENSE")]
    for name in selected:
        p=source/name
        p.parent.mkdir(parents=True,exist_ok=True)
        # Fresh timestamps supplement explicit package invalidation below.
        shutil.copyfile(repo/name,p)
    hashes=manifest(source)
    (out/"patch.diff").write_bytes(subprocess.check_output(["git","diff","HEAD","--binary"],cwd=repo))
    provenance={"status":"building","started":now(),"repository":str(repo),"head":head,"status_at_build":status,"baseline":BASELINE,"source_sha256":hashes,"driver_sha256":sha(__file__),"tail_sha256":sha(HERE/"tail.rs"),"reference_sha256":sha(HERE/"reference.rs"),"protocol_sha256":sha(HERE/"protocol.json"),"toolchain":command(["rustc","-Vv"]),"target_cache":str(TARGET),"environment":{k:v for k,v in env().items() if k.startswith(("RUST","CARGO_PROFILE")) or k in ("CARGO_ENCODED_RUSTFLAGS","RAYON_NUM_THREADS","LD_PRELOAD","MALLOC_CONF")}}
    save(out/"provenance.json",provenance)
    project=out/"runner"
    (project/"src").mkdir(parents=True)
    prefix=(HERE/"reference.rs").read_text().split("macro_rules! precision_suite")[0]
    prefix=prefix.replace("use tdigest_core::TDigest as Original;\n","")
    (project/"src/main.rs").write_text(prefix+(HERE/"tail.rs").read_text())
    (project/"Cargo.toml").write_text('[package]\nname="day-digest-probe-v3"\nversion="0.0.0"\nedition="2021"\n[workspace]\n[dependencies]\ngr-tdigest={path="../source"}\nserde_json="1.0.145"\n')
    shutil.copyfile(source/"Cargo.lock",project/"Cargo.lock")
    with (out/"clean.log").open("w") as log:
        clean=subprocess.run(["cargo","clean","--release","--offline","--package","gr-tdigest"],cwd=project,env=env(),stdout=log,stderr=subprocess.STDOUT,timeout=120)
    assert clean.returncode==0,f"Package invalidation failed: {out/'clean.log'}"
    with (out/"build.log").open("w") as log:
        result=subprocess.run(["cargo","build","--release","--offline","--jobs","2"],cwd=project,env=env(),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
    assert result.returncode==0,f"Build failed: {out/'build.log'}"
    compile_lines=[line for line in (out/"build.log").read_text().splitlines() if "Compiling gr-tdigest " in line and f"({source})" in line]
    assert compile_lines,"Build did not prove compilation of this exact source snapshot"
    assert manifest(source)==hashes,"Snapshot changed during build"
    assert all(sha(repo/name)==value for name,value in hashes.items()),"Worktree changed during build"
    verify_protocol()
    assert provenance["protocol_sha256"]==sha(HERE/"protocol.json")
    binary=out/"probe"
    shutil.copyfile(TARGET/"release/day-digest-probe-v3",binary)
    binary.chmod(0o555)
    provenance.update(status="complete",finished=now(),binary_sha256=sha(binary),resolved_lock_sha256=sha(project/"Cargo.lock"),harness_source_sha256=sha(project/"src/main.rs"),profile="cargo release defaults, opt-level3 no LTO; recorded Cargo manifests and environment govern overrides",compile_proof=compile_lines,clean_log_sha256=sha(out/"clean.log"),build_log_sha256=sha(out/"build.log"),source_timestamp_policy="copyfile fresh timestamps plus forced release-package clean")
    save(out/"provenance.json",provenance)
    print(json.dumps({"built":label,"binary":str(binary),"sha256":sha(binary)}),flush=True)

def binary(label):
    directory=EVIDENCE/"builds"/label
    metadata=read(directory/"provenance.json")
    path=directory/"probe"
    assert metadata["status"]=="complete" and sha(path)==metadata["binary_sha256"]
    assert manifest(directory/"source")==metadata["source_sha256"]
    assert sha(directory/"build.log")==metadata["build_log_sha256"]
    assert sha(directory/"clean.log")==metadata["clean_log_sha256"]
    assert sha(directory/"runner/src/main.rs")==metadata["harness_source_sha256"]
    assert sha(directory/"runner/Cargo.lock")==metadata["resolved_lock_sha256"]
    return path

def smoke(args):
    path=binary(args.label)
    out=EVIDENCE/f"smoke-{args.label}"
    assert not out.exists()
    out.mkdir()
    repros=run_native(path,["repros"],out/"repros.jsonl")
    reports=[]
    for precision in ["f32","f64"]:
        for kind in ["normal","ties","separated_modes"]:
            name=f"{kind}-{precision}"
            rows=run_native(path,["audit",kind,32,SEEDS["screen"],precision,"delta300","merge",1],out/f"{name}.jsonl")
            reports.extend(rows)
    extra=[("weighted",8,p,"cdf") for p in ["f32","f64"]]
    extra += [("atomic",1,p,"cdf") for p in ["f32","f64"]]
    extra += [("normal",128,"f64",f"frontend_owned_{fanin}") for fanin in [1,2,40]]
    extra += [("normal",128,"f64",f"batch_{order}_{count}") for order in ORDERS for count in [1,8]]
    for kind,n,precision,operation in extra:
        name=f"extra-{kind}-{n}-{precision}-{operation}"
        reports.extend(run_native(path,["audit",kind,n,SEEDS["screen"],precision,"delta300",operation,1],out/f"{name}.jsonl"))
    invariants=[(x["kind"],x["precision"],x["route"],x["invariant_failures"]) for x in reports if x["invariant_failures"]]
    save(out/"summary.json",{"label":args.label,"binary_sha256":sha(path),"repros":repros,"invariant_failures":invariants,"states":len(reports),"utc":now()})
    print(json.dumps({"smoke":str(out),"states":len(reports),"invariant_failures":invariants,"repros":repros}),flush=True)
    assert not invariants,"Unexpected ordinary smoke invariant failure"

def cases(phase,operation,experiment):
    if operation=="cdf_atomic":
        return [("atomic",n,p,"cap1000use") for n in [1,8,64,256,1000] for p in ["f32","f64"]]
    if operation=="cdf_weighted":
        return [("weighted",n,p,"cap1000use") for n in [8,64,256,1000] for p in ["f32","f64"]]
    if operation.startswith("batch_"):
        # Covers realistic compressed states and the sparse-probe scan penalty.
        cells=[("normal",100000,"f64","delta300"),("ties",1000,"f32","cap64use"),("atomic",1000,"f64","cap1000use")]
        if phase=="confirm":
            cells += [("normal",100000,"f32","delta300"),("separated_modes",100000,"f64","cap256off"),("weighted",256,"f64","cap1000use")]
        return cells
    if operation.startswith("frontend_"):
        kinds=["normal","ties","separated_modes"] if phase=="screen" else ["normal","ties","separated_modes","lognormal","pareto","spike_tail"]
        return [(k,n,p,s) for k in kinds for n in [1000,100000] for p,s in [("f32","cap64use"),("f64","delta300")]]
    controls={"E02":["build","p50","cdf_batch"],"E04":["build","p50","cdf_batch"],"E10":["build","p50","cdf_batch"],"E20":["build","p50","cdf"]}
    if operation in controls.get(experiment,[]):
        cells=[("normal",1000,"f32","delta300"),("normal",100000,"f64","delta300"),("ties",100000,"f32","cap64use"),("separated_modes",100000,"f64","cap64use")]
        if phase=="confirm":cells += [("pareto",100000,"f64","delta1200"),("lognormal",100000,"f32","cap256off")]
        return cells
    if experiment=="E04" and operation=="cdf":
        kinds=["normal","ties","separated_modes"] if phase=="screen" else ["normal","ties","separated_modes","lognormal","pareto","spike_tail"]
        return [(k,n,p,s) for k in kinds for n in [1000,100000] for p in ["f32","f64"] for s in ["delta300","cap64use"]]
    kinds=["normal","ties","separated_modes"] if phase=="screen" else ["uniform","normal","exponential","lognormal","pareto","ties","spike_tail","separated_modes"]
    sizes=[1000,100000] if phase=="screen" else [32,1000,100000]
    settings=["delta300","cap64use"] if phase=="screen" else ["delta100","delta300","delta1200","cap64use","cap256off","cap256edges"]
    if experiment=="E01":settings=["cap1use","cap2use","cap3use","cap64use","delta300"]
    if experiment=="E06":settings=["cap1off","cap2off","cap3off","cap64off","delta300"]
    return [(k,n,p,s) for k in kinds for n in sizes for p in ["f32","f64"] for s in settings]

def compare(args):
    a,b=binary(args.baseline_label),binary(args.label)
    metadata={"baseline":read(EVIDENCE/"builds"/args.baseline_label/"provenance.json"),args.label:read(EVIDENCE/"builds"/args.label/"provenance.json")}
    for key in ["harness_source_sha256","toolchain","resolved_lock_sha256","profile","environment","protocol_sha256"]:
        assert metadata["baseline"][key]==metadata[args.label][key],f"Cannot compare mismatched {key}"
    assert metadata["baseline"]["protocol_sha256"]==sha(HERE/"protocol.json")
    suffix=(f"-limit{args.limit}" if args.limit else "")+("-audit" if args.audit_only else "")+f"-scale{args.rounds_scale}"
    out=EVIDENCE/f"{args.experiment}-{args.label}-{args.phase}-{args.operation}{suffix}"
    assert not out.exists(),f"Refuse overwrite {out}"
    out.mkdir()
    protocol=verify_protocol()
    seed=SEEDS[args.phase]
    cells=cases(args.phase,args.operation,args.experiment)
    if args.limit:
        cells=cells[:args.limit]
    report={"status":"running","started":now(),"experiment":args.experiment,"baseline_label":args.baseline_label,"candidate":args.label,"phase":args.phase,"kind":args.kind,"operation":args.operation,"rounds_scale":args.rounds_scale,"protocol":protocol,"planned_cells":cells,"binary_sha256":{"baseline":sha(a),"candidate":sha(b)},"state_changes":[],"invariant_failures":[],"storage_increases":[],"quality_changes":[],"timings":[],"decision":"No automatic retention: requires full preregistered target/unchanged-control screening and held-out confirmation."}
    prior=None
    if args.rounds_scale==10:
        assert not args.audit_only,"No repetition escalation for an audit-only run"
        prior=out.with_name(out.name.removesuffix("scale10")+"scale1")
        initial=read(prior/"summary.json")
        assert initial["status"]=="complete" and initial["rounds_scale"]==1
        assert initial["protocol"]==protocol and initial["planned_cells"]==[list(c) for c in cells]
        assert initial["binary_sha256"]==report["binary_sha256"]
        assert any(t["classification"]=="short_batch_inconclusive" for t in initial["timings"]),"10x allowed only after retained short initial batches"
        report["initial_result"]=str(prior)
    save(out/"summary.json",report)
    for index,(kind,n,p,setting) in enumerate(cells):
        sub=out/f"{index:03}-{kind}-{n}-{p}-{setting}"
        sub.mkdir()
        op="cdf" if args.operation in ("cdf_atomic","cdf_weighted") else args.operation
        rows={label:run_native(path,["audit",kind,n,seed,p,setting,op,1],sub/f"{label}-audit.jsonl") for label,path in [("baseline",a),("candidate",b)]}
        expected_routes=["fixture"] if kind=="weighted" else ["direct","sequential","kway"]+(["frontend"] if op.startswith("frontend_") else [])
        assert [r["route"] for r in rows["baseline"]]==[r["route"] for r in rows["candidate"]]==expected_routes
        for before,after in zip(rows["baseline"],rows["candidate"]):
            key={"cell":index,"route":before["route"]}
            assert before["route"]==after["route"]
            changed_fields=[field for field in ["state","batch_query_bits","batch_output_bits"] if before.get(field)!=after.get(field)]
            if changed_fields: report["state_changes"].append({**key,"changed_fields":changed_fields})
            if after["invariant_failures"]: report["invariant_failures"].append({**key,"baseline":before["invariant_failures"],"candidate":after["invariant_failures"]})
            if after["logical_bytes"]>before["logical_bytes"]:report["storage_increases"].append({**key,"before":before["logical_bytes"],"after":after["logical_bytes"]})
            if before["quality"]!=after["quality"]:report["quality_changes"].append({**key,"before":before["quality"],"after":after["quality"]})
        if args.kind=="speed" and (report["state_changes"] or report["invariant_failures"]):
            report.update(status="rejected_exactness",finished=now());save(out/"summary.json",report)
            print(json.dumps({"result":str(out),"status":report["status"]}),flush=True);return
        if args.audit_only:
            save(out/"summary.json",report);continue
        rounds=64 if op.startswith("frontend_") else 8 if op.startswith("batch_") else {"build":3 if n>=100000 else 100,"merge":16,"merge2":64,"kway":16,"p50":20000,"cdf":10000,"cdf_batch":128,"encode":1000,"decode":1000}[op]
        if prior is None:
            pilot=run_native(a,["timing",kind,n,seed,p,setting,op,rounds],sub/"baseline-calibration.jsonl")[0]
            ns_per_op=statistics.median(pilot["ns_per_op"])
            rounds=max(rounds,math.ceil(60_000_000/ns_per_op))
        else:
            old_calibration=read(prior/sub.name/"calibration.json")
            ns_per_op=old_calibration["pilot_median_ns_per_op"]
            rounds=old_calibration["frozen_rounds"]*10
        assert rounds<=ROUND_CAP,"Calibration would exceed frozen round budget"
        save(sub/"calibration.json",{"source":"baseline_only" if prior is None else "preregistered_10x_initial_baseline_calibration","initial_result":str(prior) if prior else None,"rounds_scale":args.rounds_scale,"target_batch_ns":50_000_000,"safety_target_ns":60_000_000,"pilot_median_ns_per_op":ns_per_op,"frozen_rounds":rounds,"candidate_not_used":True})
        schedule=["baseline","candidate","candidate","baseline","candidate","baseline","baseline","candidate"]
        values={"baseline":[],"candidate":[]}
        samples=[]
        for order,label in enumerate(schedule):
            path=a if label=="baseline" else b
            result=run_native(path,["timing",kind,n,seed,p,setting,op,rounds],sub/f"timing-{order}-{label}.jsonl")[0]
            expected_row=next(row for row in rows[label] if row["route"]==result["audit_route"])
            assert result["state"]["state"]==expected_row["state"],"Timed state differs from audited state"
            for field in ["batch_query_bits","batch_output_bits"]:
                assert result["state"].get(field)==expected_row.get(field),f"Timed batch differs from audit: {field}"
            median=statistics.median(result["ns_per_op"])
            values[label].append(median)
            samples.append({"order":order,"label":label,"median_ns":median,"samples_ns":result["ns_per_op"],"pool_peak_logical_bytes_estimate":result.get("pool_peak_logical_bytes_estimate")})
        ratios=[]
        for j in [0,2,4,6]:
            pair=samples[j:j+2]
            va={r["label"]:r["median_ns"] for r in pair}
            ratios.append(va["candidate"]/va["baseline"])
        median_ratio=statistics.median(ratios)
        improvement=all(r<=0.97 for r in ratios)
        regression=all(r>1.03 for r in ratios)
        drift={label:max(v)/min(v) for label,v in values.items()}
        classification="improved" if improvement else "regressed" if regression else "neutral_or_noisy"
        if any(r>1.10 for r in drift.values()):classification="unstable_controls"
        if any(min(r["samples_ns"])*rounds<50_000_000 for r in samples):classification="short_batch_inconclusive"
        report["timings"].append({"cell":index,"case":[kind,n,p,setting],"paired_ratios":ratios,"median_ratio":median_ratio,"classification":classification,"within_binary_drift_ratio":drift,"samples":samples})
        save(out/"summary.json",report)
        print(f"{index+1}/{len(cells)} {kind}/{n}/{p}/{setting} ratio={median_ratio:.4f} {classification}",flush=True)
    report.update(status="complete",finished=now())
    verify_protocol()
    assert sha(a)==report["binary_sha256"]["baseline"] and sha(b)==report["binary_sha256"]["candidate"]
    save(out/"summary.json",report)
    print(json.dumps({"result":str(out),"status":"complete","state_changes":len(report["state_changes"]),"quality_changes":len(report["quality_changes"]),"invariant_failures":len(report["invariant_failures"]),"storage_increases":len(report["storage_increases"])}),flush=True)

def main():
    assert sys.platform=="linux","Linux/WSL only"
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="action",required=True)
    p=sub.add_parser("build");p.add_argument("--repo",type=Path,required=True);p.add_argument("--label",required=True)
    p=sub.add_parser("smoke");p.add_argument("--label",required=True)
    p=sub.add_parser("compare");p.add_argument("--label",required=True);p.add_argument("--baseline-label",default="baseline-v3");p.add_argument("--experiment",required=True);p.add_argument("--kind",choices=["speed","quality"],required=True);p.add_argument("--phase",choices=["screen","confirm"],default="screen");p.add_argument("--operation",choices=OPERATIONS,required=True);p.add_argument("--audit-only",action="store_true");p.add_argument("--limit",type=int);p.add_argument("--rounds-scale",type=int,choices=[1,10],default=1)
    args=parser.parse_args()
    verify_protocol()
    with (ROOT/"cpu.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        {"build":build,"smoke":smoke,"compare":compare}[args.action](args)

if __name__=="__main__":main()
