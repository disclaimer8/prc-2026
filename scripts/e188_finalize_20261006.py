"""Registered complete E188 v12072 and fixed whole-file 50:50 v12073.

Replays the arithmetic E152/E81/LOBT/protect tail, first requiring exact integer
reconstruction of v12048 from its cached v12044. No fitting or uploading.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

os.environ["POLARS_MAX_THREADS"] = "4"
os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import today_submissions_20261006 as T

ID, Y, ROWS = "MVT_ID_mvt", "TAXITIME_SEC_mvt", 344841
PROTOCOL = "E188-fixed-full-and-wholefile-half-v12072-v12073-20261006"
BEST_SHA = "8d3b589213240bd6960984d6b2a46c6716617262c635ac2dc02d35b50b876015"
PARENT_SHA = "0c72511bf5be05ecb9d68a72245e244feeb6088b92372732bb3aae0dc7dddbb1"
STAGE_SHA = "0db85b5fbc25ec6b668f81afe99bc6abb1652aa72fbe8cfd5dc02f8523dac6fc"
FIT_REGISTRATION = "d45cb738fb7671cf289ae8a400a30b5b46dfbb3d"
STAGE = "data/e188_rank_20261006_v2/stage.parquet"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8*1024**2), b""):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def frame(path, ids=None):
    d = pl.read_parquet(path)
    if (d.columns != [ID, Y] or d.height != ROWS or d[ID].n_unique() != ROWS or d[Y].null_count()
        or not np.isfinite(d[Y].to_numpy()).all()):
        raise RuntimeError(f"incomplete/nonfinite file {path}")
    if ids is not None:
        if not np.array_equal(np.sort(ids), np.sort(d[ID].to_numpy())):
            raise RuntimeError(f"different IDs {path}")
        d = pl.DataFrame({ID: ids}).join(d, on=ID, how="left", validate="1:1", maintain_order="left")
    return d


def build(root, output, registration):
    if not re.fullmatch("[a-f0-9]{40}", registration):
        raise RuntimeError("full registration SHA required")
    script = Path(__file__).resolve()
    name = str(script.relative_to(root))
    frozen = subprocess.check_output(["git", "show", f"{registration}:{name}"], cwd=root)
    if hashlib.sha256(frozen).hexdigest() != sha(script):
        raise RuntimeError("source differs from preregistration")
    T.check_registration(root, registration)
    if output.exists():
        raise RuntimeError("fresh output required")
    for version in (12072, 12073):
        for suffix in ("parquet", "manifest.json"):
            if (root/f"submissions/gentle-igloo_v{version}.{suffix}").exists():
                raise RuntimeError("version already exists")
    paths = [root/f"submissions/gentle-igloo_v{v}.parquet" for v in (44, 60, 291, 361, 12042, 12044, 12048, 12061)]
    paths += [root/"submissions/parts_v12041.npz", root/"data/features_e106.parquet",
              root/"data/prc-2026-datasets/ranking.parquet", root/STAGE,
              root/STAGE.replace(".parquet", ".json"), root/"data/e188_rank_20261006_v2/fit_manifest.json",
              root/"submissions/gentle-igloo_v12048.manifest.json"]
    inputs = {str(p.relative_to(root)): sha(p) for p in paths}
    for p, expected in {"submissions/gentle-igloo_v12061.parquet": BEST_SHA,
                        "submissions/gentle-igloo_v12048.parquet": PARENT_SHA, STAGE: STAGE_SHA}.items():
        if inputs[p] != expected:
            raise RuntimeError(f"fixed input drift {p}")
    stage_record = json.loads((root/STAGE.replace(".parquet", ".json")).read_text())
    if (stage_record["registration_commit"] != FIT_REGISTRATION or stage_record["output_sha256"] != STAGE_SHA
        or not stage_record["offmask_unchanged"] or stage_record["ranking_augmentation"]):
        raise RuntimeError("E188 stage provenance differs")
    parent_manifest = json.loads((root/"submissions/gentle-igloo_v12048.manifest.json").read_text())
    for p in ("submissions/gentle-igloo_v12042.parquet", "submissions/gentle-igloo_v12044.parquet", "submissions/parts_v12041.npz"):
        if inputs[p] != parent_manifest["outputs"][p]:
            raise RuntimeError(f"cached parent stage differs {p}")
    for p in ("data/features_e106.parquet", "submissions/gentle-igloo_v44.parquet", "submissions/gentle-igloo_v60.parquet",
              "submissions/gentle-igloo_v291.parquet", "submissions/gentle-igloo_v361.parquet"):
        if inputs[p] != parent_manifest["inputs"][p]:
            raise RuntimeError(f"cached tail input differs {p}")
    best = frame(root/"submissions/gentle-igloo_v12061.parquet")
    ids = best[ID].to_numpy()
    read = lambda v: frame(root/f"submissions/gentle-igloo_v{v}.parquet", ids)
    old, tab, donor_old, donor_new = read(12048), read(12042), read(291), read(361)
    donors = donor_old[Y].to_numpy() != donor_new[Y].to_numpy()
    if donors.sum() != 42 or not np.array_equal(tab[Y].to_numpy()[donors], donor_new[Y].to_numpy()[donors]):
        raise RuntimeError("42 donor definition or values differ")
    with np.load(root/"submissions/parts_v12041.npz", allow_pickle=False) as parts:
        if parts["proba"].shape != (ROWS, 6) or not np.isfinite(parts["proba"]).all():
            raise RuntimeError("incomplete parts")
        probabilities = pl.DataFrame({ID: parts["mvt_id"], "conf": parts["proba"][:, 1:5].max(axis=1)})
    conf = best.select(ID).join(probabilities, on=ID, how="left", validate="1:1", maintain_order="left")
    if conf.height != ROWS or conf["conf"].null_count():
        raise RuntimeError("missing probability IDs")
    gate = conf["conf"].to_numpy() >= .95
    anchor44, anchor60 = read(44)[Y].to_numpy().astype(float), read(60)[Y].to_numpy().astype(float)
    features = pl.read_parquet(root/"data/features_e106.parquet", columns=[ID, "split", "d_mvt_lobt"]).filter(pl.col("split") == "rank")
    f = best.select(ID).join(features.select(ID, "d_mvt_lobt"), on=ID, how="left", validate="1:1", maintain_order="left")
    if f.height != ROWS:
        raise RuntimeError("invalid feature join")
    lo = f["d_mvt_lobt"].to_numpy() - 3606
    raw = pl.read_parquet(root/"data/prc-2026-datasets/ranking.parquet", columns=[ID, "ADEP_mvt", "FLIGHT_ID_mvt"])
    routing = best.select(ID).join(raw, on=ID, how="left", validate="1:1", maintain_order="left")
    target = routing.select(((pl.col("ADEP_mvt") == "LIRF") & pl.col("FLIGHT_ID_mvt").is_not_null()).fill_null(False)).to_series().to_numpy()
    if np.any(target & donors) or not np.array_equal(best[Y].to_numpy()[target], old[Y].to_numpy()[target]):
        raise RuntimeError("LIRF current-best parent or donor routing differs")
    output.mkdir(parents=True, exist_ok=False)
    (output/"submissions").mkdir()
    def emit(values, version):
        d = best.with_columns(pl.Series(Y, values).cast(best.schema[Y]))
        p = output/f"submissions/gentle-igloo_v{version}.parquet"
        d.write_parquet(p)
        check = frame(p, ids)
        if check.schema != best.schema or not np.array_equal(check[Y].to_numpy(), values):
            raise RuntimeError("serialized prediction/schema mismatch")
        return d
    def tail(staged, first):
        p = staged[Y].to_numpy().copy()
        emit(p, first)
        p = np.where(gate, tab[Y].to_numpy(), p).astype(best[Y].to_numpy().dtype)
        emit(p, first+1)
        # Literal arithmetic/order of blend_submissions.py.
        p = np.rint((1. - sum([.1214, .0270])) * p.astype(float) + .1214 * anchor44 + .0270 * anchor60).astype(np.int32)
        emit(p, first+2)
        base = p.astype(float)
        g1 = np.isfinite(lo) & (np.abs(base-lo) <= 2)
        g2 = np.isfinite(lo) & (base-lo > 2) & (base-lo <= 60)
        moved = base.copy()
        moved[g1] = lo[g1]+1074
        moved[g2] = np.maximum(base[g2], lo[g2]+165)
        ok = np.isfinite(lo)
        moved[ok] = np.clip(moved[ok], lo[ok], lo[ok]+7212)
        p = np.round(moved).astype(np.int32)
        emit(p, first+3)
        p[donors] = tab[Y].to_numpy()[donors]
        emit(p, first+4)
        return p
    control = tail(read(12044), 12200)
    if not np.array_equal(control, old[Y].to_numpy()):
        save(output/"control_failure.json", dict(changed=int(np.count_nonzero(control-old[Y].to_numpy())),
             max_abs=float(np.max(np.abs(control.astype(float)-old[Y].to_numpy().astype(float))))))
        raise RuntimeError("control tail did not reproduce v12048 exactly")
    candidate = tail(frame(root/STAGE, ids), 12210)
    if not np.array_equal(candidate[~target], control[~target]):
        raise RuntimeError("candidate tail changed rows outside LIRF/NM")
    final = best[Y].to_numpy().copy()
    final[target] = candidate[target]
    half = np.rint(.5*best[Y].to_numpy().astype(float) + .5*final.astype(float)).astype(np.int32)
    results = {}
    for version, pred in [(12072, final), (12073, half)]:
        if (not np.isfinite(pred).all() or np.any(pred < 0) or not np.array_equal(pred[~target], best[Y].to_numpy()[~target])
            or not np.array_equal(pred[donors], donor_new[Y].to_numpy()[donors])):
            raise RuntimeError("final integrity/offmask/donor failure")
        emit(pred, version)
        delta = pred.astype(float) - best[Y].to_numpy().astype(float)
        results[str(version)] = dict(rows=ROWS, changed=int(np.count_nonzero(delta)), rms=float(np.sqrt(np.mean(delta**2))),
            max_abs=float(np.max(np.abs(delta))), offmask_unchanged=True, donor_rows=42, donor_exact=True)
    for p, expected in inputs.items():
        if sha(root/p) != expected:
            raise RuntimeError(f"input changed during assembly {p}")
    hashes = {str(p.relative_to(output)): sha(p) for p in (output/"submissions").glob("*.parquet")}
    manifest = dict(protocol=PROTOCOL, registration_commit=registration, sources={name: sha(script)}, inputs=inputs,
        outputs=hashes, versions=[12072, 12073], fixed_weights={"12072": "full E188 on best v12061", "12073": "0.5*v12061+0.5*v12072"},
        e188_fit_registration=FIT_REGISTRATION, control_tail_exact_v12048=True, full_chain_cv_waived=True,
        private_control_versions=list(range(12200,12205)), private_candidate_versions=list(range(12210,12215)),
        validation=results, lszh_halfblend_retained_from_v12061=True, final_submission=True)
    save(output/"manifest.json", manifest)
    for version in (12072, 12073):
        file = f"gentle-igloo_v{version}.parquet"
        T.publish(root, output/"submissions"/file, version, registration,
            {"e188_full_tail_manifest":str((output/"manifest.json").relative_to(root)),
             "e188_full_tail_manifest_sha256":sha(output/"manifest.json"),
             "control_tail_exact_v12048":True, "e188_fit_registration":FIT_REGISTRATION,
             "input_sha256":inputs, "output_sha256":hashes, "routing_validation":results[str(version)]})
    print(json.dumps(dict(event="complete", versions=[12072,12073], validation=results,
                          manifest=str(output/"manifest.json"), manifest_sha256=sha(output/"manifest.json"))), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path.cwd())
    p.add_argument("--output",type=Path,default=Path("data/e188_finalize_20261006"))
    p.add_argument("--registration-commit",required=True)
    a=p.parse_args()
    root=a.root.resolve()
    build(root,(root/a.output).resolve(),a.registration_commit)


if __name__=="__main__":
    main()
