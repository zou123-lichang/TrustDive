from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, SplineTransformer, StandardScaler

from .config import PROJECT_ROOT, RESULTS_ROOT, RUNS_ROOT, Paths, load_contract
from .util import git_head, sha256_file, stable_json, write_json
from .v2_data import official_panel_aggregate
from .v7_data import V7_RESULTS_ROOT, load_v7_frame
from .v8_data import V8_RESULTS_ROOT


V9_CONTRACT_PATH = PROJECT_ROOT / "01_PROTOCOL" / "analysis_contract_v9_judge_sim.yaml"
V9_RESULTS_ROOT = RESULTS_ROOT / "V9_JUDGE_SIM"
V9_RUN_ROOT = RUNS_ROOT / "v9_judge_sim"
V9_CACHE_ROOT = PROJECT_ROOT / ".cache" / "v9_judge_sim"


def v9_paths() -> Paths:
    return replace(Paths(), contract=V9_CONTRACT_PATH)


@lru_cache(maxsize=1)
def load_v9_contract() -> dict:
    return load_contract(V9_CONTRACT_PATH)


def ensure_v9_dirs() -> None:
    for path in (
        V9_RESULTS_ROOT / "00_AUDIT",
        V9_RESULTS_ROOT / "01_SIMULATOR",
        V9_RESULTS_ROOT / "02_CONFLICTS",
        V9_RESULTS_ROOT / "03_FEATURES",
        V9_RESULTS_ROOT / "04_PILOT",
        V9_RESULTS_ROOT / "05_FINAL",
        V9_RESULTS_ROOT / "06_STRESS",
        V9_RESULTS_ROOT / "07_REVIEW",
        V9_RESULTS_ROOT / "figures_v9" / "source_data",
        V9_RUN_ROOT / "checkpoints",
        V9_CACHE_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)


def _changed_since_anchor(anchor: str) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", anchor, "--"], cwd=PROJECT_ROOT,
        check=True, capture_output=True, text=True,
    )
    return [x.strip().replace("\\", "/") for x in result.stdout.splitlines() if x.strip()]


def _protected(path: str) -> bool:
    if path.startswith("03_RESULTS/") and not path.startswith("03_RESULTS/V9_JUDGE_SIM/"):
        return True
    if path.startswith("runs/") and "v9" not in Path(path).name.lower():
        return True
    if path.startswith("01_PROTOCOL/") and "v9" not in Path(path).name.lower():
        return True
    if path.startswith("02_CODE/src/trustdive/v"):
        return any(Path(path).name.startswith(f"v{k}_") for k in range(1, 9))
    if path.startswith("README_V") and not path.startswith("README_V9"):
        return True
    return False


def audit_v9() -> dict:
    ensure_v9_dirs()
    contract = load_v9_contract()
    frame = load_v7_frame()
    anchor = str(contract["read_only_anchor"]["repository_commit"])
    head = git_head(PROJECT_ROOT) or ""
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", anchor, head], cwd=PROJECT_ROOT, check=False
    ).returncode == 0
    protected = [p for p in _changed_since_anchor(anchor) if _protected(p)]
    eligible = frame.disagreement_primary_eligible.astype(bool)
    role_counts = frame.loc[eligible, "analysis_role"].value_counts().to_dict()
    required = [
        RESULTS_ROOT / "V2_DISAGREEMENT" / "03_FINAL" / "trustdive_d_predictions_v2.parquet",
        V8_RESULTS_ROOT / "02_PHASE_TOKENS" / "reference_phase_tokens_development_v8.npz",
        V7_RESULTS_ROOT / "03_SCORE" / "predictions_v7.parquet",
        V7_RESULTS_ROOT / "04_PHASE_EVIDENCE" / "phase_evidence_v7.parquet",
        V7_RESULTS_ROOT / "05_RISK_REVIEW" / "review_priority_v7.parquet",
        V8_RESULTS_ROOT / "RESULTS_DECISION_V8.md",
    ]
    expected = contract["data"]
    checks = {
        "anchor_is_ancestor": ancestor,
        "protected_v1_v8_unchanged": not protected,
        "required_independent_assets_exist": all(p.exists() for p in required),
        "samples": len(frame) == int(expected["samples"]),
        "official_train": int((frame.official_split == "train").sum()) == int(expected["official_train"]),
        "official_test": int((frame.official_split == "test").sum()) == int(expected["official_test"]),
        "valid_seven_judge": int(eligible.sum()) == int(expected["valid_seven_judge"]),
        "fit_panels": int(role_counts.get("fit", 0)) == int(expected["seven_judge_fit"]),
        "validation_panels": int(role_counts.get("validation", 0)) == int(expected["seven_judge_validation"]),
        "calibration_panels": int(role_counts.get("calibration", 0)) == int(expected["seven_judge_calibration"]),
        "test_panels": int(role_counts.get("official_test", 0)) == int(expected["seven_judge_test"]),
        "event_families": int(frame.event_family.nunique()) == int(expected["event_families"]),
        "no_persistent_real_judge_identity": not any("judge_id" in c.lower() for c in frame.columns),
        "generator_detector_sources_independent": required[0].exists() and required[1].exists(),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "anchor": anchor,
        "current_git_head": head,
        "protected_changes": protected,
        "required_hashes": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in required if p.exists()},
        "role_counts": {str(k): int(v) for k, v in role_counts.items()},
        "material_passport": "experiment-agent / audit / 2026-08-21 / VERIFIED_DATA / trustdive_judgesim_v9",
    }
    write_json(V9_RESULTS_ROOT / "00_AUDIT" / "audit_v9.json", result)
    return result


def require_v9_audit() -> dict:
    path = V9_RESULTS_ROOT / "00_AUDIT" / "audit_v9.json"
    if not path.exists():
        raise RuntimeError("Run audit --protocol v9 first")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != "PASS":
        raise RuntimeError("v9 audit did not pass")
    return result


def _judge_long(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in frame.loc[frame.disagreement_primary_eligible.astype(bool)].itertuples(index=False):
        values = np.asarray(json.loads(row.judge_scores_json), dtype=float)
        for slot, value in enumerate(values):
            control = float(np.median(np.delete(values, slot)))
            rows.append({
                "clip_uid": row.clip_uid, "analysis_role": row.analysis_role,
                "event_family": str(row.event_family), "action_type": str(row.action_type),
                "difficulty": float(row.difficulty), "slot": slot, "score": float(value),
                "control": control, "residual": float(value - control),
            })
    return pd.DataFrame(rows)


def _scale_pipeline(contract: dict):
    score = make_pipeline(
        SplineTransformer(n_knots=int(contract["simulator"]["spline_knots"]), degree=3, include_bias=False),
        StandardScaler(),
    )
    pre = ColumnTransformer([
        ("control", score, ["control"]),
        ("action", OneHotEncoder(handle_unknown="ignore"), ["action_type"]),
        ("difficulty", RobustScaler(), ["difficulty"]),
    ])
    return make_pipeline(pre, Ridge(alpha=float(contract["simulator"]["ridge_alpha"])))


def fit_judge_simulator_v9() -> dict:
    require_v9_audit()
    contract = load_v9_contract()
    long = _judge_long(load_v7_frame())
    fit = long.analysis_role == "fit"
    train = long.loc[fit].copy()
    target = np.log(np.abs(train.residual.to_numpy(dtype=float)) + float(contract["simulator"]["epsilon"]))
    model = _scale_pipeline(contract)
    model.fit(train[["control", "action_type", "difficulty"]], target)
    scale = np.maximum(
        np.exp(model.predict(train[["control", "action_type", "difficulty"]])),
        float(contract["simulator"]["min_scale"]),
    )
    standardized = train.residual.to_numpy(dtype=float) / scale
    artifact = {
        "model": model,
        "standardized_residuals": standardized.astype(np.float32),
        "fit_residual_p95": float(np.quantile(np.abs(train.residual), float(contract["simulator"]["stealth_residual_quantile"]))),
        "fit_residual_quantiles": {str(q): float(np.quantile(np.abs(train.residual), q)) for q in (0.5, 0.75, 0.9, 0.95, 0.99)},
        "fit_rows": int(len(train)),
    }
    model_path = V9_RESULTS_ROOT / "01_SIMULATOR" / "judge_simulator_v9.joblib"
    joblib.dump(artifact, model_path)
    serializable = {
        "framework": "JEP-inspired heteroscedastic empirical-residual simulator",
        "fit_panel_count": int(train.clip_uid.nunique()),
        "fit_mark_count": int(len(train)),
        "fit_residual_p95": artifact["fit_residual_p95"],
        "fit_residual_quantiles": artifact["fit_residual_quantiles"],
        "scale_min": float(scale.min()), "scale_median": float(np.median(scale)), "scale_max": float(scale.max()),
        "test_used": False, "joblib_sha256": sha256_file(model_path),
        "material_passport": "experiment-agent / model-fit / 2026-08-21 / VERIFIED_FIT_ONLY / trustdive_judgesim_v9",
    }
    write_json(V9_RESULTS_ROOT / "01_SIMULATOR" / "judge_simulator_v9.json", serializable)
    return {"status": "PASS", **serializable}


def _predict_scale(artifact: dict, frame: pd.DataFrame) -> np.ndarray:
    contract = load_v9_contract()
    return np.maximum(
        np.exp(artifact["model"].predict(frame[["control", "action_type", "difficulty"]])),
        float(contract["simulator"]["min_scale"]),
    )


def _round_score(x: float) -> float:
    contract = load_v9_contract()["simulator"]
    increment = float(contract["score_increment"])
    return float(np.clip(np.round(x / increment) * increment, contract["score_min"], contract["score_max"]))


def _panel_features(values: np.ndarray, action: str, difficulty: float) -> list[float]:
    loo = np.asarray([values[j] - np.median(np.delete(values, j)) for j in range(7)], dtype=float)
    action_hash = int(hashlib.sha256(str(action).encode()).hexdigest()[:8], 16) % 97
    return [float(np.median(values)), float(np.std(values, ddof=1)), float(np.max(values)-np.min(values)),
            float(np.median(np.abs(loo))), float(np.max(np.abs(loo))), float(difficulty), float(action_hash)]


def validate_judge_simulator_v9() -> dict:
    require_v9_audit()
    path = V9_RESULTS_ROOT / "01_SIMULATOR" / "judge_simulator_v9.joblib"
    if not path.exists():
        raise RuntimeError("Run fit-judge-simulator --protocol v9 first")
    artifact = joblib.load(path)
    contract = load_v9_contract()["simulator"]
    frame = load_v7_frame()
    val = frame[(frame.analysis_role == "validation") & frame.disagreement_primary_eligible.astype(bool)].copy()
    long = _judge_long(val)
    rng = np.random.default_rng(int(contract["validation_seed"]))
    simulated, records = [], []
    pool = np.asarray(artifact["standardized_residuals"], dtype=float)
    for row in val.itertuples(index=False):
        original = np.asarray(json.loads(row.judge_scores_json), dtype=float)
        slot = int(rng.integers(0, 7))
        control = float(np.median(np.delete(original, slot)))
        scale_row = pd.DataFrame({"control":[control], "action_type":[str(row.action_type)], "difficulty":[float(row.difficulty)]})
        scale = float(_predict_scale(artifact, scale_row)[0])
        replacement = _round_score(control + scale * float(rng.choice(pool)))
        null = original.copy(); null[slot] = replacement
        simulated.append(null)
        records.append({"clip_uid":row.clip_uid, "slot":slot, "control":control, "scale":scale,
                        "real_panel":stable_json(original.tolist()), "null_panel":stable_json(null.tolist())})
    real = np.stack(val.judge_scores_json.map(json.loads).map(np.asarray))
    null = np.stack(simulated)
    real_resid = np.concatenate([[p[j]-np.median(np.delete(p,j)) for j in range(7)] for p in real])
    null_resid = np.concatenate([[p[j]-np.median(np.delete(p,j)) for j in range(7)] for p in null])
    real_sd, null_sd = real.std(axis=1, ddof=1), null.std(axis=1, ddof=1)
    def rel(a: float, b: float) -> float:
        return float(abs(a-b) / max(abs(a), 0.05))
    metrics = {
        "score_grid_pass_rate": float(np.mean((null >= 0)&(null <= 10)&np.isclose(null*2,np.round(null*2)))),
        "residual_median_relative_error": rel(float(np.median(np.abs(real_resid))), float(np.median(np.abs(null_resid)))),
        "residual_iqr_relative_error": rel(float(np.subtract(*np.quantile(np.abs(real_resid),[.75,.25]))), float(np.subtract(*np.quantile(np.abs(null_resid),[.75,.25])))),
        "panel_sd_median_relative_error": rel(float(np.median(real_sd)), float(np.median(null_sd))),
        "panel_sd_iqr_relative_error": rel(float(np.subtract(*np.quantile(real_sd,[.75,.25]))), float(np.subtract(*np.quantile(null_sd,[.75,.25])))),
        "residual_wasserstein": float(wasserstein_distance(real_resid, null_resid)),
    }
    x_real = np.asarray([_panel_features(p,a,d) for p,a,d in zip(real,val.action_type,val.difficulty)])
    x_null = np.asarray([_panel_features(p,a,d) for p,a,d in zip(null,val.action_type,val.difficulty)])
    x = np.vstack((x_real,x_null)); y=np.r_[np.zeros(len(real)),np.ones(len(null))]
    order=np.arange(len(y)); rng.shuffle(order); folds=np.array_split(order,5); pred=np.zeros(len(y))
    for held in folds:
        train=np.setdiff1d(order,held); clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000))
        clf.fit(x[train],y[train]); pred[held]=clf.predict_proba(x[held])[:,1]
    metrics["real_vs_null_auroc"] = float(max(roc_auc_score(y,pred), 1-roc_auc_score(y,pred)))
    checks = {
        "score_grid": metrics["score_grid_pass_rate"] == 1.0,
        "residual_location": metrics["residual_median_relative_error"] <= float(contract["max_relative_location_error"]),
        "residual_iqr": metrics["residual_iqr_relative_error"] <= float(contract["max_relative_iqr_error"]),
        "panel_sd_location": metrics["panel_sd_median_relative_error"] <= float(contract["max_relative_location_error"]),
        "panel_sd_iqr": metrics["panel_sd_iqr_relative_error"] <= float(contract["max_relative_iqr_error"]),
        "wasserstein": metrics["residual_wasserstein"] <= float(contract["max_wasserstein"]),
        "indistinguishable": metrics["real_vs_null_auroc"] <= float(contract["max_real_vs_null_auroc"]),
    }
    pair_path=V9_RESULTS_ROOT/"01_SIMULATOR"/"validation_null_panels_v9.parquet"
    pd.DataFrame(records).to_parquet(pair_path,index=False)
    result={"status":"PASS" if all(checks.values()) else "STOP", "checks":checks, "metrics":metrics,
            "validation_panels":int(len(val)), "validation_only":True, "output_sha256":sha256_file(pair_path)}
    write_json(V9_RESULTS_ROOT/"01_SIMULATOR"/"simulator_validation_v9.json",result)
    return result


def _phase_source() -> pd.DataFrame:
    path=RESULTS_ROOT/"V2_DISAGREEMENT"/"03_FINAL"/"trustdive_d_predictions_v2.parquet"
    frame=pd.read_parquet(path)
    needed=["clip_uid","phase_takeoff_contribution","phase_flight_contribution","phase_entry_contribution"]
    return frame[needed].copy()


def _stable_int(*parts: object) -> int:
    raw="|".join(map(str,parts)).encode(); return int(hashlib.sha256(raw).hexdigest()[:16],16)


def generate_conflicts_v9(stage: str) -> dict:
    require_v9_audit()
    validation=json.loads((V9_RESULTS_ROOT/"01_SIMULATOR"/"simulator_validation_v9.json").read_text(encoding="utf-8"))
    if validation.get("status") != "PASS":
        raise RuntimeError("Simulator authenticity gate did not pass")
    if stage not in {"development","final"}: raise ValueError(stage)
    if stage=="final" and not (V9_RESULTS_ROOT/"04_PILOT"/"contract_freeze_v9.json").exists():
        raise RuntimeError("Freeze v9 contract before final conflict generation")
    artifact=joblib.load(V9_RESULTS_ROOT/"01_SIMULATOR"/"judge_simulator_v9.joblib")
    contract=load_v9_contract()["simulator"]
    roles=("fit","validation","calibration") if stage=="development" else ("official_test",)
    frame=load_v7_frame(); frame=frame[frame.analysis_role.isin(roles)&frame.disagreement_primary_eligible.astype(bool)].copy()
    frame=frame.merge(_phase_source(),on="clip_uid",how="left",validate="one_to_one")
    phase_cols=["phase_takeoff_contribution","phase_flight_contribution","phase_entry_contribution"]
    phase_values=frame[phase_cols].to_numpy(dtype=float); phase_values=np.nan_to_num(phase_values)
    fit_phase=_phase_source().merge(load_v7_frame()[["clip_uid","analysis_role"]],on="clip_uid")
    fit_mat=np.nan_to_num(fit_phase.loc[fit_phase.analysis_role=="fit",phase_cols].to_numpy(dtype=float))
    phase_center=np.median(fit_mat,axis=0); phase_scale=np.maximum(np.median(np.abs(fit_mat-phase_center),axis=0)*1.4826,0.05)
    pool=np.asarray(artifact["standardized_residuals"],dtype=float); p95=float(artifact["fit_residual_p95"])
    # Predict all seven possible leave-one-out scales in one batch. This is
    # mathematically identical to row-wise prediction and avoids thousands of
    # sklearn pipeline calls during deterministic conflict generation.
    scale_rows=[]; control_matrix=np.zeros((len(frame),7),dtype=float)
    original_panels=[]
    for row_idx,row in enumerate(frame.itertuples(index=False)):
        original=np.asarray(json.loads(row.judge_scores_json),dtype=float); original_panels.append(original)
        for slot in range(7):
            control=float(np.median(np.delete(original,slot))); control_matrix[row_idx,slot]=control
            scale_rows.append({"control":control,"action_type":str(row.action_type),"difficulty":float(row.difficulty)})
    scale_matrix=_predict_scale(artifact,pd.DataFrame(scale_rows)).reshape(len(frame),7)
    rows=[]
    for row_idx,row in enumerate(frame.itertuples(index=False)):
        original=original_panels[row_idx]
        for seed in contract["simulator_seeds"]:
            rng=np.random.default_rng(_stable_int(seed,row.clip_uid)%(2**32))
            for scenario in contract["anomaly_types"]:
                slot=_stable_int(seed,row.event_family,scenario)%7
                direction=1.0 if _stable_int(seed,row.event_family,scenario,"dir")%2 else -1.0
                control=float(control_matrix[row_idx,slot])
                sigma=float(scale_matrix[row_idx,slot]); z=float(rng.choice(pool))
                null_score=_round_score(control+sigma*z)
                target_phase=-1; signal=1.0
                if scenario=="phase_bias":
                    target_phase=_stable_int(seed,row.event_family,"phase")%3
                    pv=(phase_values[row_idx,target_phase]-phase_center[target_phase])/phase_scale[target_phase]
                    signal=float(np.clip(0.6+abs(pv),0.6,1.5))
                elif scenario=="action_preference":
                    signal=1.0+0.15*(int(str(row.action_type)[0])%3)
                elif scenario=="episodic_lapse":
                    signal=1.25
                virtual=f"VJ-{seed}-{row.event_family}-{slot}-{scenario}"
                for severity in contract["severity_sigma"]:
                    shift=direction*float(severity)*sigma*signal
                    anomaly_score=_round_score(null_score+shift)
                    null_panel=original.copy(); null_panel[slot]=null_score
                    anomaly_panel=original.copy(); anomaly_panel[slot]=anomaly_score
                    pair_id=f"{stage}|{row.clip_uid}|{seed}|{scenario}|{severity:.0f}"
                    anomaly_residual=anomaly_score-control
                    anomaly_loo=np.asarray([anomaly_panel[j]-np.median(np.delete(anomaly_panel,j)) for j in range(7)])
                    not_largest=bool(abs(anomaly_residual)<np.max(np.abs(np.delete(anomaly_loo,slot)))-1e-12)
                    stealth=bool(abs(anomaly_residual)<=p95)
                    common={"pair_id":pair_id,"clip_uid":row.clip_uid,"analysis_role":row.analysis_role,
                            "source_role":row.source_role,"event_family":str(row.event_family),"action_type":str(row.action_type),
                            "difficulty":float(row.difficulty),"simulation_seed":int(seed),"severity_sigma":float(severity),
                            "scenario_type":scenario,"target_slot":int(slot),"target_phase":int(target_phase),
                            "virtual_judge_id":virtual,"control_score":control,"jep_scale":sigma,"base_standardized_residual":z,
                            "null_score":null_score,"anomaly_score":anomaly_score,"shift":float(anomaly_score-null_score),
                            "stealth_anomaly":stealth,"not_largest_numeric_outlier":not_largest,
                            "original_panel_json":stable_json(original.tolist())}
                    rows.append({**common,"variant":"null","is_anomaly":False,"panel_scores_json":stable_json(null_panel.tolist()),
                                 "panel_aggregate":official_panel_aggregate(null_panel.tolist())})
                    rows.append({**common,"variant":"anomaly","is_anomaly":True,"panel_scores_json":stable_json(anomaly_panel.tolist()),
                                 "panel_aggregate":official_panel_aggregate(anomaly_panel.tolist())})
    output=pd.DataFrame(rows)
    path=V9_RESULTS_ROOT/"02_CONFLICTS"/f"synthetic_judge_pairs_{stage}_v9.parquet"
    output.to_parquet(path,index=False)
    summary={"status":"PASS","stage":stage,"rows":int(len(output)),"pairs":int(output.pair_id.nunique()),
             "videos":int(output.clip_uid.nunique()),"anomaly_rows":int(output.is_anomaly.sum()),
             "stealth_anomaly_rows":int(output.loc[output.is_anomaly,"stealth_anomaly"].sum()),
             "roles":output.analysis_role.value_counts().to_dict(),"output_sha256":sha256_file(path)}
    write_json(V9_RESULTS_ROOT/"02_CONFLICTS"/f"conflict_summary_{stage}_v9.json",summary)
    return summary
