from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .util import sha256_file, write_json
from .v7_data import V7_RESULTS_ROOT, load_v7_frame
from .v8_data import V8_RESULTS_ROOT
from .v9_data import V9_RESULTS_ROOT, load_v9_contract
from .v9_modeling import _baseline_scores, _fit_one, _metrics, _read_pairs
from .v9_features import load_features_v9


def stress_test_v9() -> dict:
    if not (V9_RESULTS_ROOT / "05_FINAL" / "predictions_v9.parquet").exists():
        raise RuntimeError("Run v9 final training first")
    dev=load_features_v9("development"); test=load_features_v9("final")
    contract=load_v9_contract(); pilot=json.loads((V9_RESULTS_ROOT/"04_PILOT"/"pilot_gate_v9.json").read_text(encoding="utf-8"))
    hidden=int(pilot["selected"]["hidden"]); dropout=float(pilot["selected"]["dropout"])
    rows=[]
    for held_name in ("persistent_bias","action_preference","phase_bias","episodic_lapse"):
        train=np.flatnonzero(np.isin(dev["analysis_role"],["fit","validation"]) & ~((dev["scenario_type"]==held_name)&(dev["is_anomaly"]>0)))
        val=np.flatnonzero(dev["analysis_role"]=="calibration")
        model,scalers,_,_,epochs=_fit_one(dev,train,val,hidden,dropout,20260831,True,min(35,int(contract["model"]["final_epochs"])))
        test_idx=np.flatnonzero(test["scenario_type"]==held_name)
        from .v9_modeling import _predict
        pred=_predict(model,test,test_idx,scalers,next(model.parameters()).device); metrics,_=_metrics(test,test_idx,pred)
        rows.append({"held_out_type":held_name,"epochs":epochs,"auroc":metrics["auroc"],"auprc":metrics["auprc"],
                     "judge_top1":metrics["judge_top1"],"phase_accuracy":metrics["phase_accuracy"]})
    final=pd.read_parquet(V9_RESULTS_ROOT/"05_FINAL"/"predictions_v9.parquet")
    source=[]
    for role,part in final.groupby("source_role"):
        if part.is_anomaly.nunique()<2: continue
        source.append({"source_role":str(role),"rows":int(len(part)),"auroc":float(roc_auc_score(part.is_anomaly,part.panel_risk)),
                       "auprc":float(average_precision_score(part.is_anomaly,part.panel_risk))})
    path=V9_RESULTS_ROOT/"06_STRESS"/"stress_results_v9.parquet"
    pd.DataFrame(rows).to_parquet(path,index=False)
    result={"status":"PASS","loto":rows,"loto_mean_auroc":float(np.mean([x["auroc"] for x in rows])),
            "loto_min_auroc":float(np.min([x["auroc"] for x in rows])),"source_robustness":source,"output_sha256":sha256_file(path)}
    write_json(V9_RESULTS_ROOT/"06_STRESS"/"stress_summary_v9.json",result); return result


def _queue_metrics(selected: pd.DataFrame, score_col: str, review_n: int) -> dict:
    order=np.argsort(-selected[score_col].to_numpy(dtype=float)); reviewed=np.zeros(len(selected),bool); reviewed[order[:review_n]]=True
    anomaly=selected.is_anomaly.to_numpy(dtype=bool); error=selected.aggregate_deviation.to_numpy(dtype=float)
    recall=float(anomaly[reviewed].sum()/max(anomaly.sum(),1)); enrichment=float(anomaly[reviewed].mean()/max(anomaly.mean(),1e-8))
    base=float(error.mean()); kept=float(error[~reviewed].mean()); reduction=float((base-kept)/max(base,1e-8))
    return {"recall_at_20":recall,"enrichment":enrichment,"unreviewed_error":kept,"unreviewed_error_reduction":reduction}


def analyze_review_v9() -> dict:
    final=pd.read_parquet(V9_RESULTS_ROOT/"05_FINAL"/"predictions_v9.parquet")
    baseline=_baseline_scores(_read_pairs("final"))
    final=final.merge(baseline[["pair_id","variant","max_jep_z"]],on=["pair_id","variant"],how="left",validate="one_to_one")
    v7=pd.read_parquet(V7_RESULTS_ROOT/"05_RISK_REVIEW"/"review_priority_v7.parquet").set_index("clip_uid")
    pred7=pd.read_parquet(V7_RESULTS_ROOT/"03_SCORE"/"predictions_v7.parquet").set_index("clip_uid")
    final["v7_risk"]=[float(v7.loc[x,"review_priority"]) for x in final.clip_uid]
    final["rica_uncertainty"]=[float(pred7.loc[x,"teacher_uncertainty"]) for x in final.clip_uid]
    null_aggregate=final.loc[~final.is_anomaly].set_index(["pair_id"])["panel_aggregate"].to_dict()
    final["aggregate_deviation"]=[abs(v-null_aggregate[p]) for p,v in zip(final.pair_id,final.panel_aggregate)]
    clips=np.asarray(sorted(final.clip_uid.unique())); n=len(clips); anomaly_n=int(round(n*0.20)); review_n=anomaly_n
    methods={"low_score":"low_score_risk","max_jep_residual":"max_jep_z","rica_uncertainty":"rica_uncertainty","v7_risk":"v7_risk","v9_risk":"panel_risk"}
    queue_rows=[]
    for queue in range(int(load_v9_contract()["review"]["queue_count"])):
        rng=np.random.default_rng(20260830+queue); anomaly_clips=set(rng.choice(clips,size=anomaly_n,replace=False)); chosen=[]
        type_cycle=np.resize(np.asarray(["persistent_bias","action_preference","phase_bias","episodic_lapse"]),anomaly_n)
        strength_cycle=np.resize(np.asarray([1.,2.,3.]),anomaly_n)
        for pos,clip in enumerate(clips):
            part=final[final.clip_uid==clip]
            if clip in anomaly_clips:
                cand=part[(part.is_anomaly)&(part.scenario_type==type_cycle[pos%len(type_cycle)])&(part.severity_sigma==strength_cycle[pos%len(strength_cycle)])]
            else:
                cand=part[~part.is_anomaly]
            if cand.empty: cand=part[part.is_anomaly==(clip in anomaly_clips)]
            chosen.append(cand.iloc[int(rng.integers(0,len(cand)))])
        selected=pd.DataFrame(chosen).reset_index(drop=True); selected["low_score_risk"]=-selected.panel_aggregate
        random_recall=[]; random_enrich=[]; random_reduction=[]
        for _ in range(1000):
            risk=rng.random(n); selected["_random"]=risk; met=_queue_metrics(selected,"_random",review_n)
            random_recall.append(met["recall_at_20"]); random_enrich.append(met["enrichment"]); random_reduction.append(met["unreviewed_error_reduction"])
        queue_rows.append({"queue":queue,"method":"random","recall_at_20":float(np.mean(random_recall)),"enrichment":float(np.mean(random_enrich)),"unreviewed_error_reduction":float(np.mean(random_reduction))})
        for name,col in methods.items(): queue_rows.append({"queue":queue,"method":name,**_queue_metrics(selected,col,review_n)})
    queues=pd.DataFrame(queue_rows); qpath=V9_RESULTS_ROOT/"07_REVIEW"/"operational_queues_v9.parquet"; queues.to_parquet(qpath,index=False)
    summary=queues.groupby("method")[["recall_at_20","enrichment","unreviewed_error_reduction"]].agg(["mean","std"]).round(6)
    summary.columns=["_".join(x) for x in summary.columns]; summary=summary.reset_index(); spath=V9_RESULTS_ROOT/"07_REVIEW"/"review_summary_v9.csv"; summary.to_csv(spath,index=False)
    v9=summary[summary.method=="v9_risk"].iloc[0]; result={"status":"PASS","queues":int(queues.queue.nunique()),"videos_per_queue":n,
        "review_v9":{"recall_at_20":float(v9.recall_at_20_mean),"enrichment":float(v9.enrichment_mean),"unreviewed_error_reduction":float(v9.unreviewed_error_reduction_mean)},
        "queue_sha256":sha256_file(qpath),"summary_sha256":sha256_file(spath)}
    write_json(V9_RESULTS_ROOT/"07_REVIEW"/"review_analysis_v9.json",result); return result


def final_decision_v9() -> dict:
    contract=load_v9_contract(); simulator=json.loads((V9_RESULTS_ROOT/"01_SIMULATOR"/"simulator_validation_v9.json").read_text(encoding="utf-8"))
    pilot=json.loads((V9_RESULTS_ROOT/"04_PILOT"/"pilot_gate_v9.json").read_text(encoding="utf-8"))
    final_path=V9_RESULTS_ROOT/"05_FINAL"/"final_metrics_v9.json"
    if not final_path.exists():
        result={"status":"STOP","publication_decision":"NO_GO_AT_PILOT","simulator":simulator.get("status"),
                "pilot":pilot.get("status"),"reason":"judge-phase detector did not exceed scalar/style baselines and phase localization remained weak",
                "pilot_metrics":pilot.get("selected",{}),"final_test_generated":False,"formal_figures_locked":True,
                "real_data_track":{"v7_results_preserved":True,"v8_natural_disagreement_stop_preserved":True}}
        write_json(V9_RESULTS_ROOT/"07_REVIEW"/"analysis_summary_v9.json",result)
        lines=["# TrustDive-JudgeSim v9 result decision","","- Decision: **NO-GO at development pilot**",
               f"- Simulator authenticity: **{simulator.get('status')}**","- Official-test conflicts generated: **No**",
               "- Formal manuscript figures: **Locked**","",
               "The JEP-calibrated null simulator passed every realism gate, but the Judge-Phase network did not outperform the strongest scalar/style baseline. Its phase evidence ablation was neutral and conflict-stage localization remained weak. The controlled benchmark therefore does not support the planned psychological mechanism claim.","",
               "All synthetic claims are limited to virtual controlled deviations and do not identify real judges or misconduct."]
        (V9_RESULTS_ROOT/"RESULTS_DECISION_V9.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        return result
    final=json.loads(final_path.read_text(encoding="utf-8")); stress=json.loads((V9_RESULTS_ROOT/"06_STRESS"/"stress_summary_v9.json").read_text(encoding="utf-8")); review=json.loads((V9_RESULTS_ROOT/"07_REVIEW"/"review_analysis_v9.json").read_text(encoding="utf-8"))
    baselines=_baseline_scores(_read_pairs("final")); val=baselines.is_anomaly.astype(int).to_numpy(); best_ap=max(average_precision_score(val,baselines[c]) for c in ["max_loo_residual","max_jep_z","rasch_fixed_effect_residual"] if c in baselines)
    m=final["metrics"]; r=review["review_v9"]
    gates={"simulator_authentic":simulator["status"]=="PASS","all_auroc":m["auroc"]>=float(contract["final"]["minimum_all_anomaly_auroc"]),
           "stealth_phase_auroc":m["stealth_phase_auroc"]>=float(contract["final"]["minimum_stealth_phase_auroc"]),
           "ap_gain":m["auprc"]-best_ap>=float(contract["final"]["minimum_auprc_gain_over_score_baseline"]),
           "judge_top1":m["judge_top1"]>=float(contract["final"]["minimum_judge_top1"]),"phase":m["phase_accuracy"]>=float(contract["final"]["minimum_phase_accuracy"]),
           "loto":stress["loto_mean_auroc"]>=float(contract["final"]["minimum_loto_mean_auroc"]) and stress["loto_min_auroc"]>=float(contract["final"]["minimum_loto_type_auroc"]),
           "review":r["recall_at_20"]>=float(contract["review"]["minimum_recall"]) and r["enrichment"]>=float(contract["review"]["minimum_enrichment"]) and r["unreviewed_error_reduction"]>=float(contract["review"]["minimum_unreviewed_error_reduction"])}
    strong=all(gates.values()); application=simulator["status"]=="PASS" and m["auroc"]>=.75 and m["phase_accuracy"]>.50
    decision="FRONTIERS_PSYCHOLOGY_8_5_GO" if strong else ("CONTROLLED_BENCHMARK_APPLICATION_GO" if application else "NO_GO")
    real_anchor={"v7_score_summary_sha256":sha256_file(V7_RESULTS_ROOT/"03_SCORE"/"score_summary_v7.json"),"v7_phase_evidence_sha256":sha256_file(V7_RESULTS_ROOT/"04_PHASE_EVIDENCE"/"phase_evidence_summary_v7.json"),"v8_stop_sha256":sha256_file(V8_RESULTS_ROOT/"RESULTS_DECISION_V8.md")}
    result={"status":"PASS" if strong else "CONDITIONAL","publication_decision":decision,"gates":gates,"final_metrics":m,"best_score_baseline_auprc":best_ap,"review":r,"real_data_track":real_anchor}
    write_json(V9_RESULTS_ROOT/"07_REVIEW"/"analysis_summary_v9.json",result)
    lines=["# TrustDive-JudgeSim v9 result decision","",f"- Decision: **{decision}**",f"- Simulator: **{simulator['status']}**",f"- Pilot: **{pilot['status']}**",f"- Final anomaly AUROC: `{m['auroc']:.4f}`",f"- Stealth phase AUROC: `{m['stealth_phase_auroc']:.4f}`",f"- Judge Top-1: `{m['judge_top1']:.4f}`",f"- Phase accuracy: `{m['phase_accuracy']:.4f}`",f"- Review Recall@20: `{r['recall_at_20']:.4f}`",f"- Review enrichment: `{r['enrichment']:.4f}`","","Synthetic results concern controlled virtual judging deviations only; they do not identify real judges or psychological misconduct."]
    (V9_RESULTS_ROOT/"RESULTS_DECISION_V9.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return result
