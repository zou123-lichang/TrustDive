from __future__ import annotations

import json
import math
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .util import sha256_file, write_json
from .v9_data import V9_RESULTS_ROOT, V9_RUN_ROOT, load_v9_contract
from .v9_features import PHASE_NAMES, TYPE_NAMES, load_features_v9


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if np.unique(y).size == 2 else float("nan")


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if np.unique(y).size == 2 else float("nan")


def _read_pairs(stage: str) -> pd.DataFrame:
    return pd.read_parquet(V9_RESULTS_ROOT / "02_CONFLICTS" / f"synthetic_judge_pairs_{stage}_v9.parquet")


def _baseline_scores(pairs: pd.DataFrame) -> pd.DataFrame:
    artifact = joblib.load(V9_RESULTS_ROOT / "01_SIMULATOR" / "judge_simulator_v9.joblib")
    panels=np.stack(pairs.panel_scores_json.map(json.loads).map(np.asarray)).astype(float)
    controls=np.zeros_like(panels)
    for slot in range(7): controls[:,slot]=np.median(np.delete(panels,slot,axis=1),axis=1)
    scale_frame=pd.DataFrame({"control":controls.reshape(-1),"action_type":np.repeat(pairs.action_type.astype(str).to_numpy(),7),
                              "difficulty":np.repeat(pairs.difficulty.to_numpy(dtype=float),7)})
    scales=np.maximum(np.exp(artifact["model"].predict(scale_frame)),0.20).reshape(len(pairs),7)
    rows=[]
    for i,row in enumerate(pairs.itertuples(index=False)):
        values=panels[i]
        residual=values-controls[i]; z=residual/scales[i]
        rows.append({"pair_id":row.pair_id,"clip_uid":row.clip_uid,"analysis_role":row.analysis_role,
                     "event_family":str(row.event_family),"variant":row.variant,"is_anomaly":bool(row.is_anomaly),
                     "scenario_type":row.scenario_type,"severity_sigma":float(row.severity_sigma),
                     "stealth_anomaly":bool(row.stealth_anomaly),"max_loo_residual":float(np.max(np.abs(residual))),
                     "max_jep_z":float(np.max(np.abs(z))),"target_residual":float(abs(residual[int(row.target_slot)])),
                     "target_jep_z":float(abs(z[int(row.target_slot)]))})
    return pd.DataFrame(rows)


def train_baselines_v9() -> dict:
    pairs=_read_pairs("development"); scores=_baseline_scores(pairs)
    fit=scores.analysis_role=="fit"; val=scores.analysis_role=="validation"
    # Rasch-style severity/action fixed-effect residual fitted only on normal fit panels.
    fit_pairs=pairs.loc[fit & ~pairs.is_anomaly].copy()
    target=(fit_pairs.null_score-fit_pairs.control_score).to_numpy(dtype=float)
    features=["virtual_judge_id","action_type","difficulty"]
    pre=ColumnTransformer([("cat",OneHotEncoder(handle_unknown="ignore"),features[:2]),("num",StandardScaler(),["difficulty"])])
    rasch=make_pipeline(pre,Ridge(alpha=10.0)); rasch.fit(fit_pairs[features],target)
    expected=rasch.predict(pairs[features]); scores["rasch_fixed_effect_residual"]=np.abs((pairs.anomaly_score-pairs.control_score)-expected)
    null_fit=scores.loc[fit & ~scores.is_anomaly]
    style=null_fit.groupby(pairs.loc[null_fit.index,"virtual_judge_id"])["target_residual"].median().to_dict()
    scores["kendall_style_deviation"]=[abs(v-style.get(j,0.0)) for v,j in zip(scores.target_residual,pairs.virtual_judge_id)]
    # Video-only uncertainty/risk baselines are intentionally identical within each matched pair.
    v7=pd.read_parquet(V9_RESULTS_ROOT.parent/"V7_RISK_TASK"/"05_RISK_REVIEW"/"review_priority_v7.parquet").set_index("clip_uid")
    scores["v7_risk"]=[float(v7.loc[x,"review_priority"]) for x in scores.clip_uid]
    pred7=pd.read_parquet(V9_RESULTS_ROOT.parent/"V7_RISK_TASK"/"03_SCORE"/"predictions_v7.parquet").set_index("clip_uid")
    scores["rica_uncertainty"]=[float(pred7.loc[x,"teacher_uncertainty"]) for x in scores.clip_uid]
    metric_rows=[]
    for name in ("max_loo_residual","max_jep_z","rasch_fixed_effect_residual","kendall_style_deviation","rica_uncertainty","v7_risk"):
        y=scores.loc[val,"is_anomaly"].astype(int).to_numpy(); s=scores.loc[val,name].to_numpy(dtype=float)
        metric_rows.append({"model":name,"validation_auroc":_safe_auc(y,s),"validation_auprc":_safe_ap(y,s)})
    out=V9_RESULTS_ROOT/"04_PILOT"/"baseline_predictions_v9.parquet"; scores.to_parquet(out,index=False)
    metric_path=V9_RESULTS_ROOT/"04_PILOT"/"baseline_metrics_v9.csv"; pd.DataFrame(metric_rows).to_csv(metric_path,index=False)
    joblib.dump(rasch,V9_RUN_ROOT/"checkpoints"/"rasch_baseline_v9.joblib")
    result={"status":"PASS","rows":int(len(scores)),"validation_metrics":metric_rows,
            "prediction_sha256":sha256_file(out),"metrics_sha256":sha256_file(metric_path)}
    write_json(V9_RESULTS_ROOT/"04_PILOT"/"baseline_summary_v9.json",result); return result


class JudgePhaseNet(nn.Module):
    def __init__(self, judge_dim: int, phase_dim: int, global_dim: int, action_count: int,
                 hidden: int, dropout: float, include_phase: bool=True):
        super().__init__(); self.include_phase=include_phase
        self.judge=nn.Sequential(nn.Linear(judge_dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,hidden))
        self.phase=nn.Sequential(nn.Linear(phase_dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,hidden))
        self.attn=nn.MultiheadAttention(hidden,2,batch_first=True,dropout=dropout)
        self.action=nn.Embedding(action_count+1,min(16,hidden//2)); context_dim=hidden+global_dim+self.action.embedding_dim
        self.context=nn.Sequential(nn.Linear(context_dim,hidden),nn.GELU(),nn.Dropout(dropout))
        self.panel=nn.Linear(hidden,1); self.slot=nn.Linear(hidden*2,1)
        self.type_head=nn.Linear(hidden,len(TYPE_NAMES)); self.phase_head=nn.Linear(hidden,3)
    def forward(self,j,p,g,a):
        jh=self.judge(j)
        if self.include_phase:
            ph=self.phase(p); cross,_=self.attn(jh,ph,ph,need_weights=False); jh=jh+cross
        pooled=jh.mean(1); ctx=self.context(torch.cat((pooled,g,self.action(a)),dim=1))
        slot=self.slot(torch.cat((jh,ctx[:,None,:].expand(-1,7,-1)),dim=-1)).squeeze(-1)
        return self.panel(ctx).squeeze(-1),slot,self.type_head(ctx),self.phase_head(ctx)


@dataclass
class Scalers:
    judge_mean: np.ndarray; judge_std: np.ndarray; phase_mean: np.ndarray; phase_std: np.ndarray
    global_mean: np.ndarray; global_std: np.ndarray


def _standardize(data: dict[str,np.ndarray], fit_mask: np.ndarray, scalers: Scalers|None=None):
    if scalers is None:
        def ms(x):
            m=x[fit_mask].reshape(-1,x.shape[-1]).mean(0); s=x[fit_mask].reshape(-1,x.shape[-1]).std(0); return m,np.where(s>1e-6,s,1)
        jm,js=ms(data["judge_token"]); pm,ps=ms(data["phase_token"]); gm,gs=ms(data["global_feature"])
        scalers=Scalers(jm,js,pm,ps,gm,gs)
    return ((data["judge_token"]-scalers.judge_mean)/scalers.judge_std).astype(np.float32),\
           ((data["phase_token"]-scalers.phase_mean)/scalers.phase_std).astype(np.float32),\
           ((data["global_feature"]-scalers.global_mean)/scalers.global_std).astype(np.float32),scalers


def _loss(model,batch,permutation_weight:float):
    j,p,g,a,y,slot,typ,phase=batch; panel,slot_logits,type_logits,phase_logits=model(j,p,g,a)
    bce=nn.functional.binary_cross_entropy_with_logits(panel,y)
    mask=y>0.5; loss=bce
    if mask.any():
        loss=loss+0.5*nn.functional.cross_entropy(slot_logits[mask],slot[mask])+0.25*nn.functional.cross_entropy(type_logits[mask],typ[mask])
        pmask=mask&(phase>=0)
        if pmask.any(): loss=loss+0.25*nn.functional.cross_entropy(phase_logits[pmask],phase[pmask])
    if permutation_weight>0:
        perm=torch.randperm(7,device=j.device); inv=torch.argsort(perm)
        ppanel,pslot,_,_=model(j[:,perm],p,g,a)
        loss=loss+permutation_weight*(nn.functional.mse_loss(ppanel,panel.detach())+nn.functional.mse_loss(pslot[:,inv],slot_logits.detach()))
    return loss


def _predict(model,data,indices,scalers,device):
    fit_dummy=np.ones(len(data["is_anomaly"]),bool); j,p,g,_=_standardize(data,fit_dummy,scalers)
    model.eval(); out=[[],[],[],[]]
    with torch.no_grad():
        for start in range(0,len(indices),2048):
            idx=indices[start:start+2048]
            values=model(torch.from_numpy(j[idx]).to(device),torch.from_numpy(p[idx]).to(device),
                         torch.from_numpy(g[idx]).to(device),torch.from_numpy(data["action_index"][idx]).long().to(device))
            for target,value in zip(out,values): target.append(value.cpu().numpy())
    return tuple(np.concatenate(x) for x in out)


def _metrics(data,indices,pred):
    panel,slot,type_logits,phase_logits=pred; score=1/(1+np.exp(-panel)); y=data["is_anomaly"][indices].astype(int)
    anomaly=y==1; phase_anom=anomaly&(data["target_phase"][indices]>=0)
    stealth_phase=(data["scenario_type"][indices]=="phase_bias")&((data["stealth_anomaly"][indices]>0)|(~anomaly))&(data["severity_sigma"][indices]<=2)
    metrics={"auroc":_safe_auc(y,score),"auprc":_safe_ap(y,score),
             "judge_top1":float(np.mean(np.argmax(slot[anomaly],axis=1)==data["target_slot"][indices][anomaly])) if anomaly.any() else float("nan"),
             "judge_top2":float(np.mean([t in np.argsort(s)[-2:] for s,t in zip(slot[anomaly],data["target_slot"][indices][anomaly])])) if anomaly.any() else float("nan"),
             "type_macro_f1":float(f1_score(data["type_index"][indices][anomaly],np.argmax(type_logits[anomaly],axis=1),average="macro")) if anomaly.any() else float("nan"),
             "phase_accuracy":float(np.mean(np.argmax(phase_logits[phase_anom],axis=1)==data["target_phase"][indices][phase_anom])) if phase_anom.any() else float("nan"),
             "stealth_phase_auroc":_safe_auc(y[stealth_phase],score[stealth_phase]) if stealth_phase.any() else float("nan"),
             "stealth_phase_auprc":_safe_ap(y[stealth_phase],score[stealth_phase]) if stealth_phase.any() else float("nan")}
    ids=data["pair_id"][indices]; frame=pd.DataFrame({"pair":ids,"y":y,"s":score}); pivot=frame.pivot_table(index="pair",columns="y",values="s",aggfunc="mean")
    metrics["paired_ranking_accuracy"]=float(np.mean(pivot[1]>pivot[0])) if {0,1}.issubset(pivot.columns) else float("nan")
    return metrics,score


def _fit_one(data,train_idx,val_idx,hidden,dropout,seed,include_phase,epochs):
    torch.manual_seed(seed); np.random.seed(seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fit_mask=np.zeros(len(data["is_anomaly"]),bool); fit_mask[train_idx]=True
    j,p,g,scalers=_standardize(data,fit_mask)
    model=JudgePhaseNet(j.shape[-1],p.shape[-1],g.shape[-1],int(data["action_index"].max()),hidden,dropout,include_phase).to(device)
    tensors=TensorDataset(torch.from_numpy(j[train_idx]),torch.from_numpy(p[train_idx]),torch.from_numpy(g[train_idx]),
                          torch.from_numpy(data["action_index"][train_idx]).long(),torch.from_numpy(data["is_anomaly"][train_idx]).float(),
                          torch.from_numpy(data["target_slot"][train_idx]).long(),torch.from_numpy(data["type_index"][train_idx]).long(),
                          torch.from_numpy(data["target_phase"][train_idx]).long())
    loader=DataLoader(tensors,batch_size=int(load_v9_contract()["model"]["batch_size"]),shuffle=True)
    opt=torch.optim.AdamW(model.parameters(),lr=float(load_v9_contract()["model"]["learning_rate"]),weight_decay=1e-4)
    best=None; best_ap=-1.; wait=0; patience=int(load_v9_contract()["model"]["patience"])
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            batch=tuple(x.to(device) for x in batch); opt.zero_grad(set_to_none=True); loss=_loss(model,batch,0.1); loss.backward(); opt.step()
        pred=_predict(model,data,val_idx,scalers,device); metrics,_=_metrics(data,val_idx,pred); ap=metrics["stealth_phase_auprc"]
        if np.isfinite(ap) and ap>best_ap+1e-4:
            best_ap=ap; wait=0; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else: wait+=1
        if wait>=patience: break
    if best is not None: model.load_state_dict(best)
    pred=_predict(model,data,val_idx,scalers,device); metrics,score=_metrics(data,val_idx,pred)
    return model,scalers,metrics,score,epoch+1


def pilot_v9() -> dict:
    data=load_features_v9("development"); roles=data["analysis_role"]
    train=np.flatnonzero(roles=="fit"); val=np.flatnonzero(roles=="validation")
    contract=load_v9_contract(); trials=[]; best_obj=-1; best_bundle=None
    for hidden in contract["model"]["hidden_dimensions"]:
        for dropout in contract["model"]["dropouts"]:
            model,scalers,metrics,score,epochs=_fit_one(data,train,val,int(hidden),float(dropout),20260827,True,int(contract["model"]["pilot_epochs"]))
            row={"hidden":hidden,"dropout":dropout,"include_phase":True,"epochs":epochs,**metrics}; trials.append(row)
            if metrics["stealth_phase_auprc"]>best_obj:
                best_obj=metrics["stealth_phase_auprc"]; best_bundle=(model,scalers,row)
    hidden=int(best_bundle[2]["hidden"]); dropout=float(best_bundle[2]["dropout"])
    no_phase,ns,nm,nscore,ne=_fit_one(data,train,val,hidden,dropout,20260827,False,int(contract["model"]["pilot_epochs"]))
    trials.append({"hidden":hidden,"dropout":dropout,"include_phase":False,"epochs":ne,**nm})
    baseline=pd.read_csv(V9_RESULTS_ROOT/"04_PILOT"/"baseline_metrics_v9.csv"); best_baseline=float(baseline.validation_auprc.max())
    chosen=best_bundle[2]; gates={"all_anomaly_auroc":chosen["auroc"]>=float(contract["pilot"]["minimum_all_anomaly_auroc"]),
           "stealth_phase_auroc":chosen["stealth_phase_auroc"]>=float(contract["pilot"]["minimum_stealth_phase_auroc"]),
           "phase_auprc_gain":chosen["stealth_phase_auprc"]-best_baseline>=float(contract["pilot"]["minimum_auprc_gain_over_score_baseline"]),
           "phase_accuracy":chosen["phase_accuracy"]>=float(contract["pilot"]["minimum_phase_accuracy"]),
           "phase_ablation_supported":chosen["stealth_phase_auprc"]>nm["stealth_phase_auprc"]}
    status="PASS" if all(gates.values()) else "STOP"
    trial_path=V9_RESULTS_ROOT/"04_PILOT"/"pilot_trials_v9.csv"; pd.DataFrame(trials).to_csv(trial_path,index=False)
    bundle={"state_dict":best_bundle[0].state_dict(),"scalers":best_bundle[1],"hidden":hidden,"dropout":dropout,"include_phase":True,"epochs":best_bundle[2]["epochs"]}
    torch.save(bundle,V9_RUN_ROOT/"checkpoints"/"pilot_v9.pt")
    result={"status":status,"gates":gates,"selected":chosen,"no_phase":nm,"best_scalar_baseline_auprc":best_baseline,
            "trials_sha256":sha256_file(trial_path)}
    write_json(V9_RESULTS_ROOT/"04_PILOT"/"pilot_gate_v9.json",result)
    return result


def freeze_contract_v9() -> dict:
    gate=json.loads((V9_RESULTS_ROOT/"04_PILOT"/"pilot_gate_v9.json").read_text(encoding="utf-8"))
    if gate.get("status")!="PASS": raise RuntimeError("v9 pilot did not pass")
    from .v9_data import V9_CONTRACT_PATH
    result={"status":"FROZEN","contract_sha256":sha256_file(V9_CONTRACT_PATH),"pilot_sha256":sha256_file(V9_RESULTS_ROOT/"04_PILOT"/"pilot_gate_v9.json")}
    write_json(V9_RESULTS_ROOT/"04_PILOT"/"contract_freeze_v9.json",result); return result


def train_final_v9() -> dict:
    if not (V9_RESULTS_ROOT/"04_PILOT"/"contract_freeze_v9.json").exists(): raise RuntimeError("Freeze v9 first")
    dev=load_features_v9("development"); test=load_features_v9("final")
    # Combine train/validation development rows; final remains untouched.
    train=np.flatnonzero(np.isin(dev["analysis_role"],["fit","validation"])); val=np.flatnonzero(dev["analysis_role"]=="calibration")
    pilot=json.loads((V9_RESULTS_ROOT/"04_PILOT"/"pilot_gate_v9.json").read_text(encoding="utf-8")); h=int(pilot["selected"]["hidden"]); d=float(pilot["selected"]["dropout"])
    contract=load_v9_contract(); pred_acc=[]; slot_acc=[]; type_acc=[]; phase_acc=[]; model_rows=[]
    # Test features are standardized using each development model's scalers.
    for seed in contract["model"]["model_seeds"]:
        model,scalers,_,_,epochs=_fit_one(dev,train,val,h,d,int(seed),True,int(contract["model"]["final_epochs"]))
        device=next(model.parameters()).device; pred=_predict(model,test,np.arange(len(test["is_anomaly"])),scalers,device)
        panel,slot,typ,phase=pred; pred_acc.append(panel); slot_acc.append(slot); type_acc.append(typ); phase_acc.append(phase)
        torch.save({"state_dict":model.state_dict(),"scalers":scalers,"hidden":h,"dropout":d},V9_RUN_ROOT/"checkpoints"/f"final_v9_seed{seed}.pt")
        model_rows.append({"seed":seed,"epochs":epochs})
    ensemble=(np.mean(pred_acc,axis=0),np.mean(slot_acc,axis=0),np.mean(type_acc,axis=0),np.mean(phase_acc,axis=0))
    metrics,risk=_metrics(test,np.arange(len(test["is_anomaly"])),ensemble)
    pairs=_read_pairs("final").copy(); pairs["panel_risk"]=risk; pairs["predicted_slot"]=np.argmax(ensemble[1],axis=1)
    pairs["predicted_type"]=[TYPE_NAMES[i] for i in np.argmax(ensemble[2],axis=1)]; pairs["predicted_phase"]=np.argmax(ensemble[3],axis=1)
    out=V9_RESULTS_ROOT/"05_FINAL"/"predictions_v9.parquet"; pairs.to_parquet(out,index=False)
    result={"status":"PASS","metrics":metrics,"models":model_rows,"output_sha256":sha256_file(out)}
    write_json(V9_RESULTS_ROOT/"05_FINAL"/"final_metrics_v9.json",result); return result
