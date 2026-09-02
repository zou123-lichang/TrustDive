from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from .metrics import aqa_score_metrics
from .util import set_seed, sha256_file, write_json
from .v2_data import load_panel_targets
from .v6_modeling import load_v6_assets
from .v7_data import V7_RESULTS_ROOT
from .v8_data import (
    V8_CONTRACT_PATH,
    V8_RESULTS_ROOT,
    V8_RUN_ROOT,
    load_conditional_manifest_v8,
    load_v8_contract,
)
from .v8_tokens import load_phase_tokens_v8


def _binary_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=bool)
    probability = np.asarray(probability, dtype=float)
    if len(np.unique(labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan"), "brier": float("nan")}
    return {
        "auroc": float(roc_auc_score(labels, probability)),
        "auprc": float(average_precision_score(labels, probability)),
        "brier": float(brier_score_loss(labels, np.clip(probability, 0.0, 1.0))),
    }


def _safe_probability_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return (order + 0.5) / max(len(values), 1)


def _baseline_risks(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    result = {
        "negative_predicted_score": -frame.model_predicted_quality.to_numpy(dtype=float),
        "score_action_difficulty": frame.expected_log_judge_sd.to_numpy(dtype=float),
    }
    v7 = pd.read_parquet(V7_RESULTS_ROOT / "05_RISK_REVIEW" / "review_priority_v7.parquet").set_index("clip_uid")
    result["v7_disagreement"] = v7.loc[frame.clip_uid, "risk_disagreement"].to_numpy(dtype=float)
    v2_path = V8_RESULTS_ROOT.parent / "V2_DISAGREEMENT" / "03_FINAL" / "predictions_v2.parquet"
    if v2_path.exists():
        v2 = pd.read_parquet(v2_path).set_index("clip_uid")
        result["v2_student_t_sigma"] = v2.loc[frame.clip_uid, "sigma_judge"].to_numpy(dtype=float)
    predictions_v7 = pd.read_parquet(V7_RESULTS_ROOT / "03_SCORE" / "predictions_v7.parquet").set_index("clip_uid")
    result["rica2_uncertainty"] = predictions_v7.loc[frame.clip_uid, "teacher_uncertainty"].to_numpy(dtype=float)
    return result


def train_baselines_v8() -> dict:
    manifest = load_conditional_manifest_v8()
    validation = (manifest.analysis_role == "validation") & manifest.disagreement_primary_eligible.astype(bool)
    labels = manifest.loc[validation, "high_excess_disagreement"].to_numpy(dtype=bool)
    rows = []
    for name, risk in _baseline_risks(manifest).items():
        metrics = _binary_metrics(labels, risk[validation.to_numpy()])
        rows.append({"model": name, **metrics})
    output = pd.DataFrame(rows)
    path = V8_RESULTS_ROOT / "03_PILOT" / "disagreement_baselines_validation_v8.csv"
    output.to_csv(path, index=False)
    result = {
        "status": "PASS",
        "validation_rows": int(validation.sum()),
        "best_simple_auroc": float(output.auroc.max()),
        "best_simple_model": str(output.sort_values("auroc", ascending=False).iloc[0].model),
        "output_sha256": sha256_file(path),
        "official_test_used": False,
    }
    write_json(V8_RESULTS_ROOT / "03_PILOT" / "baseline_summary_v8.json", result)
    return result


def _judge_matrix(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros((len(frame), 7), dtype=np.float32)
    mask = np.zeros((len(frame), 7), dtype=bool)
    for index, payload in enumerate(frame.judge_scores_json):
        values = json.loads(payload)
        count = min(len(values), 7)
        scores[index, :count] = np.asarray(values[:count], dtype=np.float32)
        mask[index, :count] = True
    eligible = frame.disagreement_primary_eligible.to_numpy(dtype=bool)
    mask[~eligible] = False
    return scores, mask


def _pairwise_rank_loss(prediction, target):
    import torch
    import torch.nn.functional as functional

    if len(prediction) < 2:
        return prediction.sum() * 0.0
    difference = target[:, None] - target[None, :]
    keep = torch.triu(torch.abs(difference) > 0.05, diagonal=1)
    if not torch.any(keep):
        return prediction.sum() * 0.0
    sign = torch.sign(difference[keep])
    predicted_difference = (prediction[:, None] - prediction[None, :])[keep]
    return functional.softplus(-sign * predicted_difference).mean()


class PhaseConflictNetwork:
    def __init__(self, input_dim: int, hidden: int, heads: int, dropout: float, mean: np.ndarray, scale: np.ndarray):
        import torch

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("token_mean", torch.as_tensor(mean, dtype=torch.float32))
                self.register_buffer("token_scale", torch.as_tensor(scale, dtype=torch.float32))
                self.encoder = torch.nn.Sequential(
                    torch.nn.Linear(input_dim + 1, hidden),
                    torch.nn.GELU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(hidden, hidden),
                )
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=heads,
                    dim_feedforward=hidden * 2,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.transformer = torch.nn.TransformerEncoder(layer, num_layers=1)
                self.phase_position = torch.nn.Parameter(torch.zeros(1, 3, hidden))
                torch.nn.init.normal_(self.phase_position, std=0.02)
                self.score_head = torch.nn.Linear(hidden, 1)
                self.excess_head = torch.nn.Linear(hidden, 1)
                self.excess_logit_head = torch.nn.Linear(hidden, 1)
                self.error_head = torch.nn.Linear(hidden, 1)

            def forward(self, token, reference_weight, base_quality, phase_mask):
                valid = (reference_weight > 0).float()[:, :, None, None]
                present = phase_mask[:, None, :, None]
                normalized = (token - self.token_mean) / self.token_scale
                normalized = normalized * valid * present
                x = torch.cat((normalized, present.expand(-1, token.shape[1], -1, 1) * valid), dim=-1)
                encoded = self.encoder(x)
                weight = reference_weight[:, :, None, None] * present
                pooled = (encoded * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-8)
                phase = self.transformer(pooled + self.phase_position)
                phase = phase * phase_mask[:, :, None]
                global_feature = phase.sum(dim=1) / phase_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                score_delta = 0.25 * torch.tanh(self.score_head(global_feature).squeeze(-1))
                return {
                    "quality": base_quality + score_delta,
                    "score_delta": score_delta,
                    "log_excess_scale": self.excess_head(global_feature).squeeze(-1),
                    "excess_logit": self.excess_logit_head(global_feature).squeeze(-1),
                    "error_logit": self.error_head(global_feature).squeeze(-1),
                }

        self.module = Module()

    def to(self, device):
        self.module.to(device)
        return self


@dataclass
class TrainingArrays:
    token: np.ndarray
    weights: np.ndarray
    base_quality: np.ndarray
    quality_target: np.ndarray
    expected_log_sd: np.ndarray
    excess_target: np.ndarray
    high_excess: np.ndarray
    high_error: np.ndarray
    judge_scores: np.ndarray
    judge_mask: np.ndarray
    difficulty: np.ndarray
    dive_score: np.ndarray
    open_set: np.ndarray


def _arrays(final: bool) -> tuple[TrainingArrays, pd.DataFrame, dict[str, np.ndarray]]:
    tokens = load_phase_tokens_v8(final=final)
    manifest = load_conditional_manifest_v8().set_index("clip_uid").loc[tokens["clip_uid"].astype(str)].reset_index()
    judge_scores, judge_mask = _judge_matrix(manifest)
    arrays = TrainingArrays(
        token=tokens["token"].astype(np.float32),
        weights=tokens["reference_weights"].astype(np.float32),
        base_quality=tokens["base_coalition"][:, 7].astype(np.float32),
        quality_target=manifest.execution_quality.to_numpy(dtype=np.float32),
        expected_log_sd=manifest.expected_log_judge_sd.to_numpy(dtype=np.float32),
        excess_target=manifest.excess_log_judge_sd.fillna(0.0).to_numpy(dtype=np.float32),
        high_excess=manifest.high_excess_disagreement.fillna(False).to_numpy(dtype=np.float32),
        high_error=manifest.high_error_risk.fillna(False).to_numpy(dtype=np.float32),
        judge_scores=judge_scores,
        judge_mask=judge_mask,
        difficulty=manifest.difficulty.to_numpy(dtype=np.float32),
        dive_score=manifest.dive_score.to_numpy(dtype=np.float32),
        open_set=tokens["open_set"].astype(bool),
    )
    return arrays, manifest, tokens


def _normalization(token: np.ndarray, indices: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.repeat((weights[indices] > 0)[:, :, None], 3, axis=2)
    values = token[indices][valid]
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale = np.where(scale > 1e-5, scale, 1.0).astype(np.float32)
    return mean.reshape(1, 1, 1, -1), scale.reshape(1, 1, 1, -1)


def _evaluate(model, arrays: TrainingArrays, indices: np.ndarray, device) -> dict:
    import torch

    model.module.eval()
    outputs = {key: [] for key in ("quality", "log_excess_scale", "excess_logit", "error_logit")}
    with torch.inference_mode():
        for start in range(0, len(indices), 512):
            batch = indices[start : start + 512]
            result = model.module(
                torch.from_numpy(arrays.token[batch]).to(device),
                torch.from_numpy(arrays.weights[batch]).to(device),
                torch.from_numpy(arrays.base_quality[batch]).to(device),
                torch.ones((len(batch), 3), dtype=torch.float32, device=device),
            )
            for key in outputs:
                outputs[key].append(result[key].detach().cpu().numpy())
    prediction = {key: np.concatenate(value) for key, value in outputs.items()}
    score = 3.0 * arrays.difficulty[indices] * prediction["quality"]
    score_metrics = aqa_score_metrics(arrays.dive_score[indices], score)
    judge = arrays.judge_mask[indices].any(axis=1)
    disagreement = _binary_metrics(
        arrays.high_excess[indices][judge].astype(bool),
        1.0 / (1.0 + np.exp(-prediction["excess_logit"][judge])),
    )
    continuous = float(spearmanr(
        prediction["log_excess_scale"][judge], arrays.excess_target[indices][judge]
    ).statistic) if judge.sum() > 2 else float("nan")
    return {
        "prediction": prediction,
        "score": score_metrics,
        "disagreement": disagreement,
        "continuous_excess_spearman": continuous,
    }


def _train_one(
    arrays: TrainingArrays,
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    hidden: int,
    rank_weight: float,
    seed: int,
    epochs: int,
    output_path,
) -> dict:
    import torch
    import torch.nn.functional as functional

    contract = load_v8_contract()
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, scale = _normalization(arrays.token, train_indices, arrays.weights)
    network = PhaseConflictNetwork(
        arrays.token.shape[-1], hidden, int(contract["model"]["attention_heads"]),
        float(contract["model"]["dropout"]), mean, scale,
    ).to(device)
    optimizer = torch.optim.AdamW(network.module.parameters(), lr=float(contract["model"]["learning_rate"]), weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_state = None
    best_epoch = 0
    best_objective = -np.inf
    patience = 0
    batch_size = int(contract["model"]["batch_size"])
    df = float(contract["model"]["student_t_df"])
    for epoch in range(int(epochs)):
        network.module.train()
        shuffled = rng.permutation(train_indices)
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            token = torch.from_numpy(arrays.token[batch]).to(device)
            weight = torch.from_numpy(arrays.weights[batch]).to(device)
            base = torch.from_numpy(arrays.base_quality[batch]).to(device)
            result = network.module(token, weight, base, torch.ones((len(batch), 3), device=device))
            target_q = torch.from_numpy(arrays.quality_target[batch]).to(device)
            score_loss = functional.smooth_l1_loss(result["quality"], target_q, beta=0.5)
            judge_row = torch.from_numpy(arrays.judge_mask[batch].any(axis=1)).to(device)
            if torch.any(judge_row):
                expected = torch.from_numpy(arrays.expected_log_sd[batch]).to(device)[judge_row]
                scale_t = torch.exp(expected + result["log_excess_scale"][judge_row]).clamp(0.03, 3.0)
                distribution = torch.distributions.StudentT(
                    df=torch.tensor(df, device=device),
                    loc=result["quality"][judge_row][:, None],
                    scale=scale_t[:, None],
                )
                judge_values = torch.from_numpy(arrays.judge_scores[batch]).to(device)[judge_row]
                judge_mask = torch.from_numpy(arrays.judge_mask[batch]).to(device)[judge_row]
                nll = -distribution.log_prob(judge_values)[judge_mask].mean()
                high_excess = torch.from_numpy(arrays.high_excess[batch]).to(device)[judge_row]
                excess_bce = functional.binary_cross_entropy_with_logits(
                    result["excess_logit"][judge_row], high_excess
                )
                excess_target = torch.from_numpy(arrays.excess_target[batch]).to(device)[judge_row]
                rank_loss = _pairwise_rank_loss(result["log_excess_scale"][judge_row], excess_target)
            else:
                nll = score_loss * 0.0
                excess_bce = score_loss * 0.0
                rank_loss = score_loss * 0.0
            high_error = torch.from_numpy(arrays.high_error[batch]).to(device)
            error_bce = functional.binary_cross_entropy_with_logits(result["error_logit"], high_error)
            delta_l2 = torch.mean(result["score_delta"] ** 2)
            loss = (
                score_loss
                + float(contract["model"]["student_t_weight"]) * nll
                + float(contract["model"]["excess_bce_weight"]) * excess_bce
                + float(rank_weight) * rank_loss
                + float(contract["error_risk"]["bce_weight"]) * error_bce
                + float(contract["model"]["score_delta_l2_weight"]) * delta_l2
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.module.parameters(), 5.0)
            optimizer.step()
        if validation_indices is not None:
            evaluation = _evaluate(network, arrays, validation_indices, device)
            objective = float(evaluation["disagreement"]["auroc"])
            objective += 0.1 * float(evaluation["score"]["spearman"])
            if objective > best_objective + 1e-5:
                best_objective = objective
                best_epoch = epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in network.module.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= int(contract["model"]["early_stopping_patience"]):
                break
        else:
            best_epoch = epoch + 1
    if best_state is not None:
        network.module.load_state_dict(best_state)
    payload = {
        "state_dict": network.module.state_dict(),
        "hidden": int(hidden),
        "rank_weight": float(rank_weight),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "input_dim": int(arrays.token.shape[-1]),
        "token_mean": mean,
        "token_scale": scale,
    }
    torch.save(payload, output_path)
    return {"model": network, "best_epoch": best_epoch, "artifact_sha256": sha256_file(output_path)}


def _load_network(path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    network = PhaseConflictNetwork(
        int(payload["input_dim"]), int(payload["hidden"]),
        int(load_v8_contract()["model"]["attention_heads"]),
        float(load_v8_contract()["model"]["dropout"]),
        np.asarray(payload["token_mean"]), np.asarray(payload["token_scale"]),
    )
    network.module.load_state_dict(payload["state_dict"])
    return network, payload


def pilot_v8() -> dict:
    baseline_path = V8_RESULTS_ROOT / "03_PILOT" / "baseline_summary_v8.json"
    if not baseline_path.exists():
        raise RuntimeError("Run train-baselines --protocol v8 first")
    baseline_summary = json.loads(baseline_path.read_text(encoding="utf-8"))
    arrays, manifest, _ = _arrays(final=False)
    fit = np.flatnonzero(manifest.analysis_role.to_numpy() == "fit")
    validation = np.flatnonzero(manifest.analysis_role.to_numpy() == "validation")
    base_score = 3.0 * arrays.difficulty[validation] * arrays.base_quality[validation]
    base_metrics = aqa_score_metrics(arrays.dive_score[validation], base_score)
    rows = []
    for hidden in load_v8_contract()["model"]["hidden_dimensions"]:
        for rank_weight in load_v8_contract()["model"]["excess_rank_weights"]:
            name = f"h{int(hidden)}_rank{float(rank_weight):g}"
            path = V8_RUN_ROOT / "checkpoints" / f"pilot_{name}_v8.pt"
            trained = _train_one(
                arrays, fit, validation, int(hidden), float(rank_weight),
                int(load_v8_contract()["statistics"]["seed"]),
                int(load_v8_contract()["model"]["pilot_epochs"]), path,
            )
            evaluation = _evaluate(
                trained["model"], arrays, validation,
                next(trained["model"].module.parameters()).device,
            )
            rows.append({
                "name": name,
                "hidden": int(hidden),
                "rank_weight": float(rank_weight),
                "best_epoch": int(trained["best_epoch"]),
                "spearman": float(evaluation["score"]["spearman"]),
                "mae": float(evaluation["score"]["mae"]),
                "excess_auroc": float(evaluation["disagreement"]["auroc"]),
                "excess_auprc": float(evaluation["disagreement"]["auprc"]),
                "continuous_excess_spearman": float(evaluation["continuous_excess_spearman"]),
                "artifact_sha256": trained["artifact_sha256"],
            })
    trials = pd.DataFrame(rows)
    contract = load_v8_contract()
    trials["spearman_drop"] = float(base_metrics["spearman"]) - trials.spearman
    trials["mae_increase"] = trials.mae - float(base_metrics["mae"])
    trials["auroc_gain"] = trials.excess_auroc - float(baseline_summary["best_simple_auroc"])
    trials["eligible"] = (
        (trials.spearman_drop <= float(contract["pilot"]["maximum_spearman_drop"]))
        & (trials.mae_increase <= float(contract["pilot"]["maximum_mae_increase"]))
        & (trials.auroc_gain >= float(contract["pilot"]["minimum_auroc_gain_over_best_simple_baseline"]))
    )
    path = V8_RESULTS_ROOT / "03_PILOT" / "pilot_trials_v8.csv"
    trials.to_csv(path, index=False)
    eligible = trials[trials.eligible].sort_values(
        ["excess_auroc", "spearman", "mae"], ascending=[False, False, True], kind="stable"
    )
    if eligible.empty:
        result = {
            "status": "STOP",
            "reason": "No candidate passed score and incremental-disagreement pilot gates",
            "base_score_validation": base_metrics,
            "best_simple_auroc": baseline_summary["best_simple_auroc"],
            "best_observed": trials.sort_values("excess_auroc", ascending=False).iloc[0].to_dict(),
            "trials_sha256": sha256_file(path),
            "official_test_used": False,
        }
        write_json(V8_RESULTS_ROOT / "03_PILOT" / "pilot_gate_v8.json", result)
        (V8_RESULTS_ROOT / "RESULTS_DECISION_V8.md").write_text(
            "# TrustDive-Conflict v8 decision\n\n"
            "**STOPPED_AT_PILOT.** The score head remained non-inferior, but the "
            "phase-conflict head did not improve excess-disagreement detection over "
            "the frozen simple baselines. The contract, official-test analysis, final "
            "three-seed training, dual-risk figures, and publication claims remain locked.\n",
            encoding="utf-8",
        )
        return result
    selected = eligible.iloc[0].to_dict()
    selected_path = V8_RUN_ROOT / "checkpoints" / f"pilot_{selected['name']}_v8.pt"
    result = {
        "status": "PASS",
        "selected": selected,
        "selected_artifact_sha256": sha256_file(selected_path),
        "base_score_validation": base_metrics,
        "best_simple_auroc": baseline_summary["best_simple_auroc"],
        "trials_sha256": sha256_file(path),
        "official_test_used": False,
    }
    write_json(V8_RESULTS_ROOT / "03_PILOT" / "pilot_gate_v8.json", result)
    return result


def freeze_contract_v8() -> dict:
    pilot_path = V8_RESULTS_ROOT / "03_PILOT" / "pilot_gate_v8.json"
    if not pilot_path.exists():
        raise RuntimeError("Run pilot --protocol v8 first")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("status") != "PASS":
        raise RuntimeError("v8 pilot gate did not pass; official test remains locked")
    result = {
        "status": "FROZEN",
        "contract_sha256": sha256_file(V8_CONTRACT_PATH),
        "pilot_gate_sha256": sha256_file(pilot_path),
        "selected": pilot["selected"],
        "test_targets_used_for_selection": False,
    }
    write_json(V8_RESULTS_ROOT / "03_PILOT" / "contract_freeze_v8.json", result)
    return result


def train_final_v8() -> dict:
    freeze_path = V8_RESULTS_ROOT / "03_PILOT" / "contract_freeze_v8.json"
    if not freeze_path.exists():
        raise RuntimeError("Run freeze-contract --protocol v8 first")
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    arrays, manifest, tokens = _arrays(final=True)
    development = np.flatnonzero(manifest.analysis_role.isin(("fit", "validation")).to_numpy())
    selected = frozen["selected"]
    epochs = max(int(selected["best_epoch"]), 10)
    model_paths = []
    for seed in load_v8_contract()["statistics"]["model_seeds"]:
        path = V8_RUN_ROOT / "checkpoints" / f"final_seed_{int(seed)}_v8.pt"
        _train_one(
            arrays, development, None, int(selected["hidden"]), float(selected["rank_weight"]),
            int(seed), epochs, path,
        )
        model_paths.append(path)
    all_indices = np.arange(len(manifest), dtype=int)
    predictions = []
    for path in model_paths:
        network, _ = _load_network(path)
        device = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
        network.to(device)
        predictions.append(_evaluate(network, arrays, all_indices, device)["prediction"])
    quality = np.mean(np.stack([item["quality"] for item in predictions]), axis=0)
    log_excess = np.mean(np.stack([item["log_excess_scale"] for item in predictions]), axis=0)
    excess_probability = np.mean(np.stack([1.0 / (1.0 + np.exp(-item["excess_logit"])) for item in predictions]), axis=0)
    error_probability = np.mean(np.stack([1.0 / (1.0 + np.exp(-item["error_logit"])) for item in predictions]), axis=0)
    quality[arrays.open_set] = arrays.base_quality[arrays.open_set]
    score = 3.0 * arrays.difficulty * quality
    calibration = manifest.analysis_role.to_numpy() == "calibration"
    conformal = float(np.quantile(
        np.abs(score[calibration] - arrays.dive_score[calibration]),
        float(load_v8_contract()["review"]["conformal_coverage"]), method="higher",
    ))
    output = manifest[[
        "clip_uid", "official_split", "analysis_role", "source_role", "event_family",
        "action_type", "difficulty", "dive_score", "execution_quality",
        "expected_log_judge_sd", "excess_threshold", "error_threshold",
        "disagreement_primary_eligible",
    ]].copy()
    output["base_predicted_quality"] = arrays.base_quality
    output["base_predicted_score"] = 3.0 * arrays.difficulty * arrays.base_quality
    output["predicted_quality"] = quality
    output["predicted_score"] = score
    output["predicted_excess_log_scale"] = log_excess
    output["predicted_judge_sd"] = np.exp(output.expected_log_judge_sd + log_excess)
    output["risk_excess_disagreement"] = excess_probability
    output["risk_error"] = error_probability
    output["prediction_interval_width"] = 2.0 * conformal
    output["open_set"] = arrays.open_set
    output["valid_reference_count"] = tokens["valid_reference"].sum(axis=1)
    output["seed_score_sd"] = np.std(
        np.stack([3.0 * arrays.difficulty * item["quality"] for item in predictions]), axis=0, ddof=0
    )
    path = V8_RESULTS_ROOT / "04_FINAL" / "predictions_v8.parquet"
    output.to_parquet(path, index=False)
    test = manifest.analysis_role.to_numpy() == "official_test"
    base_metrics = aqa_score_metrics(arrays.dive_score[test], output.loc[test, "base_predicted_score"])
    metrics = aqa_score_metrics(arrays.dive_score[test], output.loc[test, "predicted_score"])
    result = {
        "status": "PASS",
        "model_seeds": list(load_v8_contract()["statistics"]["model_seeds"]),
        "training_epochs": epochs,
        "base_test_metrics": base_metrics,
        "v8_test_metrics": metrics,
        "conformal_half_width": conformal,
        "prediction_sha256": sha256_file(path),
        "checkpoint_hashes": {path.name: sha256_file(path) for path in model_paths},
        "test_used_for_model_selection": False,
    }
    write_json(V8_RESULTS_ROOT / "04_FINAL" / "final_training_summary_v8.json", result)
    return result


def load_final_models_v8():
    models = []
    for seed in load_v8_contract()["statistics"]["model_seeds"]:
        path = V8_RUN_ROOT / "checkpoints" / f"final_seed_{int(seed)}_v8.pt"
        if not path.exists():
            raise RuntimeError("Run train --protocol v8 --stage final first")
        models.append(_load_network(path)[0])
    return models
