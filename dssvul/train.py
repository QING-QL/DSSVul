import copy
import csv
import json
import math
import random
import time
import platform
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .core import focal_loss, nrce_loss, entropy_confidence, align_labels
from .data import read_corpus, make_splits, save_splits
from .models import ASTGGNN, FeatureMLP, LineVul, ReVealFeatures, ReVeal
from .metrics import metrics


DEFAULTS = dict(backbone="linevul", seed=42, labeled_fraction=0.1,
                batch_size=1024, micro_batch_size=32, inference_batch_size=64,
                teacher_epochs=3000, student_epochs=3000, patience=30, max_rounds=10,
                lr=1e-3, weight_decay=1e-5, hidden=128, steps=6,
                alpha="inverse_frequency", gamma=2.0, perturbations=10, delta=0.01,
                clusters=10, balance_weight=1.0, mu=1.0, nrce_fraction=0.1,
                variant="full", selection_metric="mcc", min_delta=0.0,
                device="cpu", threads=4, grad_clip=None)

METHOD_REVISION = "dssvul_20261028"


def validate_config(config):
    if config.get("normalize_embedding", True) is False and config["backbone"] in ("linevul", "reveal"):
        raise ValueError("DSSVul uncertainty assumes normalized embeddings. Use forward_original for plain-backbone evaluation.")
    for key in ("batch_size", "micro_batch_size", "inference_batch_size", "teacher_epochs",
                "student_epochs", "patience", "max_rounds", "perturbations", "threads"):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["clusters"] <= 2 or not 0 <= config["delta"] < 0.05:
        raise ValueError("clusters must exceed 2 and 0 <= delta < 0.05")
    if config["selection_metric"] not in ("mcc", "f1"):
        raise ValueError("Select checkpoints using validation mcc or f1")
    if config["variant"] not in ("full", "nce", "ndp", "nbl", "nnrce"):
        raise ValueError("Unknown ablation variant")
    if not 0 < config["nrce_fraction"] <= 1 or config["gamma"] < 0:
        raise ValueError("Invalid NRCE fraction or focal gamma")
    if config["lr"] <= 0 or min(config["balance_weight"], config["mu"], config["weight_decay"], config["min_delta"]) < 0:
        raise ValueError("Invalid optimization parameters")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_model(config, metadata):
    kind = config["backbone"]
    if kind == "ast-ggnn":
        return ASTGGNN(metadata["num_types"], config["hidden"], config["steps"])
    if kind == "linevul":
        return LineVul(config["model_path"], config.get("normalize_embedding", True))
    if kind == "reveal":
        return ReVeal(metadata["input_dim"], metadata["num_relations"],
                      config.get("graph_hidden", 200), config["hidden"], config["steps"],
                      config.get("representation_layers", 1), config.get("dropout", 0.2),
                      config.get("normalize_embedding", True))
    if kind == "reveal-features":
        return ReVealFeatures(metadata["input_dim"], config["hidden"])
    if kind == "mlp":
        return FeatureMLP(metadata["input_dim"], config["hidden"])
    raise ValueError(kind)


@torch.no_grad()
def infer(model, corpus, indices, config, uncertainty=False, noise_seed=None):
    model.eval()
    all_z, all_p, all_w = [], [], []
    device = torch.device(config["device"])
    rng = torch.Generator(device=device).manual_seed(noise_seed or config["seed"])
    for start in range(0, len(indices), config["inference_batch_size"]):
        x = corpus.batch(indices[start:start + config["inference_batch_size"]], device)
        z = model.encode(x)
        all_z.append(z.cpu())
        all_p.append(model.decode(z).softmax(-1).cpu())
        if uncertainty:
            w = entropy_confidence(model, z, config["perturbations"], config["delta"], rng)
        else:
            w = torch.zeros(len(z), device=device)
        all_w.append(w.cpu())
    return torch.cat(all_z), torch.cat(all_p), torch.cat(all_w)


def evaluate(model, corpus, indices, config):
    _, probabilities, _ = infer(model, corpus, indices, config)
    return metrics(corpus.labels[indices], probabilities.argmax(1).numpy(), probabilities[:, 1].numpy()), probabilities


def snapshot(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def save_checkpoint(path, model, config, corpus, stage, validation):
    torch.save(dict(state_dict=snapshot(model), config=config, metadata=corpus.metadata,
                    split_fingerprint=corpus.fingerprint(), split_sha256=config.get("split_sha256"),
                    stage=stage, validation=validation), path)


def rng_state(device):
    return torch.get_rng_state(), torch.cuda.get_rng_state(device) if device.type == "cuda" else None


def restore_rng(state, device):
    torch.set_rng_state(state[0])
    if state[1] is not None:
        torch.cuda.set_rng_state(state[1], device)


def fit(model, corpus, train_indices, targets, is_pseudo, eligible, prior, valid_indices,
        config, stage, out, alpha):


    device = torch.device(config["device"])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    teacher_stage = stage == "teacher"
    epochs = config["teacher_epochs"] if teacher_stage else config["student_epochs"]
    best_metric, best_state, best_validation = -float("inf"), None, None
    stale = 0
    n_pseudo = int(is_pseudo.sum())
    correction = len(train_indices) / max(1, n_pseudo)
    generator = np.random.default_rng(config["seed"] + (0 if teacher_stage else int(stage.split("_")[-1])))
    labels = torch.as_tensor(targets, dtype=torch.long, device=device)
    mask_pseudo = torch.as_tensor(is_pseudo, dtype=torch.bool, device=device)
    mask_eligible = torch.as_tensor(eligible, dtype=torch.bool, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        order = generator.permutation(len(train_indices))
        loss_sum = 0.0
        for start in range(0, len(order), config["batch_size"]):
            logical = order[start:start + config["batch_size"]]
            chunks = [logical[a:a + config["micro_batch_size"]]
                      for a in range(0, len(logical), config["micro_batch_size"])]
            use_nrce = not teacher_stage and config["variant"] != "nnrce"
            states, probabilities = [], []
            cutoff = None
            if use_nrce:
                with torch.no_grad():
                    for chunk in chunks:
                        states.append(rng_state(device))
                        logits = model(corpus.batch(train_indices[chunk], device))
                        probabilities.append(logits.log_softmax(-1).exp().gather(1, labels[chunk, None]).squeeze(1))
                cp = torch.cat(probabilities)
                cutoff = cp.kthvalue(max(1, math.ceil(len(cp) * config["nrce_fraction"]))).values
            optimizer.zero_grad(set_to_none=True)
            for number, chunk in enumerate(chunks):
                if use_nrce:
                    restore_rng(states[number], device)
                logits = model(corpus.batch(train_indices[chunk], device))
                y = labels[chunk]
                if teacher_stage:
                    terms = focal_loss(logits, y, alpha, config["gamma"], reduction="none")
                elif use_nrce:
                    terms = nrce_loss(logits, y, config["nrce_fraction"], reduction="none", cutoff=cutoff)
                else:
                    terms = F.cross_entropy(logits, y, reduction="none")
                loss = terms.sum() / len(logical)
                if not teacher_stage and config["variant"] != "nbl" and n_pseudo:
                    adjusted = logits + config["mu"] * prior.to(device).log()
                    balance = F.cross_entropy(adjusted, y, reduction="none")
                    loss = loss + config["balance_weight"] * correction * (
                        balance * mask_pseudo[chunk] * mask_eligible[chunk]).sum() / len(logical)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at {stage} epoch {epoch}")
                loss.backward()
                loss_sum += float(loss.detach()) * len(logical)
            if config["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"], error_if_nonfinite=True)
            elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite gradient")
            optimizer.step()
        validation, _ = evaluate(model, corpus, valid_indices, config)
        row = dict(stage=stage, epoch=epoch, loss=loss_sum / len(train_indices), validation=validation)
        with (out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        value = validation[config["selection_metric"]]
        if value > best_metric + config["min_delta"]:
            best_metric, best_state, best_validation = value, snapshot(model), validation
            stale = 0
        else:
            stale += 1
            if stale >= config["patience"]:
                break
    model.load_state_dict(best_state)
    save_checkpoint(out / f"{stage}.pt", model, config, corpus, stage, best_validation)
    return best_validation


def run(config):
    started = time.monotonic()
    config = {**DEFAULTS, **config, "method_revision": METHOD_REVISION}
    validate_config(config)
    if config["backbone"] in ("linevul", "reveal") and not config.get("splits"):
        raise ValueError("A persisted split manifest is required for LineVul and ReVeal")
    out = Path(config["output"])

    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    import sklearn
    runtime = dict(python=platform.python_version(), torch=torch.__version__,
                   numpy=np.__version__, sklearn=sklearn.__version__, cuda=torch.version.cuda)
    (out / "runtime.json").write_text(json.dumps(runtime, indent=2))
    seed_all(config["seed"])
    torch.set_num_threads(config["threads"])
    corpus = read_corpus(config)
    splits = make_splits(corpus, config["labeled_fraction"], config["seed"], config.get("splits"))
    training_size = len(splits["labeled"]) + len(splits["unlabeled"])
    if not len(splits["unlabeled"]):
        raise ValueError("Self-training requires a nonempty unlabeled partition")
    if abs(len(splits["labeled"]) / training_size - config["labeled_fraction"]) > 1 / training_size:
        raise ValueError("The fixed partition has a different labeled fraction; prepare a matching partition")
    save_splits(out / "splits.json", corpus, splits)
    config["split_sha256"] = hashlib.sha256((out / "splits.json").read_bytes()).hexdigest()
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out / "data_metadata.json").write_text(json.dumps(corpus.metadata, indent=2))
    labeled, unlabeled = splits["labeled"], splits["unlabeled"]
    counts = np.bincount(corpus.labels[labeled], minlength=2)
    alpha_value = 1 - counts / counts.sum() if config["alpha"] == "inverse_frequency" else config["alpha"]
    alpha = torch.as_tensor(alpha_value, dtype=torch.float32, device=config["device"])
    if alpha.shape != (2,) or not ((alpha >= 0) & (alpha <= 1)).all():
        raise ValueError("alpha must be inverse_frequency or two weights in [0,1]")
    teacher = build_model(config, corpus.metadata).to(config["device"])
    clean = np.zeros(len(labeled), dtype=bool)
    validation = fit(teacher, corpus, labeled, corpus.labels[labeled], clean, clean,
                     torch.tensor([0.5, 0.5]), splits["validation"], config, "teacher", out, alpha)
    rounds = []
    stop_reason = "maximum_rounds"
    for iteration in range(1, config["max_rounds"] + 1):
        z_l, _, _ = infer(teacher, corpus, labeled, config)
        z_u, p_u, weights = infer(teacher, corpus, unlabeled, config,
                                  uncertainty=config["variant"] != "nce",
                                  noise_seed=config["seed"] + iteration)
        selection = align_labels(z_u, p_u, weights, z_l,
                                 torch.as_tensor(corpus.labels[labeled]), config["clusters"],
                                 config["seed"] + iteration, config["variant"] != "ndp")
        np.savez_compressed(out / f"selection_{iteration}.npz", indices=unlabeled,
                            ids=np.asarray([corpus.ids[i] for i in unlabeled]),
                            classifier=selection.classifier.numpy(), cluster=selection.cluster.numpy(),
                            confidence_weight=weights.numpy(), probabilities=p_u.numpy(),
                            selected=selection.selected.numpy(), eligible=selection.eligible.numpy(),
                            prior=selection.prior.numpy(), tau=selection.tau)
        selected = np.flatnonzero(selection.selected.numpy())
        info = dict(round=iteration, n_unlabeled=len(unlabeled), n_selected=len(selected),
                    selected_fraction=len(selected) / len(unlabeled),
                    n_eligible=int(selection.eligible.sum()), tau=selection.tau,
                    prior=selection.prior.tolist(), actual_clusters=selection.cluster_count,
                    pseudo_class_counts=torch.bincount(selection.classifier[selected], minlength=2).tolist())


        info["empty_selection"] = not len(selected)
        train_ids = np.concatenate([labeled, unlabeled[selected]])
        train_y = np.concatenate([corpus.labels[labeled], selection.classifier[selected].numpy()])
        pseudo = np.concatenate([clean, np.ones(len(selected), dtype=bool)])
        eligible = np.concatenate([clean, selection.eligible[selected].numpy()])
        student = copy.deepcopy(teacher)
        candidate = fit(student, corpus, train_ids, train_y, pseudo, eligible, selection.prior,
                        splits["validation"], config, f"student_{iteration}", out, alpha)


        promoted = candidate[config["selection_metric"]] > validation[config["selection_metric"]]
        info.update(validation=candidate, promoted=promoted)
        rounds.append(info)
        print(json.dumps(info), flush=True)
        if promoted:
            teacher, validation = student, candidate
        else:
            del student


    final_test, p_test = evaluate(teacher, corpus, splits["test"], config)
    save_checkpoint(out / "best.pt", teacher, config, corpus, "final", validation)
    with (out / "test_predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "label", "p_vulnerable", "prediction"])
        for i, p in zip(splits["test"], p_test):
            writer.writerow([corpus.ids[i], int(corpus.labels[i]), float(p[1]), int(p.argmax())])
    summary = dict(status="completed", config=config, metadata=corpus.metadata,
                   validation=validation, test=final_test, rounds=rounds, stop_reason=stop_reason,
                   elapsed_seconds=time.monotonic()-started, runtime=runtime,
                   metrics_scale="precision/recall/f1/auprc: 0..1; MCC: -1..1")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"status": "completed", "output": str(out), "test": final_test,
                      "stop_reason": stop_reason, "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    return summary
