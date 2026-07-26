"""Hierarchical Commander V4: Monte Carlo imitation + league recurrent PPO."""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
INPUTS, REGIMENTS = 48, 6
EMBED, ATTENTION, MEMORY = 64, 32, 64
DOCTRINES, FOCUS, FIRE, VERSION = 8, 7, 3, 2
LEAGUE = "crowd,neural,ppo,commander_v3,commander_v4,offensive,defensive,adaptive"


class CommanderV4(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(INPUTS, EMBED)
        self.query = nn.Linear(EMBED, ATTENTION)
        self.key = nn.Linear(EMBED, ATTENTION)
        self.attention_value = nn.Linear(EMBED, ATTENTION)
        self.mix = nn.Linear(EMBED + ATTENTION, EMBED)
        self.gru = nn.GRUCell(EMBED, MEMORY)
        self.doctrine = nn.Linear(MEMORY, DOCTRINES)
        self.focus = nn.Linear(MEMORY, FOCUS)
        self.fire = nn.Linear(MEMORY, FIRE)
        self.critic = nn.Linear(MEMORY, 1)
        for module in (self.encoder, self.query, self.key, self.attention_value, self.mix):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.zeros_(module.bias)
        for module in (self.doctrine, self.focus, self.fire):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)
        with torch.no_grad():
            self.doctrine.bias[0] = 1.5
            self.focus.bias[3] = 1.5
            self.fire.bias[0] = 1

    def step(self, features, active, hidden):
        encoded = torch.tanh(self.encoder(features))
        query, key = self.query(encoded), self.key(encoded)
        value = self.attention_value(encoded)
        scores = query @ key.transpose(-1, -2) / math.sqrt(ATTENTION)
        scores = scores.masked_fill(~active.unsqueeze(-2), -1e9)
        context = torch.softmax(scores, dim=-1) @ value
        mixed = torch.tanh(self.mix(torch.cat((encoded, context), dim=-1)))
        mask = active.unsqueeze(-1).float()
        pooled = (mixed * mask).sum(-2) / mask.sum(-2).clamp_min(1)
        hidden = self.gru(pooled, hidden)
        return (
            self.doctrine(hidden),
            self.focus(hidden),
            self.fire(hidden),
            self.critic(hidden).squeeze(-1),
            hidden,
        )

    def sequence(self, features, active, hidden):
        doctrines, focuses, fires, values = [], [], [], []
        for index in range(features.shape[1]):
            doctrine, focus, fire, value, hidden = self.step(
                features[:, index], active[:, index], hidden
            )
            doctrines.append(doctrine)
            focuses.append(focus)
            fires.append(fire)
            values.append(value)
        return (
            torch.stack(doctrines, 1),
            torch.stack(focuses, 1),
            torch.stack(fires, 1),
            torch.stack(values, 1),
        )


def flat(tensor):
    return tensor.detach().cpu().flatten().tolist()


def export_policy(model, metadata):
    state = model.state_dict()
    weights = {
        "encoderWeight": flat(state["encoder.weight"]),
        "encoderBias": flat(state["encoder.bias"]),
        "queryWeight": flat(state["query.weight"]),
        "queryBias": flat(state["query.bias"]),
        "keyWeight": flat(state["key.weight"]),
        "keyBias": flat(state["key.bias"]),
        "valueAttentionWeight": flat(state["attention_value.weight"]),
        "valueAttentionBias": flat(state["attention_value.bias"]),
        "mixWeight": flat(state["mix.weight"]),
        "mixBias": flat(state["mix.bias"]),
        "gruWeightInput": flat(state["gru.weight_ih"]),
        "gruWeightHidden": flat(state["gru.weight_hh"]),
        "gruBiasInput": flat(state["gru.bias_ih"]),
        "gruBiasHidden": flat(state["gru.bias_hh"]),
        "doctrineWeight": flat(state["doctrine.weight"]),
        "doctrineBias": flat(state["doctrine.bias"]),
        "focusWeight": flat(state["focus.weight"]),
        "focusBias": flat(state["focus.bias"]),
        "fireWeight": flat(state["fire.weight"]),
        "fireBias": flat(state["fire.bias"]),
        "criticWeight": flat(state["critic.weight"]),
        "criticBias": flat(state["critic.bias"]),
    }
    policy = {
        "version": VERSION, "inputs": INPUTS, "regiments": "variable",
        "embed": EMBED, "attention": ATTENTION, "memory": MEMORY,
        "doctrines": [
            "compact_hold", "mass_advance", "mass_assault", "elastic",
            "left_hook", "right_hook", "counterattack", "encircle"
        ],
        "focusSectors": FOCUS,
        "fireModes": ["independent", "volley", "hold_fire"],
        "weights": weights,
    }
    source = (
        "// Generated locally by train-commander-v4.py.\n"
        f"export const TRAINED_COMMANDER_V4_POLICY = {json.dumps(policy, separators=(',', ':'))};\n"
        "export const COMMANDER_V4_TRAINING_METADATA = "
        f"{json.dumps(metadata, ensure_ascii=False, indent=2)};\n"
    )
    (ROOT / "trained-commander-v4-policy.js").write_text(source, encoding="utf8")


def run(command):
    subprocess.run(command, cwd=ROOT, check=True)


def collect(node, battles, workers, seed, output, exploration=0, evaluate=False, opponents=LEAGUE):
    command = [
        node, str(ROOT / "collect-commander-v4-rollouts.js"),
        "--battles", str(battles), "--workers", str(workers),
        "--seed", str(seed), "--exploration", str(exploration),
        "--opponents", opponents, "--output", str(output),
    ]
    if evaluate:
        command.append("--evaluate")
    run(command)
    return json.loads(output.read_text(encoding="utf8"))


def make_teacher(node, scenarios, workers, seed, output, opponents=LEAGUE):
    run([
        node, str(ROOT / "monte-carlo-v4-teacher.js"),
        "--scenarios", str(scenarios), "--workers", str(workers),
        "--seed", str(seed), "--opponents", opponents, "--output", str(output)
    ])
    return json.loads(output.read_text(encoding="utf8"))


def teacher_update(model, optimizer, samples, device, epochs=5, batch_size=128):
    if not samples:
        return 0.0
    losses = []
    for _ in range(epochs):
        order = torch.randperm(len(samples)).tolist()
        for start in range(0, len(order), batch_size):
            batch = [samples[index] for index in order[start:start + batch_size]]
            features = torch.tensor([x["features"] for x in batch], dtype=torch.float32, device=device)
            active = torch.tensor([x["activeMask"] for x in batch], dtype=torch.bool, device=device)
            hidden = torch.tensor([x["hidden"] for x in batch], dtype=torch.float32, device=device)
            doctrine = torch.tensor([x["doctrine"] for x in batch], dtype=torch.long, device=device)
            focus = torch.tensor([x["focus"] for x in batch], dtype=torch.long, device=device)
            weight = torch.tensor([x["weight"] for x in batch], dtype=torch.float32, device=device)
            fire = torch.tensor([x["fireMode"] for x in batch], dtype=torch.long, device=device)
            dlogits, flogits, fire_logits, _, _ = model.step(features, active, hidden)
            loss = (
                F.cross_entropy(dlogits, doctrine, reduction="none")
                + 0.6 * F.cross_entropy(flogits, focus, reduction="none")
                + 0.7 * F.cross_entropy(fire_logits, fire, reduction="none")
            )
            loss = (loss * weight / weight.mean().clamp_min(1e-6)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()
            losses.append(loss.item())
    return sum(losses) / max(1, len(losses))


def episodes(samples):
    grouped = {}
    for item in samples:
        grouped.setdefault(item["episode"], []).append(item)
    return [sorted(items, key=lambda item: item["step"]) for items in grouped.values()]


def pad(items, mean, std, device):
    batch, steps = len(items), max(len(x) for x in items)
    features = torch.zeros(batch, steps, REGIMENTS, INPUTS, device=device)
    active = torch.zeros(batch, steps, REGIMENTS, dtype=torch.bool, device=device)
    doctrine = torch.zeros(batch, steps, dtype=torch.long, device=device)
    focus = torch.zeros(batch, steps, dtype=torch.long, device=device)
    fire = torch.zeros(batch, steps, dtype=torch.long, device=device)
    old_log = torch.zeros(batch, steps, device=device)
    advantage = torch.zeros(batch, steps, device=device)
    returns = torch.zeros(batch, steps, device=device)
    valid = torch.zeros(batch, steps, dtype=torch.bool, device=device)
    hidden = torch.zeros(batch, MEMORY, device=device)
    for row, episode in enumerate(items):
        hidden[row] = torch.tensor(episode[0]["hidden"], device=device)
        for step, item in enumerate(episode):
            features[row, step] = torch.tensor(item["features"], device=device)
            active[row, step] = torch.tensor(item["activeMask"], device=device)
            doctrine[row, step], focus[row, step] = item["doctrine"], item["focus"]
            fire[row, step] = item["fireMode"]
            old_log[row, step] = item["logProb"]
            advantage[row, step] = (item["advantage"] - mean) / std
            returns[row, step] = item["return"]
            valid[row, step] = True
    return (
        features, active, doctrine, focus, fire, old_log,
        advantage, returns, valid, hidden
    )


def mixed(logits, actions, exploration):
    policy = torch.softmax(logits, -1)
    probability = (1 - exploration) * policy + exploration / logits.shape[-1]
    selected = probability.gather(-1, actions.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)
    entropy = -(policy.clamp_min(1e-9).log() * policy).sum(-1)
    return selected.log(), entropy


def ppo_update(model, optimizer, samples, device, epochs, batch_size, exploration):
    eps = episodes(samples)
    raw = torch.tensor([x["advantage"] for e in eps for x in e], dtype=torch.float32)
    mean, std = raw.mean().item(), raw.std(unbiased=False).item() + 1e-6
    totals = {"policy": 0, "value": 0, "entropy": 0, "kl": 0, "batches": 0}
    stop = False
    for _ in range(epochs):
        order = torch.randperm(len(eps)).tolist()
        for start in range(0, len(order), batch_size):
            data = pad([eps[i] for i in order[start:start + batch_size]], mean, std, device)
            (
                features, active, doctrine, focus, fire, old,
                advantage, returns, valid, hidden
            ) = data
            dlogits, flogits, fire_logits, values = model.sequence(features, active, hidden)
            dlog, dentropy = mixed(dlogits, doctrine, exploration)
            flog, fentropy = mixed(flogits, focus, exploration)
            fire_log, fire_entropy = mixed(fire_logits, fire, exploration)
            logprob = dlog + flog + fire_log
            ratio = (logprob - old).clamp(-20, 20).exp()
            objective = torch.minimum(ratio * advantage, ratio.clamp(0.82, 1.18) * advantage)
            mask = valid.float()
            lengths = mask.sum(1).clamp_min(1)
            policy_loss = -((objective * mask).sum(1) / lengths).mean()
            value_loss = ((F.smooth_l1_loss(values, returns, reduction="none") * mask).sum(1) / lengths).mean()
            entropy = (
                (((dentropy + fentropy + fire_entropy) * mask).sum(1)) / lengths
            ).mean()
            kl = (((old - logprob) * mask).sum() / mask.sum().clamp_min(1))
            loss = policy_loss + 0.5 * value_loss - 0.025 * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.6)
            optimizer.step()
            for name, value in (("policy", policy_loss), ("value", value_loss), ("entropy", entropy), ("kl", kl)):
                totals[name] += value.item()
            totals["batches"] += 1
            if kl.item() > 0.027:
                stop = True
                break
        if stop:
            break
    count = max(1, totals.pop("batches"))
    result = {key: value / count for key, value in totals.items()}
    result["earlyStop"] = stop
    return result


def robust_score(summary):
    rates = {}
    for name, bucket in summary["byOpponent"].items():
        games = bucket["wins"] + bucket["losses"] + bucket["draws"]
        rates[name] = (bucket["wins"] + 0.5 * bucket["draws"]) / max(1, games)
    overall = (summary["wins"] + 0.5 * summary["draws"]) / max(1, summary["battles"])
    crowd = rates.get("crowd", overall)
    worst = min(rates.values(), default=overall)
    return 0.5 * overall + 0.3 * crowd + 0.2 * worst, rates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--battles", type=int, default=196)
    parser.add_argument("--validation-battles", type=int, default=140)
    parser.add_argument("--teacher-scenarios", type=int, default=18)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--episode-batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=940021)
    parser.add_argument("--node", default="node")
    parser.add_argument("--opponents", default=LEAGUE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    model = CommanderV4().to(device)
    checkpoint_path = ROOT / ".training" / "commander-v4-best.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    work = ROOT / ".training"
    work.mkdir(exist_ok=True)
    started = time.perf_counter()
    teacher_path = work / "commander-v4-teacher.json"
    teacher = make_teacher(
        args.node, args.teacher_scenarios, args.workers, args.seed + 4000000,
        teacher_path, args.opponents
    )
    imitation_loss = teacher_update(model, optimizer, teacher["samples"], device)
    history = [{"cycle": 0, "teacherSamples": len(teacher["samples"]), "imitationLoss": imitation_loss}]
    export_policy(model, {"status": "Monte Carlo pretraining", "history": history})
    validation = collect(
        args.node, args.validation_battles, args.workers, args.seed + 8000000,
        work / "commander-v4-validation.json", evaluate=True, opponents=args.opponents
    )["summary"]
    best_score, rates = robust_score(validation)
    history[0].update({"validation": validation, "rates": rates, "robustScore": best_score})
    best_state = copy.deepcopy(model.state_dict())
    exploration = 0.18
    print(f"V4 Monte Carlo: loss={imitation_loss:.4f}, robust={best_score:.3f}, rates={rates}", flush=True)
    for cycle in range(args.cycles):
        rollout = collect(
            args.node, args.battles, args.workers, args.seed + cycle * 10007,
            work / "commander-v4-rollouts.json", exploration=exploration,
            opponents=args.opponents
        )
        losses = ppo_update(
            model, optimizer, rollout["samples"], device,
            args.epochs, args.episode_batch, exploration
        )
        export_policy(model, {"status": "validating", "cycle": cycle + 1, "history": history})
        validation = collect(
            args.node, args.validation_battles, args.workers,
            args.seed + 9000000 + cycle * 20011,
            work / "commander-v4-validation.json", evaluate=True,
            opponents=args.opponents
        )["summary"]
        score_value, rates = robust_score(validation)
        entry = {
            "cycle": cycle + 1, "exploration": exploration,
            "rollout": rollout["summary"], "samples": len(rollout["samples"]),
            "losses": losses, "validation": validation,
            "rates": rates, "robustScore": score_value
        }
        history.append(entry)
        if score_value > best_score:
            best_score, best_state = score_value, copy.deepcopy(model.state_dict())
            torch.save({"model": best_state, "score": best_score, "history": history},
                       work / "commander-v4-best.pt")
        exploration = max(0.05, exploration * 0.86)
        print(
            f"V4 {cycle + 1}/{args.cycles}: robust={score_value:.3f}, "
            f"rates={rates}, KL={losses['kl']:.4f}", flush=True
        )
    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    metadata = {
        "algorithm": "root Monte Carlo plan teacher + hierarchical recurrent league PPO",
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "variableRegimentCount": True,
        "bestRobustScore": best_score,
        "elapsedSeconds": elapsed,
        "options": vars(args),
        "history": history,
    }
    export_policy(model, metadata)
    print(f"Commander V4 ready in {elapsed:.1f}s; best robust score={best_score:.3f}.")


if __name__ == "__main__":
    main()
