"""GPU PPO trainer for the separate recurrent commander strategy."""

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
INPUTS = 48
ENCODER = 64
MEMORY = 48
SECTORS = 7
STANCES = 5
VERSION = 1
LEAGUE = "crowd,neural,ppo,commander_v3,commander_v4,offensive,defensive,adaptive"


class CommanderActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(INPUTS, ENCODER)
        self.gru = nn.GRUCell(ENCODER, MEMORY)
        self.sector = nn.Linear(MEMORY, SECTORS)
        self.stance = nn.Linear(MEMORY, STANCES)
        self.value = nn.Linear(MEMORY, 1)
        for module in (self.encoder, self.sector, self.stance):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.sector.weight, gain=0.01)
        nn.init.orthogonal_(self.stance.weight, gain=0.01)
        nn.init.orthogonal_(self.value.weight, gain=1)
        nn.init.zeros_(self.value.bias)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor):
        encoded = torch.tanh(self.encoder(features))
        next_hidden = self.gru(encoded, hidden)
        return self.sector(next_hidden), self.stance(next_hidden), self.value(next_hidden).squeeze(-1)


def export_policy(model: CommanderActorCritic, metadata: dict) -> None:
    state = model.state_dict()
    weights = {
        "encoderWeight": state["encoder.weight"].detach().cpu().flatten().tolist(),
        "encoderBias": state["encoder.bias"].detach().cpu().flatten().tolist(),
        "gruWeightInput": state["gru.weight_ih"].detach().cpu().flatten().tolist(),
        "gruWeightHidden": state["gru.weight_hh"].detach().cpu().flatten().tolist(),
        "gruBiasInput": state["gru.bias_ih"].detach().cpu().flatten().tolist(),
        "gruBiasHidden": state["gru.bias_hh"].detach().cpu().flatten().tolist(),
        "sectorWeight": state["sector.weight"].detach().cpu().flatten().tolist(),
        "sectorBias": state["sector.bias"].detach().cpu().flatten().tolist(),
        "stanceWeight": state["stance.weight"].detach().cpu().flatten().tolist(),
        "stanceBias": state["stance.bias"].detach().cpu().flatten().tolist(),
        "valueWeight": state["value.weight"].detach().cpu().flatten().tolist(),
        "valueBias": state["value.bias"].detach().cpu().flatten().tolist(),
    }
    policy = {
        "version": VERSION,
        "inputs": INPUTS,
        "encoder": ENCODER,
        "memory": MEMORY,
        "sectors": SECTORS,
        "stances": STANCES,
        "weights": weights,
    }
    source = (
        "// Generated locally by train-ppo-commander.py.\n"
        f"export const TRAINED_PPO_POLICY = {json.dumps(policy, separators=(',', ':'))};\n"
        f"export const PPO_TRAINING_METADATA = {json.dumps(metadata, ensure_ascii=False, indent=2)};\n"
    )
    (ROOT / "trained-ppo-policy.js").write_text(source, encoding="utf8")


def run_collector(
    node: str, battles: int, workers: int, seed: int, output: Path,
    evaluate=False, opponents=LEAGUE
):
    command = [
        node,
        "collect-ppo-rollouts.js",
        "--battles", str(battles),
        "--workers", str(workers),
        "--seed", str(seed),
        "--opponents", opponents,
        "--output", str(output),
    ]
    if evaluate:
        command.append("--evaluate")
    process = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    return json.loads(output.read_text(encoding="utf8")), process.stdout.strip()


def tensors_from_samples(samples: list[dict], device: torch.device):
    count = len(samples)
    features = torch.tensor([sample["features"] for sample in samples], dtype=torch.float32, device=device)
    hidden = torch.tensor([sample["hidden"] for sample in samples], dtype=torch.float32, device=device)
    sectors = torch.tensor([sample["sector"] for sample in samples], dtype=torch.long, device=device)
    stances = torch.tensor([sample["stance"] for sample in samples], dtype=torch.long, device=device)
    old_log_prob = torch.tensor([sample["logProb"] for sample in samples], dtype=torch.float32, device=device)
    advantages = torch.tensor([sample["advantage"] for sample in samples], dtype=torch.float32, device=device)
    returns = torch.tensor([sample["return"] for sample in samples], dtype=torch.float32, device=device)
    sector_mask = torch.zeros((count, SECTORS), dtype=torch.bool, device=device)
    stance_mask = torch.zeros((count, STANCES), dtype=torch.bool, device=device)
    for row, sample in enumerate(samples):
        sector_mask[row, sample["sectorMask"]] = True
        stance_mask[row, sample["stanceMask"]] = True
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)
    return features, hidden, sectors, stances, old_log_prob, advantages, returns, sector_mask, stance_mask


def ppo_update(model, optimizer, samples, device, epochs, batch_size, clip, entropy_weight):
    data = tensors_from_samples(samples, device)
    count = len(samples)
    totals = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "batches": 0}
    for _ in range(epochs):
        for indices in torch.randperm(count, device=device).split(batch_size):
            batch = [tensor[indices] for tensor in data]
            features, hidden, sectors, stances, old_log, advantages, returns, sector_mask, stance_mask = batch
            sector_logits, stance_logits, values = model(features, hidden)
            sector_logits = sector_logits.masked_fill(~sector_mask, -1e9)
            stance_logits = stance_logits.masked_fill(~stance_mask, -1e9)
            sector_dist = torch.distributions.Categorical(logits=sector_logits)
            stance_dist = torch.distributions.Categorical(logits=stance_logits)
            log_prob = sector_dist.log_prob(sectors) + stance_dist.log_prob(stances)
            ratio = (log_prob - old_log).exp()
            objective = torch.minimum(
                ratio * advantages,
                ratio.clamp(1 - clip, 1 + clip) * advantages,
            )
            policy_loss = -objective.mean()
            value_loss = F.smooth_l1_loss(values, returns)
            entropy = (sector_dist.entropy() + stance_dist.entropy()).mean()
            loss = policy_loss + value_loss * 0.5 - entropy * entropy_weight
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.7)
            optimizer.step()
            totals["policy"] += policy_loss.item()
            totals["value"] += value_loss.item()
            totals["entropy"] += entropy.item()
            totals["batches"] += 1
    batches = max(1, totals.pop("batches"))
    return {name: value / batches for name, value in totals.items()}


def score(summary: dict) -> float:
    rates = {}
    for name, bucket in summary.get("byOpponent", {}).items():
        games = bucket["wins"] + bucket["losses"] + bucket["draws"]
        rates[name] = (bucket["wins"] + 0.5 * bucket["draws"]) / max(1, games)
    overall = (summary["wins"] + summary["draws"] * 0.5) / max(1, summary["battles"])
    crowd = rates.get("crowd", overall)
    worst = min(rates.values(), default=overall)
    return 0.5 * overall + 0.3 * crowd + 0.2 * worst


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=8)
    parser.add_argument("--battles", type=int, default=192)
    parser.add_argument("--validation-battles", type=int, default=96)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.012)
    parser.add_argument("--seed", type=int, default=240726)
    parser.add_argument("--node", default="node")
    parser.add_argument("--opponents", default=LEAGUE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна: PPO-обучение ожидает установленный GPU PyTorch")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    model = CommanderActorCritic().to(device)
    work = ROOT / ".training"
    work.mkdir(exist_ok=True)
    checkpoint_path = work / "ppo-best.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()
    export_policy(model, {"status": "league baseline", "history": history})
    baseline, _ = run_collector(
        args.node, args.validation_battles, args.workers,
        args.seed + 8_500_000, work / "ppo-league-baseline.json",
        evaluate=True, opponents=args.opponents
    )
    best_score = score(baseline["summary"])
    history.append({
        "cycle": 0,
        "type": "league baseline",
        "validation": baseline["summary"],
        "validationScore": best_score,
    })

    for cycle in range(args.cycles):
        export_policy(model, {"status": "collecting", "cycle": cycle, "history": history})
        rollout_path = work / "ppo-rollouts.json"
        payload, _ = run_collector(
            args.node, args.battles, args.workers, args.seed + cycle * 10007,
            rollout_path, opponents=args.opponents
        )
        losses = ppo_update(
            model, optimizer, payload["samples"], device, args.epochs,
            args.batch_size, args.clip, args.entropy
        )
        export_policy(model, {"status": "validating", "cycle": cycle + 1, "history": history})
        validation_path = work / "ppo-validation.json"
        validation, _ = run_collector(
            args.node, args.validation_battles, args.workers,
            args.seed + 9_000_000 + cycle * 20011, validation_path,
            evaluate=True, opponents=args.opponents
        )
        validation_summary = validation["summary"]
        current_score = score(validation_summary)
        entry = {
            "cycle": cycle + 1,
            "rollout": payload["summary"],
            "samples": len(payload["samples"]),
            "losses": losses,
            "validation": validation_summary,
            "validationScore": current_score,
        }
        history.append(entry)
        if current_score > best_score:
            best_score = current_score
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model": best_state,
                "score": best_score,
                "history": history,
                "options": vars(args),
            }, checkpoint_path)
        print(
            f"Цикл {cycle + 1}/{args.cycles}: rollout "
            f"{payload['summary']['wins']}-{payload['summary']['losses']}-{payload['summary']['draws']}, "
            f"validation {validation_summary['wins']}-{validation_summary['losses']}-"
            f"{validation_summary['draws']}, score={current_score:.3f}, "
            f"policy={losses['policy']:.4f}, value={losses['value']:.4f}"
        )

    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    metadata = {
        "algorithm": "PPO Actor-Critic + GRU",
        "device": torch.cuda.get_device_name(0),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "bestValidationScore": best_score,
        "elapsedSeconds": elapsed,
        "options": vars(args),
        "history": history,
    }
    export_policy(model, metadata)
    print(f"Готово за {elapsed:.1f} с; лучшая PPO-политика сохранена в trained-ppo-policy.js")


if __name__ == "__main__":
    main()
