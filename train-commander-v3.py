"""Full-sequence recurrent PPO trainer for Commander V3."""

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
EMBED = 64
ATTENTION = 32
MEMORY = 64
ACTOR = 64
SECTORS = 7
STANCES = 5
REGIMENTS = 6
VERSION = 1


class CommanderV3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(INPUTS, EMBED)
        self.query = nn.Linear(EMBED, ATTENTION)
        self.key = nn.Linear(EMBED, ATTENTION)
        self.attention_value = nn.Linear(EMBED, ATTENTION)
        self.mix = nn.Linear(EMBED + ATTENTION, EMBED)
        self.gru = nn.GRUCell(EMBED, MEMORY)
        self.actor = nn.Linear(EMBED + MEMORY, ACTOR)
        self.sector = nn.Linear(ACTOR, SECTORS)
        self.stance = nn.Linear(ACTOR, STANCES)
        self.critic = nn.Linear(MEMORY, 1)
        for module in (self.encoder, self.query, self.key, self.attention_value, self.mix, self.actor):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.sector.weight, gain=0.01)
        nn.init.orthogonal_(self.stance.weight, gain=0.01)
        nn.init.zeros_(self.sector.bias)
        nn.init.zeros_(self.stance.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1)
        nn.init.zeros_(self.critic.bias)
        # A doctrine prior, not a combat bonus: begin from a compact central
        # hold instead of an arbitrary edge sector, then let PPO change it.
        with torch.no_grad():
            self.sector.bias[3] = 1.5
            self.stance.bias[0] = 1.5

    def step(self, features, active, hidden):
        encoded = torch.tanh(self.encoder(features))
        query = self.query(encoded)
        key = self.key(encoded)
        value = self.attention_value(encoded)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(ATTENTION)
        scores = scores.masked_fill(~active.unsqueeze(-2), -1e9)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, value)
        mixed = torch.tanh(self.mix(torch.cat((encoded, context), dim=-1)))
        active_float = active.unsqueeze(-1).float()
        pooled = (mixed * active_float).sum(dim=-2) / active_float.sum(dim=-2).clamp_min(1)
        next_hidden = self.gru(pooled, hidden)
        broadcast_hidden = next_hidden.unsqueeze(-2).expand(-1, mixed.shape[-2], -1)
        actor = torch.tanh(self.actor(torch.cat((mixed, broadcast_hidden), dim=-1)))
        return self.sector(actor), self.stance(actor), self.critic(next_hidden).squeeze(-1), next_hidden

    def sequence(self, features, active, initial_hidden):
        hidden = initial_hidden
        sector_logits, stance_logits, values = [], [], []
        for step in range(features.shape[1]):
            sector, stance, value, hidden = self.step(
                features[:, step], active[:, step], hidden
            )
            sector_logits.append(sector)
            stance_logits.append(stance)
            values.append(value)
        return (
            torch.stack(sector_logits, dim=1),
            torch.stack(stance_logits, dim=1),
            torch.stack(values, dim=1),
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
        "actorWeight": flat(state["actor.weight"]),
        "actorBias": flat(state["actor.bias"]),
        "sectorWeight": flat(state["sector.weight"]),
        "sectorBias": flat(state["sector.bias"]),
        "stanceWeight": flat(state["stance.weight"]),
        "stanceBias": flat(state["stance.bias"]),
        "criticWeight": flat(state["critic.weight"]),
        "criticBias": flat(state["critic.bias"]),
    }
    policy = {
        "version": VERSION,
        "inputs": INPUTS,
        "regiments": "variable",
        "embed": EMBED,
        "attention": ATTENTION,
        "memory": MEMORY,
        "actor": ACTOR,
        "weights": weights,
    }
    source = (
        "// Generated locally by train-commander-v3.py.\n"
        f"export const TRAINED_COMMANDER_V3_POLICY = {json.dumps(policy, separators=(',', ':'))};\n"
        f"export const COMMANDER_V3_TRAINING_METADATA = "
        f"{json.dumps(metadata, ensure_ascii=False, indent=2)};\n"
    )
    (ROOT / "trained-commander-v3-policy.js").write_text(source, encoding="utf8")


def run_collector(node, battles, workers, seed, output, exploration, evaluate=False, opponents="crowd"):
    command = [
        node, str(ROOT / "collect-commander-v3-rollouts.js"),
        "--battles", str(battles),
        "--workers", str(workers),
        "--seed", str(seed),
        "--exploration", str(exploration),
        "--opponents", opponents,
        "--output", str(output),
    ]
    if evaluate:
        command.append("--evaluate")
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(output.read_text(encoding="utf8"))


def make_episode_batches(samples, device):
    grouped = {}
    for sample in samples:
        grouped.setdefault(sample["episode"], []).append(sample)
    episodes = [sorted(values, key=lambda item: item["step"]) for values in grouped.values()]
    for episode in episodes:
        for item in episode:
            item["advantage"] = float(item["advantage"])
    advantages = torch.tensor(
        [item["advantage"] for episode in episodes for item in episode],
        dtype=torch.float32,
    )
    mean, std = advantages.mean().item(), advantages.std(unbiased=False).item() + 1e-6
    return episodes, mean, std


def pad_batch(episodes, advantage_mean, advantage_std, device):
    batch = len(episodes)
    steps = max(len(episode) for episode in episodes)
    features = torch.zeros((batch, steps, REGIMENTS, INPUTS), dtype=torch.float32, device=device)
    active = torch.zeros((batch, steps, REGIMENTS), dtype=torch.bool, device=device)
    sectors = torch.zeros((batch, steps, REGIMENTS), dtype=torch.long, device=device)
    stances = torch.zeros((batch, steps, REGIMENTS), dtype=torch.long, device=device)
    sector_mask = torch.zeros((batch, steps, REGIMENTS, SECTORS), dtype=torch.bool, device=device)
    stance_mask = torch.zeros((batch, steps, REGIMENTS, STANCES), dtype=torch.bool, device=device)
    old_action_log = torch.zeros((batch, steps, REGIMENTS), dtype=torch.float32, device=device)
    advantages = torch.zeros((batch, steps), dtype=torch.float32, device=device)
    returns = torch.zeros((batch, steps), dtype=torch.float32, device=device)
    time_mask = torch.zeros((batch, steps), dtype=torch.bool, device=device)
    initial_hidden = torch.zeros((batch, MEMORY), dtype=torch.float32, device=device)
    for row, episode in enumerate(episodes):
        initial_hidden[row] = torch.tensor(episode[0]["hidden"], dtype=torch.float32, device=device)
        for step, item in enumerate(episode):
            features[row, step] = torch.tensor(item["features"], dtype=torch.float32, device=device)
            active[row, step] = torch.tensor(item["activeMask"], dtype=torch.bool, device=device)
            sectors[row, step] = torch.tensor(
                [max(0, value) for value in item["sectors"]], dtype=torch.long, device=device
            )
            stances[row, step] = torch.tensor(
                [max(0, value) for value in item["stances"]], dtype=torch.long, device=device
            )
            old_action_log[row, step] = torch.tensor(
                item["actionLogProbs"], dtype=torch.float32, device=device
            )
            for regiment in range(REGIMENTS):
                sector_mask[row, step, regiment, item["sectorMasks"][regiment]] = True
                stance_mask[row, step, regiment, item["stanceMasks"][regiment]] = True
            advantages[row, step] = (item["advantage"] - advantage_mean) / advantage_std
            returns[row, step] = item["return"]
            time_mask[row, step] = True
    return {
        "features": features,
        "active": active,
        "sectors": sectors,
        "stances": stances,
        "sector_mask": sector_mask,
        "stance_mask": stance_mask,
        "old_action_log": old_action_log,
        "advantages": advantages,
        "returns": returns,
        "time_mask": time_mask,
        "initial_hidden": initial_hidden,
    }


def mixed_log_prob(logits, mask, actions, exploration):
    masked_logits = logits.masked_fill(~mask, -1e9)
    policy = torch.softmax(masked_logits, dim=-1)
    allowed = mask.sum(dim=-1).clamp_min(1)
    probabilities = (1 - exploration) * policy + exploration * mask.float() / allowed.unsqueeze(-1)
    selected = probabilities.gather(-1, actions.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9)
    entropy = -(policy.clamp_min(1e-9).log() * policy).sum(dim=-1)
    return selected.log(), entropy


def ppo_update(
    model, optimizer, samples, device, epochs, episode_batch,
    clip, entropy_weight, exploration, target_kl
):
    episodes, advantage_mean, advantage_std = make_episode_batches(samples, device)
    totals = {
        "policy": 0.0, "value": 0.0, "robustness": 0.0,
        "entropy": 0.0, "kl": 0.0, "batches": 0
    }
    stopped_early = False
    for _ in range(epochs):
        order = torch.randperm(len(episodes)).tolist()
        for start in range(0, len(order), episode_batch):
            selected = [episodes[index] for index in order[start:start + episode_batch]]
            data = pad_batch(selected, advantage_mean, advantage_std, device)
            sector_logits, stance_logits, values = model.sequence(
                data["features"], data["active"], data["initial_hidden"]
            )
            sector_log, sector_entropy = mixed_log_prob(
                sector_logits, data["sector_mask"], data["sectors"], exploration
            )
            stance_log, stance_entropy = mixed_log_prob(
                stance_logits, data["stance_mask"], data["stances"], exploration
            )
            regiment_mask = data["active"].float()
            log_prob = ((sector_log + stance_log) * regiment_mask).sum(dim=-1)
            old_log_prob = (data["old_action_log"] * regiment_mask).sum(dim=-1)
            ratio = (log_prob - old_log_prob).clamp(-20, 20).exp()
            objective = torch.minimum(
                ratio * data["advantages"],
                ratio.clamp(1 - clip, 1 + clip) * data["advantages"],
            )
            time_mask = data["time_mask"].float()
            per_episode_steps = time_mask.sum(dim=1).clamp_min(1)
            policy_loss = -((objective * time_mask).sum(dim=1) / per_episode_steps).mean()
            value_error = F.smooth_l1_loss(values, data["returns"], reduction="none")
            value_loss = ((value_error * time_mask).sum(dim=1) / per_episode_steps).mean()
            augmented_active = data["active"] & (
                torch.rand_like(data["active"].float()) > 0.10
            )
            empty = ~augmented_active.any(dim=-1) & data["active"].any(dim=-1)
            if empty.any():
                first_active = data["active"].float().argmax(dim=-1)
                fallback = F.one_hot(first_active, num_classes=REGIMENTS).bool()
                augmented_active |= fallback & empty.unsqueeze(-1)
            _, _, augmented_values = model.sequence(
                data["features"], augmented_active, data["initial_hidden"]
            )
            robustness_error = F.smooth_l1_loss(
                augmented_values, data["returns"], reduction="none"
            )
            robustness_loss = (
                (robustness_error * time_mask).sum(dim=1) / per_episode_steps
            ).mean()
            entropy = (
                ((sector_entropy + stance_entropy) * regiment_mask).sum(dim=-1)
                / regiment_mask.sum(dim=-1).clamp_min(1)
            )
            entropy = ((entropy * time_mask).sum(dim=1) / per_episode_steps).mean()
            approx_kl = (
                ((old_log_prob - log_prob) * time_mask).sum() / time_mask.sum().clamp_min(1)
            )
            loss = (
                policy_loss + value_loss * 0.5 + robustness_loss * 0.06
                - entropy * entropy_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.6)
            optimizer.step()
            totals["policy"] += policy_loss.item()
            totals["value"] += value_loss.item()
            totals["robustness"] += robustness_loss.item()
            totals["entropy"] += entropy.item()
            totals["kl"] += approx_kl.item()
            totals["batches"] += 1
            if approx_kl.item() > target_kl * 1.5:
                stopped_early = True
                break
        if stopped_early:
            break
    batches = max(1, totals.pop("batches"))
    result = {name: value / batches for name, value in totals.items()}
    result["earlyStop"] = stopped_early
    return result


def validation_score(summary):
    return (summary["wins"] + summary["draws"] * 0.5) / max(1, summary["battles"])


def action_share(summary, names):
    total = sum(summary.get("actions", {}).values()) or 1
    return sum(summary.get("actions", {}).get(name, 0) for name in names) / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--battles", type=int, default=192)
    parser.add_argument("--validation-battles", type=int, default=120)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--episode-batch", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--clip", type=float, default=0.18)
    parser.add_argument("--target-kl", type=float, default=0.018)
    parser.add_argument("--seed", type=int, default=730031)
    parser.add_argument("--node", default="node")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    model = CommanderV3().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=1e-5)
    work = ROOT / ".training"
    work.mkdir(exist_ok=True)
    history = []
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()
    exploration = 0.18
    export_policy(model, {"status": "fresh initialization", "history": []})
    baseline_path = work / "commander-v3-baseline.json"
    baseline = run_collector(
        args.node, args.validation_battles, args.workers,
        args.seed + 8_500_000, baseline_path, 0, evaluate=True
    )["summary"]
    best_score = validation_score(baseline)
    history.append({
        "cycle": 0,
        "type": "doctrine baseline",
        "validation": baseline,
        "validationScore": best_score,
        "validationAttackShare": action_share(baseline, ("assault", "flank")),
    })
    torch.save({
        "model": best_state,
        "score": best_score,
        "history": history,
        "options": vars(args),
    }, work / "commander-v3-best.pt")
    print(
        f"V3 baseline: {baseline['wins']}-{baseline['losses']}-{baseline['draws']} "
        f"score={best_score:.3f}",
        flush=True,
    )

    for cycle in range(args.cycles):
        rollout_path = work / "commander-v3-rollouts.json"
        rollout = run_collector(
            args.node, args.battles, args.workers,
            args.seed + cycle * 10007, rollout_path, exploration
        )
        entropy_weight = max(0.012, 0.03 * (0.86 ** cycle))
        losses = ppo_update(
            model, optimizer, rollout["samples"], device, args.epochs,
            args.episode_batch, args.clip, entropy_weight, exploration, args.target_kl
        )
        export_policy(model, {"status": "validating", "cycle": cycle + 1, "history": history})
        validation_path = work / "commander-v3-validation.json"
        validation = run_collector(
            args.node, args.validation_battles, args.workers,
            args.seed + 9_000_000 + cycle * 20011,
            validation_path, 0, evaluate=True
        )["summary"]
        current_score = validation_score(validation)
        attack_share = action_share(rollout["summary"], ("assault", "flank"))
        validation_attack_share = action_share(validation, ("assault", "flank"))
        entry = {
            "cycle": cycle + 1,
            "exploration": exploration,
            "entropyWeight": entropy_weight,
            "rollout": rollout["summary"],
            "samples": len(rollout["samples"]),
            "losses": losses,
            "validation": validation,
            "validationScore": current_score,
            "attackShare": attack_share,
            "validationAttackShare": validation_attack_share,
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
            }, work / "commander-v3-best.pt")
        exploration = max(0.06, exploration * 0.88)
        if attack_share < 0.08 or validation_attack_share < 0.08:
            exploration = max(exploration, 0.18)
        print(
            f"V3 {cycle + 1}/{args.cycles}: rollout "
            f"{rollout['summary']['wins']}-{rollout['summary']['losses']}-"
            f"{rollout['summary']['draws']}, validation "
            f"{validation['wins']}-{validation['losses']}-{validation['draws']} "
            f"score={current_score:.3f}, attack={attack_share:.3f}/"
            f"{validation_attack_share:.3f}, "
            f"KL={losses['kl']:.4f}, entropy={losses['entropy']:.3f}",
            flush=True,
        )

    model.load_state_dict(best_state)
    elapsed = time.perf_counter() - started
    metadata = {
        "algorithm": "full-sequence recurrent PPO + self-attention",
        "device": torch.cuda.get_device_name(0),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "variableRegimentCount": True,
        "bestValidationScore": best_score,
        "elapsedSeconds": elapsed,
        "options": vars(args),
        "history": history,
    }
    export_policy(model, metadata)
    print(f"Готово за {elapsed:.1f} с; лучший Commander V3 сохранён.")


if __name__ == "__main__":
    main()
