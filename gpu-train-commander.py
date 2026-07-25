import argparse
import copy
import json
import pathlib
import subprocess
import time

import torch
from torch import nn
from torch.nn import functional as F


ROOT = pathlib.Path(__file__).resolve().parent
POLICY_INPUTS = 48
HIDDEN_1 = 32
HIDDEN_2 = 24
SECTORS = 7
STANCES = 5
OUTPUTS = SECTORS + STANCES
VERSION = 3


class CommanderPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(POLICY_INPUTS, HIDDEN_1)
        self.second = nn.Linear(HIDDEN_1, HIDDEN_2)
        self.output = nn.Linear(HIDDEN_2, OUTPUTS)

    def forward(self, values):
        values = torch.tanh(self.first(values))
        values = torch.tanh(self.second(values))
        return self.output(values)


def load_flat_weights(model, policy):
    if not policy or policy.get("version") != VERSION:
        return
    flat = torch.tensor(policy["weights"], dtype=torch.float32)
    if float(flat.abs().max()) < 1e-8:
        return
    cursor = 0
    with torch.no_grad():
        for parameter in [
            model.first.weight, model.first.bias,
            model.second.weight, model.second.bias,
            model.output.weight, model.output.bias,
        ]:
            count = parameter.numel()
            parameter.copy_(flat[cursor:cursor + count].reshape(parameter.shape))
            cursor += count


def flat_weights(model):
    tensors = [
        model.first.weight, model.first.bias,
        model.second.weight, model.second.bias,
        model.output.weight, model.output.bias,
    ]
    return torch.cat([tensor.detach().cpu().reshape(-1) for tensor in tensors]).tolist()


def stance_mask(features, device):
    roles = features[:, 10:16].argmax(dim=1)
    allowed = torch.zeros((features.shape[0], STANCES), dtype=torch.bool, device=device)
    allowed[roles <= 2, 0:3] = True
    allowed[roles == 3, 0:2] = True
    allowed[roles == 4, 0] = True
    allowed[roles == 4, 2:4] = True
    allowed[roles == 5, 0:3] = True
    allowed[roles == 5, 4] = True
    return allowed


def train_dataset(model, payload, device, epochs, batch_size, learning_rate):
    samples = payload["samples"]
    features = torch.tensor([item["features"] for item in samples], dtype=torch.float32)
    sectors = torch.tensor([item["sector"] for item in samples], dtype=torch.long)
    stances = torch.tensor([item["stance"] for item in samples], dtype=torch.long)
    rewards = torch.tensor([item["reward"] for item in samples], dtype=torch.float32)

    cutoff = torch.quantile(rewards, 0.58)
    elite = rewards >= cutoff
    if elite.sum() < 256:
        elite = rewards >= torch.quantile(rewards, 0.4)
    features = features[elite]
    sectors = sectors[elite]
    stances = stances[elite]
    rewards = rewards[elite]
    weights = (rewards - rewards.min() + 0.08)
    weights = weights / weights.mean().clamp_min(1e-6)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    model.train()
    generator = torch.Generator().manual_seed(99173)
    last_loss = 0.0
    for _ in range(epochs):
        order = torch.randperm(features.shape[0], generator=generator)
        for start in range(0, features.shape[0], batch_size):
            indices = order[start:start + batch_size]
            batch_x = features[indices].to(device, non_blocking=True)
            batch_sector = sectors[indices].to(device, non_blocking=True)
            batch_stance = stances[indices].to(device, non_blocking=True)
            batch_weight = weights[indices].to(device, non_blocking=True)
            logits = model(batch_x)
            sector_logits = logits[:, :SECTORS]
            stance_logits = logits[:, SECTORS:]
            mask = stance_mask(batch_x, device)
            stance_logits = stance_logits.masked_fill(~mask, -1e9)
            sector_loss = F.cross_entropy(sector_logits, batch_sector, reduction="none")
            stance_loss = F.cross_entropy(stance_logits, batch_stance, reduction="none")
            entropy = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
            loss = ((sector_loss + stance_loss) * batch_weight).mean() - entropy.mean() * 0.003
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            last_loss = float(loss.detach())
    return last_loss, int(features.shape[0]), len(samples)


def save_policy(model, metadata):
    policy = {
        "version": VERSION,
        "inputs": POLICY_INPUTS,
        "hidden": [HIDDEN_1, HIDDEN_2],
        "outputs": OUTPUTS,
        "weights": flat_weights(model),
    }
    source = (
        "// Generated locally by gpu-train-commander.py.\n"
        f"export const TRAINED_POLICY = {json.dumps(policy, separators=(',', ':'))};\n"
        f"export const TRAINING_METADATA = {json.dumps(metadata, ensure_ascii=False, indent=2)};\n"
    )
    (ROOT / "trained-policy.js").write_text(source, encoding="utf8")


def validate_policy(node, workers, trials, seed):
    summary = {"wins": 0, "losses": 0, "draws": 0, "battles": 0}
    for units in (100, 200, 400):
        for neural_team in (0, 1):
            blue, red = ("neural", "crowd") if neural_team == 0 else ("crowd", "neural")
            command = [
                node, str(ROOT / "diagnostic.js"),
                "--blue", blue,
                "--red", red,
                "--units", str(units),
                "--trials", str(trials),
                "--workers", str(workers),
                "--seed", str(seed + units * 97 + neural_team * 100003),
                "--json",
            ]
            process = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            result = json.loads(process.stdout)["results"][0]
            neural_wins = result["blueWins"] if neural_team == 0 else result["redWins"]
            crowd_wins = result["redWins"] if neural_team == 0 else result["blueWins"]
            summary["wins"] += neural_wins
            summary["losses"] += crowd_wins
            summary["draws"] += result["draws"]
            summary["battles"] += trials
    summary["score"] = (
        summary["wins"] + summary["draws"] * 0.5
    ) / max(1, summary["battles"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--battles", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.34)
    parser.add_argument("--epsilon", type=float, default=0.16)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=94103)
    parser.add_argument("--node", default="node")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--validation-trials", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA недоступна в установленной сборке PyTorch")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model = CommanderPolicy().to(device)
    started = time.perf_counter()
    history = []
    best_score = -1.0
    best_state = copy.deepcopy(model.state_dict())
    if args.fresh:
        save_policy(model, {"status": "fresh initialization", "history": []})

    for cycle in range(args.cycles):
        rollout_path = ROOT / ".training" / "rollouts.json"
        command = [
            args.node, str(ROOT / "collect-rollouts.js"),
            "--battles", str(args.battles),
            "--workers", str(args.workers),
            "--seed", str(args.seed + cycle * 104729),
            "--temperature", str(args.temperature * (0.9 ** cycle)),
            "--epsilon", str(max(0.06, args.epsilon * (0.88 ** cycle))),
            "--output", str(rollout_path),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        payload = json.loads(rollout_path.read_text(encoding="utf8"))
        if cycle == 0 and not args.fresh:
            load_flat_weights(model, payload.get("policy"))
        loss, elite_samples, total_samples = train_dataset(
            model, payload, device, args.epochs, args.batch_size, args.learning_rate
        )
        save_policy(model, {"status": "validating", "cycle": cycle + 1, "history": history})
        validation = validate_policy(
            args.node,
            args.workers,
            args.validation_trials,
            args.seed + 8_000_000 + cycle * 20011,
        )
        cycle_result = {
            "cycle": cycle + 1,
            "rollouts": payload["summary"],
            "samples": total_samples,
            "eliteSamples": elite_samples,
            "loss": loss,
            "validation": validation,
        }
        history.append(cycle_result)
        if validation["score"] > best_score:
            best_score = validation["score"]
            best_state = copy.deepcopy(model.state_dict())
        metadata = {
            "backend": "PyTorch CUDA",
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "options": vars(args),
            "history": history,
            "elapsedSeconds": time.perf_counter() - started,
        }
        save_policy(model, metadata)
        print(
            f"GPU-цикл {cycle + 1}/{args.cycles}: "
            f"{payload['summary']['wins']}-{payload['summary']['losses']}-{payload['summary']['draws']}, "
            f"samples={total_samples}, elite={elite_samples}, loss={loss:.4f}, "
            f"validation={validation['wins']}-{validation['losses']}-{validation['draws']} "
            f"({validation['score']:.3f})",
            flush=True,
        )

    model.load_state_dict(best_state)
    metadata = {
        "backend": "PyTorch CUDA",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "options": vars(args),
        "history": history,
        "bestValidationScore": best_score,
        "elapsedSeconds": time.perf_counter() - started,
    }
    save_policy(model, metadata)
    print(
        f"Готово за {time.perf_counter() - started:.1f} с; "
        f"лучшая проверенная модель сохранена в trained-policy.js"
    )


if __name__ == "__main__":
    main()
