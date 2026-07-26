"""Coevolution league trainer with self-play and best-of-branch selection."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / ".venv-gpu" / "Scripts" / "python.exe"
LEAGUE = "crowd,neural,ppo,commander_v3,commander_v4,offensive,defensive,adaptive"


@dataclass(frozen=True)
class Model:
    name: str
    policy: str
    checkpoint: str | None
    trainer: str
    collector: str


MODELS = (
    Model("neural", "trained-policy.js", None, "gpu-train-commander.py", "collect-rollouts.js"),
    Model("ppo", "trained-ppo-policy.js", ".training/ppo-best.pt",
          "train-ppo-commander.py", "collect-ppo-rollouts.js"),
    Model("commander_v3", "trained-commander-v3-policy.js", ".training/commander-v3-best.pt",
          "train-commander-v3.py", "collect-commander-v3-rollouts.js"),
    Model("commander_v4", "trained-commander-v4-policy.js", ".training/commander-v4-best.pt",
          "train-commander-v4.py", "collect-commander-v4-rollouts.js"),
)


def run(command: list[str]) -> None:
    print(">", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def robust_score(summary: dict) -> tuple[float, dict[str, float]]:
    rates = {}
    for name, bucket in summary.get("byOpponent", {}).items():
        games = bucket["wins"] + bucket["losses"] + bucket["draws"]
        rates[name] = (bucket["wins"] + 0.5 * bucket["draws"]) / max(1, games)
    overall = (summary["wins"] + 0.5 * summary["draws"]) / max(1, summary["battles"])
    crowd = rates.get("crowd", overall)
    worst = min(rates.values(), default=overall)
    return 0.5 * overall + 0.3 * crowd + 0.2 * worst, rates


def evaluate(
    model: Model, battles: int, workers: int, seed: int, output: Path
) -> dict:
    run([
        "node", model.collector,
        "--battles", str(battles),
        "--workers", str(workers),
        "--seed", str(seed),
        "--opponents", LEAGUE,
        "--output", str(output),
        "--evaluate",
    ])
    summary = json.loads(output.read_text(encoding="utf8"))["summary"]
    score, rates = robust_score(summary)
    return {"score": score, "rates": rates, "summary": summary}


def snapshot(model: Model) -> tuple[bytes, bytes | None]:
    policy = (ROOT / model.policy).read_bytes()
    checkpoint_path = ROOT / model.checkpoint if model.checkpoint else None
    checkpoint = checkpoint_path.read_bytes() if checkpoint_path and checkpoint_path.exists() else None
    return policy, checkpoint


def restore(model: Model, policy: bytes, checkpoint: bytes | None) -> None:
    (ROOT / model.policy).write_bytes(policy)
    if model.checkpoint and checkpoint is not None:
        path = ROOT / model.checkpoint
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(checkpoint)


def trainer_command(
    model: Model, cycles: int, battles: int, validation_battles: int,
    workers: int, seed: int, teacher_scenarios: int
) -> list[str]:
    command = [
        str(PYTHON), model.trainer,
        "--cycles", str(cycles),
        "--battles", str(battles),
        "--workers", str(workers),
        "--seed", str(seed),
        "--opponents", LEAGUE,
    ]
    if model.name == "neural":
        command += ["--validation-trials", str(validation_battles)]
    else:
        command += ["--validation-battles", str(validation_battles), "--resume"]
    if model.name == "commander_v4":
        command += ["--teacher-scenarios", str(teacher_scenarios)]
    return command


def train_model(
    model: Model, round_index: int, branches: int, cycles: int, battles: int,
    validation_battles: int, workers: int, seed: int, teacher_scenarios: int,
    run_dir: Path
) -> dict:
    model_dir = run_dir / f"round-{round_index + 1}" / model.name
    model_dir.mkdir(parents=True, exist_ok=True)
    base_policy, base_checkpoint = snapshot(model)
    (model_dir / "baseline-policy.js").write_bytes(base_policy)
    if base_checkpoint is not None:
        (model_dir / "baseline-checkpoint.pt").write_bytes(base_checkpoint)
    fixed_seed = seed + round_index * 10_000_000 + MODELS.index(model) * 1_000_000
    baseline = evaluate(
        model, validation_battles, workers, fixed_seed + 900_000,
        model_dir / "baseline-evaluation.json"
    )
    candidates = [{
        "branch": 0,
        "kind": "baseline",
        "evaluation": baseline,
        "policy": base_policy,
        "checkpoint": base_checkpoint,
    }]
    print(
        f"{model.name} baseline robust={baseline['score']:.3f} "
        f"rates={baseline['rates']}", flush=True
    )

    for branch in range(1, branches + 1):
        restore(model, base_policy, base_checkpoint)
        branch_seed = fixed_seed + branch * 104_729
        run(trainer_command(
            model, cycles, battles, validation_battles, workers,
            branch_seed, teacher_scenarios
        ))
        policy, checkpoint = snapshot(model)
        branch_path = model_dir / f"branch-{branch}-policy.js"
        branch_path.write_bytes(policy)
        if checkpoint is not None:
            (model_dir / f"branch-{branch}-checkpoint.pt").write_bytes(checkpoint)
        result = evaluate(
            model, validation_battles, workers, fixed_seed + 900_000,
            model_dir / f"branch-{branch}-evaluation.json"
        )
        candidates.append({
            "branch": branch,
            "kind": "trained",
            "evaluation": result,
            "policy": policy,
            "checkpoint": checkpoint,
        })
        print(
            f"{model.name} branch {branch}/{branches}: "
            f"robust={result['score']:.3f} rates={result['rates']}", flush=True
        )

    best = max(candidates, key=lambda item: item["evaluation"]["score"])
    restore(model, best["policy"], best["checkpoint"])
    shutil.copy2(ROOT / model.policy, model_dir / "selected-policy.js")
    if model.checkpoint and (ROOT / model.checkpoint).exists():
        shutil.copy2(ROOT / model.checkpoint, model_dir / "selected-checkpoint.pt")
    report = {
        "model": model.name,
        "selectedBranch": best["branch"],
        "selectedKind": best["kind"],
        "score": best["evaluation"]["score"],
        "rates": best["evaluation"]["rates"],
        "summary": best["evaluation"]["summary"],
        "candidates": [{
            "branch": item["branch"],
            "kind": item["kind"],
            **item["evaluation"],
        } for item in candidates],
    }
    (model_dir / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(
        f"{model.name}: selected branch {best['branch']} "
        f"robust={best['evaluation']['score']:.3f}", flush=True
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--branches", type=int, default=2)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--battles", type=int, default=192)
    parser.add_argument("--validation-battles", type=int, default=160)
    parser.add_argument("--final-battles", type=int, default=320)
    parser.add_argument("--teacher-scenarios", type=int, default=12)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=26072641)
    args = parser.parse_args()
    if not PYTHON.exists():
        raise RuntimeError(f"GPU Python was not found: {PYTHON}")

    started = time.perf_counter()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = ROOT / ".training" / "league" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for round_index in range(args.rounds):
        for model in MODELS:
            reports.append(train_model(
                model, round_index, args.branches, args.cycles,
                args.battles, args.validation_battles, args.workers,
                args.seed, args.teacher_scenarios, run_dir
            ))

    final = {}
    for index, model in enumerate(MODELS):
        final[model.name] = evaluate(
            model, args.final_battles, args.workers,
            args.seed + 80_000_000 + index * 100_003,
            run_dir / f"final-{model.name}.json"
        )
    result = {
        "league": LEAGUE.split(","),
        "options": vars(args),
        "elapsedSeconds": time.perf_counter() - started,
        "selections": reports,
        "final": final,
    }
    (run_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
    )
    (ROOT / ".training" / "latest-league-report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8"
    )
    print(f"League training completed: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
