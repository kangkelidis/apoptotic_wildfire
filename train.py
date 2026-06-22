#!/usr/bin/env python3
"""
MAPPO Training Script.

This script trains a Multi-Agent Proximal Policy Optimization (MAPPO) agent
to control the wildfire suppression drones in the simulation environment.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from tqdm import tqdm

from src.core.batch_runner import BatchRunner
from src.core.engine import SimulationEngine
from src.rl.env import WildfireEnv
from src.rl.policy import SwarmPolicy
from src.rl.rewards import available_reward_profiles
from src.rl.trainer import PPOConfig, PPOTrainer, RolloutBuffer
from src.utils.analysis import MultiExperimentAnalyzer
from src.utils.config_loader import load_config
from src.utils.hardware import generate_seeds
from src.utils.outputs import create_output_directory, output_paths_from_root

# rollout steps are always synced from simulation.max_steps at runtime
ROLLOUT_STEPS = "AUTO_FROM_MAX_STEPS"
TRAIN_BATCH_SIZE = 512
LEARNING_RATE = 3e-4
PPO_EPOCHS = 4
MINI_BATCH_SIZE = 8192
ENTROPY_COEFF = 0.05
TOTAL_STEPS = 100_000
UPDATES_PER_RUN = 50

SEED_MODE = "cycle"      # "static" or "cycle"
SEED_REPEAT = 8          # CONSECUTIVE episodes per seed in cycle mode
SEED_POOL_SIZE = 64      # Number of deterministic seeds in cycle mode
LOG_EVERY_UPDATES = 1
SAVE_EVERY_UPDATES = 5
VISUALIZE_PROGRESS_EVERY_UPDATES = 0
FINAL_ANALYSIS_STRATEGIES = (
    "mappo",
)

# Terminal utility anneal: reward term starts weak to stabilize early PPO.
TERMINAL_UTILITY_START_WEIGHT = 0.0
TERMINAL_UTILITY_TARGET_WEIGHT = 1.0
TERMINAL_UTILITY_ANNEAL_UPDATES = 100


def _derive_training_master_seed(base_master_seed: int) -> int:
    """Derive a deterministic training-only seed stream from config seed."""
    derived = int(generate_seeds(int(base_master_seed), 2)[-1])
    if derived == int(base_master_seed):
        derived = int((int(base_master_seed) + 1) % (2**31))
    return derived


def _checkpoint_payload(
    *,
    policy: SwarmPolicy,
    optimizer: torch.optim.Optimizer,
    update: int,
    steps: int,
    env_episode_idx: int,
    training_sessions: int,
    training_base_seed: int,
    training_master_seed: int,
    output_dir: Path,
    checkpoint_stage: str,
    metrics: dict | None = None,
    actor_feature_names: list[str] | None = None,
    critic_feature_names: list[str] | None = None,
    reward_profile: str | None = None,
) -> dict:
    payload = {
        'policy': policy.state_dict(),
        'optimizer': optimizer.state_dict(),
        'update': int(update),
        'steps': int(steps),
        'env_episode_idx': int(env_episode_idx),
        'training_sessions': int(training_sessions),
        'training_base_seed': int(training_base_seed),
        'training_master_seed': int(training_master_seed),
        'output_dir': str(output_dir.resolve()),
        'checkpoint_stage': str(checkpoint_stage),
    }
    if actor_feature_names is not None:
        payload['actor_feature_names'] = list(actor_feature_names)
    if critic_feature_names is not None:
        payload['critic_feature_names'] = list(critic_feature_names)
    if reward_profile is not None:
        payload['reward_profile'] = str(reward_profile)
    if metrics is not None:
        payload['metrics'] = dict(metrics)
    return payload


def get_action_mask(engine: SimulationEngine) -> torch.Tensor:
    """
    """
    mgr = engine.swarm

    # 1. Base Mask (All True)
    mask = torch.ones(
        (engine.batch_size, engine.n_drones, 2),
        dtype=torch.bool, device=engine.device
    )

    # Action 0 (stay/return) remains always legal.
    can_issue_go = mgr.decision_requests.squeeze(-1)
    mask[:, :, 1] = can_issue_go

    return mask


def _collect_rollout(
    *,
    env: WildfireEnv,
    policy: SwarmPolicy,
    ppo_config: PPOConfig,
    device: torch.device,
    buffer: RolloutBuffer,
    desc: str = "rollout",
) -> tuple[torch.Tensor, int]:
    """Collect one rollout without mutating optimizer state."""
    buffer.reset()
    (actor_obs, critic_obs), decision_mask = env.reset()
    env_steps = 0

    with torch.no_grad():
        for _ in tqdm(range(ppo_config.rollout_steps), desc=desc, leave=False):
            action_mask = get_action_mask(env.engine)
            if actor_obs is None:
                raise RuntimeError(
                    "Environment returned None for actor_obs during rollout.")
            actions, log_probs, _ = policy.get_action(actor_obs, action_mask)
            if critic_obs is None:
                raise RuntimeError(
                    "Environment returned None for critic_obs during rollout.")
            values = policy.get_value(critic_obs)

            (next_actor_obs, next_critic_obs), rewards, next_decision_mask, done, info = env.step(
                actions)
            step_decision_mask = info.get('decision_mask_step', decision_mask)

            done_mask = torch.full(
                (env.batch_size, env.n_drones, 1),
                done,
                dtype=torch.bool,
                device=device
            )
            buffer.add(
                actor_obs, critic_obs, actions, log_probs, rewards, values,
                done_mask,
                action_mask, step_decision_mask
            )

            if done:
                (next_actor_obs, next_critic_obs), next_decision_mask = env.reset()

            actor_obs = next_actor_obs
            critic_obs = next_critic_obs
            decision_mask = next_decision_mask
            env_steps += env.batch_size

        if actor_obs is None:
            raise RuntimeError("actor_obs is None after rollout loop")
        if critic_obs is None:
            raise RuntimeError("critic_obs is None after rollout loop")
        last_value = policy.get_value(critic_obs)

    return last_value, env_steps


def _run_progress_visualization_subprocess(
    *,
    preset: str,
    scenario: str,
    output_dir: Path,
    update: int,
    total_env_steps: int,
    seed: int,
    model_selector: str = "latest",
) -> None:
    """Run a single MAPPO visualize pass to snapshot training progress."""
    tag = f"upd{int(update):05d}_step{int(total_env_steps)}"
    cmd = [
        sys.executable,
        str((Path(__file__).resolve().parent / "main.py")),
        "--_visualize_once",
        "--preset", preset,
        "--strategies", "mappo",
        "--scenarios", scenario,
        "--model", model_selector,
        "--_output_dir", str(output_dir),
        "--_video_tag", tag,
        "--_seed", str(int(seed)),
    ]
    subprocess.run(cmd, check=True)


def _summarize_scenario_deployment(env: WildfireEnv) -> dict:
    scenario = getattr(env.engine.events, "_scenario_data", {}) or {}
    reserved = scenario.get("reserved_drones", {}) or {}
    deploy_events = []
    for event in scenario.get("events", []) or []:
        if str(event.get("type", "")).lower() != "deploy_drones":
            continue
        deploy_events.append({
            "step": event.get("step"),
            "step_percentage": event.get("step_percentage"),
            "count": event.get("count"),
            "percentage": event.get("percentage"),
        })
    return {
        "reserved_drones": reserved,
        "deploy_events": deploy_events,
    }


def _write_training_report(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    sim_cfg: dict,
    device: torch.device,
    env: WildfireEnv,
    ppo_config: PPOConfig,
    total_steps: int,
    steps_per_update: int,
    n_updates: int,
    start_update: int,
    total_env_steps: int,
    resume_path: str | None,
    training_base_seed: int,
    training_master_seed: int,
) -> Path:
    """Persist a compact report of effective training parameters."""
    report_path = output_dir / "training_report.json"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir.resolve()),
        "cli": {
            "preset": args.preset,
            "scenario": args.scenario,
            "steps_override": args.steps,
            "updates_override": args.updates,
            "resume": resume_path,
            "learning_rate_override": args.learning_rate,
            "entropy_coeff_override": args.entropy_coeff,
            "train_batch_size_override": args.train_batch_size,
            "reward_profile_override": args.reward_profile,
            "final_video": bool(args.final_video),
            "final_batch_analysis": bool(args.final_batch_analysis),
            "grid_size_override": args.grid_size,
            "max_steps_override": args.max_steps,
            "n_drones_override": args.n_drones,
            "reload_time_override": args.reload_time,
            "decision_interval_override": args.decision_interval,
            "splash_size_override": args.splash_size,
        },
        "constants": {
            "ROLLOUT_STEPS": int(ppo_config.rollout_steps),
            "TRAIN_BATCH_SIZE": TRAIN_BATCH_SIZE,
            "LEARNING_RATE": LEARNING_RATE,
            "PPO_EPOCHS": PPO_EPOCHS,
            "MINI_BATCH_SIZE": MINI_BATCH_SIZE,
            "ENTROPY_COEFF": ENTROPY_COEFF,
            "TOTAL_STEPS": TOTAL_STEPS,
            "UPDATES_PER_RUN": UPDATES_PER_RUN,
            "SEED_MODE": SEED_MODE,
            "SEED_REPEAT": SEED_REPEAT,
            "SEED_POOL_SIZE": SEED_POOL_SIZE,
            "LOG_EVERY_UPDATES": LOG_EVERY_UPDATES,
            "SAVE_EVERY_UPDATES": SAVE_EVERY_UPDATES,
            "VISUALIZE_PROGRESS_EVERY_UPDATES": VISUALIZE_PROGRESS_EVERY_UPDATES,
            "TERMINAL_UTILITY_START_WEIGHT": TERMINAL_UTILITY_START_WEIGHT,
            "TERMINAL_UTILITY_TARGET_WEIGHT": TERMINAL_UTILITY_TARGET_WEIGHT,
            "TERMINAL_UTILITY_ANNEAL_UPDATES": TERMINAL_UTILITY_ANNEAL_UPDATES,
        },
        "effective_training": {
            "device": str(device),
            "scenario": env.scenario_name,
            "seed_mode": env.seed_mode,
            "seed_repeat": env.seed_repeat,
            "seed_pool_size": len(env.seed_pool),
            "training_base_seed": int(training_base_seed),
            "training_master_seed": int(training_master_seed),
            "actor_obs_dim": env.actor_obs_dim,
            "critic_obs_dim": env.critic_obs_dim,
            "actor_feature_names": list(env.engine.swarm.perception.actor_features),
            "critic_feature_names": list(env.engine.swarm.perception.critic_features),
            "total_steps_target": int(total_steps),
            "steps_per_update": int(steps_per_update),
            "planned_updates": int(n_updates),
            "resume_start_update": int(start_update),
            "resume_start_steps": int(total_env_steps),
            "rollout_steps": int(ppo_config.rollout_steps),
            "ppo_epochs": int(ppo_config.ppo_epochs),
            "mini_batch_size": int(ppo_config.mini_batch_size),
            "learning_rate": float(ppo_config.learning_rate),
            "entropy_coeff": float(ppo_config.entropy_coeff),
            "reward_profile": str(getattr(env.reward_engine, "profile_name", "default")),
        },
        "simulation": {
            "batch_size": int(sim_cfg["simulation"]["batch_size"]),
            "max_steps": int(sim_cfg["simulation"]["max_steps"]),
            "grid_size": int(sim_cfg["simulation"]["grid_size"]),
            "physics_substeps": int(sim_cfg["simulation"]["physics_substeps"]),
            "seed": int(sim_cfg["simulation"]["seed"]),
        },
        "swarm": {
            "n_drones": int(sim_cfg["swarm"]["n_drones"]),
            "initial_in_air_fraction": float(
                sim_cfg.get("swarm", {}).get("initial_in_air_fraction", 0.0)
            ),
            "initial_force_go_decisions": int(
                sim_cfg.get("swarm", {}).get("initial_force_go_decisions", 0)
            ),
            "launch_commitment_decisions": int(
                sim_cfg.get("swarm", {}).get("launch_commitment_decisions", 0)
            ),
            "congestion_effects": dict(
                sim_cfg.get("swarm", {}).get("congestion_effects", {})
            ),
            "congestion_enabled": bool(
                sim_cfg.get("swarm", {}).get(
                    "congestion_effects", {}).get("enabled", False)
            ),
        },
        "scenario_deployment": _summarize_scenario_deployment(env),
        "rewards": (
            asdict(env.reward_engine.cfg)
            if is_dataclass(env.reward_engine.cfg)
            else dict(env.reward_engine.cfg.__dict__)
        ),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _run_final_batch_analysis(
    *,
    output_paths,
    preset: str,
    scenario: str,
    device: torch.device,
    final_model_path: Path,
) -> None:
    """
    Run an end-of-training comparison sweep and summarize it without plots.

    This reuses the same batch pipeline as `main.py -b`, but keeps training
    outputs plot-free. Generate paper plots separately with `main.py -a`.
    """
    eval_cfg = load_config(preset=preset)
    eval_cfg['simulation']['device'] = str(device)
    eval_cfg.setdefault('runtime', {})
    eval_cfg['runtime']['mappo_model_path'] = str(final_model_path.resolve())

    batch_cfg = eval_cfg.setdefault('batch_run', {})
    batch_cfg['strategies'] = list(FINAL_ANALYSIS_STRATEGIES)
    batch_cfg['scenarios'] = [str(scenario)]

    print(
        "📊 Final batch analysis: "
        f"strategies={batch_cfg['strategies']} scenarios={batch_cfg['scenarios']}"
    )
    runner = BatchRunner(eval_cfg, output_paths, default_preset=preset)
    runner.run_suite(batch_cfg['strategies'], batch_cfg['scenarios'])

    exp_analyzer = MultiExperimentAnalyzer(output_paths)
    if not exp_analyzer.analyze(generate_plots=False):
        print("⚠️ Final batch analysis did not generate summary outputs.")
        return


def _append_training_metrics_row(
    path: Path,
    fieldnames: list[str],
    row: dict,
) -> None:
    """Append one row to the per-update training metrics CSV (header written once)."""
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(line_buffering=True)
        except Exception:
            pass
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", "-p", type=str, default="training")
    parser.add_argument("--steps", "-s", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--scenario", type=str, default="baseline")
    parser.add_argument("--resume", "-r", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coeff", type=float, default=None)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument(
        "--reward-profile",
        type=str,
        choices=available_reward_profiles(),
        default=None,
    )
    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-drones", type=int, default=None)
    parser.add_argument("--reload-time", type=int, default=None)
    parser.add_argument("--decision-interval", type=int, default=None)
    parser.add_argument("--splash-size", type=int, default=None)
    parser.add_argument(
        "--final-video",
        action="store_true",
        help="After training, render a final MAPPO progress video.",
        default=True
    )
    parser.add_argument(
        "--final-batch-analysis",
        action="store_true",
        help="After training, run the end-of-run batch analysis sweep."
    )
    parser.add_argument(
        "--fresh-output-on-resume",
        action="store_true",
        help="Resume model/optimizer state from a checkpoint but write to a new training output directory.",
    )
    args = parser.parse_args()

    if args.resume and not Path(args.resume).exists():
        print("looking for default checkpoint at 'models/mappo_latest.pt'...")
        if Path("models/mappo_latest.pt").exists():
            args.resume = "models/mappo_latest.pt"
            print(f"Found checkpoint: {args.resume}")
        else:
            print("No checkpoint found. Starting fresh.")
            args.resume = None

    sim_cfg = load_config(preset=args.preset)
    sim_cfg['simulation']['batch_size'] = (
        int(args.train_batch_size)
        if args.train_batch_size is not None
        else int(TRAIN_BATCH_SIZE)
    )
    if args.reward_profile is not None:
        sim_cfg.setdefault('rl', {})
        sim_cfg['rl']['reward_profile'] = str(args.reward_profile)
    if args.grid_size is not None:
        sim_cfg['simulation']['grid_size'] = int(args.grid_size)
    if args.max_steps is not None:
        sim_cfg['simulation']['max_steps'] = int(args.max_steps)
    if args.n_drones is not None:
        sim_cfg['swarm']['n_drones'] = int(args.n_drones)
    if args.reload_time is not None:
        sim_cfg['swarm']['reload_time'] = int(args.reload_time)
    if args.decision_interval is not None:
        sim_cfg['swarm']['decision_interval'] = int(args.decision_interval)
    if args.splash_size is not None:
        sim_cfg['swarm']['splash_size'] = int(args.splash_size)
    splash_size = int(sim_cfg['swarm']['splash_size'])
    if splash_size <= 0 or splash_size % 2 == 0:
        parser.error(
            f"--splash-size must be a positive odd integer; got {splash_size}"
        )
    device = torch.device(sim_cfg['simulation']['device'])
    train_batch_size = int(sim_cfg['simulation']['batch_size'])
    base_master_seed = int(sim_cfg['simulation']['seed'])

    print(
        f"Initializing {train_batch_size} Parallel Environments on {device}...")

    resume_ckpt = None
    if args.resume and Path(args.resume).exists():
        resume_ckpt = torch.load(args.resume, map_location=device)

    resume_training_seed = None
    if isinstance(resume_ckpt, dict):
        raw_training_seed = resume_ckpt.get("training_master_seed")
        if raw_training_seed is not None:
            try:
                resume_training_seed = int(raw_training_seed)
            except (TypeError, ValueError):
                resume_training_seed = None

    if resume_training_seed is not None:
        training_master_seed = int(resume_training_seed)
        seed_source = "checkpoint"
    else:
        training_master_seed = _derive_training_master_seed(base_master_seed)
        seed_source = "derived"
    sim_cfg['simulation']['seed'] = int(training_master_seed)
    print(
        "Training seed stream: "
        f"base={base_master_seed} -> train={training_master_seed} "
        f"({seed_source})"
    )

    resume_output_dir = None
    if isinstance(resume_ckpt, dict):
        raw_output_dir = resume_ckpt.get("output_dir")
        if isinstance(raw_output_dir, str) and raw_output_dir.strip():
            candidate = Path(raw_output_dir).expanduser()
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            resume_output_dir = candidate

    if resume_output_dir is not None and not args.fresh_output_on_resume:
        output_paths = output_paths_from_root(resume_output_dir)
        print("Output directory source: checkpoint metadata")
        print(
            f"Resuming output directory from checkpoint: {output_paths.root}")
    else:
        output_paths = create_output_directory(suffix="training")
        if resume_output_dir is not None and args.fresh_output_on_resume:
            print("Output directory source: new training run (resume output overridden)")
        else:
            print("Output directory source: new training run")
    print(f"Training output: {output_paths.root}")

    env = WildfireEnv(
        sim_cfg,
        scenario_name=args.scenario,
        seed_mode=SEED_MODE,
        seed_repeat=SEED_REPEAT,
        seed_pool_size=SEED_POOL_SIZE
    )
    if isinstance(resume_ckpt, dict):
        raw_episode_idx = resume_ckpt.get("env_episode_idx")
        inferred_episode_idx = None
        if raw_episode_idx is None:
            raw_update = resume_ckpt.get("update")
            if raw_update is not None:
                try:
                    update_idx = int(raw_update)
                    inferred_episode_idx = 0 if update_idx < 0 else (
                        2 * (update_idx + 1))
                except (TypeError, ValueError):
                    inferred_episode_idx = None

        episode_idx_to_use = raw_episode_idx if raw_episode_idx is not None else inferred_episode_idx
        if episode_idx_to_use is not None:
            try:
                env.episode_idx = max(0, int(episode_idx_to_use))
                source = "checkpoint" if raw_episode_idx is not None else "inferred"
                print(
                    f"Resumed seed schedule episode index: {env.episode_idx} "
                    f"({source})"
                )
            except (TypeError, ValueError):
                print(
                    "⚠️ Checkpoint env_episode_idx is invalid; "
                    "seed schedule restarted from 0."
                )
    print(f"Training scenario: {env.scenario_name}")
    actor_feature_names = list(env.engine.swarm.perception.actor_features)
    critic_feature_names = list(env.engine.swarm.perception.critic_features)
    print(
        f"Actor features ({len(actor_feature_names)}): {actor_feature_names}")
    print(
        f"Critic features ({len(critic_feature_names)}): {critic_feature_names}")
    preflight = _summarize_scenario_deployment(env)
    print(
        "Deployment preflight: "
        f"initial_in_air_fraction={sim_cfg['swarm'].get('initial_in_air_fraction', 0)} "
        f"initial_force_go_decisions={sim_cfg['swarm'].get('initial_force_go_decisions', 0)} "
        f"launch_commitment_decisions={sim_cfg['swarm'].get('launch_commitment_decisions', 0)}"
    )
    print(
        "Scenario deployment: "
        f"reserved={preflight['reserved_drones']} "
        f"deploy_events={preflight['deploy_events']}"
    )
    if (
        int(sim_cfg['swarm'].get('initial_force_go_decisions', 0)) == 0 and
        float(sim_cfg['swarm'].get('initial_in_air_fraction', 0.0)) == 0.0 and
        any(
            (evt.get("step") == 1 and float(evt.get("percentage", 0.0) or 0.0) >= 1.0)
            for evt in preflight["deploy_events"]
        )
    ):
        print(
            "⚠️ Early-launch symmetry warning: this scenario releases all alive drones almost "
            "immediately with no forced initial deployment. Staggering must come from the learned policy."
        )
    congestion_cfg = dict(sim_cfg.get(
        "swarm", {}).get("congestion_effects", {}))
    print(
        f"Reward profile: {getattr(env.reward_engine, 'profile_name', 'default')}"
    )
    print(
        "Congestion effects: "
        f"enabled={bool(congestion_cfg.get('enabled', False))} "
        f"cfg={congestion_cfg}"
    )
    print(
        f"Seed schedule: mode={env.seed_mode}, repeat={env.seed_repeat}, "
        f"pool={len(env.seed_pool)}"
    )
    if env.seed_mode == "cycle":
        preview_count = min(5, len(env.seed_pool))
        preview = ", ".join(str(s) for s in env.seed_pool[:preview_count])
        print(f"Seed preview: [{preview}]")

    # CTDE: Separate observation dimensions for actor and critic
    actor_obs_dim = env.actor_obs_dim
    critic_obs_dim = env.critic_obs_dim

    rollout_steps = int(sim_cfg['simulation']['max_steps'])
    learning_rate = float(
        args.learning_rate) if args.learning_rate is not None else float(LEARNING_RATE)
    entropy_coeff = float(
        args.entropy_coeff) if args.entropy_coeff is not None else float(ENTROPY_COEFF)
    save_final_video = bool(args.final_video)
    run_final_batch_analysis = bool(args.final_batch_analysis)
    policy = SwarmPolicy(config={'rl': {
        'actor_obs_dim': actor_obs_dim,
        'critic_obs_dim': critic_obs_dim
    }}).to(device)

    ppo_config = PPOConfig(
        rollout_steps=rollout_steps,
        learning_rate=learning_rate,
        entropy_coeff=entropy_coeff,
        ppo_epochs=PPO_EPOCHS,
        mini_batch_size=MINI_BATCH_SIZE
    )
    trainer = PPOTrainer(policy, ppo_config, device)

    # Resume
    start_update = 0
    resumed_training = False
    total_env_steps = 0
    training_sessions_completed = 0
    if args.resume and Path(args.resume).exists():
        ckpt = resume_ckpt if resume_ckpt is not None else torch.load(
            args.resume, map_location=device
        )
        policy.load_state_dict(ckpt['policy'])
        if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
            trainer.optimizer.load_state_dict(ckpt['optimizer'])
        else:
            print("⚠️ Checkpoint has no optimizer state; optimizer reset.")
        start_update = ckpt.get('update', 0)
        total_env_steps = ckpt.get('steps', 0)
        training_sessions_completed = int(ckpt.get('training_sessions', 0))
        resumed_training = True
        print(f"📂 Resumed from Update {start_update}")

    # CTDE: Buffer needs both actor and critic obs dimensions
    buffer = RolloutBuffer(
        config=ppo_config,
        batch_size=env.batch_size,
        n_agents=env.n_drones,
        actor_obs_dim=actor_obs_dim,
        critic_obs_dim=critic_obs_dim,
        device=device
    )

    steps_per_update = ppo_config.rollout_steps * env.batch_size
    if args.steps is None:
        n_updates = int(args.updates) if args.updates is not None else int(
            UPDATES_PER_RUN)
        total_steps = int(total_env_steps + (n_updates * steps_per_update))
    else:
        total_steps = int(args.steps)
        remaining_steps = max(0, total_steps - int(total_env_steps))
        n_updates = remaining_steps // steps_per_update
    loop_start_update = int(start_update) + (1 if resumed_training else 0)
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    run_model_dir = output_paths.root / "models"
    run_model_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = run_model_dir / "checkpoints"
    archive_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting Training ({n_updates} updates)")
    print(
        f"Rollout Steps: {ppo_config.rollout_steps}, "
        f"PPO Epochs: {ppo_config.ppo_epochs}, "
        f"Mini Batch Size: {ppo_config.mini_batch_size}, "
    )
    print(
        "Update plan: "
        f"start={loop_start_update} count={n_updates} "
        f"end={max(loop_start_update + n_updates - 1, loop_start_update)}"
    )
    print(
        f"Scale: batch={train_batch_size} x n_drones={sim_cfg['swarm']['n_drones']} "
        f"x {ppo_config.rollout_steps} rollout steps/update on {device}. "
        "Each update can take several minutes at this scale (a per-rollout "
        "progress bar follows). For a quick smoke test, run e.g.: "
        "--n-drones 100 --train-batch-size 64 --max-steps 64 --updates 3"
    )
    report_path = _write_training_report(
        output_dir=output_paths.root,
        args=args,
        sim_cfg=sim_cfg,
        device=device,
        env=env,
        ppo_config=ppo_config,
        total_steps=total_steps,
        steps_per_update=steps_per_update,
        n_updates=n_updates,
        start_update=loop_start_update,
        total_env_steps=total_env_steps,
        resume_path=args.resume,
        training_base_seed=base_master_seed,
        training_master_seed=training_master_seed,
    )
    print(f"📝 Training report: {report_path}")

    if (not resumed_training) and n_updates > 0:
        torch.save(
            _checkpoint_payload(
                policy=policy,
                optimizer=trainer.optimizer,
                update=-1,
                steps=0,
                env_episode_idx=0,
                training_sessions=int(training_sessions_completed),
                training_base_seed=int(base_master_seed),
                training_master_seed=int(training_master_seed),
                output_dir=output_paths.root,
                checkpoint_stage="baseline_pre_training",
                metrics={"row_type": "baseline_pre_training"},
                actor_feature_names=actor_feature_names,
                critic_feature_names=critic_feature_names,
                reward_profile=str(
                    getattr(env.reward_engine, "profile_name", "default")),
            ),
            run_model_dir / "mappo_baseline.pt",
        )

    last_completed_update = int(start_update)
    last_update_metrics: dict[str, float] | None = None
    metrics_csv_path = output_paths.data / "training_metrics.csv"
    metrics_fieldnames = [
        "update", "total_env_steps", "reward_mean",
        "policy_loss", "value_loss", "entropy",
        "decision_sample_fraction", "terminal_utility_weight",
    ]
    for update in range(loop_start_update, loop_start_update + n_updates):
        terminal_weight = env.reward_engine.cfg.terminal_utility_weight

        last_value, env_steps = _collect_rollout(
            env=env,
            policy=policy,
            ppo_config=ppo_config,
            device=device,
            buffer=buffer,
            desc=f"Update {update:05d} | rollout",
        )
        total_env_steps += env_steps
        save_checkpoint_due = (update % SAVE_EVERY_UPDATES == 0)
        if save_checkpoint_due:
            archive_name = f"mappo_update_{int(update):05d}.pt"
            torch.save(
                _checkpoint_payload(
                    policy=policy,
                    optimizer=trainer.optimizer,
                    update=update,
                    steps=total_env_steps,
                    env_episode_idx=int(env.episode_idx),
                    training_sessions=int(training_sessions_completed),
                    training_base_seed=int(base_master_seed),
                    training_master_seed=int(training_master_seed),
                    output_dir=output_paths.root,
                    checkpoint_stage="periodic_pre_update",
                    actor_feature_names=actor_feature_names,
                    critic_feature_names=critic_feature_names,
                    reward_profile=str(
                        getattr(env.reward_engine, "profile_name", "default")),
                ),
                archive_dir / archive_name,
            )

        # Update
        stats = trainer.update(buffer, last_value)
        reward_mean = (
            float(buffer.rewards[:buffer.ptr].mean().detach().cpu())
            if buffer.ptr > 0
            else 0.0
        )
        last_update_metrics = {
            **{str(key): float(value) for key, value in stats.items()},
            "reward_mean": reward_mean,
            "terminal_utility_weight": float(terminal_weight),
            "total_env_steps": float(total_env_steps),
        }
        if update % LOG_EVERY_UPDATES == 0:
            print(
                "Update "
                f"{update:05d} | steps={total_env_steps} "
                f"| reward={reward_mean:.4f} "
                f"| policy_loss={last_update_metrics.get('policy_loss', 0.0):.4f} "
                f"| value_loss={last_update_metrics.get('value_loss', 0.0):.4f} "
                f"| entropy={last_update_metrics.get('entropy', 0.0):.4f} "
                f"| decision_frac={last_update_metrics.get('decision_sample_fraction', 0.0):.4f}",
                flush=True,
            )
        _append_training_metrics_row(metrics_csv_path, metrics_fieldnames, {
            "update": int(update),
            "total_env_steps": int(total_env_steps),
            "reward_mean": reward_mean,
            "policy_loss": last_update_metrics.get("policy_loss", 0.0),
            "value_loss": last_update_metrics.get("value_loss", 0.0),
            "entropy": last_update_metrics.get("entropy", 0.0),
            "decision_sample_fraction": last_update_metrics.get(
                "decision_sample_fraction", 0.0),
            "terminal_utility_weight": float(terminal_weight),
        })
        last_completed_update = int(update)

        # Save
        if save_checkpoint_due:
            latest_payload = _checkpoint_payload(
                policy=policy,
                optimizer=trainer.optimizer,
                update=update,
                steps=total_env_steps,
                env_episode_idx=int(env.episode_idx),
                training_sessions=int(training_sessions_completed),
                training_base_seed=int(base_master_seed),
                training_master_seed=int(training_master_seed),
                output_dir=output_paths.root,
                checkpoint_stage="periodic_post_update",
                metrics=last_update_metrics,
                actor_feature_names=actor_feature_names,
                critic_feature_names=critic_feature_names,
                reward_profile=str(
                    getattr(env.reward_engine, "profile_name", "default")),
            )
            torch.save(latest_payload, run_model_dir / "mappo_latest.pt")
            torch.save(latest_payload, save_dir / "mappo_latest.pt")
            print(f"💾 Checkpoint saved at Update {update}")

        if (
            VISUALIZE_PROGRESS_EVERY_UPDATES > 0 and
            update > 0 and
            update % VISUALIZE_PROGRESS_EVERY_UPDATES == 0
        ):
            print(
                "🎬 Training progress visualization "
                f"(update={update}, steps={total_env_steps})"
            )
            try:
                _run_progress_visualization_subprocess(
                    preset=args.preset,
                    scenario=args.scenario,
                    output_dir=output_paths.root,
                    update=update,
                    total_env_steps=total_env_steps,
                    seed=int(sim_cfg['simulation']['seed']),
                    model_selector="latest",
                )
            except subprocess.CalledProcessError as exc:
                print(f"⚠️ Progress visualization failed: {exc}")

    training_sessions = int(
        training_sessions_completed + (1 if n_updates > 0 else 0))
    final_ckpt = _checkpoint_payload(
        policy=policy,
        optimizer=trainer.optimizer,
        update=last_completed_update,
        steps=total_env_steps,
        env_episode_idx=int(env.episode_idx),
        training_sessions=training_sessions,
        training_base_seed=int(base_master_seed),
        training_master_seed=int(training_master_seed),
        output_dir=output_paths.root,
        checkpoint_stage="final_post_update",
        metrics=last_update_metrics,
        actor_feature_names=actor_feature_names,
        critic_feature_names=critic_feature_names,
        reward_profile=str(
            getattr(env.reward_engine, "profile_name", "default")),
    )
    final_archive = archive_dir / \
        f"mappo_final_update_{int(last_completed_update):05d}.pt"
    torch.save(final_ckpt, final_archive)
    torch.save(final_ckpt, run_model_dir / "mappo_final.pt")
    torch.save(final_ckpt, run_model_dir / "mappo_latest.pt")
    torch.save(final_ckpt, save_dir / "mappo_final.pt")
    torch.save(final_ckpt, save_dir / "mappo_latest.pt")

    if save_final_video and n_updates > 0:
        print(
            "🎬 Final visualization "
            f"(update={last_completed_update}, steps={total_env_steps})"
        )
        try:
            _run_progress_visualization_subprocess(
                preset=args.preset,
                scenario=args.scenario,
                output_dir=output_paths.root,
                update=last_completed_update,
                total_env_steps=total_env_steps,
                seed=int(sim_cfg['simulation']['seed']),
                model_selector="final",
            )
        except subprocess.CalledProcessError as exc:
            print(f"⚠️ Final visualization failed: {exc}")

    if run_final_batch_analysis and n_updates > 0:
        try:
            _run_final_batch_analysis(
                output_paths=output_paths,
                preset=args.preset,
                scenario=args.scenario,
                device=device,
                final_model_path=save_dir / "mappo_final.pt",
            )
        except Exception as exc:
            print(f"⚠️ Final batch analysis failed: {exc}")

    print(
        "💾 Final checkpoints: "
        f"update={last_completed_update} training_sessions={training_sessions} "
        f"archive_dir={archive_dir}"
    )
    print("✅ Training Complete.")


if __name__ == "__main__":
    main()
