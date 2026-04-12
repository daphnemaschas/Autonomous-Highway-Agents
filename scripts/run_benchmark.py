import argparse
import glob
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import DQN as SB3DQN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environment.shared_core_config import SHARED_CORE_CONFIG, SHARED_CORE_ENV_ID, make_env
from src.agents.dqn.dqn_model import QNetwork
from src.utils.evaluate import evaluate_policy_on_seeds


def _resolve_paths(csv_paths, pattern):
    if csv_paths:
        paths = [p.strip() for p in csv_paths.split(",") if p.strip()]
    else:
        paths = sorted(glob.glob(pattern))
    return [p for p in paths if os.path.exists(p)]


def _rolling_mean(values, window=100):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    if arr.size < window:
        return np.array([arr.mean()], dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(arr, kernel, mode="valid")


def _package_versions():
    pkgs = ["gymnasium", "highway-env", "torch", "stable-baselines3", "numpy", "pandas", "matplotlib"]
    versions = {}
    for pkg in pkgs:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def _load_dqn_checkpoint(model, model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    try:
        model.load_state_dict(checkpoint)
        return
    except RuntimeError:
        pass

    # Backward compatibility with older checkpoints using fc1/fc2/fc3 names.
    if all(k in checkpoint for k in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias", "fc3.weight", "fc3.bias"]):
        remapped = {
            "network.0.weight": checkpoint["fc1.weight"],
            "network.0.bias": checkpoint["fc1.bias"],
            "network.2.weight": checkpoint["fc2.weight"],
            "network.2.bias": checkpoint["fc2.bias"],
            "network.4.weight": checkpoint["fc3.weight"],
            "network.4.bias": checkpoint["fc3.bias"],
        }
        model.load_state_dict(remapped, strict=True)
        return

    raise RuntimeError(f"Unsupported DQN checkpoint format for: {model_path}")


def _evaluate_dqn_model(model_path, hidden_size, eval_seeds):
    env = make_env()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = QNetwork(obs_dim, action_dim, hidden_size=hidden_size).to(device)
    _load_dqn_checkpoint(model=model, model_path=model_path, device=device)
    model.eval()

    def policy(obs):
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = model(obs_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    results = evaluate_policy_on_seeds(env=env, policy_func=policy, eval_seeds=eval_seeds)
    env.close()
    return results


def _evaluate_sb3_model(model_path, eval_seeds):
    env = make_env()
    model = SB3DQN.load(model_path)

    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    results = evaluate_policy_on_seeds(env=env, policy_func=policy, eval_seeds=eval_seeds)
    env.close()
    return results


def _build_row(algorithm, model_path, eval_result, n_eval):
    rewards, lengths, mean_reward, std_reward, crash_rate, failure_seeds = eval_result
    return {
        "algorithm": algorithm,
        "model_id": os.path.basename(model_path),
        "model_path": model_path,
        "n_eval_episodes": n_eval,
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "mean_length": float(np.mean(lengths)),
        "std_length": float(np.std(lengths)),
        "crash_rate_pct": float(crash_rate),
        "n_crashes": int(len(failure_seeds)),
        "failure_seeds_preview": ",".join(str(s) for s in failure_seeds[:10]),
    }


def _plot_training_curves(dqn_reward_files, sb3_reward_files, output_path):
    plt.figure(figsize=(10, 6))

    for path in dqn_reward_files:
        rewards = np.load(path)
        curve = _rolling_mean(rewards, window=100)
        x = np.arange(len(curve)) + 100 if len(rewards) >= 100 else np.arange(len(curve))
        plt.plot(x, curve, alpha=0.9, linewidth=2, label=f"DQN {os.path.basename(path)}")

    for path in sb3_reward_files:
        rewards = np.load(path)
        curve = _rolling_mean(rewards, window=100)
        x = np.arange(len(curve)) + 100 if len(rewards) >= 100 else np.arange(len(curve))
        plt.plot(x, curve, alpha=0.6, linewidth=1.6, linestyle="--", label=f"SB3 {os.path.basename(path)}")

    plt.title("Training Curves (Rolling Mean Reward, window=100)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(fontsize=7)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _plot_benchmark_summary(summary_df, output_path):
    plt.figure(figsize=(7, 5))
    x = np.arange(len(summary_df))
    plt.bar(
        x,
        summary_df["mean_reward"].values,
        yerr=summary_df["std_reward"].values,
        capsize=6,
        alpha=0.85,
    )
    plt.xticks(x, summary_df["algorithm"].values)
    plt.ylabel("Mean Reward (50 shared eval seeds)")
    plt.title("Benchmark Summary by Algorithm")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def _write_markdown_summary(path, detailed_df, summary_df, metadata_dict):
    def dataframe_to_markdown(df):
        headers = list(df.columns)
        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in df.iterrows():
            vals = [str(row[h]) for h in headers]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    summary_fr = summary_df.rename(
        columns={
            "algorithm": "algorithme",
            "mean_reward": "reward_moyenne",
            "std_reward": "ecart_type_reward",
            "mean_crash_rate_pct": "crash_rate_moyen_pct",
            "n_models": "nb_modeles",
        }
    )
    detailed_fr = detailed_df.rename(
        columns={
            "algorithm": "algorithme",
            "model_id": "modele",
            "model_path": "chemin",
            "n_eval_episodes": "n_eval",
            "mean_reward": "reward_moyenne",
            "std_reward": "reward_std",
            "mean_length": "longueur_moyenne",
            "std_length": "longueur_std",
            "crash_rate_pct": "crash_rate_pct",
            "n_crashes": "nb_crash",
            "failure_seeds_preview": "seeds_echec_apercu",
        }
    )

    lines = []
    lines.append("# Resume Du Benchmark")
    lines.append("")
    lines.append("## Protocole")
    lines.append(f"- Environnement: `{metadata_dict['environment_id']}`")
    lines.append(f"- Seeds d'evaluation partagees: `{metadata_dict['eval_seeds'][0]}..{metadata_dict['eval_seeds'][-1]}`")
    lines.append(f"- Nombre d'episodes d'evaluation par modele: `{metadata_dict['num_eval_episodes']}`")
    lines.append(f"- Hash de configuration (reproductibilite): `{metadata_dict['shared_config_sha256']}`")
    lines.append("")
    lines.append("### Interpretation Du Protocole")
    lines.append("- Meme configuration pour tous les modeles: comparaison equitable.")
    lines.append("- Memes seeds d'evaluation: variance aleatoire controlee.")
    lines.append("- Hash fourni: preuve de configuration stable.")
    lines.append("")
    lines.append("## Visualisations")
    lines.append("")
    lines.append("### Comparaison Finale (Reward)")
    lines.append("![Comparaison benchmark](benchmark_comparison.png)")
    lines.append("- Lecture: compare la reward moyenne agregée entre methodes.")
    lines.append("")
    lines.append("### Courbes D'entrainement")
    lines.append("![Courbes entrainement](training_curves.png)")
    lines.append("- Lecture: dynamique de convergence et stabilite des runs.")
    lines.append("")
    lines.append("## Resultats Agreges")
    lines.append(dataframe_to_markdown(summary_fr))
    lines.append("")
    lines.append("### Interpretation Des Resultats Agreges")
    lines.append("- `reward_moyenne`: performance brute moyenne.")
    lines.append("- `crash_rate_moyen_pct`: indicateur securite (plus bas = mieux).")
    lines.append("- `ecart_type_reward`: variabilite entre modeles d'une meme methode.")
    lines.append("")
    lines.append("## Resultats Par Modele")
    lines.append(dataframe_to_markdown(detailed_fr))
    lines.append("")
    lines.append("### Interpretation Par Modele")
    lines.append("- Permet d'identifier le meilleur checkpoint et les cas instables.")
    lines.append("- `seeds_echec_apercu` liste des seeds a rejouer pour analyse d'echec.")
    lines.append("")
    lines.append("## Notes De Reproductibilite")
    lines.append("- Meme configuration environnementale pour toutes les evaluations.")
    lines.append("- Meme liste de seeds deterministes pour tous les modeles.")
    lines.append("- Metadonnees completes dans `benchmark_metadata.json`.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Run a fair benchmark between custom DQN and SB3 DQN.")
    parser.add_argument("--config", type=str, default="configs/dqn_params.yaml")
    parser.add_argument("--dqn-models", type=str, default="")
    parser.add_argument("--sb3-models", type=str, default="")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--seed-offset", type=int, default=100)
    parser.add_argument("--output-dir", type=str, default="results/benchmark")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    hidden_size = int(cfg["model"]["hidden_size"])

    dqn_models = _resolve_paths(args.dqn_models, "results/dqn/*_last.pth")
    sb3_models = _resolve_paths(args.sb3_models, "results/sb3/*_last.zip")
    if not dqn_models or not sb3_models:
        raise FileNotFoundError(
            "Missing models. Expected at least one custom DQN and one SB3 model. "
            "Use --dqn-models / --sb3-models or place files under results/dqn and results/sb3."
        )

    os.makedirs(args.output_dir, exist_ok=True)
    eval_seeds = [args.seed_offset + i for i in range(args.num_episodes)]

    rows = []
    for model_path in dqn_models:
        print(f"[DQN] Evaluating {model_path}")
        eval_result = _evaluate_dqn_model(model_path=model_path, hidden_size=hidden_size, eval_seeds=eval_seeds)
        rows.append(_build_row("custom_dqn", model_path, eval_result, args.num_episodes))

    for model_path in sb3_models:
        print(f"[SB3] Evaluating {model_path}")
        eval_result = _evaluate_sb3_model(model_path=model_path, eval_seeds=eval_seeds)
        rows.append(_build_row("sb3_dqn", model_path, eval_result, args.num_episodes))

    detailed_df = pd.DataFrame(rows).sort_values(["algorithm", "model_id"]).reset_index(drop=True)
    summary_df = (
        detailed_df.groupby("algorithm", as_index=False)
        .agg(
            mean_reward=("mean_reward", "mean"),
            std_reward=("mean_reward", "std"),
            mean_crash_rate_pct=("crash_rate_pct", "mean"),
            n_models=("model_id", "count"),
        )
        .fillna(0.0)
    )

    detailed_csv_path = os.path.join(args.output_dir, "benchmark_table.csv")
    summary_csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    md_path = os.path.join(args.output_dir, "benchmark_summary.md")
    metadata_path = os.path.join(args.output_dir, "benchmark_metadata.json")
    training_plot_path = os.path.join(args.output_dir, "training_curves.png")
    benchmark_plot_path = os.path.join(args.output_dir, "benchmark_comparison.png")

    detailed_df.to_csv(detailed_csv_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    metadata_dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _package_versions(),
        "environment_id": SHARED_CORE_ENV_ID,
        "shared_config_sha256": hashlib.sha256(
            json.dumps(SHARED_CORE_CONFIG, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "num_eval_episodes": args.num_episodes,
        "eval_seeds": eval_seeds,
        "config_file": args.config,
        "dqn_models": dqn_models,
        "sb3_models": sb3_models,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_dict, f, indent=2)

    dqn_reward_files = sorted(glob.glob("results/dqn/*_rewards.npy"))
    sb3_reward_files = sorted(glob.glob("results/sb3/*_rewards.npy"))
    if dqn_reward_files or sb3_reward_files:
        _plot_training_curves(dqn_reward_files, sb3_reward_files, training_plot_path)
    _plot_benchmark_summary(summary_df, benchmark_plot_path)

    _write_markdown_summary(md_path, detailed_df, summary_df, metadata_dict)

    print("\nBenchmark completed.")
    print(f"- Detailed table: {detailed_csv_path}")
    print(f"- Aggregate summary: {summary_csv_path}")
    print(f"- Markdown report: {md_path}")
    print(f"- Metadata: {metadata_path}")
    print(f"- Benchmark plot: {benchmark_plot_path}")
    if os.path.exists(training_plot_path):
        print(f"- Training curves plot: {training_plot_path}")


if __name__ == "__main__":
    main()
