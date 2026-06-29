"""
Run from the project root:
    python results/generate_plots.py
"""

import json
import csv
import matplotlib

#from src import model
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


SCRIPT_DIR  = Path(__file__).parent.resolve()
ROOT        = SCRIPT_DIR.parent
OUTPUTS_DIR = ROOT / "outputs"
PLOTS_DIR   = SCRIPT_DIR / "plots"
METRICS_DIR = SCRIPT_DIR / "metrics"

PLOTS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

# ── Model name normalisation ───────────────────────────────────────────────────
# Translates raw HuggingFace model IDs (stored in metrics.json) to short labels.
# Add entries here whenever a new base model is introduced.
_MODEL_ID_MAP = {
    "bert-base-uncased":                "BERT",
    "allenai/scibert_scivocab_uncased": "SciBERT",
}

def _normalise_model(model_id: str, task: str | None) -> str:
    """Return a human-readable model label, appending '(Multitask)' for joint tasks."""
    base = _MODEL_ID_MAP.get(model_id, model_id)
    return f"{base} (Multitask)" if task == "joint" else base

MODEL_COLORS = {
    "BERT":                   "#FF5722",
    "SciBERT":                "#2196F3"
}

# ── Discovery ──────────────────────────────────────────────────────────────────
def discover_experiments():
    """
    Scans every subdirectory of OUTPUTS_DIR. A folder is included if it
    contains metrics/metrics.json. Labels (dataset, model) are read from
    that file, so no hardcoded experiment names are needed.

    Loads per experiment:
      - metrics/metrics.json             (required)
      - checkpoint-N/trainer_state.json  (highest checkpoint, if present)
      - cross_domain/metrics/metrics.json (if present)

    Returns a dict keyed by folder name.
    """
    experiments = {}

    if not OUTPUTS_DIR.exists():
        print(f"[WARN] outputs/ directory not found at {OUTPUTS_DIR}")
        return experiments

    for exp_dir in sorted(OUTPUTS_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue

        metrics_path = exp_dir / "metrics" / "metrics.json"
        if not metrics_path.exists():
            continue

        raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        # Derive human-readable labels from the metrics file itself.
        dataset = raw_metrics.get("dataset", exp_dir.name)
        task    = raw_metrics.get("task")           # "joint" for multitask, absent otherwise
        model   = _normalise_model(raw_metrics.get("model", exp_dir.name), task)

        info = {
            "name":    exp_dir.name,
            "path":    exp_dir,
            "dataset": dataset,
            "model":   model,
            "metrics": raw_metrics,
        }

        # Latest checkpoint's trainer_state.json
        checkpoints = sorted(
            [d for d in exp_dir.iterdir()
             if d.is_dir() and d.name.startswith("checkpoint-")],
            key=lambda d: int(d.name.split("-")[-1]),
        )
        for ckpt in reversed(checkpoints):
            state_path = ckpt / "trainer_state.json"
            if state_path.exists():
                info["trainer_state"] = json.loads(state_path.read_text(encoding="utf-8"))
                info["trainer_state_source"] = ckpt.name
                break

        # Cross-domain metrics (SOFT experiments)
        xd_path = exp_dir / "cross_domain" / "metrics" / "metrics.json"
        if xd_path.exists():
            info["cross_domain_metrics"] = json.loads(xd_path.read_text(encoding="utf-8"))

        experiments[exp_dir.name] = info

        state_src = info.get("trainer_state_source", "none")
        print(f"  {exp_dir.name:40s}  model={model:25s}  checkpoint={state_src}")

    return experiments


# ── Log-history helpers ────────────────────────────────────────────────────────
def split_log_history(trainer_state):
    """
    HuggingFace Trainer interleaves training and eval entries in log_history.
    Training entries contain 'loss'; eval entries contain 'eval_macro_f1'.
    """
    train_log, eval_log = [], []
    for entry in trainer_state.get("log_history", []):
        if "loss" in entry:
            train_log.append(entry)
        if "eval_macro_f1" in entry:
            eval_log.append(entry)
    return train_log, eval_log


# ── Shared style helpers ───────────────────────────────────────────────────────
def _annotate_bars(ax, bars, fmt=".3f"):
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.013,
                format(h, fmt),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )


def _finish_bar_ax(ax, title, ylabel, xtick_labels, ylim=(0, 1.08)):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    ax.set_xticks(np.arange(len(xtick_labels)))
    ax.set_xticklabels(xtick_labels, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ── Plot 1: Training curves ────────────────────────────────────────────────────
def plot_training_curves(experiments):
    """
    One figure per dataset with two subplots:
      Left  — training loss vs epoch
      Right — eval Macro-F1 vs epoch
    Source: trainer_state.json → log_history
    """
    by_dataset = {}
    for exp in experiments.values():
        if "trainer_state" not in exp:
            continue
        by_dataset.setdefault(exp["dataset"], []).append(exp)

    for dataset, exps in by_dataset.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Training Curves — {dataset}", fontsize=14, fontweight="bold", y=1.01)

        any_train, any_eval = False, False
        for exp in exps:
            train_log, eval_log = split_log_history(exp["trainer_state"])
            color = MODEL_COLORS.get(exp["model"], "grey")
            label = exp["model"]

            if train_log:
                epochs = [e["epoch"] for e in train_log]
                losses = [e["loss"] for e in train_log]
                axes[0].plot(epochs, losses, marker="o", linewidth=2, markersize=6,
                             label=label, color=color)
                any_train = True

            if eval_log:
                epochs = [e["epoch"] for e in eval_log]
                f1s    = [e["eval_macro_f1"] for e in eval_log]
                axes[1].plot(epochs, f1s, marker="o", linewidth=2, markersize=6,
                             label=label, color=color)
                any_eval = True

        for ax, title, ylabel, has_data in [
            (axes[0], "Training Loss vs Epoch",  "Cross-Entropy Loss", any_train),
            (axes[1], "Eval Macro-F1 vs Epoch",  "Macro-F1",           any_eval),
        ]:
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("Epoch", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            if has_data:
                ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.tight_layout()
        slug = dataset.lower().replace("-", "_").replace(" ", "_")
        fname = f"training_curves_{slug}.png"
        fig.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Shared metric resolver ─────────────────────────────────────────────────────
def _resolve_metrics(m: dict) -> tuple[float, float] | tuple[None, None]:
    """
    Returns (accuracy, macro_f1) from any metrics.json regardless of key names.
    Standard runs:   accuracy / macro_f1
    Multitask runs:  intent_accuracy / intent_macro_f1  (used as representative)
    Returns (None, None) if neither key set is present.
    """
    acc = m.get("accuracy") if "accuracy" in m else m.get("intent_accuracy")
    f1  = m.get("macro_f1") if "macro_f1"  in m else m.get("intent_macro_f1")
    if acc is None or f1 is None:
        return None, None
    return acc, f1


# ── Plot 2: BERT vs SciBERT per dataset ───────────────────────────────────────
def plot_model_comparison(experiments):
    """
    One figure per dataset comparing all models found for that dataset.
    Grouped bars: Accuracy and Macro-F1 per model.
    Works for both standard (accuracy/macro_f1) and multitask
    (intent_accuracy/intent_macro_f1) metrics.json schemas.
    """
    by_dataset = {}
    for exp in experiments.values():
        acc, f1 = _resolve_metrics(exp.get("metrics", {}))
        if acc is None:
            continue
        model = exp["model"]

        if "SciBERT" in model:
            model = "SciBERT"
        elif "BERT" in model:
            model = "BERT"

        by_dataset.setdefault(exp["dataset"], {})[model] = (acc, f1)

    for dataset, model_map in by_dataset.items():
        models   = list(model_map.keys())
        accuracy = [model_map[m][0] for m in models]
        macro_f1 = [model_map[m][1] for m in models]

        x     = np.arange(len(models))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))

        # Accuracy bars: model color + hatch to distinguish from F1
        acc_bars = ax.bar(
            x - width / 2, accuracy, width,
            color=[MODEL_COLORS.get(m, "grey") for m in models],
            hatch="//", edgecolor="white", linewidth=0.8, label="Accuracy",
        )
        # Macro-F1 bars: same model color, solid
        f1_bars = ax.bar(
            x + width / 2, macro_f1, width,
            color=[MODEL_COLORS.get(m, "grey") for m in models],
            edgecolor="white", linewidth=0.8, label="Macro-F1",
        )

        _annotate_bars(ax, acc_bars)
        _annotate_bars(ax, f1_bars)

        # Custom legend: one entry per model (color) + metric style
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=MODEL_COLORS.get(m, "grey"), label=m) for m in models
        ] + [
            Patch(facecolor="grey", hatch="//", edgecolor="white", label="Accuracy style"),
            Patch(facecolor="grey", edgecolor="white", label="Macro-F1 style"),
        ]
        ax.legend(handles=legend_handles, fontsize=8.5, ncol=2)

        _finish_bar_ax(ax, f"Model Comparison — {dataset}", "Score", models)

        plt.tight_layout()
        slug = dataset.lower().replace("-", "_").replace(" ", "_")
        fname = f"model_comparison_{slug}.png"
        fig.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Plot 3: Dataset comparison across all experiments ─────────────────────────
def plot_dataset_comparison(experiments):
    """
    Grouped bar chart with datasets on x-axis and one bar group per model.
    Two subplots: Accuracy (left) and Macro-F1 (right).
    Source: metrics/metrics.json
    """
    rows = []
    for exp in experiments.values():
        m = exp.get("metrics", {})
        if not m:
            continue
        # Resolve keys: standard models use accuracy/macro_f1;
        # SOFT multitask uses intent_accuracy/intent_macro_f1 as representative.
        acc = m.get("accuracy") if "accuracy" in m else m.get("intent_accuracy")
        f1  = m.get("macro_f1") if "macro_f1" in m else m.get("intent_macro_f1")
        if acc is None or f1 is None:
            continue
        model = exp["model"]

        # Normalize model names
        if "SciBERT" in model:
            model = "SciBERT"
        elif "BERT" in model:
            model = "BERT"
        rows.append({
            "dataset": exp["dataset"],
            "model":   model,
            "acc":     acc,
            "f1":      f1,
        })

    datasets = sorted(set(r["dataset"] for r in rows))
    models   = sorted(set(r["model"]   for r in rows))

    def lookup(ds, mdl, key):
        for r in rows:
            if r["dataset"] == ds and r["model"] == mdl:
                return r[key]
        return 0.0

    x       = np.arange(len(datasets))
    n       = len(models)
    bar_w   = 0.72 / n

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Dataset Comparison Across All Experiments",
                 fontsize=14, fontweight="bold", y=1.01)

    for i, model in enumerate(models):
        offset    = (i - n / 2 + 0.5) * bar_w
        color     = MODEL_COLORS.get(model, f"C{i}")
        acc_vals  = [lookup(ds, model, "acc") for ds in datasets]
        f1_vals   = [lookup(ds, model, "f1")  for ds in datasets]

        axes[0].bar(x + offset, acc_vals, bar_w, label=model,
                    color=color, alpha=0.88, edgecolor="white")
        axes[1].bar(x + offset, f1_vals,  bar_w, label=model,
                    color=color, alpha=0.88, edgecolor="white")

    for ax, title in [(axes[0], "Accuracy"), (axes[1], "Macro-F1")]:
        _finish_bar_ax(ax, title, title, datasets)

    plt.tight_layout()
    fname = "dataset_comparison.png"
    fig.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Plot 4: SOFT multitask breakdown ──────────────────────────────────────────
def plot_soft_breakdown(experiments):
    """
    One subplot per SOFT experiment (all models discovered automatically).
    Each subplot shows in-domain vs cross-domain bars for:
      intent_accuracy, intent_macro_f1, content_accuracy, content_macro_f1,
      joint_accuracy
    A single figure is saved as soft_multitask_breakdown.png.
    """
    soft_exps = [e for e in experiments.values() if e["dataset"] == "SOFT"]
    if not soft_exps:
        print("  [SKIP] no SOFT experiments found")
        return

    metric_labels = [
        "Intent\nAccuracy",
        "Intent\nMacro-F1",
        "Content\nAccuracy",
        "Content\nMacro-F1",
        "Joint\nAccuracy",
    ]
    metric_keys = [
        "intent_accuracy",
        "intent_macro_f1",
        "content_accuracy",
        "content_macro_f1",
        "joint_accuracy",
    ]

    def extract(m):
        return [m.get(k, 0) for k in metric_keys]

    n_exps  = len(soft_exps)
    fig, axes = plt.subplots(1, n_exps, figsize=(11 * n_exps, 6), squeeze=False)
    fig.suptitle("SOFT — In-Domain vs Cross-Domain",
                 fontsize=14, fontweight="bold", y=1.02)

    x     = np.arange(len(metric_labels))
    width = 0.35

    for ax, exp in zip(axes[0], soft_exps):
        in_vals = extract(exp.get("metrics", {}))
        xd_vals = extract(exp.get("cross_domain_metrics", {}))
        IN_DOMAIN_COLOR = "#2196F3"   # Blue
        CROSS_DOMAIN_COLOR = "#FF5722" # Dark Orange

        bars_in = ax.bar(
            x - width / 2,
            in_vals,
            width,
            label="In-Domain",
            color=IN_DOMAIN_COLOR,
            alpha=0.88,
            edgecolor="white",
        )

        bars_xd = ax.bar(
            x + width / 2,
            xd_vals,
            width,
            label="Cross-Domain",
            color=CROSS_DOMAIN_COLOR,
            alpha=0.88,
            edgecolor="white",
        )

        _annotate_bars(ax, bars_in)
        _annotate_bars(ax, bars_xd)

        model = exp["model"]

        if "SciBERT" in model:
            model = "SciBERT"
        elif "BERT" in model:
            model = "BERT"

        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fname = "soft_multitask_breakdown.png"
    fig.savefig(PLOTS_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


# ── Aggregate metrics summary ──────────────────────────────────────────────────
def save_summary(experiments):
    """
    Writes results/metrics/summary.json and summary.csv aggregating
    all test-set metrics and best training-time metrics.
    """
    summary = []
    for exp in experiments.values():
        m = exp.get("metrics", {})
        row = {
            "experiment":        exp["name"],
            "dataset":           exp["dataset"],
            "model":             exp["model"],
            "test_accuracy":     m.get("accuracy") or m.get("intent_accuracy"),
            "test_macro_f1":     m.get("macro_f1") or m.get("intent_macro_f1"),
        }
        if exp.get("trainer_state"):
            row["train_steps"]   = exp["trainer_state"].get("global_step")
            row["best_eval_f1"]  = exp["trainer_state"].get("best_metric")
            row["checkpoint"]    = exp.get("trainer_state_source")
        if "cross_domain_metrics" in exp:
            xd = exp["cross_domain_metrics"]
            row["xd_intent_f1"]  = xd.get("intent_macro_f1")
            row["xd_content_f1"] = xd.get("content_macro_f1")
        summary.append(row)

    (METRICS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if summary:
        keys = list(summary[0].keys())
        with open(METRICS_DIR / "summary.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(summary)

    print(f"  Saved summary.json and summary.csv  ({len(summary)} rows)")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Project root : {ROOT}")
    print(f"Outputs dir  : {OUTPUTS_DIR}")
    print(f"Plots dir    : {PLOTS_DIR}")
    print(f"Metrics dir  : {METRICS_DIR}\n")

    print("Discovering experiments...")
    experiments = discover_experiments()
    print(f"\nFound {len(experiments)} experiment(s).\n")

    if not experiments:
        print("No experiments found. Ensure outputs/ exists under the project root.")
        raise SystemExit(1)

    print("--- Plot 1: Training curves ---")
    plot_training_curves(experiments)

    print("\n--- Plot 2: BERT vs SciBERT model comparison ---")
    plot_model_comparison(experiments)

    print("\n--- Plot 3: Dataset comparison ---")
    plot_dataset_comparison(experiments)

    print("\n--- Plot 4: SOFT multitask breakdown ---")
    plot_soft_breakdown(experiments)

    print("\n--- Aggregating metrics summary ---")
    save_summary(experiments)

    print(f"\nDone.\n  Plots   -> {PLOTS_DIR}\n  Metrics -> {METRICS_DIR}")
