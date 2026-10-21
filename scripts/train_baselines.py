"""
Paper 1: Genuine Empirical Baseline Training & Statistical Evaluation Suite
=============================================================================
Trains NativeEEGNet and NativeShallowConvNet on the identical 10 PhysioNet subjects
(S001 to S010) and 5-fold cross-validation splits used by Wav-KambaNet.

Produces:
1. results/genuine_baseline_comparison_results.csv
2. results/baseline_statistical_table.tex (Formal camera-ready LaTeX table)
3. results/baseline_comparison_plot.png (Publication-grade boxplot)
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.physionet_loader import PhysioNetMIDataset
from src.data.dataset import EEGDataModule
from src.models.baselines import NativeEEGNet, NativeShallowConvNet
from src.analysis.statistical_tests import (
    compare_models,
    compute_cohens_d,
    compute_cliffs_delta,
    approximate_bayes_factor_paired,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainBaselines")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def train_and_eval_model(model_cls, X, y, n_classes, n_timepoints, n_epochs=30, batch_size=8, seed=42):
    """5-Fold CV evaluation of a baseline architecture on a single subject."""
    dm = EEGDataModule(X=X, y=y, batch_size=batch_size, n_folds=5, pin_memory=False, seed=seed)
    fold_accs = []

    for fold_idx in range(5):
        dm.set_fold(fold_idx)
        torch.manual_seed(seed + fold_idx)
        
        model = model_cls(n_channels=64, n_classes=n_classes, n_timepoints=n_timepoints).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(n_epochs):
            model.train()
            for bx, by in dm.train_dataloader():
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

        # Evaluate on test fold
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in dm.test_dataloader():
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                preds = model(bx).argmax(dim=-1)
                correct += (preds == by).sum().item()
                total += len(by)
                
        acc = correct / max(1, total)
        fold_accs.append(acc)

    return np.mean(fold_accs), np.std(fold_accs), fold_accs


def main():
    logger.info("=" * 75)
    logger.info("Starting Empirical Baseline Training across 10 PhysioNet Subjects")
    logger.info(f"Target Device: {DEVICE}")
    logger.info("=" * 75)

    data_dir = PROJECT_ROOT.parent / "Motor"
    subjects = list(range(1, 11))
    dataset = PhysioNetMIDataset(root_dir=str(data_dir), subjects=subjects)

    # Load Wav-KambaNet empirical scores from genuine cohort CSV
    wavkamba_csv = RESULTS_DIR / "genuine_physionet_cohort_results.csv"
    if not wavkamba_csv.exists():
        raise FileNotFoundError(f"Missing {wavkamba_csv}")
    df_wav = pd.read_csv(wavkamba_csv)
    wavkamba_scores = df_wav["Mean_Acc"].values # (10,)

    eegnet_subj_means = []
    shallow_subj_means = []
    eegnet_records = []
    shallow_records = []

    t0 = time.time()
    for s_idx in subjects:
        subj_str = f"S{s_idx:03d}"
        logger.info(f"--- Training Baselines on Subject {subj_str} (5 Folds) ---")
        s_data = dataset.load_subject(subj_str)
        X, y = s_data["X"], s_data["y"]
        n_classes = len(np.unique(y))
        n_timepoints = X.shape[2]

        # 1. NativeEEGNet
        eeg_mean, eeg_std, eeg_folds = train_and_eval_model(
            NativeEEGNet, X, y, n_classes, n_timepoints, n_epochs=35, batch_size=8, seed=42
        )
        eegnet_subj_means.append(eeg_mean)
        logger.info(f"Subject {subj_str} NativeEEGNet: {eeg_mean*100:.2f}% ± {eeg_std*100:.2f}%")

        # 2. NativeShallowConvNet
        shal_mean, shal_std, shal_folds = train_and_eval_model(
            NativeShallowConvNet, X, y, n_classes, n_timepoints, n_epochs=35, batch_size=8, seed=42
        )
        shallow_subj_means.append(shal_mean)
        logger.info(f"Subject {subj_str} NativeShallowConvNet: {shal_mean*100:.2f}% ± {shal_std*100:.2f}%")

        eegnet_records.append({"Subject": subj_str, "Mean_Acc": eeg_mean, "Std_Acc": eeg_std, "Folds": eeg_folds})
        shallow_records.append({"Subject": subj_str, "Mean_Acc": shal_mean, "Std_Acc": shal_std, "Folds": shal_folds})

    elapsed = time.time() - t0
    logger.info(f"All 10 subjects baseline training completed in {elapsed:.1f} seconds!")

    eegnet_scores = np.array(eegnet_subj_means)
    shallow_scores = np.array(shallow_subj_means)
    chance_scores = np.full(10, 0.25) # 4-class chance

    # Save detailed CSV
    rows = []
    for i, s_idx in enumerate(subjects):
        rows.append({
            "Subject": f"S{s_idx:03d}",
            "WavKambaNet": round(wavkamba_scores[i] * 100.0, 2),
            "EEGNet": round(eegnet_scores[i] * 100.0, 2),
            "ShallowConvNet": round(shallow_scores[i] * 100.0, 2),
            "ChanceLevel": 25.00
        })
    df_cohort = pd.DataFrame(rows)
    df_cohort.to_csv(RESULTS_DIR / "genuine_baseline_comparison_results.csv", index=False)
    logger.info(f"Saved genuine baseline CSV to {RESULTS_DIR / 'genuine_baseline_comparison_results.csv'}")

    # Statistical battery comparing Wav-KambaNet against baselines
    baselines_dict = {
        "Theoretical Chance": chance_scores,
        "EEGNet-8,2 (2018)": eegnet_scores,
        "ShallowConvNet (2017)": shallow_scores,
    }

    stat_rows = []
    for b_name, b_scores in baselines_dict.items():
        diff = (wavkamba_scores - b_scores) * 100.0
        mean_diff = np.mean(diff)
        
        res = compare_models(wavkamba_scores, b_scores, alternative="greater")
        p_val = res.p_value
        d_val = res.effect_size
        bf10 = res.bayes_factor_10 if res.bayes_factor_10 is not None else 1.0
        
        # Cohen's kappa for Wav-KambaNet vs chance
        # kappa = (acc - p_e) / (1 - p_e)
        kappa_wav = (np.mean(wavkamba_scores) - 0.25) / (1.0 - 0.25)
        kappa_b = (np.mean(b_scores) - 0.25) / (1.0 - 0.25)

        stat_rows.append({
            "Baseline": b_name,
            "Baseline_Acc": f"{np.mean(b_scores)*100:.2f} $\\pm$ {np.std(b_scores)*100:.2f}",
            "WavKamba_Acc": f"{np.mean(wavkamba_scores)*100:.2f} $\\pm$ {np.std(wavkamba_scores)*100:.2f}",
            "Gain": f"{mean_diff:+.2f}\\%",
            "p_val": p_val,
            "Cohens_d": d_val,
            "BF10": bf10,
            "Kappa_Base": kappa_b,
            "Kappa_Wav": kappa_wav,
        })

    # Holm-Bonferroni correction
    p_vals = [r["p_val"] for r in stat_rows]
    sorted_indices = np.argsort(p_vals)
    m = len(p_vals)
    for rank, idx in enumerate(sorted_indices):
        p_hb = min(1.0, p_vals[idx] * (m - rank))
        stat_rows[idx]["p_HB"] = p_hb

    # Build LaTeX table
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Empirical Performance Comparison of Wav-KambaNet against Baseline Architectures on Held-Out PhysioNet Subjects ($N=10$, 5-Fold CV). ($^{***}: p < 0.001$, $^{**}: p < 0.01$, $^{*}: p < 0.05$)}",
        r"\label{tab:baseline_comparison}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\textbf{Model / Architecture} & \textbf{Cohort Acc (\%)} & \textbf{Cohen's $\kappa$} & \textbf{Gain vs. Base ($\Delta$)} & \textbf{$p_{\text{HB}}$} & \textbf{Cohen's $d$} & \textbf{$BF_{10}$} \\",
        r"\midrule",
        f"\\textbf{{Wav-KambaNet (Proposed)}} & \\textbf{{{np.mean(wavkamba_scores)*100:.2f} $\\pm$ {np.std(wavkamba_scores)*100:.2f}}} & \\textbf{{{kappa_wav:.3f}}} & --- & --- & --- & --- \\\\",
        r"\midrule"
    ]

    for r in stat_rows:
        sig = "$^{***}$" if r["p_HB"] < 0.001 else "$^{**}$" if r["p_HB"] < 0.01 else "$^{*}$" if r["p_HB"] < 0.05 else ""
        bf_str = f"{r['BF10']:.1f}" if r['BF10'] < 1e4 else f"{r['BF10']:.1e}"
        latex_lines.append(
            f"{r['Baseline']} & {r['Baseline_Acc']} & {r['Kappa_Base']:.3f} & {r['Gain']} & {r['p_HB']:.4f}{sig} & {r['Cohens_d']:.2f} & {bf_str} \\\\"
        )

    latex_lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ]

    with open(RESULTS_DIR / "baseline_statistical_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines))
    logger.info(f"Saved LaTeX statistical table to {RESULTS_DIR / 'baseline_statistical_table.tex'}")

    # Generate Publication-Grade Boxplot
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)
    data_to_plot = [
        chance_scores * 100.0,
        eegnet_scores * 100.0,
        shallow_scores * 100.0,
        wavkamba_scores * 100.0,
    ]
    labels = ["Theoretical\nChance", "Native\nEEGNet", "Native\nShallowNet", "Wav-KambaNet\n(Proposed)"]
    colors = ["#9ca3af", "#60a5fa", "#a78bfa", "#10b981"]

    bplot = ax.boxplot(
        data_to_plot,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", linewidth=1.5),
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white"),
        showmeans=True
    )

    for patch, color in zip(bplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    # Overlay strip plot points
    for idx, d in enumerate(data_to_plot):
        jitter = np.random.normal(0, 0.04, size=len(d))
        ax.scatter(np.full_like(d, idx + 1) + jitter, d, color="black", alpha=0.6, s=25, zorder=3)

    ax.set_ylabel("5-Fold Test Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Empirical Multiclass MI Accuracy: Wav-KambaNet vs. Deep Learning Baselines ($N=10$)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.5, axis="y")
    plt.tight_layout()

    out_plot = RESULTS_DIR / "baseline_comparison_plot.png"
    plt.savefig(out_plot, dpi=300)
    plt.close()
    logger.info(f"Saved updated genuine baseline comparison plot to {out_plot}")


if __name__ == "__main__":
    main()
