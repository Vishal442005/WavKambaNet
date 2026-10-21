"""
Paper 1: Statistical Analysis Suite for BCI Decoding
===================================================
Provides rigorous statistical testing compliant with top-tier neuroengineering journals
(IEEE TNSRE, NeuroImage, JNE, NeurIPS):

1. Non-parametric Wilcoxon Signed-Rank Test (handles non-normal BCI subject distributions).
2. Paired Student's t-test with normality verification (Shapiro-Wilk).
3. Effect Size Estimation: Cohen's d (parametric) and Cliff's Delta (non-parametric).
4. Multiple Testing Correction: Bonferroni, Holm-Bonferroni, Benjamini-Hochberg (FDR).
5. Bayesian Evidence Factor (BF10) via JZS prior estimation.
6. Automated LaTeX Table Exporter for camera-ready manuscripts.
"""

from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import scipy.stats as stats
from dataclasses import dataclass


@dataclass
class StatisticalResult:
    test_name: str
    statistic: float
    p_value: float
    effect_size_name: str
    effect_size: float
    effect_magnitude: str
    normality_p1: float
    normality_p2: float
    is_normal: bool
    bayes_factor_10: Optional[float]
    significant_p05: bool
    significant_p01: bool
    significant_p001: bool

    def summary(self) -> str:
        sig = "***" if self.significant_p001 else "**" if self.significant_p01 else "*" if self.significant_p05 else "n.s."
        bf_str = f", BF10={self.bayes_factor_10:.2f}" if self.bayes_factor_10 is not None else ""
        return (
            f"[{self.test_name}] stat={self.statistic:.4f}, p={self.p_value:.4e} ({sig}), "
            f"{self.effect_size_name}={self.effect_size:.3f} ({self.effect_magnitude}){bf_str}"
        )


def compute_cliffs_delta(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """Computes non-parametric Cliff's delta effect size between paired samples."""
    diff = x - y
    n_pos = np.sum(diff > 0)
    n_neg = np.sum(diff < 0)
    n_total = len(diff)
    delta = (n_pos - n_neg) / n_total

    abs_d = abs(delta)
    if abs_d < 0.147:
        magnitude = "negligible"
    elif abs_d < 0.33:
        magnitude = "small"
    elif abs_d < 0.474:
        magnitude = "medium"
    else:
        magnitude = "large"
    return float(delta), magnitude


def compute_cohens_d(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
    """Computes paired Cohen's d effect size."""
    diff = x - y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0:
        return 0.0, "negligible"
    d = mean_diff / std_diff

    abs_d = abs(d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    return float(d), magnitude


def approximate_bayes_factor_paired(t_stat: float, n: int) -> float:
    """
    Approximates the JZS Bayes Factor (BF10) for a paired t-test using the BIC approximation.
    BF10 > 3 indicates substantial evidence for H1, BF10 > 10 strong evidence, BF10 > 100 decisive.
    """
    if n <= 1:
        return 1.0
    # BIC-based BF10 approximation (Wagenmakers, 2007)
    df = n - 1
    r2 = (t_stat ** 2) / (t_stat ** 2 + df)
    # Savage-Dickey / BIC approximation
    bic_diff = n * np.log(1 - r2) + np.log(n)
    bf10 = np.exp(-0.5 * bic_diff)
    return float(np.clip(bf10, 1e-6, 1e8))


def compare_models(
    scores_proposed: Union[List[float], np.ndarray],
    scores_baseline: Union[List[float], np.ndarray],
    alternative: str = "greater"
) -> StatisticalResult:
    """
    Performs full statistical comparison between proposed model and baseline across subjects.
    
    Args:
        scores_proposed: Per-subject accuracies of proposed model (e.g. Wav-KambaNet)
        scores_baseline: Per-subject accuracies of comparative baseline (e.g. EEGNet)
        alternative: 'greater' (proposed > baseline), 'two-sided', or 'less'
    """
    x = np.asarray(scores_proposed, dtype=np.float64)
    y = np.asarray(scores_baseline, dtype=np.float64)
    assert len(x) == len(y), f"Mismatched subject counts: {len(x)} vs {len(y)}"
    n = len(x)

    # 1. Normality test on differences (Shapiro-Wilk)
    diff = x - y
    if n >= 3:
        _, p_norm1 = stats.shapiro(x)
        _, p_norm2 = stats.shapiro(y)
        _, p_norm_diff = stats.shapiro(diff)
        is_normal = (p_norm_diff >= 0.05)
    else:
        p_norm1, p_norm2, p_norm_diff = 1.0, 1.0, 1.0
        is_normal = False

    # 2. Statistical Test Selection
    # If differences are normally distributed, use paired t-test; otherwise Wilcoxon signed-rank
    if is_normal and n >= 10:
        test_name = "Paired Student's t-test"
        res_t = stats.ttest_rel(x, y, alternative=alternative)
        statistic = float(res_t.statistic)
        p_val = float(res_t.pvalue)
        es_val, es_mag = compute_cohens_d(x, y)
        es_name = "Cohen's d"
        bf10 = approximate_bayes_factor_paired(statistic, n)
    else:
        test_name = "Wilcoxon Signed-Rank Test"
        # Handle zero differences
        nonzero_diff = diff[diff != 0]
        if len(nonzero_diff) == 0:
            statistic, p_val = 0.0, 1.0
        else:
            res_w = stats.wilcoxon(x, y, alternative=alternative)
            statistic = float(res_w.statistic)
            p_val = float(res_w.pvalue)
        es_val, es_mag = compute_cliffs_delta(x, y)
        es_name = "Cliff's Delta"
        # Compute paired t statistic for approximate BF
        t_stat = np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(n)) if np.std(diff) > 0 else 0.0
        bf10 = approximate_bayes_factor_paired(t_stat, n)

    return StatisticalResult(
        test_name=test_name,
        statistic=statistic,
        p_value=p_val,
        effect_size_name=es_name,
        effect_size=es_val,
        effect_magnitude=es_mag,
        normality_p1=float(p_norm1),
        normality_p2=float(p_norm2),
        is_normal=is_normal,
        bayes_factor_10=bf10,
        significant_p05=(p_val < 0.05),
        significant_p01=(p_val < 0.01),
        significant_p001=(p_val < 0.001)
    )


def apply_multiple_corrections(p_values: List[float], method: str = "holm") -> List[float]:
    """
    Applies multiple comparison p-value adjustments:
    - 'bonferroni': Standard family-wise error rate control
    - 'holm': Holm-Bonferroni step-down (more powerful than Bonferroni)
    - 'fdr_bh': Benjamini-Hochberg False Discovery Rate
    """
    p_arr = np.asarray(p_values, dtype=np.float64)
    m = len(p_arr)

    if method == "bonferroni":
        return list(np.clip(p_arr * m, 0.0, 1.0))

    elif method == "holm":
        order = np.argsort(p_arr)
        p_sorted = p_arr[order]
        adjusted_sorted = np.zeros(m)
        cum_max = 0.0
        for i, p in enumerate(p_sorted):
            adj = p * (m - i)
            cum_max = max(cum_max, adj)
            adjusted_sorted[i] = min(cum_max, 1.0)
        adjusted = np.zeros(m)
        adjusted[order] = adjusted_sorted
        return list(adjusted)

    elif method == "fdr_bh":
        order = np.argsort(p_arr)
        p_sorted = p_arr[order]
        adjusted_sorted = np.zeros(m)
        cum_min = 1.0
        for i in range(m - 1, -1, -1):
            adj = (p_sorted[i] * m) / (i + 1)
            cum_min = min(cum_min, adj)
            adjusted_sorted[i] = min(cum_min, 1.0)
        adjusted = np.zeros(m)
        adjusted[order] = adjusted_sorted
        return list(adjusted)

    else:
        raise ValueError(f"Unknown correction method: {method}")


def generate_latex_comparison_table(
    proposed_name: str,
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    dataset_name: str = "BCI Competition IV-2a"
) -> str:
    """
    Generates a publication-grade LaTeX table with bold markers for significance.
    results_dict format: {baseline_name: (proposed_scores, baseline_scores)}
    """
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{Statistical Significance Analysis on {dataset_name}: {proposed_name} vs. State-of-the-Art Baselines.}}",
        r"\label{tab:stat_analysis}",
        r"\begin{tabular}{lccccrc}",
        r"\toprule",
        rf"\textbf{{Baseline Model}} & \textbf{{Baseline Acc (\%)}} & \textbf{{{proposed_name} (\%)}} & \textbf{{Diff ($\Delta$)}} & \textbf{{Test Stat}} & \textbf{{$p$-value}} & \textbf{{Effect Size}} \\",
        r"\midrule"
    ]

    raw_p_values = []
    comparisons = []
    for base_name, (prop_scores, base_scores) in results_dict.items():
        res = compare_models(prop_scores, base_scores, alternative="greater")
        diff_mean = np.mean(prop_scores) - np.mean(base_scores)
        raw_p_values.append(res.p_value)
        comparisons.append((base_name, np.mean(base_scores), np.mean(prop_scores), diff_mean, res))

    adj_p_values = apply_multiple_corrections(raw_p_values, method="holm")

    for (base_name, b_mean, p_mean, diff_m, res), p_adj in zip(comparisons, adj_p_values):
        sig_star = r"$^{***}$" if p_adj < 0.001 else r"$^{**}$" if p_adj < 0.01 else r"$^{*}$" if p_adj < 0.05 else ""
        lines.append(
            f"{base_name} & {b_mean*100:.2f} $\\pm$ {np.std(results_dict[base_name][1])*100:.2f} & "
            f"\\textbf{{{p_mean*100:.2f} $\\pm$ {np.std(results_dict[base_name][0])*100:.2f}}} & "
            f"+{diff_m*100:.2f}\\% & {res.statistic:.2f} & {p_adj:.4f}{sig_star} & "
            f"{res.effect_size_name[0]}={res.effect_size:.2f} ({res.effect_magnitude}) \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{1mm}",
        r"\raggedright{\footnotesize * $p < 0.05$, ** $p < 0.01$, *** $p < 0.001$ (Holm-Bonferroni corrected).}",
        r"\end{table}"
    ])
    return "\n".join(lines)
