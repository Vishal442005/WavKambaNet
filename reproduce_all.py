"""
Paper 1: Master Reproducibility Script
======================================
Executes the complete experimental and analysis pipeline for Wav-KambaNet in a single command:
1. Generates Genuine Neurophysiological Explainability (XAI) figures from learned checkpoints:
   - Morlet wavelet Fourier frequency response
   - Spatial Chebyshev graph topographic scalp maps
   - Time-frequency ERD/ERS dynamics
2. Generates all IEEE Publication Figures and Tables:
   - Figure 1: End-to-end 4-stage tripartite architecture diagram
   - Figure 2: Multiclass diagnostic confusion matrix (4 MI classes)
   - Figure 3: Granular per-subject decoding breakdown (S1 to S10 vs. baselines)
   - Figure 4: Epoch-by-epoch loss & accuracy convergence curves
   - Table: Cross-subject (LOSO) transferability LaTeX table
3. Systematic Architectural Ablation Matrix & LaTeX table
4. Hyperparameter Sensitivity Analysis across 5 dimensions

Usage:
    python reproduce_all.py
"""

import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Paper1_ReproduceAll")

PROJECT_ROOT = Path(__file__).parent.resolve()

def run_step(step_name: str, script_name: str):
    logger.info("=" * 70)
    logger.info("Executing Step: %s (%s)", step_name, script_name)
    logger.info("=" * 70)
    script_path = PROJECT_ROOT / script_name
    result = subprocess.run([sys.executable, str(script_path)], cwd=str(PROJECT_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Step failed: %s\nStderr:\n%s", step_name, result.stderr)
        sys.exit(result.returncode)
    else:
        logger.info("Step completed successfully!\n%s", result.stdout.strip())

def main():
    logger.info("Starting Paper 1 (Wav-KambaNet) End-to-End Master Reproducibility Pipeline...")
    
    # 1. Genuine Neurophysiological Explainability Suite (XAI)
    run_step("1. Genuine Neurophysiological XAI Extraction", "scripts/generate_genuine_xai.py")
    
    # 2. Master Publication Assets (Architecture, Confusion Matrix, Per-Subject, Curves, LOSO)
    run_step("2. Master Publication Figures & Tables Generator", "scripts/generate_publication_assets.py")
    
    # 3. Systematic Ablation Study
    run_step("3. Systematic Architectural Ablation Study", "scripts/run_ablations.py")
    
    # 4. Hyperparameter Sensitivity Sweeps
    run_step("4. Hyperparameter Sensitivity Analysis", "scripts/run_sensitivity.py")
    
    logger.info("=" * 70)
    logger.info("All Paper 1 Experimental Analyses Successfully Reproduced!")
    logger.info("Results stored under: %s", PROJECT_ROOT / "results")
    logger.info("Manuscript ready under: %s", PROJECT_ROOT / "manuscript")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
