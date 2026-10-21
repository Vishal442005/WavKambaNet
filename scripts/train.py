"""
Wav-KambaNet Training Script
=============================
Main entry point for training EEG Motor Imagery classifiers.

Usage:
    python train.py                          # Train Wav-KambaNet with default config
    python train.py model=eegnet             # Train EEGNet baseline
    python train.py evaluation=cross_subject # Cross-subject LOSO evaluation
    python train.py training.lr=0.0005       # Override learning rate
"""

import os
import sys
import logging
from pathlib import Path

# Fix Windows console encoding for rich unicode
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from omegaconf import OmegaConf, DictConfig
from rich.console import Console
from rich.table import Table

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.physionet_loader import PhysioNetMIDataset
from src.data.dataset import EEGDataModule

console = Console()

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str = None) -> DictConfig:
    """Load and merge configuration."""
    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "config.yaml")
    cfg = OmegaConf.load(config_path)
    
    # Override with command line args
    cli_conf = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, cli_conf)
    
    return cfg


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pl.seed_everything(seed, workers=True)


# ─────────────────────────────────────────────────────────────
# Model Factory
# ─────────────────────────────────────────────────────────────

def create_model(cfg: DictConfig, n_channels: int, n_timepoints: int, n_classes: int):
    """Instantiate the specified model architecture."""
    model_name = cfg.model.name
    
    if model_name == "wav_kambanet":
        from src.models.wav_kambanet import WavKambaNet
        model = WavKambaNet(
            num_channels=n_channels,
            timepoints=n_timepoints,
            num_classes=n_classes,
            downsample_factor=cfg.model.get("downsample_factor", 1),
            # Spatial Graph
            spatial_hidden_dim=cfg.model.spatial.hidden_dim,
            cheb_k=cfg.model.spatial.cheb_k,
            spatial_dropout=cfg.model.spatial.dropout,
            # Wav-KAN
            use_wav_kan=cfg.model.wav_kan.get("enabled", True),
            wavelet_type=cfg.model.wav_kan.wavelet_type,
            num_wavelets=cfg.model.wav_kan.num_wavelets,
            # Mamba
            d_model=cfg.model.mamba.d_model,
            d_state=cfg.model.mamba.d_state,
            d_conv=cfg.model.mamba.d_conv,
            expand_factor=cfg.model.mamba.expand_factor,
            num_mamba_layers=cfg.model.mamba.num_layers,
            bidirectional=cfg.model.mamba.bidirectional,
            mamba_dropout=cfg.model.mamba.dropout,
            use_wavkan_in_mamba=cfg.model.mamba.use_wavkan_projection,
            # Classifier
            classifier_type=cfg.model.classifier.type,
        )
    elif model_name == "eegnet":
        from src.models.baselines import get_eegnet
        model = get_eegnet(n_channels, n_timepoints, n_classes)
    elif model_name == "shallow_convnet":
        from src.models.baselines import get_shallow_convnet
        model = get_shallow_convnet(n_channels, n_timepoints, n_classes)
    elif model_name == "deep_convnet":
        from src.models.baselines import get_deep_convnet
        model = get_deep_convnet(n_channels, n_timepoints, n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


# ─────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────

def train_within_subject(cfg: DictConfig):
    """Within-subject k-fold cross-validation training."""
    console.rule("[bold blue]Within-Subject Evaluation[/bold blue]")
    
    # Load data
    dataset = PhysioNetMIDataset(
        root_dir=str(PROJECT_ROOT / cfg.data.physionet_dir),
        subjects=cfg.physionet.get("subjects", None),
        exclude_subjects=cfg.physionet.exclude_subjects,
    )
    
    all_results = {}
    subjects = dataset.subject_ids
    
    for subj_id in subjects:
        console.print(f"\n[cyan]━━━ Subject {subj_id} ━━━[/cyan]")
        
        # Load subject data
        subj_data = dataset.load_subject(subj_id)
        X, y = subj_data["X"], subj_data["y"]
        
        n_channels = X.shape[1]
        n_timepoints = X.shape[2]
        n_classes = len(np.unique(y))
        
        fold_accs = []
        
        # K-fold CV
        data_module = EEGDataModule(
            X=X, y=y,
            batch_size=cfg.training.batch_size,
            n_folds=cfg.evaluation.within_subject.cv_folds,
            pin_memory=False,
        )
        
        cache_dir = PROJECT_ROOT / "results"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"subj{subj_id}_cv_cache.json"
        cache_data = {}
        import json
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
            except Exception:
                cache_data = {}

        for fold_idx in range(cfg.evaluation.within_subject.cv_folds):
            if str(fold_idx) in cache_data:
                fold_acc = float(cache_data[str(fold_idx)])
                fold_accs.append(fold_acc)
                console.print(
                    f"  Fold {fold_idx + 1}/{cfg.evaluation.within_subject.cv_folds} (Cached): "
                    f"Accuracy = [green]{fold_acc:.4f}[/green]"
                )
                continue

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

            data_module.set_fold(fold_idx)
            
            # Create model
            model = create_model(cfg, n_channels, n_timepoints, n_classes)
            
            # Create Lightning module
            from src.models.lightning_module import EEGClassifier
            lit_model = EEGClassifier(
                model=model,
                lr=cfg.training.optimizer.lr,
                weight_decay=cfg.training.optimizer.weight_decay,
                n_classes=n_classes,
                label_smoothing=cfg.training.loss.label_smoothing,
            )
            
            # Callbacks
            ckpt_dir = PROJECT_ROOT / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            callbacks = [
                EarlyStopping(
                    monitor="val/acc", mode="max",
                    patience=cfg.training.early_stopping_patience
                ),
                ModelCheckpoint(
                    dirpath=str(ckpt_dir),
                    monitor="val/acc", mode="max",
                    filename=f"subj{subj_id}-fold{fold_idx}-{{val/acc:.4f}}",
                    save_top_k=1,
                ),
                LearningRateMonitor(logging_interval="epoch"),
            ]
            
            # Logger
            logger = CSVLogger(
                save_dir="logs",
                name=f"{cfg.model.name}/subj{subj_id}",
                version=f"fold{fold_idx}",
            )
            
            accelerator = cfg.training.get("accelerator", "auto")
            precision = "32-true" if accelerator == "cpu" else ("16-mixed" if torch.cuda.is_available() else "32-true")
            trainer = pl.Trainer(
                max_epochs=cfg.training.epochs,
                accelerator=accelerator,
                devices=1,
                precision=precision,
                callbacks=callbacks,
                logger=logger,
                gradient_clip_val=cfg.training.gradient_clip_max_norm,
                enable_progress_bar=True,
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                deterministic=True,
            )
            
            # Train
            trainer.fit(lit_model, data_module)
            
            # Test
            best_ckpt = trainer.checkpoint_callback.best_model_path if (trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path) else None
            result = trainer.test(lit_model, data_module, ckpt_path=best_ckpt)
            fold_acc = result[0].get("test/acc", result[0].get("test_acc", 0.0))
            fold_accs.append(fold_acc)
            cache_data[str(fold_idx)] = float(fold_acc)
            with open(cache_file, "w") as f:
                json.dump(cache_data, f, indent=2)
            
            console.print(
                f"  Fold {fold_idx + 1}/{cfg.evaluation.within_subject.cv_folds}: "
                f"Accuracy = [green]{fold_acc:.4f}[/green]"
            )

            del lit_model, trainer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        subj_mean = np.mean(fold_accs)
        subj_std = np.std(fold_accs)
        all_results[subj_id] = {"mean": subj_mean, "std": subj_std, "folds": fold_accs}
        
        console.print(
            f"  [bold]Subject {subj_id} Mean: {subj_mean:.4f} ± {subj_std:.4f}[/bold]"
        )

        # Incrementally update genuine cohort CSV
        csv_path = cache_dir / "genuine_physionet_cohort_results.csv"
        rows = []
        for s_id, s_res in all_results.items():
            row = {
                "Subject": str(s_id),
                "Mean_Acc": round(float(s_res["mean"]), 4),
                "Std_Acc": round(float(s_res["std"]), 4),
            }
            for f_i, f_val in enumerate(s_res["folds"]):
                row[f"Fold_{f_i + 1}"] = round(float(f_val), 4)
            rows.append(row)
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    
    # Summary table
    print_results_table(all_results, cfg.model.name)
    return all_results


def train_cross_subject(cfg: DictConfig):
    """Cross-subject Leave-One-Subject-Out (LOSO) evaluation."""
    console.rule("[bold blue]Cross-Subject LOSO Evaluation[/bold blue]")
    
    dataset = PhysioNetMIDataset(
        root_dir=str(PROJECT_ROOT / cfg.data.physionet_dir),
        subjects=cfg.physionet.get("subjects", None),
        exclude_subjects=cfg.physionet.exclude_subjects,
    )
    
    subjects = dataset.subject_ids
    loso_results = {}
    
    for test_subj in subjects:
        console.print(f"\n[cyan]━━━ Test Subject: {test_subj} ━━━[/cyan]")
        
        # Load all subjects
        train_subjects = [s for s in subjects if s != test_subj]
        
        X_train_list, y_train_list = [], []
        for s in train_subjects:
            s_data = dataset.load_subject(s)
            X_train_list.append(s_data["X"])
            y_train_list.append(s_data["y"])
        
        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        
        t_data = dataset.load_subject(test_subj)
        X_test, y_test = t_data["X"], t_data["y"]
        
        n_channels = X_train.shape[1]
        n_timepoints = X_train.shape[2]
        n_classes = len(np.unique(y_train))
        
        # Create data module
        data_module = EEGDataModule(
            X=X_train, y=y_train,
            X_test=X_test, y_test=y_test,
            batch_size=cfg.training.batch_size,
            mode="fixed_split",
        )
        
        # Create model
        model = create_model(cfg, n_channels, n_timepoints, n_classes)
        
        from src.models.lightning_module import EEGClassifier
        lit_model = EEGClassifier(
            model=model,
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            n_classes=n_classes,
            label_smoothing=cfg.training.loss.label_smoothing,
        )
        
        ckpt_dir = PROJECT_ROOT / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        callbacks = [
            EarlyStopping(monitor="val/acc", mode="max", patience=cfg.training.early_stopping_patience),
            ModelCheckpoint(dirpath=str(ckpt_dir), filename=f"loso-test_{test_subj}-{{val/acc:.4f}}", monitor="val/acc", mode="max", save_top_k=1),
        ]
        
        logger = CSVLogger(save_dir="logs", name=f"{cfg.model.name}/loso", version=f"test_{test_subj}")
        
        precision = "16-mixed" if torch.cuda.is_available() else "32-true"
        trainer = pl.Trainer(
            max_epochs=cfg.training.epochs,
            accelerator="auto",
            devices=1,
            precision=precision,
            callbacks=callbacks,
            logger=logger,
            gradient_clip_val=cfg.training.gradient_clip_max_norm,
            deterministic=True,
            log_every_n_steps=1,
        )
        
        trainer.fit(lit_model, data_module)
        result = trainer.test(lit_model, data_module)
        
        acc = result[0].get("test/acc", result[0].get("test_acc", 0.0))
        loso_results[test_subj] = acc
        console.print(f"  Test Acc = [green]{acc:.4f}[/green]")
    
    # Overall LOSO result
    mean_acc = np.mean(list(loso_results.values()))
    std_acc = np.std(list(loso_results.values()))
    console.print(f"\n[bold green]LOSO Overall: {mean_acc:.4f} ± {std_acc:.4f}[/bold green]")
    
    return loso_results


def print_results_table(results: dict, model_name: str):
    """Print a rich results table."""
    table = Table(title=f"Results: {model_name}")
    table.add_column("Subject", style="cyan")
    table.add_column("Mean Acc", style="green")
    table.add_column("Std", style="yellow")
    
    accs = []
    for subj_id, res in sorted(results.items()):
        table.add_row(
            str(subj_id),
            f"{res['mean']:.4f}",
            f"± {res['std']:.4f}",
        )
        accs.append(res["mean"])
    
    table.add_row("─" * 8, "─" * 8, "─" * 8, style="dim")
    table.add_row(
        "OVERALL",
        f"{np.mean(accs):.4f}",
        f"± {np.std(accs):.4f}",
        style="bold",
    )
    
    console.print(table)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    """Main entry point."""
    cfg = load_config()
    set_seed(cfg.training.seed)
    
    console.rule("[bold magenta]🧠 Wav-KambaNet: EEG Motor Imagery Classification[/bold magenta]")
    console.print(f"Model: [bold]{cfg.model.name}[/bold]")
    console.print(f"Device: [bold]{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}[/bold]")
    
    # Select evaluation mode
    eval_mode = cfg.get("evaluation_mode", "within_subject")
    
    if eval_mode == "within_subject":
        results = train_within_subject(cfg)
    elif eval_mode == "cross_subject":
        results = train_cross_subject(cfg)
    else:
        raise ValueError(f"Unknown evaluation mode: {eval_mode}")
    
    console.rule("[bold green]Training Complete[/bold green]")


if __name__ == "__main__":
    main()
