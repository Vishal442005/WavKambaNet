"""
PyTorch Dataset and Lightning DataModule for EEG Motor Imagery Decoding.

Provides structured PyTorch data handling with electrode spatial coordinates for
Graph Neural Networks (ChebConv) and cross-validation / LOSO splitting routines.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import logging
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class EEGDataset(Dataset):
    """
    PyTorch Dataset for preprocessed EEG motor imagery data.

    Supports multichannel time series (N, C, T), labels, subject IDs,
    electrode 3D positions, spatial adjacency computation for graph neural networks,
    and on-the-fly data augmentations.

    Parameters
    ----------
    X : Union[np.ndarray, torch.Tensor]
        EEG data array of shape (n_trials, n_channels, n_timepoints).
    y : Union[np.ndarray, torch.Tensor]
        Target labels of shape (n_trials,).
    subject_ids : Optional[Union[np.ndarray, Sequence[Union[int, str]]]], default=None
        Subject identifier for each trial, shape (n_trials,).
    run_ids : Optional[Union[np.ndarray, Sequence[Union[int, str]]]], default=None
        Run identifier for each trial, shape (n_trials,).
    transform : Optional[Callable], default=None
        Transform or augmentation callable applied to each trial during __getitem__.
    ch_names : Optional[List[str]], default=None
        List of EEG channel names corresponding to the channels dimension.
    electrode_positions : Optional[Union[np.ndarray, Dict[str, Sequence[float]]]], default=None
        3D electrode positions of shape (n_channels, 3) or dictionary mapping channel names to coordinates.
    class_to_idx : Optional[Dict[str, int]], default=None
        Mapping from class names to integer indices.
    idx_to_class : Optional[Dict[int, str]], default=None
        Mapping from integer indices to class names.
    """

    def __init__(
        self,
        X: Union[np.ndarray, torch.Tensor],
        y: Union[np.ndarray, torch.Tensor],
        subject_ids: Optional[Union[np.ndarray, Sequence[Union[int, str]]]] = None,
        run_ids: Optional[Union[np.ndarray, Sequence[Union[int, str]]]] = None,
        transform: Optional[Callable] = None,
        ch_names: Optional[List[str]] = None,
        electrode_positions: Optional[Union[np.ndarray, Dict[str, Sequence[float]]]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
        idx_to_class: Optional[Dict[int, str]] = None,
    ) -> None:
        super().__init__()

        # Ensure X is float tensor
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X).float()
        elif isinstance(X, torch.Tensor):
            self.X = X.clone().float()
        else:
            raise TypeError(f"Unsupported type for X: {type(X)}. Expected np.ndarray or torch.Tensor.")

        # Ensure y is long tensor
        if isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y).long()
        elif isinstance(y, torch.Tensor):
            self.y = y.clone().long()
        else:
            raise TypeError(f"Unsupported type for y: {type(y)}. Expected np.ndarray or torch.Tensor.")

        if len(self.X) != len(self.y):
            raise ValueError(f"Size mismatch: X has {len(self.X)} samples, y has {len(self.y)} samples.")

        if self.X.ndim != 3:
            raise ValueError(f"Expected X to have shape (n_trials, n_channels, n_timepoints), got {self.X.shape}")

        self.num_trials, self.num_channels, self.timepoints = self.X.shape

        # Metadata
        if subject_ids is not None:
            self.subject_ids = np.array(subject_ids)
        else:
            self.subject_ids = np.zeros(self.num_trials, dtype=int)

        if run_ids is not None:
            self.run_ids = np.array(run_ids)
        else:
            self.run_ids = np.zeros(self.num_trials, dtype=int)

        self.transform = transform
        self.ch_names = list(ch_names) if ch_names is not None else [f"EEG{i+1:03d}" for i in range(self.num_channels)]

        # Electrode positions handling
        self.electrode_positions: Optional[np.ndarray] = None
        if electrode_positions is not None:
            if isinstance(electrode_positions, dict):
                coords = []
                for name in self.ch_names:
                    if name in electrode_positions:
                        coords.append(electrode_positions[name])
                    else:
                        coords.append([0.0, 0.0, 0.0])
                self.electrode_positions = np.array(coords, dtype=np.float32)
            elif isinstance(electrode_positions, np.ndarray):
                self.electrode_positions = electrode_positions.astype(np.float32)
            elif isinstance(electrode_positions, torch.Tensor):
                self.electrode_positions = electrode_positions.cpu().numpy().astype(np.float32)

        # Class mappings
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class
        if self.idx_to_class is None and self.class_to_idx is not None:
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self) -> int:
        return self.num_trials

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve sample and target.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            EEG tensor (C, T) and label scalar.
        """
        x = self.X[idx]
        y = self.y[idx]

        if self.transform is not None:
            x = self.transform(x)
            if isinstance(x, np.ndarray):
                x = torch.from_numpy(x).float()

        return x, y

    def get_item_with_metadata(self, idx: int) -> Dict[str, Any]:
        """Retrieve sample with associated subject and run metadata."""
        x, y = self.__getitem__(idx)
        return {
            "x": x,
            "y": y,
            "subject_id": self.subject_ids[idx],
            "run_id": self.run_ids[idx],
        }

    def get_electrode_positions(self) -> Optional[np.ndarray]:
        """Return 3D coordinates array (n_channels, 3)."""
        return self.electrode_positions

    def get_adjacency_matrix(
        self,
        distance_metric: str = "euclidean",
        sigma: Optional[float] = None,
        threshold: Optional[float] = None,
        k_neighbors: Optional[int] = None,
        self_loop: bool = True,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Compute spatial graph adjacency matrix from electrode 3D positions.

        Used by Chebyshev spectral graph convolutional layers (ChebConv).

        Parameters
        ----------
        distance_metric : str, default='euclidean'
            Distance metric for electrode positions.
        sigma : Optional[float], default=None
            Gaussian kernel bandwidth. If None, set to median pairwise distance.
        threshold : Optional[float], default=None
            Pruning threshold. Distances above or affinities below this threshold are zeroed.
        k_neighbors : Optional[int], default=None
            If specified, keeps only the k nearest neighbors for each electrode.
        self_loop : bool, default=True
            Whether to add self-loops (diagonal = 1.0).
        normalize : bool, default=False
            Whether to return symmetric normalized adjacency D^(-1/2) * A * D^(-1/2).

        Returns
        -------
        np.ndarray
            Adjacency matrix of shape (n_channels, n_channels).
        """
        if self.electrode_positions is None:
            # Fallback to fully-connected identity with self loops
            adj = np.eye(self.num_channels, dtype=np.float32)
            return adj

        coords = self.electrode_positions  # (C, 3)
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (C, C, 3)
        dist = np.linalg.norm(diff, axis=-1)  # (C, C)

        # Compute Gaussian affinity
        if sigma is None:
            # Use median of upper triangle distances
            triu_dists = dist[np.triu_indices_from(dist, k=1)]
            sigma = float(np.median(triu_dists)) if len(triu_dists) > 0 else 1.0
            if sigma < 1e-6:
                sigma = 1.0

        adj = np.exp(-(dist ** 2) / (2.0 * (sigma ** 2)))

        # k-nearest neighbor filtering if requested
        if k_neighbors is not None and 0 < k_neighbors < self.num_channels:
            k_adj = np.zeros_like(adj)
            for i in range(self.num_channels):
                # Top k closest nodes (including self if self_loop, or excluding)
                closest_indices = np.argsort(dist[i])[: k_neighbors + 1]
                k_adj[i, closest_indices] = adj[i, closest_indices]
            adj = np.maximum(k_adj, k_adj.T)  # Symmetrize

        if threshold is not None:
            adj[adj < threshold] = 0.0

        if self_loop:
            np.fill_diagonal(adj, 1.0)
        else:
            np.fill_diagonal(adj, 0.0)

        if normalize:
            degrees = np.sum(adj, axis=1)
            degrees_inv_sqrt = np.power(np.maximum(degrees, 1e-8), -0.5)
            deg_mat = np.diag(degrees_inv_sqrt)
            adj = deg_mat @ adj @ deg_mat

        return adj.astype(np.float32)

    def get_pyg_edge_index_and_weight(
        self,
        threshold: float = 0.1,
        **adj_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get PyTorch Geometric edge_index (2, num_edges) and edge_weight (num_edges,).

        Parameters
        ----------
        threshold : float, default=0.1
            Minimum edge weight to include in edge_index.
        **adj_kwargs :
            Additional arguments passed to get_adjacency_matrix.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (edge_index, edge_weight) as torch tensors.
        """
        adj = self.get_adjacency_matrix(**adj_kwargs)
        adj[adj < threshold] = 0.0

        rows, cols = np.where(adj > 0.0)
        edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
        edge_weight = torch.tensor(adj[rows, cols], dtype=torch.float32)

        return edge_index, edge_weight

    def subset(self, indices: Sequence[int], transform: Optional[Callable] = None) -> "EEGDataset":
        """
        Create a new EEGDataset with a subset of trial indices.

        Parameters
        ----------
        indices : Sequence[int]
            Indices of trials to select.
        transform : Optional[Callable], default=None
            Optional new transform to assign to the subset dataset.
        """
        indices_arr = np.array(indices)
        return EEGDataset(
            X=self.X[indices_arr],
            y=self.y[indices_arr],
            subject_ids=self.subject_ids[indices_arr] if self.subject_ids is not None else None,
            run_ids=self.run_ids[indices_arr] if self.run_ids is not None else None,
            transform=transform if transform is not None else self.transform,
            ch_names=self.ch_names,
            electrode_positions=self.electrode_positions,
            class_to_idx=self.class_to_idx,
            idx_to_class=self.idx_to_class,
        )


class EEGDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for EEG Motor Imagery decoding.

    Handles dataset preparation, within-subject splits, cross-subject Leave-One-Subject-Out (LOSO)
    splits, stratified k-fold cross-validation, and high-performance DataLoaders.

    Parameters
    ----------
    dataset : Optional[EEGDataset], default=None
        Pre-built EEGDataset. If not provided, physionet_loader will construct it.
    physionet_dataset : Optional[Any], default=None
        Optional PhysioNetMIDataset instance to build datasets from.
    split_type : str, default='within_subject'
        Type of split: 'within_subject', 'cross_subject' (or 'loso'), 'kfold', or 'holdout'.
    test_subject : Optional[Union[int, str]], default=None
        Test subject identifier for LOSO cross-subject evaluation.
    subject_id : Optional[Union[int, str]], default=None
        Specific subject identifier for within-subject evaluation.
    k_folds : int, default=5
        Number of folds for k-fold cross validation.
    fold_idx : int, default=0
        Active fold index for k-fold cross validation (0 <= fold_idx < k_folds).
    val_ratio : float, default=0.2
        Ratio of training data used for validation in holdout / within_subject modes.
    batch_size : int, default=64
        Mini-batch size.
    num_workers : int, default=0
        Number of workers for data loading (0 recommended on Windows).
    pin_memory : bool, default=True
        Whether to pin memory in GPU dataloading.
    drop_last : bool, default=False
        Whether to drop last incomplete mini-batch in train dataloader.
    train_transform : Optional[Callable], default=None
        Data augmentation pipeline for training set.
    val_transform : Optional[Callable], default=None
        Optional transform for validation set.
    test_transform : Optional[Callable], default=None
        Optional transform for test set.
    seed : int, default=42
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dataset: Optional[EEGDataset] = None,
        physionet_dataset: Optional[Any] = None,
        X: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y: Optional[Union[np.ndarray, torch.Tensor]] = None,
        X_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
        y_test: Optional[Union[np.ndarray, torch.Tensor]] = None,
        split_type: str = "kfold",
        mode: Optional[str] = None,
        test_subject: Optional[Union[int, str]] = None,
        subject_id: Optional[Union[int, str]] = None,
        k_folds: int = 5,
        n_folds: Optional[int] = None,
        fold_idx: int = 0,
        val_ratio: float = 0.2,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["dataset", "physionet_dataset", "train_transform", "val_transform", "test_transform"])

        if dataset is not None:
            self.full_dataset = dataset
        elif X is not None and y is not None:
            self.full_dataset = EEGDataset(X=X, y=y)
        else:
            self.full_dataset = None

        self.test_dataset_fixed = EEGDataset(X=X_test, y=y_test) if (X_test is not None and y_test is not None) else None
        self.physionet_dataset = physionet_dataset
        
        effective_mode = mode if mode is not None else split_type
        self.split_type = effective_mode.lower()
        self.test_subject = test_subject
        self.subject_id = subject_id
        self.k_folds = n_folds if n_folds is not None else k_folds
        self.fold_idx = fold_idx
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.drop_last = drop_last
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.seed = seed

        self.train_dataset: Optional[EEGDataset] = None
        self.val_dataset: Optional[EEGDataset] = None
        self.test_dataset: Optional[EEGDataset] = None

    def set_fold(self, fold_idx: int) -> None:
        """Dynamically set the active fold index for cross validation."""
        self.fold_idx = fold_idx
        self.setup()

    def prepare_data(self) -> None:
        """Verify data source availability."""
        if self.full_dataset is None and self.physionet_dataset is None:
            raise ValueError("Either `dataset`, `(X, y)`, or `physionet_dataset` must be provided to EEGDataModule.")

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Split datasets according to split_type with strict zero data leakage.

        Parameters
        ----------
        stage : Optional[str], default=None
            Pipeline stage: 'fit', 'validate', 'test', or None.
        """
        if self.full_dataset is None and self.physionet_dataset is not None:
            logger.info("Building full EEGDataset from PhysioNetMIDataset...")
            self.full_dataset = self.physionet_dataset.to_pytorch_dataset()

        base_dataset = self.full_dataset
        if base_dataset is None:
            raise RuntimeError("Failed to obtain EEGDataset for data module setup.")

        # Filter by subject_id if within_subject and subject_id is specified
        if self.split_type == "within_subject" and self.subject_id is not None:
            sub_id_str = str(self.subject_id)
            sub_mask = np.array([str(s) == sub_id_str for s in base_dataset.subject_ids])
            if not np.any(sub_mask):
                # Try formatted Sxxx match
                sub_id_int = int("".join(filter(str.isdigit, sub_id_str))) if any(c.isdigit() for c in sub_id_str) else None
                if sub_id_int is not None:
                    sub_mask = np.array([
                        (str(s) == f"S{sub_id_int:03d}" or str(s) == str(sub_id_int))
                        for s in base_dataset.subject_ids
                    ])
            indices = np.where(sub_mask)[0]
            if len(indices) == 0:
                raise ValueError(f"No trials found for subject {self.subject_id} in dataset.")
            base_dataset = base_dataset.subset(indices)

        n_samples = len(base_dataset)
        labels = base_dataset.y.numpy()

        if self.split_type in ("cross_subject", "loso"):
            # Leave-One-Subject-Out evaluation
            if self.test_subject is None:
                raise ValueError("`test_subject` must be specified for 'cross_subject' / 'loso' split_type.")

            test_sub_str = str(self.test_subject)
            sub_ids = base_dataset.subject_ids

            # Normalize test subject comparison
            test_mask = np.array([str(s) == test_sub_str for s in sub_ids])
            if not np.any(test_mask) and any(c.isdigit() for c in test_sub_str):
                sub_int = int("".join(filter(str.isdigit, test_sub_str)))
                test_mask = np.array([
                    (str(s) == f"S{sub_int:03d}" or str(s) == str(sub_int))
                    for s in sub_ids
                ])

            test_indices = np.where(test_mask)[0]
            train_val_indices = np.where(~test_mask)[0]

            if len(test_indices) == 0:
                raise ValueError(f"Subject {self.test_subject} not found in dataset for LOSO split.")

            # Stratified split train_val into train and validation
            train_val_labels = labels[train_val_indices]
            train_idx_rel, val_idx_rel = train_test_split(
                np.arange(len(train_val_indices)),
                test_size=self.val_ratio,
                stratify=train_val_labels,
                random_state=self.seed,
            )
            train_indices = train_val_indices[train_idx_rel]
            val_indices = train_val_indices[val_idx_rel]

            self.train_dataset = base_dataset.subset(train_indices, transform=self.train_transform)
            self.val_dataset = base_dataset.subset(val_indices, transform=self.val_transform)
            self.test_dataset = base_dataset.subset(test_indices, transform=self.test_transform)

        elif self.split_type == "kfold":
            # Stratified K-Fold cross validation
            skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)
            folds = list(skf.split(np.zeros(n_samples), labels))

            if not (0 <= self.fold_idx < self.k_folds):
                raise ValueError(f"fold_idx {self.fold_idx} out of range for k_folds={self.k_folds}")

            train_indices, test_indices = folds[self.fold_idx]

            # Split train further to obtain validation fold
            train_labels = labels[train_indices]
            train_idx_rel, val_idx_rel = train_test_split(
                np.arange(len(train_indices)),
                test_size=self.val_ratio,
                stratify=train_labels,
                random_state=self.seed,
            )
            val_indices = train_indices[val_idx_rel]
            train_indices = train_indices[train_idx_rel]

            self.train_dataset = base_dataset.subset(train_indices, transform=self.train_transform)
            self.val_dataset = base_dataset.subset(val_indices, transform=self.val_transform)
            self.test_dataset = base_dataset.subset(test_indices, transform=self.test_transform)

        elif self.split_type in ("fixed_split", "fixed"):
            if self.test_dataset_fixed is None:
                raise ValueError("`X_test` and `y_test` must be provided for 'fixed_split' mode.")
            # Stratified split base_dataset into train and validation
            train_idx, val_idx = train_test_split(
                np.arange(n_samples),
                test_size=self.val_ratio,
                stratify=labels,
                random_state=self.seed,
            )
            self.train_dataset = base_dataset.subset(train_idx, transform=self.train_transform)
            self.val_dataset = base_dataset.subset(val_idx, transform=self.val_transform)
            self.test_dataset = self.test_dataset_fixed

        elif self.split_type in ("within_subject", "holdout"):
            # Standard stratified holdout split
            indices = np.arange(n_samples)
            train_val_idx, test_idx = train_test_split(
                indices,
                test_size=0.2,
                stratify=labels,
                random_state=self.seed,
            )
            train_val_labels = labels[train_val_idx]
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=self.val_ratio,
                stratify=train_val_labels,
                random_state=self.seed,
            )

            self.train_dataset = base_dataset.subset(train_idx, transform=self.train_transform)
            self.val_dataset = base_dataset.subset(val_idx, transform=self.val_transform)
            self.test_dataset = base_dataset.subset(test_idx, transform=self.test_transform)
        else:
            raise ValueError(
                f"Unknown split_type: '{self.split_type}'. Choose from 'within_subject', 'cross_subject', 'kfold', 'fixed_split', 'holdout'."
            )

        logger.info(
            f"EEGDataModule setup complete ({self.split_type}): "
            f"train={len(self.train_dataset)}, val={len(self.val_dataset)}, test={len(self.test_dataset)}"
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule train_dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last and len(self.train_dataset) > self.batch_size,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule val_dataset not initialized. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule test_dataset not initialized. Call setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    @property
    def electrode_positions(self) -> Optional[np.ndarray]:
        """Electrode 3D positions array of shape (C, 3)."""
        if self.full_dataset is not None:
            return self.full_dataset.get_electrode_positions()
        return None

    @property
    def adjacency_matrix(self) -> Optional[np.ndarray]:
        """Graph adjacency matrix for Chebyshev spectral graph conv."""
        if self.full_dataset is not None:
            return self.full_dataset.get_adjacency_matrix()
        return None

    @property
    def num_classes(self) -> int:
        """Number of unique target classes."""
        if self.full_dataset is not None:
            return len(torch.unique(self.full_dataset.y))
        return 0

    @property
    def num_channels(self) -> int:
        """Number of EEG channels."""
        if self.full_dataset is not None:
            return self.full_dataset.num_channels
        return 0

    @property
    def timepoints(self) -> int:
        """Number of temporal samples per trial."""
        if self.full_dataset is not None:
            return self.full_dataset.timepoints
        return 0

    @property
    def ch_names(self) -> List[str]:
        """EEG channel names."""
        if self.full_dataset is not None:
            return self.full_dataset.ch_names
        return []
