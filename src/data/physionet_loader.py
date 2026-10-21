"""
Comprehensive PhysioNet EEG Motor Movement/Imagery Dataset Loader.

Dataset structure:
- 109 subjects (S001-S109), excluding S088, S092, S100 (known timing anomalies).
- 64 EEG channels (10-20 standard system), recorded at 160 Hz.
- 14 experimental runs per subject:
  - R01, R02: Baseline (eyes open, eyes closed)
  - R03, R07, R11: Motor execution (left/right fist)
  - R04, R08, R12: Motor IMAGERY (left/right fist) [Left/Right Task]
  - R05, R09, R13: Motor execution (both fists/both feet)
  - R06, R10, R14: Motor IMAGERY (both fists/both feet) [Fists/Feet Task]
- Events:
  - T0: Rest
  - In Left/Right runs: T1 = Left fist, T2 = Right fist
  - In Fists/Feet runs: T1 = Both fists, T2 = Both feet
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import logging
import os
import mne
import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch

from src.data.dataset import EEGDataset

logger = logging.getLogger(__name__)

# Default subjects with timing / marker anomalies to exclude
DEFAULT_EXCLUDE_SUBJECTS = (88, 92, 100)

# Run categorizations
IMAGERY_RUNS_LR = [4, 8, 12]
IMAGERY_RUNS_FF = [6, 10, 14]
EXECUTION_RUNS_LR = [3, 7, 11]
EXECUTION_RUNS_FF = [5, 9, 13]
BASELINE_RUNS = [1, 2]

# Unambiguous global event codes
EVENT_CODE_REST = 1
EVENT_CODE_LEFT_FIST = 10
EVENT_CODE_RIGHT_FIST = 11
EVENT_CODE_BOTH_FISTS = 20
EVENT_CODE_BOTH_FEET = 21


def format_subject_id(subject_id: Union[int, str]) -> str:
    """Format subject identifier to standard 'S001' string."""
    if isinstance(subject_id, int):
        return f"S{subject_id:03d}"
    s = str(subject_id).strip()
    if s.isdigit():
        return f"S{int(s):03d}"
    if s.upper().startswith("S") and s[1:].isdigit():
        return f"S{int(s[1:]):03d}"
    return s


def format_run_id(run_id: Union[int, str]) -> str:
    """Format run identifier to standard 'R01' string."""
    if isinstance(run_id, int):
        return f"R{run_id:02d}"
    s = str(run_id).strip()
    if s.isdigit():
        return f"R{int(s):02d}"
    if s.upper().startswith("R") and s[1:].isdigit():
        return f"R{int(s[1:]):02d}"
    return s


def parse_run_number(run_id: Union[int, str]) -> int:
    """Extract integer run number from run identifier."""
    if isinstance(run_id, int):
        return run_id
    s = str(run_id).strip().upper()
    if s.startswith("R"):
        return int(s[1:])
    return int(s)


class PhysioNetMIDataset:
    """
    Loader and preprocessor for PhysioNet EEG Motor Movement/Imagery Database.

    Parameters
    ----------
    root_dir : Union[str, Path]
        Path to the root directory containing subject folders (e.g. S001, S002).
    subjects : Optional[Sequence[Union[int, str]]], default=None
        List of subject IDs to load. If None, loads all available subjects (1..109)
        excluding `exclude_subjects`.
    runs : Union[str, Sequence[Union[int, str]]], default='imagery'
        Runs to load:
        - 'imagery': R04, R08, R12 (left/right) and/or R06, R10, R14 (fists/feet) based on task
        - 'execution': R03, R07, R11 and/or R05, R09, R13
        - 'all': All 14 runs or all task runs
        - Sequence of ints or strings: e.g. [4, 8, 12] or ['R04', 'R08', 'R12']
    task : str, default='left_right'
        Target classification task:
        - 'left_right' (2-class: left fist vs right fist)
        - 'fists_feet' (2-class: both fists vs both feet)
        - '4_class' or 'four_class' (4-class: left fist, right fist, both fists, both feet)
    l_freq : float, default=0.5
        Lower cutoff frequency for bandpass filter in Hz.
    h_freq : float, default=45.0
        Upper cutoff frequency for bandpass filter in Hz.
    notch_freq : float, default=60.0
        Notch filter frequency in Hz (60.0 Hz for US PhysioNet dataset).
    resample_freq : Optional[float], default=250.0
        Target sampling rate in Hz. If None, keeps original 160 Hz.
    tmin : float, default=0.0
        Epoch start time relative to event in seconds.
    tmax : float, default=4.0
        Epoch end time relative to event in seconds.
    baseline : Optional[Tuple[float, float]], default=None
        Baseline correction interval in seconds, or None for no baseline correction.
    normalize : str, default='zscore'
        Normalization strategy: 'zscore' (per-channel per-trial) or None.
    apply_car : bool, default=True
        Whether to apply Common Average Referencing (CAR).
    apply_ica : bool, default=False
        Whether to apply FastICA for ocular/motion artifact rejection.
    ica_n_components : int, default=20
        Number of principal components for ICA.
    exclude_subjects : Optional[Sequence[Union[int, str]]], default=(88, 92, 100)
        List of subjects to exclude due to timing anomalies.
    include_rest : bool, default=False
        Whether to include the rest (T0) event as an additional target class.
    cache_dir : Optional[Union[str, Path]], default=None
        Directory to store preprocessed arrays for high-speed caching.
    use_cache : bool, default=True
        Whether to load from and save to disk cache when available.
    montage_name : str, default='standard_1020'
        Standard electrode montage name in MNE.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        subjects: Optional[Sequence[Union[int, str]]] = None,
        runs: Union[str, Sequence[Union[int, str]]] = "imagery",
        task: str = "left_right",
        l_freq: float = 0.5,
        h_freq: float = 45.0,
        notch_freq: float = 60.0,
        resample_freq: Optional[float] = 250.0,
        tmin: float = 0.0,
        tmax: float = 4.0,
        baseline: Optional[Tuple[float, float]] = None,
        normalize: str = "zscore",
        apply_car: bool = True,
        apply_ica: bool = False,
        ica_n_components: int = 20,
        exclude_subjects: Optional[Sequence[Union[int, str]]] = DEFAULT_EXCLUDE_SUBJECTS,
        include_rest: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
        use_cache: bool = True,
        montage_name: str = "standard_1020",
    ) -> None:
        self.root_dir = Path(root_dir)
        self.task = task.lower().strip()
        self.l_freq = float(l_freq) if l_freq is not None else None
        self.h_freq = float(h_freq) if h_freq is not None else None
        self.notch_freq = float(notch_freq) if notch_freq is not None else None
        self.resample_freq = float(resample_freq) if resample_freq is not None else None
        self.tmin = float(tmin)
        self.tmax = float(tmax)
        self.baseline = baseline
        self.normalize = normalize
        self.apply_car = apply_car
        self.apply_ica = apply_ica
        self.ica_n_components = int(ica_n_components)
        self.include_rest = include_rest
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.use_cache = use_cache
        self.montage_name = montage_name

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Standard montage
        self.montage = mne.channels.make_standard_montage(self.montage_name)
        self.montage_ch_map = {ch.upper(): ch for ch in self.montage.ch_names}

        # Resolve excluded subjects
        self.exclude_subjects = set()
        if exclude_subjects is not None:
            for s in exclude_subjects:
                self.exclude_subjects.add(format_subject_id(s))

        # Resolve runs based on task and user choice
        self.run_ids = self._resolve_runs(runs, self.task)

        # Resolve subjects
        self.subject_ids = self._resolve_subjects(subjects)

        # Task and class definitions
        self.class_to_idx, self.idx_to_class = self._build_class_mappings()

        # Cache for loaded subject data
        self._loaded_data: Optional[Dict[str, Dict[str, Any]]] = None
        self.ch_names: Optional[List[str]] = None
        self.electrode_positions: Optional[np.ndarray] = None

    def _resolve_runs(self, runs: Union[str, Sequence[Union[int, str]]], task: str) -> List[int]:
        """Determine run numbers based on user specification and task."""
        if isinstance(runs, str):
            r_str = runs.lower().strip()
            if r_str == "imagery":
                if task == "left_right":
                    return list(IMAGERY_RUNS_LR)
                elif task == "fists_feet":
                    return list(IMAGERY_RUNS_FF)
                elif task in ("4_class", "four_class"):
                    return list(IMAGERY_RUNS_LR + IMAGERY_RUNS_FF)
                else:
                    return list(IMAGERY_RUNS_LR + IMAGERY_RUNS_FF)
            elif r_str == "execution":
                if task == "left_right":
                    return list(EXECUTION_RUNS_LR)
                elif task == "fists_feet":
                    return list(EXECUTION_RUNS_FF)
                elif task in ("4_class", "four_class"):
                    return list(EXECUTION_RUNS_LR + EXECUTION_RUNS_FF)
                else:
                    return list(EXECUTION_RUNS_LR + EXECUTION_RUNS_FF)
            elif r_str == "all":
                return list(range(1, 15))
            else:
                raise ValueError(f"Unknown runs specifier: '{runs}'. Use 'imagery', 'execution', 'all', or a list of runs.")
        else:
            return [parse_run_number(r) for r in runs]

    def _resolve_subjects(self, subjects: Optional[Sequence[Union[int, str]]]) -> List[str]:
        """Determine subject IDs to load."""
        if subjects is not None:
            if isinstance(subjects, (int, str)):
                subjects = [subjects]
            resolved = [format_subject_id(s) for s in subjects if format_subject_id(s) not in self.exclude_subjects]
            return sorted(resolved)

        # Auto-discover from root_dir
        if not self.root_dir.exists():
            raise FileNotFoundError(f"PhysioNet root directory not found: {self.root_dir}")

        all_dirs = [
            d.name for d in self.root_dir.iterdir()
            if d.is_dir() and d.name.upper().startswith("S") and d.name[1:].isdigit()
        ]
        resolved = [
            format_subject_id(d) for d in all_dirs
            if format_subject_id(d) not in self.exclude_subjects
        ]
        return sorted(resolved, key=lambda s: int(s[1:]))

    def _build_class_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Construct class label mappings for the selected task."""
        if self.task == "left_right":
            mapping = {"left_fist": 0, "right_fist": 1}
        elif self.task == "fists_feet":
            mapping = {"both_fists": 0, "both_feet": 1}
        elif self.task in ("4_class", "four_class"):
            mapping = {
                "left_fist": 0,
                "right_fist": 1,
                "both_fists": 2,
                "both_feet": 3,
            }
        else:
            raise ValueError(f"Unsupported task: '{self.task}'. Choose 'left_right', 'fists_feet', or '4_class'.")

        if self.include_rest:
            mapping["rest"] = len(mapping)

        idx_mapping = {v: k for k, v in mapping.items()}
        return mapping, idx_mapping

    def _clean_channel_names(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Strip trailing dots and map channel names to standard 10-20 case."""
        rename_dict = {}
        for ch in raw.ch_names:
            clean = ch.rstrip(".").upper()
            if clean in self.montage_ch_map:
                rename_dict[ch] = self.montage_ch_map[clean]
            else:
                rename_dict[ch] = ch.rstrip(".")
        raw.rename_channels(rename_dict)
        raw.set_montage(self.montage, on_missing="ignore", verbose=False)
        return raw

    def load_raw(self, subject_id: Union[int, str], run_id: Union[int, str]) -> mne.io.Raw:
        """
        Load a single EDF file from disk for a given subject and run.

        Parameters
        ----------
        subject_id : Union[int, str]
            Subject ID (e.g. 1 or 'S001').
        run_id : Union[int, str]
            Run ID (e.g. 4 or 'R04').

        Returns
        -------
        mne.io.Raw
            Loaded Raw EDF object with standard montage applied.
        """
        sub_str = format_subject_id(subject_id)
        run_str = format_run_id(run_id)

        filename = f"{sub_str}{run_str}.edf"
        file_path = self.root_dir / sub_str / filename

        # Case-insensitive search fallback
        if not file_path.exists():
            file_path_upper = self.root_dir / sub_str / f"{sub_str}{run_str}.EDF"
            if file_path_upper.exists():
                file_path = file_path_upper
            else:
                raise FileNotFoundError(f"EDF file not found: {file_path}")

        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        raw = self._clean_channel_names(raw)

        # Cache channel names and positions on first load
        if self.ch_names is None:
            self.ch_names = list(raw.ch_names)
            montage_positions = self.montage.get_positions()["ch_pos"]
            coords = []
            for ch in self.ch_names:
                if ch in montage_positions:
                    coords.append(montage_positions[ch])
                else:
                    coords.append([0.0, 0.0, 0.0])
            self.electrode_positions = np.array(coords, dtype=np.float32)

        return raw

    def preprocess(
        self,
        raw: mne.io.Raw,
        l_freq: Optional[float] = None,
        h_freq: Optional[float] = None,
        notch: Optional[float] = None,
        resample: Optional[float] = None,
        apply_car: Optional[bool] = None,
        apply_ica: Optional[bool] = None,
        copy: bool = True,
    ) -> mne.io.Raw:
        """
        Apply bandpass filtering, notch filtering, Common Average Referencing (CAR),
        resampling, and optional FastICA artifact rejection.

        Parameters
        ----------
        raw : mne.io.Raw
            Input raw EEG object.
        l_freq : Optional[float]
            Lower passband edge (Hz).
        h_freq : Optional[float]
            Upper passband edge (Hz).
        notch : Optional[float]
            Notch filter frequency (Hz).
        resample : Optional[float]
            Target sampling frequency (Hz).
        apply_car : Optional[bool]
            Whether to apply Common Average Referencing.
        apply_ica : Optional[bool]
            Whether to fit and apply FastICA.
        copy : bool, default=True
            Whether to operate on a copy of raw.

        Returns
        -------
        mne.io.Raw
            Preprocessed Raw EEG object.
        """
        if copy:
            raw = raw.copy()

        l_freq = l_freq if l_freq is not None else self.l_freq
        h_freq = h_freq if h_freq is not None else self.h_freq
        notch = notch if notch is not None else self.notch_freq
        resample = resample if resample is not None else self.resample_freq
        apply_car = apply_car if apply_car is not None else self.apply_car
        apply_ica = apply_ica if apply_ica is not None else self.apply_ica

        # 1. Bandpass filter
        if l_freq is not None or h_freq is not None:
            raw.filter(
                l_freq=l_freq,
                h_freq=h_freq,
                fir_design="firwin",
                skip_by_annotation="edge",
                verbose=False,
            )

        # 2. Notch filter for powerline hum
        if notch is not None and notch > 0.0:
            raw.notch_filter(
                freqs=notch,
                fir_design="firwin",
                verbose=False,
            )

        # 3. Common Average Referencing (CAR)
        if apply_car:
            raw.set_eeg_reference(ref_channels="average", projection=False, verbose=False)

        # 4. Optional FastICA artifact removal
        if apply_ica:
            try:
                n_comp = min(self.ica_n_components, len(raw.ch_names))
                ica = mne.preprocessing.ICA(
                    n_components=n_comp,
                    method="fastica",
                    random_state=42,
                    verbose=False,
                )
                ica.fit(raw, verbose=False)

                # Attempt finding EOG artifacts from frontal channels
                frontal_chs = [ch for ch in ["Fpz", "Fp1", "Fp2"] if ch in raw.ch_names]
                exclude_idx = []
                for f_ch in frontal_chs:
                    eog_idx, _ = ica.find_bads_eog(raw, ch_name=f_ch, verbose=False)
                    exclude_idx.extend(eog_idx)

                ica.exclude = list(set(exclude_idx))
                if ica.exclude:
                    ica.apply(raw, verbose=False)
            except Exception as e:
                logger.warning(f"ICA artifact rejection skipped due to error: {e}")

        # 5. Resampling
        if resample is not None and resample > 0.0 and abs(raw.info["sfreq"] - resample) > 1e-2:
            raw.resample(sfreq=resample, npad="auto", verbose=False)

        return raw

    def create_epochs(
        self,
        raw: mne.io.Raw,
        run_id: Union[int, str],
        tmin: Optional[float] = None,
        tmax: Optional[float] = None,
        baseline: Optional[Tuple[float, float]] = None,
    ) -> Optional[mne.Epochs]:
        """
        Extract motor imagery epochs around annotations with task-specific event mapping.

        Parameters
        ----------
        raw : mne.io.Raw
            Preprocessed raw EEG object.
        run_id : Union[int, str]
            Run identifier for correct event semantic mapping.
        tmin : Optional[float]
            Epoch start time relative to event onset (seconds).
        tmax : Optional[float]
            Epoch end time relative to event onset (seconds).
        baseline : Optional[Tuple[float, float]]
            Baseline interval (seconds).

        Returns
        -------
        Optional[mne.Epochs]
            Extracted epochs, or None if no matching task events found.
        """
        tmin = tmin if tmin is not None else self.tmin
        tmax = tmax if tmax is not None else self.tmax
        baseline = baseline if baseline is not None else self.baseline

        # Fixed annotation mapping to ensure consistent IDs
        annotation_map = {"T0": 1, "T1": 2, "T2": 3}
        events, event_dict = mne.events_from_annotations(raw, event_id=annotation_map, verbose=False)

        if len(events) == 0:
            return None

        run_num = parse_run_number(run_id)
        is_lr = run_num in (3, 4, 7, 8, 11, 12)
        is_ff = run_num in (5, 6, 9, 10, 13, 14)

        # Remap event codes to unambiguous global codes
        remapped_events = events.copy()
        target_event_id: Dict[str, int] = {}

        if is_lr:
            # T1 = left_fist, T2 = right_fist
            remapped_events[events[:, 2] == 2, 2] = EVENT_CODE_LEFT_FIST
            remapped_events[events[:, 2] == 3, 2] = EVENT_CODE_RIGHT_FIST

            if "left_fist" in self.class_to_idx:
                target_event_id["left_fist"] = EVENT_CODE_LEFT_FIST
            if "right_fist" in self.class_to_idx:
                target_event_id["right_fist"] = EVENT_CODE_RIGHT_FIST

        elif is_ff:
            # T1 = both_fists, T2 = both_feet
            remapped_events[events[:, 2] == 2, 2] = EVENT_CODE_BOTH_FISTS
            remapped_events[events[:, 2] == 3, 2] = EVENT_CODE_BOTH_FEET

            if "both_fists" in self.class_to_idx:
                target_event_id["both_fists"] = EVENT_CODE_BOTH_FISTS
            if "both_feet" in self.class_to_idx:
                target_event_id["both_feet"] = EVENT_CODE_BOTH_FEET

        if self.include_rest and "rest" in self.class_to_idx:
            remapped_events[events[:, 2] == 1, 2] = EVENT_CODE_REST
            target_event_id["rest"] = EVENT_CODE_REST

        if not target_event_id:
            return None

        # Filter remapped_events to only include target_event_id values
        valid_codes = set(target_event_id.values())
        mask = np.isin(remapped_events[:, 2], list(valid_codes))
        filtered_events = remapped_events[mask]

        if len(filtered_events) == 0:
            return None

        epochs = mne.Epochs(
            raw,
            events=filtered_events,
            event_id=target_event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            preload=True,
            verbose=False,
        )

        return epochs

    def _get_subject_cache_path(self, subject_id: str) -> Optional[Path]:
        """Get cache file path for a subject."""
        if self.cache_dir is None:
            return None
        sfreq_str = f"{int(self.resample_freq)}Hz" if self.resample_freq is not None else "160Hz"
        return self.cache_dir / f"{subject_id}_{self.task}_{sfreq_str}.npz"

    def load_subject(self, subject_id: Union[int, str]) -> Dict[str, Any]:
        """
        Load, preprocess, and epoch all configured runs for a single subject.

        Parameters
        ----------
        subject_id : Union[int, str]
            Subject identifier.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing 'X', 'y', 'run_ids', 'ch_names', 'sfreq'.
        """
        sub_str = format_subject_id(subject_id)
        cache_path = self._get_subject_cache_path(sub_str)

        # 1. Attempt loading from cache
        if self.use_cache and cache_path is not None and cache_path.exists():
            try:
                loaded = np.load(cache_path, allow_pickle=True)
                return {
                    "X": loaded["X"],
                    "y": loaded["y"],
                    "run_ids": loaded["run_ids"],
                    "ch_names": list(loaded["ch_names"]),
                    "sfreq": float(loaded["sfreq"]),
                }
            except Exception as e:
                logger.warning(f"Failed to load cache for {sub_str}: {e}. Reloading raw data.")

        # 2. Process raw EDF runs
        trial_data_list: List[np.ndarray] = []
        trial_labels_list: List[int] = []
        trial_runs_list: List[int] = []
        last_sfreq = self.resample_freq or 160.0

        for r in self.run_ids:
            try:
                raw = self.load_raw(sub_str, r)
                raw = self.preprocess(raw)
                epochs = self.create_epochs(raw, r)

                if epochs is None or len(epochs) == 0:
                    continue

                last_sfreq = epochs.info["sfreq"]
                data = epochs.get_data()  # (n_epochs, n_channels, n_times)
                event_codes = epochs.events[:, 2]

                # Map event codes to task integer labels
                inv_event_id = {v: k for k, v in epochs.event_id.items()}
                for i in range(len(epochs)):
                    class_name = inv_event_id[event_codes[i]]
                    label_idx = self.class_to_idx[class_name]
                    trial_data_list.append(data[i])
                    trial_labels_list.append(label_idx)
                    trial_runs_list.append(parse_run_number(r))
            except Exception as e:
                logger.warning(f"Error processing subject {sub_str} run {r}: {e}")

        if len(trial_data_list) == 0:
            raise RuntimeError(f"No valid trials extracted for subject {sub_str} under task '{self.task}'.")

        X = np.stack(trial_data_list, axis=0).astype(np.float32)
        y = np.array(trial_labels_list, dtype=np.int64)
        run_ids = np.array(trial_runs_list, dtype=np.int64)

        # 3. Save to cache
        if self.use_cache and cache_path is not None:
            try:
                np.savez_compressed(
                    cache_path,
                    X=X,
                    y=y,
                    run_ids=run_ids,
                    ch_names=np.array(self.ch_names or []),
                    sfreq=last_sfreq,
                )
            except Exception as e:
                logger.warning(f"Failed to write cache for {sub_str}: {e}")

        return {
            "X": X,
            "y": y,
            "run_ids": run_ids,
            "ch_names": self.ch_names,
            "sfreq": last_sfreq,
        }

    def load_all_subjects(self, verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Load and preprocess all configured subjects.

        Parameters
        ----------
        verbose : bool, default=True
            Whether to log loading progress.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping subject ID to processed data dictionary.
        """
        if self._loaded_data is not None:
            return self._loaded_data

        data_dict = {}
        total = len(self.subject_ids)
        for i, sub_id in enumerate(self.subject_ids, 1):
            if verbose and (i % 10 == 0 or i == 1 or i == total):
                logger.info(f"Loading subject [{i}/{total}]: {sub_id}")
            data_dict[sub_id] = self.load_subject(sub_id)

        self._loaded_data = data_dict
        return data_dict

    def to_pytorch_dataset(self, transform: Optional[Callable] = None) -> EEGDataset:
        """
        Convert loaded data into a PyTorch EEGDataset with trial-level z-score normalization.

        Parameters
        ----------
        transform : Optional[Callable], default=None
            Optional augmentation pipeline applied on-the-fly.

        Returns
        -------
        EEGDataset
            PyTorch Dataset ready for training.
        """
        all_data = self.load_all_subjects(verbose=False)

        all_X: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        all_subs: List[str] = []
        all_runs: List[np.ndarray] = []

        for sub_id, sub_data in all_data.items():
            X = sub_data["X"]  # (N_trials, C, T)
            y = sub_data["y"]
            runs = sub_data["run_ids"]

            # Per-channel per-trial z-score normalization
            if self.normalize == "zscore":
                mean = np.mean(X, axis=-1, keepdims=True)
                std = np.std(X, axis=-1, keepdims=True)
                X = (X - mean) / (std + 1e-8)

            all_X.append(X)
            all_y.append(y)
            all_subs.extend([sub_id] * len(y))
            all_runs.append(runs)

        X_cat = np.concatenate(all_X, axis=0)
        y_cat = np.concatenate(all_y, axis=0)
        runs_cat = np.concatenate(all_runs, axis=0)
        subs_arr = np.array(all_subs)

        return EEGDataset(
            X=X_cat,
            y=y_cat,
            subject_ids=subs_arr,
            run_ids=runs_cat,
            transform=transform,
            ch_names=self.ch_names,
            electrode_positions=self.electrode_positions,
            class_to_idx=self.class_to_idx,
            idx_to_class=self.idx_to_class,
        )

    def get_subject_split(
        self,
        test_subject_id: Union[int, str],
        transform_train: Optional[Callable] = None,
        transform_test: Optional[Callable] = None,
    ) -> Tuple[EEGDataset, EEGDataset]:
        """
        Leave-One-Subject-Out (LOSO) split.

        Parameters
        ----------
        test_subject_id : Union[int, str]
            Held-out subject identifier for testing.
        transform_train : Optional[Callable], default=None
            Transform for training set.
        transform_test : Optional[Callable], default=None
            Transform for testing set.

        Returns
        -------
        Tuple[EEGDataset, EEGDataset]
            (train_dataset, test_dataset)
        """
        test_sub_str = format_subject_id(test_subject_id)
        full_ds = self.to_pytorch_dataset()

        sub_mask = full_ds.subject_ids == test_sub_str
        test_idx = np.where(sub_mask)[0]
        train_idx = np.where(~sub_mask)[0]

        if len(test_idx) == 0:
            raise ValueError(f"Test subject {test_sub_str} has no trials in the dataset.")

        train_ds = full_ds.subset(train_idx, transform=transform_train)
        test_ds = full_ds.subset(test_idx, transform=transform_test)

        return train_ds, test_ds

    def get_kfold_splits(
        self,
        n_folds: int = 5,
        subject_id: Optional[Union[int, str]] = None,
        seed: int = 42,
        transform_train: Optional[Callable] = None,
        transform_val: Optional[Callable] = None,
    ) -> List[Tuple[EEGDataset, EEGDataset]]:
        """
        Generate stratified k-fold splits within-subject or across subjects.

        Parameters
        ----------
        n_folds : int, default=5
            Number of folds.
        subject_id : Optional[Union[int, str]], default=None
            If provided, performs within-subject stratified k-fold for that subject.
        seed : int, default=42
            Random seed for shuffling.
        transform_train : Optional[Callable], default=None
            Transform applied to training folds.
        transform_val : Optional[Callable], default=None
            Transform applied to validation folds.

        Returns
        -------
        List[Tuple[EEGDataset, EEGDataset]]
            List of (train_dataset, val_dataset) pairs for each fold.
        """
        full_ds = self.to_pytorch_dataset()

        if subject_id is not None:
            sub_str = format_subject_id(subject_id)
            sub_mask = full_ds.subject_ids == sub_str
            indices = np.where(sub_mask)[0]
            if len(indices) == 0:
                raise ValueError(f"Subject {sub_str} not found in dataset.")
            base_ds = full_ds.subset(indices)
        else:
            base_ds = full_ds

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = []

        labels = base_ds.y.numpy()
        for train_idx, val_idx in skf.split(np.zeros(len(base_ds)), labels):
            train_ds = base_ds.subset(train_idx, transform=transform_train)
            val_ds = base_ds.subset(val_idx, transform=transform_val)
            splits.append((train_ds, val_ds))

        return splits
