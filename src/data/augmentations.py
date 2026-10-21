"""
EEG Data Augmentations for Motor Imagery Decoding.

This module provides domain-specific data augmentation techniques tailored for
multichannel electroencephalogram (EEG) signals, supporting both PyTorch tensors
and NumPy arrays.
"""

from typing import List, Optional, Sequence, Tuple, Union
import math
import numpy as np
import scipy.signal
import torch


def _to_tensor(x: Union[np.ndarray, torch.Tensor]) -> Tuple[torch.Tensor, bool]:
    """
    Convert input to PyTorch tensor if necessary and record whether input was numpy.

    Parameters
    ----------
    x : Union[np.ndarray, torch.Tensor]
        Input EEG signal of shape (C, T) or (B, C, T).

    Returns
    -------
    Tuple[torch.Tensor, bool]
        Converted tensor and boolean flag indicating if original input was NumPy.
    """
    is_numpy = isinstance(x, np.ndarray)
    if is_numpy:
        tensor = torch.from_numpy(x).float()
    elif isinstance(x, torch.Tensor):
        tensor = x.clone().float()
    else:
        raise TypeError(f"Unsupported data type: {type(x)}. Expected numpy.ndarray or torch.Tensor.")
    return tensor, is_numpy


def _to_output(tensor: torch.Tensor, was_numpy: bool) -> Union[np.ndarray, torch.Tensor]:
    """Convert tensor back to NumPy array if the original input was NumPy."""
    if was_numpy:
        return tensor.cpu().numpy()
    return tensor


class TimeShift:
    """
    Randomly shift EEG time series temporally along the time axis.

    Simulates temporal latency jittering in event-related synchronization/desynchronization (ERS/ERD).

    Parameters
    ----------
    max_shift_ms : float, default=50.0
        Maximum time shift in milliseconds (+/- max_shift_ms).
    sfreq : float, default=250.0
        Sampling frequency in Hz.
    mode : str, default='circular'
        Shift mode: 'circular' (wrap around using roll) or 'zero_pad' (pad with zeros).
    p : float, default=0.5
        Probability of applying the augmentation.
    """

    def __init__(
        self,
        max_shift_ms: float = 50.0,
        sfreq: float = 250.0,
        mode: str = "circular",
        p: float = 0.5,
    ) -> None:
        self.max_shift_ms = float(max_shift_ms)
        self.sfreq = float(sfreq)
        self.max_shift_samples = int(math.ceil((self.max_shift_ms / 1000.0) * self.sfreq))
        self.mode = mode.lower()
        self.p = float(p)

        if self.mode not in ("circular", "zero_pad"):
            raise ValueError(f"Unsupported mode: {mode}. Choose 'circular' or 'zero_pad'.")

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() > self.p or self.max_shift_samples <= 0:
            return x

        tensor, was_numpy = _to_tensor(x)
        shift = int(np.random.randint(-self.max_shift_samples, self.max_shift_samples + 1))
        if shift == 0:
            return x

        time_dim = -1
        t_len = tensor.shape[time_dim]

        if self.mode == "circular":
            shifted = torch.roll(tensor, shifts=shift, dims=time_dim)
        elif self.mode == "zero_pad":
            shifted = torch.zeros_like(tensor)
            if shift > 0:
                # Shift right (positive delay)
                shifted[..., shift:] = tensor[..., : t_len - shift]
            else:
                # Shift left (negative delay)
                shifted[..., : t_len + shift] = tensor[..., -shift:]
        else:
            shifted = tensor

        return _to_output(shifted, was_numpy)

    def __repr__(self) -> str:
        return (
            f"TimeShift(max_shift_ms={self.max_shift_ms}, sfreq={self.sfreq}, "
            f"mode='{self.mode}', p={self.p})"
        )


class GaussianNoise:
    """
    Add zero-mean Gaussian noise at a specified Signal-to-Noise Ratio (SNR).

    Parameters
    ----------
    snr_db : Union[float, Tuple[float, float]], default=(15.0, 30.0)
        Signal-to-Noise Ratio in decibels. If a tuple is provided, SNR is sampled uniformly
        between (snr_min, snr_max).
    p : float, default=0.5
        Probability of applying the augmentation.
    """

    def __init__(
        self,
        snr_db: Union[float, Tuple[float, float]] = (15.0, 30.0),
        p: float = 0.5,
    ) -> None:
        if isinstance(snr_db, (int, float)):
            self.snr_db_range = (float(snr_db), float(snr_db))
        else:
            self.snr_db_range = (float(snr_db[0]), float(snr_db[1]))
        self.p = float(p)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() > self.p:
            return x

        tensor, was_numpy = _to_tensor(x)
        snr = float(np.random.uniform(self.snr_db_range[0], self.snr_db_range[1]))

        # Calculate signal power per trial
        signal_power = torch.mean(tensor ** 2, dim=(-2, -1), keepdim=True)
        # Avoid division by zero
        signal_power = torch.clamp(signal_power, min=1e-8)

        # SNR_dB = 10 * log10(P_signal / P_noise) => P_noise = P_signal / (10 ** (SNR / 10))
        noise_power = signal_power / (10.0 ** (snr / 10.0))
        noise_std = torch.sqrt(noise_power)

        noise = torch.randn_like(tensor) * noise_std
        augmented = tensor + noise

        return _to_output(augmented, was_numpy)

    def __repr__(self) -> str:
        return f"GaussianNoise(snr_db={self.snr_db_range}, p={self.p})"


class ChannelDropout:
    """
    Randomly zero out channels to make model robust to missing or faulty electrodes.

    Parameters
    ----------
    p_channel : float, default=0.1
        Probability of dropping an individual channel.
    max_drop : Optional[int], default=6
        Maximum number of channels that can be dropped per trial.
    p : float, default=0.5
        Probability of applying the augmentation.
    """

    def __init__(
        self,
        p_channel: float = 0.1,
        max_drop: Optional[int] = 6,
        p: float = 0.5,
    ) -> None:
        self.p_channel = float(p_channel)
        self.max_drop = max_drop
        self.p = float(p)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() > self.p:
            return x

        tensor, was_numpy = _to_tensor(x)
        num_channels = tensor.shape[-2]

        # Determine drop mask
        drop_mask = np.random.rand(num_channels) < self.p_channel
        dropped_indices = np.where(drop_mask)[0]

        if self.max_drop is not None and len(dropped_indices) > self.max_drop:
            # Subsample dropped channels to not exceed max_drop
            dropped_indices = np.random.choice(dropped_indices, size=self.max_drop, replace=False)

        if len(dropped_indices) == 0:
            return x

        tensor[..., dropped_indices, :] = 0.0
        return _to_output(tensor, was_numpy)

    def __repr__(self) -> str:
        return f"ChannelDropout(p_channel={self.p_channel}, max_drop={self.max_drop}, p={self.p})"


class FrequencyShift:
    """
    Slight frequency jittering using analytic signal (Hilbert transform) modulation.

    The signal is converted to its analytic representation and multiplied by e^(j 2 pi Delta f t)
    before taking the real part, cleanly shifting all frequency components by Delta f without
    introducing boundary artifacts.

    Parameters
    ----------
    max_shift_hz : float, default=1.0
        Maximum frequency shift in Hz (+/- max_shift_hz).
    sfreq : float, default=250.0
        Sampling frequency in Hz.
    p : float, default=0.5
        Probability of applying the augmentation.
    """

    def __init__(
        self,
        max_shift_hz: float = 1.0,
        sfreq: float = 250.0,
        p: float = 0.5,
    ) -> None:
        self.max_shift_hz = float(max_shift_hz)
        self.sfreq = float(sfreq)
        self.p = float(p)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() > self.p or self.max_shift_hz <= 0.0:
            return x

        tensor, was_numpy = _to_tensor(x)
        delta_f = float(np.random.uniform(-self.max_shift_hz, self.max_shift_hz))
        if abs(delta_f) < 1e-4:
            return x

        time_dim = -1
        t_len = tensor.shape[time_dim]
        device = tensor.device
        dtype = tensor.dtype

        # Compute analytic signal along time dimension via FFT
        # X(f) = FFT(x)
        X = torch.fft.fft(tensor, dim=time_dim)
        h = torch.zeros(t_len, device=device, dtype=torch.float32)
        if t_len % 2 == 0:
            h[0] = 1.0
            h[t_len // 2] = 1.0
            h[1 : t_len // 2] = 2.0
        else:
            h[0] = 1.0
            h[1 : (t_len + 1) // 2] = 2.0

        # Hilbert analytic signal
        analytic = torch.fft.ifft(X * h, dim=time_dim)

        # Modulation: exp(j 2 pi delta_f t)
        t = torch.arange(t_len, device=device, dtype=torch.float32) / self.sfreq
        phase = 2.0 * math.pi * delta_f * t
        modulation = torch.polar(torch.ones_like(phase), phase).to(analytic.dtype)

        shifted_complex = analytic * modulation
        augmented = torch.real(shifted_complex).to(dtype)

        return _to_output(augmented, was_numpy)

    def __repr__(self) -> str:
        return f"FrequencyShift(max_shift_hz={self.max_shift_hz}, sfreq={self.sfreq}, p={self.p})"


class SignalScaling:
    """
    Random amplitude scaling to simulate inter-trial and inter-session amplitude variability.

    Parameters
    ----------
    scale_range : Tuple[float, float], default=(0.8, 1.2)
        Range of scaling factors [scale_min, scale_max].
    p : float, default=0.5
        Probability of applying the augmentation.
    """

    def __init__(
        self,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
    ) -> None:
        self.scale_range = (float(scale_range[0]), float(scale_range[1]))
        self.p = float(p)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() > self.p:
            return x

        scale = float(np.random.uniform(self.scale_range[0], self.scale_range[1]))
        return x * scale

    def __repr__(self) -> str:
        return f"SignalScaling(scale_range={self.scale_range}, p={self.p})"


class TimeMasking:
    """
    Randomly zero out a continuous temporal window within the trial.

    Parameters
    ----------
    max_mask_ms : float, default=100.0
        Maximum duration of masked window in milliseconds.
    sfreq : float, default=250.0
        Sampling frequency in Hz.
    p : float, default=0.3
        Probability of applying the augmentation.
    """

    def __init__(
        self,
        max_mask_ms: float = 100.0,
        sfreq: float = 250.0,
        p: float = 0.3,
    ) -> None:
        self.max_mask_ms = float(max_mask_ms)
        self.sfreq = float(sfreq)
        self.max_mask_samples = int(math.ceil((self.max_mask_ms / 1000.0) * self.sfreq))
        self.p = float(p)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() > self.p or self.max_mask_samples <= 0:
            return x

        tensor, was_numpy = _to_tensor(x)
        t_len = tensor.shape[-1]
        mask_len = int(np.random.randint(1, min(self.max_mask_samples, t_len) + 1))
        start_idx = int(np.random.randint(0, t_len - mask_len + 1))

        tensor[..., start_idx : start_idx + mask_len] = 0.0
        return _to_output(tensor, was_numpy)

    def __repr__(self) -> str:
        return f"TimeMasking(max_mask_ms={self.max_mask_ms}, sfreq={self.sfreq}, p={self.p})"


class Compose:
    """
    Composes several EEG augmentations together in sequence.

    Parameters
    ----------
    transforms : Sequence[Callable]
        List or tuple of transformation callables.
    """

    def __init__(self, transforms: Sequence) -> None:
        self.transforms = list(transforms)

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        for t in self.transforms:
            x = t(x)
        return x

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "(\n"
        for t in self.transforms:
            format_string += f"    {t},\n"
        format_string += ")"
        return format_string
