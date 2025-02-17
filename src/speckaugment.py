import numpy as np
import torch
import torchaudio
from torch.nn import functional as F

class SpecAugment:
    def __init__(self, 
                 time_shift_prob=0.8,
                 max_time_shift_percent=0.4,
                 freq_mask_prob=0.8, 
                 time_mask_prob=0.8,
                 max_freq_width=8,  # About 10% of 64 mel bands
                 max_time_width=6,  # About 10% of time steps
                 n_freq_masks=2,
                 n_time_masks=2):
        """
        Initialize SpecAugment with configurable parameters
        
        Args:
            time_shift_prob: Probability of applying time shift
            max_time_shift_percent: Maximum percentage to shift in time
            freq_mask_prob: Probability of applying frequency masking
            time_mask_prob: Probability of applying time masking
            max_freq_width: Maximum width of frequency mask
            max_time_width: Maximum width of time mask
            n_freq_masks: Number of frequency masks to apply
            n_time_masks: Number of time masks to apply
        """
        self.time_shift_prob = time_shift_prob
        self.max_time_shift_percent = max_time_shift_percent
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_prob = time_mask_prob
        self.max_freq_width = max_freq_width
        self.max_time_width = max_time_width
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def time_shift(self, waveform):
        """Apply random time shift to waveform"""
        if np.random.random() < self.time_shift_prob:
            shift_amount = int(waveform.shape[1] * self.max_time_shift_percent * np.random.uniform(-1, 1))
            return torch.roll(waveform, shifts=shift_amount, dims=1)
        return waveform

    def mask_spectrogram(self, spec, max_mask_width, n_masks, mask_dim):
        """Apply masking to spectrogram along specified dimension"""
        masked_spec = spec.clone()
        
        spec_size = spec.shape[mask_dim]
        for _ in range(n_masks):
            mask_width = np.random.randint(1, max_mask_width + 1)
            mask_start = np.random.randint(0, spec_size - mask_width + 1)
            
            if mask_dim == 1:  # Frequency masking
                masked_spec[:, mask_start:mask_start+mask_width, :] = masked_spec.mean()
            else:  # Time masking
                masked_spec[:, :, mask_start:mask_start+mask_width] = masked_spec.mean()
                
        return masked_spec

    def __call__(self, waveform):
        """
        Apply SpecAugment transformations to input waveform
        
        Args:
            waveform: Input audio waveform tensor (1, samples)
            
        Returns:
            Augmented waveform and mel spectrogram
        """
        # Apply time shift
        shifted_waveform = self.time_shift(waveform)
        
        # Convert to mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_mels=64,
            n_fft=1024,
            hop_length=500
        )(shifted_waveform)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Apply frequency masking
        if np.random.random() < self.freq_mask_prob:
            mel_spec = self.mask_spectrogram(
                mel_spec, 
                self.max_freq_width,
                self.n_freq_masks,
                mask_dim=1
            )
            
        # Apply time masking
        if np.random.random() < self.time_mask_prob:
            mel_spec = self.mask_spectrogram(
                mel_spec,
                self.max_time_width, 
                self.n_time_masks,
                mask_dim=2
            )
            
        return shifted_waveform, mel_spec

# Example usage:
"""
augmenter = SpecAugment()

# Load and preprocess audio
waveform, sample_rate = torchaudio.load("keystroke.wav")
waveform = F.pad(waveform, (0, max_length - waveform.shape[1])) # Pad to fixed length

# Apply augmentation
augmented_waveform, augmented_spec = augmenter(waveform)
"""
