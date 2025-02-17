import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
from torch.nn import functional as F

class MelSpectrogramExtractor:
    def __init__(self,
                 sample_rate: int = 44100,
                 n_mels: int = 64,
                 n_fft: int = 1024,
                 hop_length: int = 500,
                 power: float = 2.0,
                 normalized: bool = True,
                 norm: Optional[str] = 'slaney',
                 mel_scale: str = 'htk',
                 window_fn = torch.hann_window):
        """
        Initialize mel-spectrogram feature extractor for keystroke audio.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            n_mels: Number of mel filterbanks
            n_fft: Size of FFT
            hop_length: Number of samples between successive frames
            power: Power of the magnitude spectrogram
            normalized: Whether to normalize mel filterbanks
            norm: Type of normalization ('slaney' or None)
            mel_scale: Scale to use for mel conversion ('htk' or 'slaney') 
            window_fn: Window function to use for STFT
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Initialize mel spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=power,
            normalized=normalized,
            norm=norm,
            mel_scale=mel_scale,
            window_fn=window_fn
        )

    def load_audio(self, 
                   file_path: Union[str, Path], 
                   target_length: Optional[int] = None) -> torch.Tensor:
        """
        Load and preprocess audio file
        
        Args:
            file_path: Path to audio file
            target_length: Optional target length to pad/trim to
            
        Returns:
            Preprocessed waveform tensor
        """
        waveform, sr = torchaudio.load(file_path)
        
        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Convert stereo to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Pad or trim to target length if specified
        if target_length is not None:
            if waveform.shape[1] < target_length:
                waveform = F.pad(waveform, (0, target_length - waveform.shape[1]))
            else:
                waveform = waveform[:, :target_length]
                
        return waveform

    def extract_features(self, 
                        waveform: torch.Tensor,
                        normalize: bool = True,
                        add_delta: bool = False) -> torch.Tensor:
        """
        Extract mel-spectrogram features from waveform
        
        Args:
            waveform: Input waveform tensor
            normalize: Whether to normalize features
            add_delta: Whether to compute and append delta features
            
        Returns:
            Mel-spectrogram features tensor
        """
        # Convert to mel spectrogram
        mel_spec = self.mel_transform(waveform)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Normalize if requested
        if normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
            
        # Add delta features if requested
        if add_delta:
            delta = torchaudio.transforms.ComputeDeltas()(mel_spec)
            delta2 = torchaudio.transforms.ComputeDeltas()(delta)
            mel_spec = torch.cat([mel_spec, delta, delta2], dim=1)
            
        return mel_spec

    def process_keystroke(self,
                         input_data: Union[str, Path, torch.Tensor],
                         target_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process single keystroke audio file or waveform
        
        Args:
            input_data: Audio file path or waveform tensor
            target_length: Optional target length to pad/trim to
            
        Returns:
            Tuple of (waveform, mel-spectrogram features)
        """
        # Load audio if path provided
        if isinstance(input_data, (str, Path)):
            waveform = self.load_audio(input_data, target_length)
        else:
            waveform = input_data
            
        # Extract features
        features = self.extract_features(waveform)
        
        return waveform, features

    def process_batch(self,
                     file_paths: list,
                     target_length: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process batch of keystroke audio files
        
        Args:
            file_paths: List of audio file paths
            target_length: Optional target length to pad/trim to
            
        Returns:
            Tuple of (batch waveforms, batch features)
        """
        waveforms = []
        features = []
        
        for file_path in file_paths:
            wave, feat = self.process_keystroke(file_path, target_length)
            waveforms.append(wave)
            features.append(feat)
            
        return torch.stack(waveforms), torch.stack(features)

# Example usage:
"""
# Initialize feature extractor
extractor = MelSpectrogramExtractor()

# Process single keystroke
waveform, features = extractor.process_keystroke("keystroke.wav")

# Process batch of keystrokes
file_paths = ["key1.wav", "key2.wav", "key3.wav"]
batch_waveforms, batch_features = extractor.process_batch(file_paths)
"""
