import numpy as np
from scipy.io import wavfile
from scipy.fft import fft
import librosa

class KeystrokeIsolator:
    def __init__(self, sample_rate=44100, window_size=1024, hop_length=512):
        """
        Initialize the keystroke isolator
        
        Args:
            sample_rate (int): Audio sample rate (default: 44100 Hz)
            window_size (int): FFT window size
            hop_length (int): Number of samples between successive FFT windows
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length
        
    def load_audio(self, file_path):
        """
        Load an audio file and convert to mono if necessary
        
        Args:
            file_path (str): Path to the audio file
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        # Load audio file
        audio_data, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return audio_data, sr
    
    def calculate_energy(self, audio_data):
        """
        Calculate the energy of the audio signal using FFT
        
        Args:
            audio_data (np.array): Audio time series
            
        Returns:
            np.array: Energy values
        """
        # Calculate spectrogram
        D = librosa.stft(audio_data, 
                        n_fft=self.window_size,
                        hop_length=self.hop_length)
        
        # Calculate energy
        energy = np.sum(np.abs(D)**2, axis=0)
        return energy
    
    def find_keystrokes(self, audio_data, threshold_factor=0.3, min_distance=0.1):
        """
        Identify keystroke positions in the audio
        
        Args:
            audio_data (np.array): Audio time series
            threshold_factor (float): Factor to multiply mean energy for threshold
            min_distance (float): Minimum time (seconds) between keystrokes
            
        Returns:
            list: List of keystroke start positions (in samples)
        """
        # Calculate energy
        energy = self.calculate_energy(audio_data)
        
        # Calculate threshold
        threshold = np.mean(energy) * threshold_factor
        
        # Find peaks above threshold
        peaks = []
        min_samples = int(min_distance * self.sample_rate / self.hop_length)
        
        for i in range(1, len(energy) - 1):
            if (energy[i] > threshold and 
                energy[i] > energy[i-1] and 
                energy[i] > energy[i+1]):
                if not peaks or (i - peaks[-1]) >= min_samples:
                    peaks.append(i)
        
        # Convert peak positions to sample positions
        keystroke_positions = [p * self.hop_length for p in peaks]
        
        return keystroke_positions
    
    def extract_keystrokes(self, audio_data, positions, duration=0.33):
        """
        Extract fixed-length keystroke segments from the audio
        
        Args:
            audio_data (np.array): Audio time series
            positions (list): List of keystroke start positions
            duration (float): Duration of each keystroke segment in seconds
            
        Returns:
            list: List of keystroke audio segments
        """
        segment_length = int(duration * self.sample_rate)
        keystrokes = []
        
        for pos in positions:
            if pos + segment_length <= len(audio_data):
                segment = audio_data[pos:pos + segment_length]
                keystrokes.append(segment)
                
        return keystrokes

# Example usage
if __name__ == "__main__":
    # Initialize isolator
    isolator = KeystrokeIsolator()
    
    # Load audio file
    audio_data, sr = isolator.load_audio("keyboard_recording.wav")
    
    # Find keystroke positions
    positions = isolator.find_keystrokes(audio_data)
    
    # Extract keystroke segments
    keystrokes = isolator.extract_keystrokes(audio_data, positions)
    
    print(f"Found {len(keystrokes)} keystrokes in the recording")
