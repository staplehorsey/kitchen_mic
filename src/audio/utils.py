"""Audio utilities for Kitchen Mic."""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

@dataclass(frozen=True)
class TimeRange:
    """Represents a time range with start and end times.
    
    Attributes:
        start: Start time in seconds
        end: End time in seconds
    """
    start: float
    end: float
    
    def contains(self, time: float) -> bool:
        """Check if a time point is within this range.
        
        Args:
            time: Time point to check
        
        Returns:
            True if time is within range
        """
        return self.start <= time <= self.end
    
    @property
    def duration(self) -> float:
        """Get duration of time range in seconds."""
        return self.end - self.start

def chunk_audio(
    audio_data: np.ndarray,
    chunk_size: int,
    pad: bool = True
) -> List[np.ndarray]:
    """Split audio data into chunks.
    
    Args:
        audio_data: Audio data to split
        chunk_size: Size of each chunk in samples
        pad: If True, pad last chunk with zeros if incomplete
    
    Returns:
        List of audio chunks
    """
    chunks = []
    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i + chunk_size]
        if pad and len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        chunks.append(chunk)
    return chunks

def resample_audio(
    audio_data: np.ndarray,
    orig_rate: int,
    target_rate: int
) -> Tuple[np.ndarray, float]:
    """Resample audio data to a new sample rate.
    
    Args:
        audio_data: Audio data to resample
        orig_rate: Original sample rate in Hz
        target_rate: Target sample rate in Hz
    
    Returns:
        Tuple of (resampled audio, duration in seconds)
    """
    import librosa
    resampled = librosa.resample(
        audio_data,
        orig_sr=orig_rate,
        target_sr=target_rate
    )
    duration = len(resampled) / target_rate
    return resampled, duration
