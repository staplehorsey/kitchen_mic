"""Audio visualization components."""

import time
import collections
from typing import Optional, Tuple

import numpy as np

from .base import BaseVisualizer

class AudioVisualizer(BaseVisualizer):
    """Real-time audio waveform visualization.
    
    Features:
    - Scrolling waveform display
    - Time-based x-axis
    - Configurable window size and display duration
    """
    
    def __init__(
        self,
        window_size: int = 2000,
        samples_per_second: int = 50,  # Display update rate
        title: str = "Audio Waveform"
    ):
        """Initialize audio visualizer.
        
        Args:
            window_size: Number of samples in the data buffer
            samples_per_second: How many samples to show per second of display
            title: Plot title
        """
        super().__init__(
            num_subplots=1,
            title=title,
            window_size=window_size
        )
        
        # Data storage
        self.audio_data = collections.deque(maxlen=window_size)
        self.start_time = time.time()
        self.samples_per_second = samples_per_second
        
        # Initialize with zeros
        for _ in range(window_size):
            self.audio_data.append(0)
        
        # Set up audio plot
        self.line_audio, = self.axes[0].plot(
            [], [], 'cyan',
            label='Audio',
            linewidth=1
        )
        
        # Configure axis
        self.configure_axis(
            self.axes[0],
            "Audio Waveform",
            "Amplitude",
            (-1, 1)
        )
    
    def add_audio(self, audio_chunk: np.ndarray) -> None:
        """Add new audio data to visualization.
        
        Args:
            audio_chunk: Audio samples to add
        """
        with self.data_lock:
            for sample in audio_chunk:
                self.audio_data.append(sample)
    
    def _update(self, frame) -> Tuple:
        """Update animation frame.
        
        Args:
            frame: Animation frame number
        
        Returns:
            Tuple containing the audio line artist
        """
        with self.data_lock:
            # Calculate relative time and time windows
            relative_time = time.time() - self.start_time
            times = np.linspace(
                relative_time - self.window_size / self.samples_per_second,
                relative_time,
                len(self.audio_data)
            )
            
            # Update line data
            self.line_audio.set_data(times, list(self.audio_data))
            
            # Update x-axis limits
            self.axes[0].set_xlim(times[0], times[-1])
        
        return (self.line_audio,)
