"""Voice Activity Detection visualization."""

import collections
import time
from typing import Optional, Tuple

import numpy as np

from src.vad.processor import VADState
from .base import BaseVisualizer

class VADVisualizer(BaseVisualizer):
    """Real-time visualization of Voice Activity Detection.
    
    Features:
    - Audio waveform display
    - Speech probability plot
    - Speech state indicators
    - Time-synchronized display
    """
    
    def __init__(
        self,
        window_size: int = 2000,
        samples_per_second: int = 50,  # Display update rate
        time_window: float = 10.0,  # Show 10 seconds of data
        title: str = "Voice Activity Detection"
    ):
        """Initialize VAD visualizer.
        
        Args:
            window_size: Number of samples to show
            samples_per_second: How many samples to show per second
            time_window: Time window to show in seconds
            title: Plot title
        """
        # Initialize base with 2 subplots (audio + VAD)
        super().__init__(
            num_subplots=2,
            title=title,
            window_size=window_size
        )
        
        # Data storage
        self.audio_data = collections.deque(maxlen=window_size)
        self.speech_probs = collections.deque(maxlen=window_size)
        self.is_speech = collections.deque(maxlen=window_size)
        self.conversation_state = collections.deque(maxlen=window_size)
        self.samples_per_second = samples_per_second
        self.time_window = time_window
        self.start_time = time.time()
        
        # Initialize with zeros
        current_time = time.time()
        for _ in range(window_size):
            self.audio_data.append(0.0)
            self.speech_probs.append((current_time, 0.0))
            self.is_speech.append((current_time, 0.0))
            self.conversation_state.append((current_time, 0.0))
        
        # Set up audio plot
        self.line_audio, = self.axes[0].plot(
            [], [], 'cyan',
            label='Audio',
            linewidth=1
        )
        self.configure_axis(
            self.axes[0],
            "Audio Waveform",
            "Amplitude",
            (-1, 1)
        )
        
        # Set up VAD plot
        self.line_prob, = self.axes[1].plot(
            [], [], 'lime',
            label='Speech Prob',
            linewidth=2
        )
        self.line_conv_prob, = self.axes[1].plot(
            [], [], 'yellow',
            label='Conversation',
            linewidth=2,
            alpha=0.7
        )
        self.line_speech, = self.axes[1].plot(
            [], [], 'red',
            label='Speech',
            linewidth=1,
            alpha=0.7
        )
        self.configure_axis(
            self.axes[1],
            "Voice Activity Detection",
            "Probability",
            (-0.1, 1.1)
        )
    
    def add_data(
        self,
        audio_chunk: np.ndarray,
        speech_prob: float,
        is_speech: bool,
        is_conversation: bool
    ) -> None:
        """Add new VAD data to visualization.
        
        Args:
            audio_chunk: Audio samples
            speech_prob: Speech probability [0-1]
            is_speech: Speech detection state
            is_conversation: Conversation detection state
        """
        with self.data_lock:
            # Add audio samples
            for sample in audio_chunk:
                self.audio_data.append(sample)
            
            # Add VAD data with current timestamp
            current_time = time.time()
            self.speech_probs.append((current_time, speech_prob))
            self.is_speech.append((current_time, float(is_speech)))
            self.conversation_state.append((current_time, float(is_conversation)))
    
    def _update(self, frame) -> Tuple:
        """Update animation frame.
        
        Args:
            frame: Animation frame number
        
        Returns:
            Tuple of artists that were modified
        """
        with self.data_lock:
            # Calculate current time window
            current_time = time.time()
            window_start = current_time - self.time_window
            window_end = current_time
            
            # Filter data to current time window
            if len(self.speech_probs) > 0:
                # Get all data points
                prob_times, probs = zip(*self.speech_probs)
                speech_times, speech = zip(*self.is_speech)
                conv_times, conv = zip(*self.conversation_state)
                
                # Convert to numpy arrays for easier filtering
                prob_times = np.array(prob_times)
                probs = np.array(probs)
                speech_times = np.array(speech_times)
                speech = np.array(speech)
                conv_times = np.array(conv_times)
                conv = np.array(conv)
                
                # Filter to current window
                prob_mask = (prob_times >= window_start) & (prob_times <= window_end)
                speech_mask = (speech_times >= window_start) & (speech_times <= window_end)
                conv_mask = (conv_times >= window_start) & (conv_times <= window_end)
                
                # Update probability plot
                self.line_prob.set_data(
                    prob_times[prob_mask] - window_start,  # Normalize to [0, time_window]
                    probs[prob_mask]
                )
                
                # Update conversation state
                self.line_conv_prob.set_data(
                    conv_times[conv_mask] - window_start,
                    conv[conv_mask]
                )
                
                # Update speech state plot
                self.line_speech.set_data(
                    speech_times[speech_mask] - window_start,
                    speech[speech_mask]
                )
            
            # Update audio waveform (keep this simple for now)
            times = np.linspace(0, self.time_window, len(self.audio_data))
            self.line_audio.set_data(times, list(self.audio_data))
            
            # Update axis limits to show scrolling window
            self.axes[0].set_xlim(0, self.time_window)
            self.axes[1].set_xlim(0, self.time_window)
        
        return self.line_audio, self.line_prob, self.line_conv_prob, self.line_speech
