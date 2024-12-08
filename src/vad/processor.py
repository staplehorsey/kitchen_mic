"""Voice Activity Detection processor using Silero VAD.

This module handles real-time voice activity detection using the Silero VAD model.
It processes 16kHz mono audio and provides configurable parameters for detection.
"""

import logging
import time
from pathlib import Path
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
import threading

import numpy as np
import torch
from silero_vad import load_silero_vad, VADIterator

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""
    threshold: float = 0.3  # Speech probability threshold (lowered for better sensitivity)
    window_size_samples: int = 512  # Window size in samples (must be 512 for 16kHz)
    min_speech_duration_ms: int = 100  # Minimum speech duration (lowered to catch quick speech)
    min_silence_duration_ms: int = 50  # Minimum silence duration (lowered to be more responsive)
    speech_pad_ms: int = 50  # Padding around speech segments (increased to smooth detection)
    sample_rate: int = 16000  # Expected sample rate

class VADProcessor:
    """Processes audio chunks for voice activity detection."""
    
    def __init__(
        self,
        config: Optional[VADConfig] = None,
        on_speech_start: Optional[Callable[[float], None]] = None,
        on_speech_end: Optional[Callable[[float, float], None]] = None
    ):
        """Initialize VAD processor.
        
        Args:
            config: VAD configuration parameters
            on_speech_start: Callback when speech starts (timestamp)
            on_speech_end: Callback when speech ends (start_time, duration)
        """
        self.config = config or VADConfig()
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        
        # State tracking
        self.speech_started = False
        self.speech_start_time = 0.0
        self.last_speech_end_time = 0.0
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Load model and create iterator
        logger.info("Loading Silero VAD model...")
        torch.set_num_threads(1)  # Optimize for real-time processing
        self.model = load_silero_vad()
        self.vad_iterator = VADIterator(
            model=self.model,
            threshold=self.config.threshold,
            sampling_rate=self.config.sample_rate,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms
        )
        
        logger.info("VAD processor initialized")
        
    def _get_speech_prob(self, audio_chunk: np.ndarray) -> float:
        """Get speech probability for audio chunk.
        
        Args:
            audio_chunk: Audio samples (16kHz, mono, float32)
            
        Returns:
            Speech probability (0.0 to 1.0)
        """
        # Process audio in 512-sample chunks
        if len(audio_chunk) != self.config.window_size_samples:
            # Pad or truncate to exactly 512 samples
            if len(audio_chunk) < self.config.window_size_samples:
                audio_chunk = np.pad(
                    audio_chunk,
                    (0, self.config.window_size_samples - len(audio_chunk))
                )
            else:
                audio_chunk = audio_chunk[:self.config.window_size_samples]
            
        # Ensure correct type
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
            
        # Convert to tensor and get probability
        try:
            tensor = torch.from_numpy(audio_chunk)
            with torch.no_grad():
                prob = self.model(tensor, self.config.sample_rate).item()
            return prob
        except Exception as e:
            logger.error(f"Error getting speech probability: {e}")
            return 0.0

    def process_audio(self, audio_chunk: np.ndarray) -> List[Tuple[float, float]]:
        """Process audio chunk and detect speech segments.
        
        Args:
            audio_chunk: Audio samples (16kHz, mono, float32)
            
        Returns:
            List of (start_time, duration) tuples for detected speech
        """
        speech_segments = []
        current_time = time.time()
        
        with self.buffer_lock:
            # Add to buffer
            self.audio_buffer.extend(audio_chunk.tolist())
            
            # Process complete windows
            while len(self.audio_buffer) >= self.config.window_size_samples:
                # Get window
                window = np.array(
                    self.audio_buffer[:self.config.window_size_samples],
                    dtype=np.float32
                )
                self.audio_buffer = self.audio_buffer[self.config.window_size_samples:]
                
                # Get speech probability using iterator for better smoothing
                tensor = torch.from_numpy(window)
                speech_dict = self.vad_iterator(tensor, return_seconds=True)
                
                # Handle speech events if present
                if speech_dict is not None:
                    if speech_dict.get('start') is not None:
                        self.speech_started = True
                        self.speech_start_time = current_time
                        if self.on_speech_start:
                            self.on_speech_start(current_time)
                    
                    if speech_dict.get('end') is not None:
                        if self.speech_started:
                            speech_duration = current_time - self.speech_start_time
                            if speech_duration >= self.config.min_speech_duration_ms / 1000:
                                speech_segments.append(
                                    (self.speech_start_time, speech_duration)
                                )
                                if self.on_speech_end:
                                    self.on_speech_end(
                                        self.speech_start_time,
                                        speech_duration
                                    )
                            self.speech_started = False
                            self.last_speech_end_time = current_time
                
        return speech_segments
        
    def reset(self) -> None:
        """Reset VAD state."""
        with self.buffer_lock:
            self.speech_started = False
            self.speech_start_time = 0.0
            self.last_speech_end_time = 0.0
            self.audio_buffer.clear()
            self.vad_iterator.reset_states()
