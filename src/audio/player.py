"""Audio playback utilities."""

import logging
import collections
from typing import Optional

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)

class AudioPlayer:
    """Real-time audio playback using sounddevice."""
    
    def __init__(self, sample_rate: int = 44100):
        """Initialize audio player.
        
        Args:
            sample_rate: Sample rate for playback
        """
        self.sample_rate = sample_rate
        self.is_muted = False
        self.audio_buffer = collections.deque(maxlen=100)  # Buffer ~3 seconds
        
        def audio_callback(outdata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags) -> None:
            """Process audio data from stream."""
            if status:
                logger.warning(f"Audio output status: {status}")
            
            if self.is_muted or not self.audio_buffer:
                outdata.fill(0)
                return
            
            try:
                # Get audio data from buffer
                data = self.audio_buffer.popleft()
                if len(data) < len(outdata):
                    # Pad with zeros if not enough data
                    data = np.pad(data, (0, len(outdata) - len(data)))
                elif len(data) > len(outdata):
                    # Truncate if too much data
                    data = data[:len(outdata)]
                outdata[:] = data.reshape(-1, 1)
            except Exception as e:
                logger.error(f"Error in audio callback: {e}")
                outdata.fill(0)
        
        # Open stream with callback
        self.stream = sd.OutputStream(
            channels=1,
            samplerate=sample_rate,
            callback=audio_callback
        )
        self.stream.start()
    
    def play(self, audio: np.ndarray) -> None:
        """Add audio data to playback buffer.
        
        Args:
            audio: Audio samples to play
        """
        try:
            self.audio_buffer.append(audio)
        except Exception as e:
            logger.error(f"Error adding audio to playback buffer: {e}")
    
    def toggle_mute(self) -> None:
        """Toggle audio mute state."""
        self.is_muted = not self.is_muted
        logger.info(f"Audio {'muted' if self.is_muted else 'unmuted'}")
    
    def stop(self) -> None:
        """Stop audio playback."""
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
