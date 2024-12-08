"""Conversation detector module.

This module provides conversation detection by combining audio capture and VAD.
It outputs ConversationMessage objects containing audio and metadata.
"""

import logging
import threading
from typing import Optional, Dict, Protocol, Callable
from collections import deque
import numpy as np
import uuid

from ..messages import ConversationMessage

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class AudioProcessor(Protocol):
    """Protocol for audio processors."""
    def add_callback(self, callback: Callable[[float, np.ndarray, np.ndarray], None]) -> None:
        """Add callback for audio data (timestamp, original, downsampled)."""
        ...
    
    def remove_callback(self, callback: Callable[[float, np.ndarray, np.ndarray], None]) -> None:
        """Remove callback for audio data."""
        ...
    
    @property
    def sample_rate(self) -> int:
        """Get sample rate."""
        ...

class VADProcessor(Protocol):
    """Protocol for VAD processors."""
    def add_audio(self, audio: np.ndarray) -> None:
        """Add audio for VAD processing."""
        ...
    
    def get_state(self) -> any:  # Returns VADState but avoid circular import
        """Get current VAD state."""
        ...
    
    def stop(self) -> None:
        """Stop VAD processing."""
        ...

class ConversationDetector:
    """Detects conversations using audio capture and VAD."""
    
    def __init__(
        self,
        audio_processor: AudioProcessor,
        vad_processor: VADProcessor,
        buffer_duration_sec: float = 30.0,
        pre_speech_sec: float = 3.0,  # Capture 3s before first speech
        on_conversation: Optional[Callable[[ConversationMessage], None]] = None
    ):
        """Initialize conversation detector.
        
        Args:
            audio_processor: Any audio processor that provides callbacks
            vad_processor: Any VAD processor that can process audio
            buffer_duration_sec: Duration to buffer audio before conversation (default: 30s)
            pre_speech_sec: Seconds of audio to keep before first speech (default: 3s)
            on_conversation: Optional callback for detected conversations
        """
        self.audio_processor = audio_processor
        self.vad_processor = vad_processor
        self.on_conversation = on_conversation
        
        # Calculate buffer sizes
        self.buffer_duration = buffer_duration_sec
        self.pre_speech_duration = pre_speech_sec
        samples_per_sec = self.audio_processor.sample_rate
        self.buffer_samples = int(buffer_duration_sec * samples_per_sec)
        
        # Ring buffer for pre-conversation audio
        self._audio_buffer = deque(maxlen=self.buffer_samples)
        
        # Conversation state
        self._lock = threading.Lock()
        self._running = False
        self._current_audio = []  # Only grows during active conversation
        self._conversation_start = None
        
        logger.info(f"Conversation detector initialized with {pre_speech_sec}s pre-speech buffer")
    
    def _handle_audio(self, timestamp: float, original_chunk: np.ndarray, downsampled_chunk: np.ndarray) -> None:
        """Handle incoming audio data from capture.
        
        Args:
            timestamp: Current wall clock time
            original_chunk: Original audio data (44kHz)
            downsampled_chunk: Downsampled audio data (16kHz)
        """
        # Process through VAD
        self.vad_processor.add_audio(downsampled_chunk)
        vad_state = self.vad_processor.get_state()
        
        with self._lock:
            # Add to ring buffer if no active conversation
            if not self._conversation_start:
                for sample in original_chunk:
                    self._audio_buffer.append(sample)
            else:
                # Add to growing conversation buffer
                self._current_audio.extend(original_chunk)
            
            # Handle conversation state
            self._handle_conversation_state(vad_state, timestamp)
    
    def _handle_conversation_state(self, vad_state: any, current_time: float) -> None:
        """Handle VAD state changes and manage conversations.
        
        Args:
            vad_state: Current VAD state
            current_time: Current wall clock time
        """
        # Start new conversation when VAD detects enough speech
        if vad_state.is_conversation and not self._conversation_start:
            logger.info("Starting new conversation")
            self._conversation_start = current_time
            
            # Calculate how many samples to keep from buffer based on first speech
            if vad_state.first_speech_time is not None:
                # Keep pre_speech_duration seconds before first speech
                buffer_start_time = vad_state.first_speech_time - self.pre_speech_duration
                samples_to_keep = int((current_time - buffer_start_time) * self.audio_processor.sample_rate)
                
                buffer_list = list(self._audio_buffer)
                if samples_to_keep < len(buffer_list):
                    self._current_audio = buffer_list[-samples_to_keep:]
                else:
                    self._current_audio = buffer_list
                    
                logger.debug(f"Captured {len(self._current_audio)/self.audio_processor.sample_rate:.1f}s of audio including {self.pre_speech_duration}s before first speech at {vad_state.first_speech_time}")
            else:
                # Fallback if no first_speech_time
                self._current_audio = list(self._audio_buffer)
                logger.debug(f"No first speech time, using entire buffer ({len(self._current_audio)} samples)")
        
        # End conversation when VAD says conversation is over
        if self._conversation_start and not vad_state.is_conversation:
            # VAD has already handled cooldown timing
            self._end_conversation(current_time, vad_state.speech_segments)
    
    def _end_conversation(self, end_time: float, speech_segments: list) -> None:
        """End current conversation and emit message.
        
        Args:
            end_time: Time conversation ended
            speech_segments: List of speech segments from VAD
        """
        if not self._conversation_start:
            logger.warning("Trying to end conversation that hasn't started")
            return
        
        logger.info("Ending conversation")
        
        try:
            # Create conversation message
            message = ConversationMessage(
                id=str(uuid.uuid4()),
                audio_data=np.array(self._current_audio),
                start_time=self._conversation_start,
                end_time=end_time,
                metadata={
                    'sample_rate': self.audio_processor.sample_rate,
                    'speech_segments': speech_segments,
                    'first_speech_time': self._conversation_start  # When we started recording
                }
            )
            
            # Send to callback if provided
            if self.on_conversation:
                self.on_conversation(message)
            
        except Exception as e:
            logger.error(f"Error creating conversation message: {e}")
            
        finally:
            # Reset state
            self._conversation_start = None
            self._current_audio = []
    
    def start(self) -> None:
        """Start conversation detection."""
        if self._running:
            logger.warning("Conversation detector already running")
            return
        
        logger.info("Starting conversation detector")
        self._running = True
        
        # Register audio callback
        self.audio_processor.add_callback(self._handle_audio)
    
    def stop(self) -> None:
        """Stop conversation detection."""
        if not self._running:
            return
        
        logger.info("Stopping conversation detector")
        self._running = False
        
        # Clean up
        self.audio_processor.remove_callback(self._handle_audio)
        self.vad_processor.stop()
        
        logger.info("Conversation detector stopped")
