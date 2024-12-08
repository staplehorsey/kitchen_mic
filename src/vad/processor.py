"""Voice Activity Detection processor using Silero VAD and WebRTC.

This module handles real-time voice activity detection using the Silero VAD model and WebRTC for streaming.
It processes 16kHz mono audio and provides configurable parameters for detection.
"""

import logging
import time
from pathlib import Path
from typing import Optional, List, Tuple, Callable, NamedTuple
from dataclasses import dataclass
import threading
import collections
from queue import Queue, Empty, Full
import datetime

import numpy as np
import torch
import torch.hub

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class VADConfig:
    """Voice Activity Detection configuration."""
    threshold: float = 0.3  # Speech probability threshold
    window_size_samples: int = 512  # Window size in samples (must be 512 for 16kHz)
    min_speech_duration_ms: int = 250  # Minimum speech duration
    min_silence_duration_ms: int = 3000  # Cooldown period for speech (3 seconds)
    speech_pad_ms: int = 100  # Padding around speech segments
    sample_rate: int = 16000  # Expected sample rate
    
    # Conversation detection parameters
    conversation_window_sec: float = 10.0  # Window to aggregate speech probabilities
    conversation_threshold: float = 0.3  # Threshold for conversation detection
    conversation_cooldown_sec: float = 15.0  # End conversation after 15 seconds of silence
    min_conversation_duration_sec: float = 5.0  # Minimum duration to consider it a conversation

class VADState(NamedTuple):
    """Current state of voice activity detection."""
    speech_probability: float
    is_speech: bool
    is_conversation: bool
    conversation_start: Optional[float]  # When conversation was officially started (after 5s of speech)
    conversation_end: Optional[float]    # When conversation ended (after 15s of no speech)
    first_speech_time: Optional[float]   # When first speech was detected
    last_speech_time: Optional[float]    # When last speech was detected
    speech_segments: List[Tuple[float, float]]  # List of (start_time, duration) for speech segments

class VADProcessor:
    """Process audio chunks and detect voice activity."""

    def __init__(self, config: Optional[VADConfig] = None):
        """Initialize VAD processor.
        
        Args:
            config: VAD configuration parameters
        """
        self.config = config or VADConfig()
        
        # Load model
        logger.info("Loading Silero VAD model...")
        torch.set_num_threads(1)  # Optimize for real-time processing
        
        # Load model from hub
        self.model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.model = self.model.float()
        
        # Processing state
        self._speech_started = False
        self._speech_start_time = None
        self._last_speech_end_time = None
        self._conversation_started = False
        self._conversation_start_time = None
        self._last_speech_prob_time = None
        self._first_speech_time = None
        
        # Input queue and processing thread
        self._input_queue = Queue(maxsize=100)
        self._lock = threading.Lock()
        self._running = True
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()
        
        # State tracking
        window_size = int(self.config.conversation_window_sec * self.config.sample_rate / self.config.window_size_samples)
        self._recent_probs = collections.deque(maxlen=100)
        self._conversation_probs = collections.deque(maxlen=window_size)
        self._speech_segments = []
        self._current_state = VADState(0.0, False, False, None, None, None, None, [])
        
        # Audio buffer for VAD processing
        self._audio_buffer = collections.deque(maxlen=self.config.window_size_samples * 4)  # Allow for larger buffer
        
        logger.info("VAD processor initialized")
    
    def add_audio(self, audio_chunk: np.ndarray) -> None:
        """Add audio chunk for processing.
        
        Args:
            audio_chunk: Audio samples (16kHz, mono, float32)
        """
        if not self._running:
            return
            
        try:
            self._input_queue.put_nowait(audio_chunk)
        except Full:
            logger.warning("Input queue full, dropping audio chunk")
    
    def get_state(self) -> VADState:
        """Get current VAD state.
        
        Returns:
            Current VADState with all detection information
        """
        with self._lock:
            return self._current_state
    
    def _process_loop(self) -> None:
        """Background processing loop."""
        while self._running:
            try:
                # Get audio chunk from queue with timeout
                try:
                    audio_chunk = self._input_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Process audio and update state
                self._process_audio(audio_chunk)
                
            except Exception as e:
                logger.error(f"Error in VAD processing: {e}")
                time.sleep(0.1)
    
    def _process_audio(self, audio_chunk: np.ndarray) -> None:
        """Process single audio chunk and update state."""
        current_time = time.time()
        
        # Convert input to float32 if needed
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Add to buffer
        self._audio_buffer.extend(audio_chunk)
        
        # Process complete chunks of exactly window_size_samples
        while len(self._audio_buffer) >= self.config.window_size_samples:
            # Get the required number of samples
            chunk = np.array(list(self._audio_buffer)[:self.config.window_size_samples])
            
            # Remove processed samples
            for _ in range(self.config.window_size_samples):
                self._audio_buffer.popleft()
            
            # Convert to tensor and get probability
            try:
                with torch.no_grad():
                    tensor = torch.from_numpy(chunk).float()
                    prob = float(self.model(tensor, self.config.sample_rate).item())
                    
                    # Apply exponential smoothing to probability
                    if self._recent_probs:
                        last_prob = self._recent_probs[-1][1]
                        alpha = 0.3  # Smoothing factor
                        prob = alpha * prob + (1 - alpha) * last_prob
                    
                    # Update probability windows
                    self._recent_probs.append((current_time, prob))
                    self._conversation_probs.append(prob)
                    
                    # Update speech detection state
                    first_speech_time = None
                    if prob >= self.config.threshold:
                        # Speech detected
                        if not self._speech_started:
                            self._speech_started = True
                            self._speech_start_time = current_time
                            if not self._first_speech_time:
                                self._first_speech_time = current_time
                                first_speech_time = current_time
                        self._last_speech_prob_time = current_time
                    elif self._speech_started:
                        # Check if silence duration exceeds cooldown
                        silence_duration = current_time - self._last_speech_prob_time
                        if silence_duration >= self.config.min_silence_duration_ms / 1000:
                            # End speech segment
                            speech_duration = current_time - self._speech_start_time
                            if speech_duration >= self.config.min_speech_duration_ms / 1000:
                                self._speech_segments.append((self._speech_start_time, speech_duration))
                            self._speech_started = False
                            self._last_speech_end_time = current_time
                    
                    # Update conversation state
                    conversation_start = None
                    conversation_end = None
                    
                    if not self._conversation_started:
                        # Check continuous speech duration
                        if self._first_speech_time and not self._conversation_start_time:
                            speech_duration = current_time - self._first_speech_time
                            if speech_duration >= self.config.min_conversation_duration_sec:
                                self._conversation_started = True
                                self._conversation_start_time = self._first_speech_time  # Start from first speech
                                conversation_start = self._conversation_start_time
                                logger.info(f"Conversation started at {datetime.datetime.fromtimestamp(self._conversation_start_time).strftime('%H:%M:%S')}")
                    else:
                        # Check if we should end the conversation
                        silence_duration = current_time - self._last_speech_prob_time
                        
                        # End conversation after consecutive silence
                        if silence_duration >= self.config.conversation_cooldown_sec:
                            if self._conversation_start_time:
                                logger.info(f"Conversation ended at {datetime.datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}")
                                self._conversation_started = False
                                conversation_end = current_time
                                self._conversation_start_time = None
                                # Reset first speech time for next conversation
                                self._first_speech_time = None
                    
                    # Update current state atomically
                    with self._lock:
                        last_speech = None
                        if self._speech_started:
                            last_speech = self._last_speech_prob_time
                        elif self._last_speech_end_time:
                            last_speech = self._last_speech_end_time
                            
                        self._current_state = VADState(
                            speech_probability=prob,
                            is_speech=self._speech_started,
                            is_conversation=self._conversation_started,
                            conversation_start=self._conversation_start_time,
                            conversation_end=conversation_end,
                            first_speech_time=first_speech_time or self._first_speech_time,  # Use existing if no new
                            last_speech_time=last_speech,
                            speech_segments=list(self._speech_segments)
                        )
            except Exception as e:
                logger.error(f"Error processing audio chunk: {e}")
                # Keep the state as is and continue processing
    
    def _process_speech_state(self, is_speech: bool, current_time: float):
        """Process speech state changes and update conversation tracking."""
        if is_speech:
            # Update speech timing
            if not self._speech_start_time:
                self._speech_start_time = current_time
                logger.info(f"First speech detected at {datetime.datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}")
            
            self._last_speech_prob_time = current_time
            self._silence_duration = 0.0
            
            if not self._conversation_started:
                # Check if we've had enough continuous speech
                self._speech_duration = current_time - self._speech_start_time
                if self._speech_duration >= self.config.min_conversation_duration_sec:
                    self._conversation_started = True
                    self._conversation_start_time = self._speech_start_time  # Use first speech time
                    logger.info(f"Conversation started at {datetime.datetime.fromtimestamp(self._conversation_start_time).strftime('%H:%M:%S')}")
        else:
            # Update silence timing
            if self._last_speech_prob_time:
                self._silence_duration = current_time - self._last_speech_prob_time
                
                # Check if silence threshold reached
                if self._conversation_started and self._silence_duration >= self.config.conversation_cooldown_sec:
                    self._conversation_end = current_time
                    logger.info(f"Conversation ended at {datetime.datetime.fromtimestamp(current_time).strftime('%H:%M:%S')}")
                    
                    # Reset state for next conversation
                    self._conversation_started = False
                    self._speech_start_time = None
                    self._conversation_start_time = None
                    self._last_speech_prob_time = None
                    self._speech_duration = None
                    self._silence_duration = None
    
    def stop(self) -> None:
        """Stop the processor and clean up."""
        if not self._running:
            return
            
        self._running = False
        
        # Wait for processing thread to finish
        if self._process_thread and self._process_thread.is_alive():
            self._process_thread.join(timeout=1.0)
        
        # Clear queues and state
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except Empty:
                break
        
        with self._lock:
            self._recent_probs.clear()
            self._conversation_probs.clear()
            self._speech_segments.clear()
            self._current_state = VADState(0.0, False, False, None, None, None, None, [])
