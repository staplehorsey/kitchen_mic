"""Audio capture module for Kitchen Mic.

Handles raw audio capture from network stream with dual-stream processing:
- Original 44kHz high-fidelity stream for storage
- Downsampled 16kHz stream for voice activity detection
"""

import logging
import socket
import threading
import time
import signal
from typing import Optional, Tuple, Callable
import struct
import errno

import numpy as np
import librosa

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def visualize_audio_level(audio_chunk: np.ndarray, width: int = 50) -> str:
    """Create a visual representation of audio level.
    
    Args:
        audio_chunk: Audio samples
        width: Width of visualization
        
    Returns:
        ASCII visualization of audio level
    """
    level = float(np.abs(audio_chunk).mean())
    normalized = min(1.0, level * 5)  # Amplify for better visibility
    bars = int(normalized * width)
    return f"[{'=' * bars}{' ' * (width - bars)}] ({level:.4f})"

class AudioCapture:
    """Captures audio from network stream and provides dual-stream processing."""
    
    def __init__(
        self,
        host: str = "staple.local",
        port: int = 12345,
        buffer_size: int = 4096,
        original_rate: int = 44000,  # Match server's sample rate
        target_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 512,  # Match server's chunk size
        visualize: bool = True
    ):
        """Initialize audio capture.
        
        Args:
            host: Source audio stream hostname
            port: Source audio stream port
            buffer_size: Socket buffer size
            original_rate: Original sample rate (Hz)
            target_rate: Target sample rate for downsampling (Hz)
            channels: Number of audio channels
            chunk_size: Audio chunk size in samples
            visualize: Whether to show audio level visualization
        """
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.original_rate = original_rate
        self.target_rate = target_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.visualize = visualize
        
        # Socket and thread state
        self.socket: Optional[socket.socket] = None
        self.running = False
        self.capture_thread: Optional[threading.Thread] = None
        
        # Connection state
        self.connected = False
        self.last_data_time = 0
        self.reconnect_delay = 1.0
        self.max_reconnect_delay = 5.0
        self.data_timeout = 2.0  # Consider connection dead after 2 seconds of no data
        
        # Audio format info (float32)
        self.bytes_per_sample = channels * 4  # 4 bytes per float32
        self.samples_per_buffer = buffer_size // self.bytes_per_sample
        
        # Callbacks for audio data
        self._callbacks = []
        self._lock = threading.Lock()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def add_callback(self, callback: Callable[[float, np.ndarray, np.ndarray], None]) -> None:
        """Add callback for audio data.
        
        Args:
            callback: Function(timestamp, original_chunk, downsampled_chunk)
                timestamp: Current wall clock time
                original_chunk: Original audio data (44kHz)
                downsampled_chunk: Downsampled audio data (16kHz)
        """
        with self._lock:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[float, np.ndarray, np.ndarray], None]) -> None:
        """Remove callback for audio data."""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def _connect(self) -> bool:
        """Establish connection to audio source.
        
        Returns:
            True if connection successful
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            self.last_data_time = time.time()
            self.reconnect_delay = 1.0  # Reset delay on successful connection
            logger.info(f"Connected to audio source at {self.host}:{self.port}")
            return True
        
        except socket.error as e:
            if e.errno == errno.ECONNREFUSED:
                logger.warning(f"Connection refused to {self.host}:{self.port}")
            else:
                logger.error(f"Socket error: {e}")
            self.socket = None
            self.connected = False
            return False
    
    def _process_audio(self, data: bytes) -> None:
        """Process raw audio data and notify callbacks.
        
        Args:
            data: Raw audio bytes
        """
        try:
            # Convert to float32 array
            audio = np.frombuffer(data, dtype=np.float32)
            
            # Visualize audio level
            if self.visualize and len(audio) > 0:
                logger.info(f"Audio Level: {visualize_audio_level(audio)}")
            
            # Current wall clock time
            current_time = time.time()
            
            # Downsample for VAD
            downsampled = librosa.resample(
                audio,
                orig_sr=self.original_rate,
                target_sr=self.target_rate
            )
            
            # Notify callbacks
            with self._lock:
                for callback in self._callbacks:
                    try:
                        callback(current_time, audio, downsampled)
                    except Exception as e:
                        logger.error(f"Error in audio callback: {e}")
            
            self.last_data_time = current_time
            
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
    
    def _capture_loop(self) -> None:
        """Main capture loop."""
        while self.running:
            try:
                # Check connection
                if not self.connected:
                    if not self._connect():
                        time.sleep(self.reconnect_delay)
                        self.reconnect_delay = min(
                            self.reconnect_delay * 2,
                            self.max_reconnect_delay
                        )
                        continue
                
                # Read data
                data = self.socket.recv(self.buffer_size)
                if not data:
                    logger.warning("No data received, reconnecting...")
                    self._cleanup()
                    continue
                
                # Process audio
                self._process_audio(data)
                
            except socket.error as e:
                logger.error(f"Socket error: {e}")
                self._cleanup()
                time.sleep(1.0)
            
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                time.sleep(0.1)
    
    def _cleanup(self) -> None:
        """Clean up socket connection."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.socket = None
        self.connected = False
    
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    def start(self) -> None:
        """Start audio capture."""
        if self.running:
            logger.warning("Audio capture already running")
            return
        
        logger.info("Starting audio capture")
        self.running = True
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.start()
    
    def stop(self) -> None:
        """Stop audio capture."""
        if not self.running:
            return
        
        logger.info("Stopping audio capture")
        self.running = False
        
        # Clean up
        self._cleanup()
        
        # Wait for thread
        if self.capture_thread:
            self.capture_thread.join(timeout=5.0)
            self.capture_thread = None
        
        logger.info("Audio capture stopped")
    
    @property
    def sample_rate(self) -> int:
        """Get the original sample rate."""
        return self.original_rate
